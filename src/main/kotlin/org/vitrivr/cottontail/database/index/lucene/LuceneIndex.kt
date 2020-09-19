package org.vitrivr.cottontail.database.index.lucene

import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document.*
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.IndexReader
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.store.Directory
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.store.NativeFSLockFactory
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.events.DataChangeEventType
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexTransaction
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.hash.UniqueHashIndex
import org.vitrivr.cottontail.database.queries.components.*
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.database.queries.planning.cost.Costs
import org.vitrivr.cottontail.model.basics.*
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.values.StringValue
import org.vitrivr.cottontail.utilities.extensions.read
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path

/**
 * Represents a Apache Lucene based index in the Cottontail DB data model. The [LuceneIndex] allows
 * for string comparisons using the EQUAL or LIKE operator.
 *
 * @author Luca Rossetto & Ralph Gasser
 * @version 1.2
 */
class LuceneIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>) : Index() {

    companion object {
        /** [ColumnDef] of the _tid column. */
        const val TID_COLUMN = "_tid"

        /**
         * Maps a [Record] to a [Document] that can be processed by Lucene.
         *
         * @param record The [Record]
         * @return The resulting [Document]
         */
        private fun documentFromRecord(record: Record): Document = Document().apply {
            add(NumericDocValuesField(TID_COLUMN, record.tupleId))
            add(StoredField(TID_COLUMN, record.tupleId))
            record.columns.forEach {
                val value = record[it]
                if (value is StringValue) {
                    add(TextField("${it.name}_txt", value.value, Field.Store.NO))
                    add(StringField("${it.name}_str", value.value, Field.Store.NO))
                }
            }
        }

        private val LOGGER = LoggerFactory.getLogger(LuceneIndex::class.java)
    }

    /** The [LuceneIndex] implementation produces an additional score column. */
    override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("score"), ColumnType.forName("FLOAT")))

    /** The path to the directory that contains the data for this [LuceneIndex]. */
    override val path: Path = this.parent.path.resolve("idx_lucene_$name")

    /** The type of this [Index]. */
    override val type: IndexType = IndexType.LUCENE

    /** Flag indicating whether or not this [LuceneIndex] is open and usable. */
    @Volatile
    override var closed: Boolean = false
        private set

    /** The [Directory] containing the data for this [LuceneIndex]. */
    private val directory: Directory = FSDirectory.open(this.path, NativeFSLockFactory.getDefault())

    /** Initial commit in case writer was created. */
    init {
        val writer = IndexWriter(this.directory, IndexWriterConfig(StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND).setCommitOnClose(true))
        writer.close()
    }

    /** The [IndexReader] instance used for accessing the [LuceneIndex]. */
    private var indexReader = DirectoryReader.open(this.directory)

    /**
     * Returns true, if the [LuceneIndex] supports incremental updates, and false otherwise.
     *
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate(): Boolean = true

    /**
     * Opens and returns a new [IndexTransaction] object that can be used to interact with this [Index].
     *
     * @param parent If the [Entity.Tx] that requested the [IndexTransaction].
     */
    override fun begin(parent: Entity.Tx): IndexTransaction = Tx(parent)

    /**
     * Closes this [LuceneIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!this.closed) {
            this.indexReader.close()
            this.directory.close()
            this.closed = true
        }
    }

    /**
     * A [IndexTransaction] that affects this [Index].
     */
    private inner class Tx(parent: Entity.Tx) : Index.Tx(parent) {
        /**
         * (Re-)builds the [LuceneIndex].
         */
        override fun rebuild() = this.localLock.read {
            LOGGER.trace("Rebuilding lucene index {}", name)
            val writer = IndexWriter(this@LuceneIndex.directory, IndexWriterConfig(StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.APPEND).setMaxBufferedDocs(100_000).setCommitOnClose(true))
            writer.deleteAll()
            var count = 0
            this.parent.scan().forEach { tid ->
                val record = this.parent.read(tid)
                writer.addDocument(documentFromRecord(record))
                count++
            }
            writer.close()

            /* Open new IndexReader and close new one. */
            val oldReader = this@LuceneIndex.indexReader
            this@LuceneIndex.indexReader = DirectoryReader.open(this@LuceneIndex.directory)
            oldReader.close()
        }

        /**
         * Updates the [UniqueHashIndex] with the provided [DataChangeEvent]s. This method determines,
         * whether the [Record] affected by the [DataChangeEvent] should be added or updated
         *
         * @param update Collection of [DataChangeEvent]s to process.
         */
        override fun update(update: Collection<DataChangeEvent>) = this.localLock.read {

            this.checkValidForWrite()

            val writer = IndexWriter(this@LuceneIndex.directory, IndexWriterConfig(StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.APPEND).setMaxBufferedDocs(100_000).setCommitOnClose(true))

            /* Define action for inserting an entry based on a DataChangeEvent. */
            fun atomicInsert(event: DataChangeEvent) {
                writer.addDocument(documentFromRecord(event.new!!))
            }


            /* Define action for deleting an entry based on a DataChangeEvent. */
            fun atomicDelete(event: DataChangeEvent) {
                writer.deleteDocuments(NumericDocValuesField.newSlowExactQuery(TID_COLUMN, event.old!!.tupleId))
            }

            /* Process the DataChangeEvents. */
            for (event in update) {
                when (event.type) {
                    DataChangeEventType.INSERT -> atomicInsert(event)
                    DataChangeEventType.UPDATE -> {
                        if (event.new?.get(this.columns[0]) != event.old?.get(this.columns[0])) {
                            atomicDelete(event)
                            atomicInsert(event)
                        }
                    }
                    DataChangeEventType.DELETE -> atomicDelete(event)
                    else -> {
                    }
                }
            }

            val oldReader = this@LuceneIndex.indexReader
            this@LuceneIndex.indexReader = DirectoryReader.open(this@LuceneIndex.directory)
            oldReader.close()
        }


        /**
         * Performs a lookup through this [LuceneIndex.Tx] and returns a [CloseableIterator] of
         * all [TupleId]s that match the [Predicate]. Only supports [AtomicBooleanPredicate]s.
         *
         * The [CloseableIterator] is not thread safe!
         *
         * <strong>Important:</strong> It remains to the caller to close the [CloseableIterator]
         *
         * @param predicate The [Predicate] for the lookup*
         * @return The resulting [CloseableIterator]
         */
        override fun filter(predicate: Predicate) = object : CloseableIterator<TupleId> {
            /** Cast [BooleanPredicate] (if such a cast is possible). */
            private val predicate = if (predicate !is BooleanPredicate) {
                throw QueryException.UnsupportedPredicateException("Index '${this@LuceneIndex.name}' (lucene index) does not support predicates of type '${predicate::class.simpleName}'.")
            } else {
                predicate
            }

            /* Performs some sanity checks. */
            init {
                checkValidForRead()

                if (!this.predicate.columns.all { this@LuceneIndex.columns.contains(it) })
                    throw QueryException.UnsupportedPredicateException("Index '${this@LuceneIndex.name}' (lucene-index) is lacking certain fields the provided predicate requires.")

                if (!this.predicate.atomics.all { it.operator == ComparisonOperator.LIKE || it.operator == ComparisonOperator.EQUAL })
                    throw QueryException.UnsupportedPredicateException("Index '${this@LuceneIndex.name}' (lucene-index) can only process LIKE comparisons.")
            }

            /** Generates a shared lock on the enclosing [Tx]. This lock is kept until the [CloseableIterator] is closed. */
            private val stamp = this@Tx.localLock.readLock()

            /** Number of [TupleId]s returned by this [CloseableIterator]. */
            @Volatile
            private var returned = 0

            /** Flag indicating whether this [CloseableIterator] has been closed. */
            @Volatile
            private var closed = false

            /* Lucene query. */
            private val query = when (this.predicate) {
                is AtomicBooleanPredicate<*> -> this.predicate.toLuceneQuery()
                is CompoundBooleanPredicate -> this.predicate.toLuceneQuery()
            }

            /** [IndexSearcher] instance used for lookup. */
            private val searcher = IndexSearcher(this@LuceneIndex.indexReader)

            /* Execute query and add results. */
            private val results = this.searcher.search(this.query, Integer.MAX_VALUE)


            /**
             * Returns `true` if the iteration has more elements.
             */
            override fun hasNext(): Boolean {
                check(!this.closed) { "Illegal invocation of next(): This CloseableIterator has been closed." }
                return this.returned < this.results.totalHits
            }

            /**
             * Returns the next element in the iteration.
             */
            override fun next(): TupleId {
                check(!this.closed) { "Illegal invocation of next(): This CloseableIterator has been closed." }
                val doc = this.searcher.doc(this.results.scoreDocs[this.returned++].doc)
                return doc[TID_COLUMN].toLong()
            }

            /**
             * Closes this [CloseableIterator] and releases all locks and resources associated with it.
             */
            override fun close() {
                if (!this.closed) {
                    this@Tx.localLock.unlock(this.stamp)
                    this.closed = true
                }
            }
        }
    }

    /**
     * Checks if this [LuceneIndex] can process the given [Predicate].
     *
     * @param predicate [Predicate] to test.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate): Boolean = if (predicate is BooleanPredicate) {
        predicate.columns.all { this.columns.contains(it) } && predicate.atomics.all { it.operator == ComparisonOperator.LIKE || it.operator == ComparisonOperator.EQUAL }
    } else {
        false
    }

    /**
     * Calculates the cost estimate of this [UniqueHashIndex] processing the provided [Predicate].
     *
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate): Cost = when {
        canProcess(predicate) -> {
            val searcher = IndexSearcher(this.indexReader)
            var cost = Cost.ZERO
            predicate.columns.forEach {
                cost += Cost(Costs.DISK_ACCESS_READ, Costs.DISK_ACCESS_READ, it.physicalSize.toFloat()) * searcher.collectionStatistics(it.name.simple).sumTotalTermFreq()
            }
            cost
        }
        else -> Cost.INVALID
    }
}
