package org.vitrivr.cottontail.database.index.lsh.superbit

import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.Column
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.general.TransactionStatus
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexTransaction
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.hash.NonUniqueHashIndex
import org.vitrivr.cottontail.database.index.hash.UniqueHashIndex
import org.vitrivr.cottontail.database.index.lsh.LSHIndex
import org.vitrivr.cottontail.database.index.lucene.LuceneIndex
import org.vitrivr.cottontail.database.queries.components.AtomicBooleanPredicate
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.math.knn.metrics.CosineDistance
import org.vitrivr.cottontail.model.basics.*
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.exceptions.StoreException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.types.VectorValue
import org.vitrivr.cottontail.utilities.extensions.read
import org.vitrivr.cottontail.utilities.extensions.write

/**
 * Represents a LSH based index in the Cottontail DB data model. An [Index] belongs to an [Entity] and can be used to
 * index one to many [Column]s. Usually, [Index]es allow for faster data access. They process [Predicate]s and return
 * [Recordset]s.
 *
 * @author Manuel Huerbin & Ralph Gasser
 * @version 1.2
 */
class SuperBitLSHIndex<T : VectorValue<*>>(name: Name.IndexName, parent: Entity, columns: Array<ColumnDef<*>>, params: Map<String, String>? = null) : LSHIndex<T>(name, parent, columns) {

    companion object {
        const val CONFIG_NAME = "lsh_config"
        const val CONFIG_NAME_STAGES = "stages"
        const val CONFIG_NAME_BUCKETS = "buckets"
        const val CONFIG_NAME_SEED = "seed"
        private const val CONFIG_DEFAULT_STAGES = 3
        private const val CONFIG_DEFAULT_BUCKETS = 10
        private val LOGGER = LoggerFactory.getLogger(SuperBitLSHIndex::class.java)
    }

    /** Internal configuration object for [SuperBitLSHIndex]. */
    val config = this.db.atomicVar(CONFIG_NAME, SuperBitLSHIndexConfig.Serializer).createOrOpen()

    override val type = IndexType.SUPERBIT_LSH

    init {
        if (!columns.all { it.type.vector }) {
            throw DatabaseException.IndexNotSupportedException(name, "Because only vector columns are supported for SuperBitLSHIndex.")
        }
        if (params != null) {
            val buckets = params[CONFIG_NAME_BUCKETS]?.toIntOrNull() ?: CONFIG_DEFAULT_BUCKETS
            val stages = params[CONFIG_NAME_STAGES]?.toIntOrNull() ?: CONFIG_DEFAULT_STAGES
            val seed = params[CONFIG_NAME_SEED]?.toLongOrNull() ?: System.currentTimeMillis()
            this.config.set(SuperBitLSHIndexConfig(buckets, stages, seed))
        } else {
            if (config.get() == null) {
                throw StoreException("No parameters supplied, and the config from disk was also empty.")
            }
        }
    }

    /**
     * Checks if the provided [Predicate] can be processed by this instance of [SuperBitLSHIndex].
     *
     * @param predicate The [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate): Boolean = if (predicate is KnnPredicate<*>) {
        predicate.columns.first() == this.columns[0] && predicate.distance is CosineDistance
    } else {
        false
    }

    /**
     * Calculates the cost estimate of this [SuperBitLSHIndex] processing the provided [Predicate].
     *
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate): Cost = if (canProcess(predicate)) {
        Cost.ZERO /* TODO: Determine. */
    } else {
        Cost.INVALID
    }

    /**
     * Returns true since [SuperBitLSHIndex] supports incremental updates.
     *
     * @return True
     */
    override fun supportsIncrementalUpdate(): Boolean = true

    /**
     * Opens and returns a new [IndexTransaction] object that can be used to interact with this [Index].
     *
     * @param parent If the [Entity.Tx] that requested the [IndexTransaction].
     */
    override fun begin(parent: Entity.Tx): IndexTransaction = Tx(parent)

    /**
     * A [IndexTransaction] that affects this [Index].
     */
    private inner class Tx(parent: Entity.Tx) : Index.Tx(parent) {

        /**
         * (Re-)builds the [SuperBitLSHIndex].
         */
        override fun rebuild() = this.localLock.read {

            checkValidForWrite()

            LOGGER.trace("Rebuilding SB-LSH index {}", this@SuperBitLSHIndex.name)

            /* LSH. */
            val specimen = this.acquireSpecimen(this.parent) ?: return
            val lsh = SuperBitLSH(this@SuperBitLSHIndex.config.get().stages, this@SuperBitLSHIndex.config.get().buckets, this.columns[0].logicalSize, this@SuperBitLSHIndex.config.get().seed, specimen)

            /* (Re-)create index entries locally. */
            val local = Array(this@SuperBitLSHIndex.config.get().buckets) { mutableListOf<Long>() }
            this.parent.scan().use { s->
                s.forEach {
                    val record = this.parent.read(it)
                    val value = record[this.columns[0]]
                    if (value is VectorValue<*>) {
                        val bucket: Int = lsh.hash(value).last()
                        local[bucket].add(it)
                    }
                }
            }

            /* Replace existing map. */
            this@SuperBitLSHIndex.map.clear()
            local.forEachIndexed { bucket, list -> this@SuperBitLSHIndex.map[bucket] = list.toLongArray() }
        }

        /**
         * Updates the [SuperBitLSHIndex] with the provided [DataChangeEvent]s. This method determines,
         * whether the [Record] affected by the [DataChangeEvent] should be added or updated
         *
         * @param update Collection of [DataChangeEvent]s to process.
         */
        override fun update(update: Collection<DataChangeEvent>) = this.localLock.read {
            TODO()
        }

        /**
         * Performs a lookup through this [SuperBitLSHIndex] and returns a [CloseableIterator] of
         * all [TupleId]s that match the [Predicate]. Only supports [KnnPredicate]s.
         *
         * The [CloseableIterator] is not thread safe!
         *
         * <strong>Important:</strong> It remains to the caller to close the [CloseableIterator]
         *
         * @param predicate The [Predicate] for the lookup*
         * @return The resulting [CloseableIterator]
         */
        override fun filter(predicate: Predicate) = object : CloseableIterator<TupleId> {

            /** Cast [AtomicBooleanPredicate] (if such a cast is possible).  */
            val predicate = if (predicate is KnnPredicate<*>) {
                predicate
            } else {
                throw QueryException.UnsupportedPredicateException("Index '${this@SuperBitLSHIndex.name}' (LSH Index) does not support predicates of type '${predicate::class.simpleName}'.")
            }

            /* Performs some sanity checks. */
            init {
                checkValidForRead()

                if (this.predicate.columns.first() != this@SuperBitLSHIndex.columns[0] || this.predicate.distance !is CosineDistance) {
                    throw QueryException.UnsupportedPredicateException("Index '${this@SuperBitLSHIndex.name}' (lsh-index) does not support the provided predicate.")
                }
            }

            /** Generates a shared lock on the enclosing [Tx]. This lock is kept until the [CloseableIterator] is closed. */
            private val stamp = this@Tx.localLock.readLock()

            /** Flag indicating whether this [CloseableIterator] has been closed. */
            @Volatile
            private var closed = false

            /** [SuperBitLSH] data structure to calculate the bucket index. */
            private val lsh = SuperBitLSH(this@SuperBitLSHIndex.config.get().stages, this@SuperBitLSHIndex.config.get().buckets, this@SuperBitLSHIndex.columns.first().logicalSize, this@SuperBitLSHIndex.config.get().seed, this.predicate.query.first())

            /** List of [TupleId]s returned by this [CloseableIterator]. */
            private val tupleIds = this.predicate.query.mapNotNull {
                this@SuperBitLSHIndex.map[this.lsh.hash(it).last()]
            }.flatMap {
                it.asIterable()
            }.toMutableList()

            override fun hasNext(): Boolean {
                check(!this.closed) { "Illegal invocation of hasNext(): This CloseableIterator has been closed." }
                return this.tupleIds.isNotEmpty()
            }

            override fun next(): TupleId {
                check(!this.closed) { "Illegal invocation of next(): This CloseableIterator has been closed." }
                return this.tupleIds.removeFirst()
            }

            override fun close() {
                if (!this.closed) {
                    this@Tx.localLock.unlock(this.stamp)
                    this.closed = true
                }
            }
        }

        /** Performs the actual COMMIT operation by rolling back the [DB]. */
        override fun performCommit() {
            this@SuperBitLSHIndex.db.commit()
        }

        /** Performs the actual ROLLBACK operation by rolling back the [DB]. */
        override fun performRollback() {
            this@SuperBitLSHIndex.db.rollback()
        }

        override fun cleanup() {
            /* No Op. */
        }

        /**
         * Tries to find a specimen of the [VectorValue] in the [Entity] underpinning this [SuperBitLSHIndex]
         *
         * @param tx [Entity.Tx] used to read from [Entity]
         * @return A specimen of the [VectorValue] that should be indexed.
         */
        private fun acquireSpecimen(tx: Entity.Tx): VectorValue<*>? {
            for (index in 2L until tx.count()) {
                val read = tx.read(index)[this.columns[0]]
                if (read is VectorValue<*>) {
                    return read
                }
            }
            return null
        }
    }
}