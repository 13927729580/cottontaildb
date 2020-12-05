package org.vitrivr.cottontail.database.index.random

import org.mapdb.DBMaker
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.Column
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.lsh.LSHIndex
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.CosineDistance
import org.vitrivr.cottontail.math.knn.metrics.RealInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.math.knn.selection.Selection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.Complex32VectorValue
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * Represents a random index in the Cottontail DB data model. An [Index] belongs to an [Entity] and can be used to
 * index one to many [Column]s. Usually, [Index]es allow for faster data access. They process [Predicate]s and return
 * [Recordset]s.
 * This index just returns a random set of tids for testing/comparing
 *
 * @author Gabriel Zihlmann
 * @version 1.0
 */
class RandomIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>, config: RandomIndexConfig? = null) : Index() {

    companion object {
        private const val CONFIG_NAME = "random_config"
        private val LOGGER = LoggerFactory.getLogger(RandomIndex::class.java)
    }
    final override val path: Path = this.parent.path.resolve("idx_random_$name.db")
    final override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("distance"), ColumnType.forName("DOUBLE")))


    /** The internal [DB] reference. */
    protected val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }

    /** Internal configuration object for [RandomIndex]. */
    val config: RandomIndexConfig
    val configOnDisk = this.db.atomicVar(CONFIG_NAME, RandomIndexConfig.Serializer).createOrOpen()

    override val type = IndexType.RANDOM

    val rng: SplittableRandom

    val tids = mutableListOf<Long>()
    private val tidsStore = db.hashMap("tids", Serializer.INTEGER, Serializer.LONG_ARRAY).counterEnable().createOrOpen()

    init {
        if (columns.size != 1) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing a single column")
        }
        if (!columns.all { it.type == ColumnType.forName("COMPLEX32_VEC") }) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing complex32 vector columns, not ${columns.first()::class.java}")
        }
        val cod = configOnDisk.get()
        if (config == null) {
            if (cod == null) {
                LOGGER.warn("No config supplied, but the config from disk was null!! Using a dummy config. Please consider this index invalid!")
                this.config = RandomIndexConfig(0.1, 1234L)
            } else {
                this.config = cod
            }
        } else {
            configOnDisk.set(config)
            this.config = config
        }

        rng = SplittableRandom(this.config.seed)
        tids.clear()
        tids.addAll(tidsStore.getOrDefault(0, longArrayOf()).toList())
    }

    /** Flag indicating if this [LSHIndex] has been closed. */
    @Volatile
    final override var closed: Boolean = false
        private set

    /**
     * Closes this [RandomIndex] and the associated data structures.
     */
    final override fun close() = this.globalLock.write {
        if (!this.closed) {
            this.db.close()
            this.closed = true
        }
    }
    /**
     * Performs a lookup through this [RandomIndex].
     *
     * @param predicate The [Predicate] for the lookup
     * @return The resulting [Recordset]
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        if (predicate is KnnPredicate<*>) {
            /* Guard: Only process predicates that are supported. */
            require(this.canProcess(predicate)) { throw QueryException.UnsupportedPredicateException("Index '${this.name}' does not support the provided predicate.") }

            LOGGER.debug("RandomIndex ${this.name.simple} Filtering ${predicate.query.size} queries.")

            val knns = filterParallel(predicate, tx)

            LOGGER.debug("Done.")
            return KnnUtilities.selectToRecordset(this.produces.first(), knns)
        } else {
            throw QueryException.UnsupportedPredicateException("Index '${this.name}' (random Index) does not support predicates of type '${predicate::class.simpleName}'.")
        }
    }

    private fun filterPartSimpleNoParallel(start: Int, endInclusive: Int, predicate: KnnPredicate<*>, tx: Entity.Tx): List<Selection<ComparablePair<Long, DoubleValue>>> {
        LOGGER.trace("filtering from $start to $endInclusive")
        val knns = predicate.query.map {
            if (predicate.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection(predicate.k)
        }
        for (t in start .. endInclusive) {
            val tid = tids[t]
            val vec = tx.read(tid)[columns.first()] as ComplexVectorValue<*>
            predicate.query.forEachIndexed { i, query ->
                val distance = predicate.distance(vec, query)
                if (knns[i].size < predicate.k || knns[i].peek()!!.second > distance) {
                    knns[i].offer(ComparablePair(tid, distance))
                }
            }
        }
        LOGGER.trace("done filtering from $start to $endInclusive")
        return knns
    }

    private fun filterParallel(predicate: KnnPredicate<*>, tx: Entity.Tx): List<Selection<ComparablePair<Long,DoubleValue>>> {
        //split signatures to threads
        val numThreads = Runtime.getRuntime().availableProcessors()
        val elemsPerThread = tids.size / numThreads
        LOGGER.debug("Filtering with $numThreads threads ($elemsPerThread TIDs per thread).")
        val remaining = tids.size % numThreads
        val exec = Executors.newFixedThreadPool(numThreads)
        val tasks = (0 until numThreads).map {
            Callable { filterPartSimpleNoParallel(it * elemsPerThread,
                    it * elemsPerThread + elemsPerThread - 1 + if (it == numThreads - 1) remaining else 0,
                    predicate, tx)}
        }
        val fresults = exec.invokeAll(tasks)
        val res = fresults.map { it.get()}
        exec.shutdownNow()
        // merge
        LOGGER.debug("Merging results")
        return res.reduce { acc, perThread ->
            (perThread zip acc).map { (knnPerThread, knnAcc) ->
                knnAcc.apply {
                    for (i in 0 until knnPerThread.size) offer(knnPerThread[i])
                }
            }
        }
    }

    /** (Re-)builds the [RandomIndex]. */
    override fun rebuild(tx: Entity.Tx) {
        LOGGER.debug("Rebuilding ${this.name}")
        tids.clear()
        tx.forEach {
            if (rng.nextDouble() < config.fraction)
                tids.add(it.tupleId)
        }
        /* Commit local database. */
        tidsStore[0] = tids.toLongArray()
        this.db.commit()
        LOGGER.debug("Done.")
    }

    /**
     * Updates the [RandomIndex] with the provided [Record]. This method determines, whether the [Record] should be added or updated
     *
     * @param update List of [DataChangeEvent]s that should be considered for the update.
     * @param tx: [Entity.Tx] used to make the change.
     */
    override fun update(update: Collection<DataChangeEvent>, tx: Entity.Tx) = try {
        // TODO
    } catch (e: Throwable) {
        this.db.rollback()
        throw e
    }

    /**
     * Checks if the provided [Predicate] can be processed by this instance of [RandomIndex].
     * note: only use the innerproduct distances with normalized vectors!
     *
     * @param predicate The [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate): Boolean =
        predicate is KnnPredicate<*>
        && predicate.columns.first() == this.columns[0]
        && (predicate.distance is CosineDistance
            || predicate.distance is RealInnerProductDistance
            || predicate.distance is AbsoluteInnerProductDistance)
        && predicate.query.all { it is ComplexVectorValue<*> }

    /**
     * Calculates the cost estimate of this [RandomIndex] processing the provided [Predicate].
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
     * Returns true since [RandomIndex] supports incremental updates.
     *
     * @return True
     */
    override fun supportsIncrementalUpdate(): Boolean = true
}
