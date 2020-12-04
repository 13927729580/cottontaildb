package org.vitrivr.cottontail.database.index

import org.mapdb.DBMaker
import org.mapdb.HTreeMap
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.Column
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.lsh.LSHIndex
import org.vitrivr.cottontail.database.index.random.RandomIndexConfig
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

            val recordset = Recordset(this.produces, (predicate.k * predicate.query.size).toLong())

            val knns = predicate.query.map { _ ->
                if (predicate.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection(predicate.k)
            }

            predicate.query.indices.toList().parallelStream().forEach { i ->
                val knn = knns[i]
                val query = predicate.query[i] as Complex32VectorValue
                tids.forEach { tid ->
                    val distance = predicate.distance.invoke(query, tx.read(tid)[columns[0]] as Complex32VectorValue)
                    if (knn.size < knn.k || knn.peek()!!.second > distance)
                        knn.offer(ComparablePair(tid, distance))
                }
            }

            /* Generate record set .*/

            for (knn in knns) {
                for (j in 0 until knn.size) {
                    recordset.addRowUnsafe(knn[j].first, arrayOf(knn[j].second))
                }
            }
            LOGGER.debug("Done.")
            return KnnUtilities.selectToRecordset(this.produces.first(), knns)
        } else {
            throw QueryException.UnsupportedPredicateException("Index '${this.name}' (random Index) does not support predicates of type '${predicate::class.simpleName}'.")
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
