package org.vitrivr.cottontail.database.index.lsh.superbit

import org.mapdb.HTreeMap
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.Column
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.lsh.LSHIndex
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities.selectToRecordset
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.CosineDistance
import org.vitrivr.cottontail.math.knn.metrics.RealInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.exceptions.StoreException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import kotlin.math.abs

/**
 * Represents a LSH based index in the Cottontail DB data model. An [Index] belongs to an [Entity] and can be used to
 * index one to many [Column]s. Usually, [Index]es allow for faster data access. They process [Predicate]s and return
 * [Recordset]s.
 *
 * The [NonBucketingSuperBitLSHIndex] uses a [SuperBit] instance to derive bit-signatures
 * from database and query vectors. The database and query bit-signatures
 * are compared and used to filter candidate database vectors that are likely to
 * have a small cosine distance (or inner product distnace for normalized vectors)
 *
 *
 * @author Gabriel Zihlmann
 * @version 0.1
 */
class NonBucketingSuperBitLSHIndex<T : VectorValue<*>> (name: Name.IndexName, parent: Entity, columns: Array<ColumnDef<*>>, config: NonBucketingSuperBitLSHIndexConfig? = null) : LSHIndex<T>(name, parent, columns) {

    companion object {
        const val CONFIG_NAME = "nbsblsh_config"
        private val LOGGER = LoggerFactory.getLogger("org.vitrivr.cottontail.database.index.lsh.superbit.NonBucketingSuperBitLSHIndex")
    }

    val config: NonBucketingSuperBitLSHIndexConfig
    val configOnDisk = this.db.atomicVar(CONFIG_NAME, NonBucketingSuperBitLSHIndexConfig.Serializer).createOrOpen()
    val superBit: SuperBit

    override val type = IndexType.NONBUCKETING_SUPERBIT_LSH

    /* todo: find better way to store the maps (we have List<Boolean>, not String!)
        what about ByteArray? Is this serializable by mapdb that it can be used as map keys?
        for kotlin it should be a List<Boolean> as these are structurally comapared for equality, not
        by reference
        as a proof of concept for correctness, use strings...
        storage overhead is factor of 8! maybe this is too much...
     */
    private val maps: List<HTreeMap<String, LongArray>>

    init {
        if (!columns.all {it.type.vector}) {
            throw DatabaseException.IndexNotSupportedException(name, "Because only vector columns are supported for SuperBitLSHIndex.")
        }
        val cod = configOnDisk.get()
        if (cod == null) {
            if (config != null) {
                this.config = config
                this.configOnDisk.set(config)
            } else {
                throw StoreException("No config supplied, and the config from disk was also empty.")
            }
        }
        else {
            this.config = cod
        }
        this.maps = List(this.config.stages) {
            this.db.hashMap(MAP_FIELD_NAME + "_stage$it", Serializer.STRING_ASCII, Serializer.LONG_ARRAY).counterEnable().createOrOpen()
        }
        superBit = SuperBit(this.config.superBitDepth, this.config.superBitsPerStage * this.config.stages, this.config.seed, this.config.samplingMethod, columns[0].defaultValue() as VectorValue<*>)
    }

    /**
     * Checks if this [Index] can process the provided [Predicate] and returns true if so and false otherwise.
     * innerproduct distances only with normalized vectors!
     * currently, we don't support considerImaginary if any query vector is real!
     *
     * @param predicate [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate): Boolean =
            predicate is KnnPredicate<*>
            && predicate.columns.first() == this.columns.first()
            && (predicate.distance is CosineDistance
                || abs(predicate.query.first().norm2().asDouble().value - 1.0) < 1e-15
                && (predicate.distance is RealInnerProductDistance
                    || predicate.distance is AbsoluteInnerProductDistance)
            )
            && (!config.considerImaginary || predicate.query.all { it is ComplexVectorValue<*> })

    /**
     * Calculates the cost estimate if this [Index] processing the provided [Predicate].
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
     * Returns true, if the [Index] supports incremental updates, and false otherwise.
     * should technically be true, as it's supported by LSH in general, but I have not
     * implemented it, so False...
     *
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate(): Boolean = false

    /**
     * (Re-)builds the [Index]. Invoking this method should rebuild the [Index] immediately, without the
     * need to commit (i.e. commit actions must take place inside).
     *
     * This is an internal method! External invocation is only possible through a [Index.Tx] object.
     *
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun rebuild(tx: Entity.Tx) {
        LOGGER.debug("Rebuilding ${name}")

        val local = List(config.stages) {
            HashMap<String, MutableList<Long>>()
        }
        tx.forEach {
            val value = it[this.columns[0]]!! as VectorValue<*>
            val signature = if (config.considerImaginary && value is ComplexVectorValue<*>) {
                superBit.signatureComplex(value)
            } else {
                superBit.signature(value)
            }
            (signature.toList().chunked(config.superBitDepth * config.superBitsPerStage) zip local).forEach { (stageSignature, tIdsForStageSignature) ->
                tIdsForStageSignature.getOrPut(stageSignature.joinToString(separator = "", transform = ::boolToString)) { mutableListOf() }.add(it.tupleId)
            }
        }

        /* clear existing maps. */
        (this.maps zip local).forEach { (map, localdata) ->
            map.clear()
            localdata.forEach { (subSignature, tIds) ->
                map[subSignature] = tIds.toLongArray()
            }
        }


        /* Commit local database. */
        this.db.commit()
        LOGGER.debug("Done.")
    }

    /**
     * Updates the [Index] with the provided [DataChangeEvent]s. The updates take effect immediately, without the need to
     * commit (i.e. commit actions must take place inside).
     *
     * Not all [Index] implementations support incremental updates. Should be indicated by [IndexTransaction#supportsIncrementalUpdate()]
     *
     * @param update [Record]s to update this [Index] with wrapped in the corresponding [DataChangeEvent].
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @throws [ValidationException.IndexUpdateException] If update of [Index] fails for some reason.
     */
    override fun update(update: Collection<DataChangeEvent>, tx: Entity.Tx) {
        TODO("Not yet implemented")
    }

    /**
     * Performs a lookup through this [Index] and returns [Recordset]. This is an internal method! External
     * invocation is only possible through a [Index.Tx] object.
     *
     * This is the minimal method any [Index] implementation must support.
     *
     * @param predicate The [Predicate] to perform the lookup.
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @return The resulting [Recordset].
     *
     * @throws QueryException.UnsupportedPredicateException If predicate is not supported by [Index].
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        require(predicate is KnnPredicate<*> && this.canProcess(predicate)) { throw QueryException.UnsupportedPredicateException("Index '${this.name}' (${this::class.qualifiedName}) does not support the provided predicate of type '${predicate::class.simpleName}'.") }

        /* for each query, we want a set with the tuples that have the same bit signature for at least one stage */

        val knns = Array(predicate.query.size) {
            if (predicate.k == 1) {
                MinSingleSelection<ComparablePair<Long, DoubleValue>>()
            } else {
                MinHeapSelection<ComparablePair<Long, DoubleValue>>(predicate.k)
            }
        }

        /* Generate record set .*/

        /* now find identical signatures that can be treated the same */
        val queryIndicesPerBucketSignature = HashMap<List<Boolean>, MutableList<Int>>()
        // we need to store the hash as list because lists are compared structurally, not by reference as arrays are
        predicate.query.forEachIndexed {queryIndex, query ->
            if (this.config.considerImaginary) {
                queryIndicesPerBucketSignature.getOrPut(superBit.signatureComplex(query as ComplexVectorValue<*>).toList()) { mutableListOf() }.add(queryIndex)
            }
            else {
                queryIndicesPerBucketSignature.getOrPut(superBit.signature(query).toList()) { mutableListOf() }.add(queryIndex)
            }
        }

        LOGGER.debug("Processing ${queryIndicesPerBucketSignature.size} overall unique signatures for ${predicate.query.size} queries.")
        queryIndicesPerBucketSignature.toList().parallelStream().forEach { (bitSignature, queryIndexes) ->
            LOGGER.debug("Building TIDs for bucketSignature")
            LOGGER.trace("bucketSignature ${bitSignature} with ${queryIndexes.size} queries")
            val tupleIds = HashSet<Long>()
            bitSignature.chunked(config.superBitDepth * config.superBitsPerStage).forEachIndexed { stage, subSignature ->
                val elements = this.maps[stage][subSignature.joinToString(separator = "", transform = ::boolToString)]
                if (elements != null) {
                    tupleIds.addAll(elements.toList())
                }
            }
            if (LOGGER.isTraceEnabled) LOGGER.trace("BitSignature ${bitSignature} has ${tupleIds.size} tIds")
            if (tupleIds.isEmpty()) {
                LOGGER.warn("no tIds found in index. Adding last tuple of entity as default!")
                tupleIds.add(tx.maxTupleId())
            }
            tupleIds.forEach {
                val record = tx.read(it)
                val value = record[predicate.column]
                if (value is VectorValue<*>) {
                    queryIndexes.forEach {queryIndex ->
                        val query = predicate.query[queryIndex]
                        if (predicate.weights != null) {
                            knns[queryIndex].offer(ComparablePair(it, predicate.distance(query, value, predicate.weights[queryIndex])))
                        } else {
                            knns[queryIndex].offer(ComparablePair(it, predicate.distance(query, value)))
                        }
                    }
                }
            }
        }
        LOGGER.debug("Done. Creating Recordset.")
        val recordset = selectToRecordset(this.produces.first(), knns.toList())

        LOGGER.debug("Done.")
        return recordset
    }

    private fun boolToString(b: Boolean): String = if (b) "1" else "0"
}