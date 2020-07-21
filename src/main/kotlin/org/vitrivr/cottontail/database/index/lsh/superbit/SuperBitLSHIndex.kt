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
import org.vitrivr.cottontail.math.knn.metrics.CosineDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.exceptions.StoreException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import kotlin.collections.HashMap
import kotlin.collections.HashSet

/**
 * Represents a LSH based index in the Cottontail DB data model. An [Index] belongs to an [Entity] and can be used to
 * index one to many [Column]s. Usually, [Index]es allow for faster data access. They process [Predicate]s and return
 * [Recordset]s.
 *
 * @author Manuel Huerbin & Ralph Gasser
 * @version 1.1
 */
class SuperBitLSHIndex<T : VectorValue<*>>(name: Name.IndexName, parent: Entity, columns: Array<ColumnDef<*>>, params: Map<String, String>? = null) : LSHIndex<T>(name, parent, columns) {

    companion object {
        const val CONFIG_NAME = "lsh_config"
        const val CONFIG_NAME_STAGES = "stages"
        const val CONFIG_NAME_BUCKETS = "buckets"
        const val CONFIG_NAME_SEED = "seed"
        private const val CONFIG_DEFAULT_STAGES = 3
        private const val CONFIG_DEFAULT_BUCKETS = 10
        private val LOGGER = LoggerFactory.getLogger("org.vitrivr.cottontail.database.index.lsh.superbit.SuperBitLSHIndex")
    }

    /** Internal configuration object for [SuperBitLSHIndex]. */
    val config = this.db.atomicVar(CONFIG_NAME, SuperBitLSHIndexConfig.Serializer).createOrOpen()
    private val considerImaginary = false // todo: put in config
    private val samplingMethod = SuperBit.SamplingMethod.UNIFORM // todo: this as well

    override val type = IndexType.SUPERBIT_LSH

    private var maps: List<HTreeMap<Int, LongArray>>

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
        this.maps = List(this.config.get().stages) {
            this.db.hashMap(MAP_FIELD_NAME + "_stage$it", Serializer.INTEGER, Serializer.LONG_ARRAY).counterEnable().createOrOpen()
        }
    }

    /**
     * Performs a lookup through this [SuperBitLSHIndex].
     *
     * @param predicate The [Predicate] for the lookup
     * @return The resulting [Recordset]
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        if (predicate is KnnPredicate<*>) {
            /* Guard: Only process predicates that are supported. */
            require(this.canProcess(predicate)) { throw QueryException.UnsupportedPredicateException("Index '${this.name}' (lsh-index) does not support the provided predicate.") }

            /* Prepare empty Recordset and LSH object. */
            val recordset = Recordset(this.produces, (predicate.k * predicate.query.size).toLong())
            val lsh = SuperBitLSH(this.config.get().stages, this.config.get().buckets, this.columns.first().logicalSize, this.config.get().seed, predicate.query.first(), considerImaginary, samplingMethod)

             /* for each query, we want a set with the tuples that were in the same bucket in one stage during the
               we can no longer do it per-bucket, because of teh different stages. we could try it per stage, and then
               merge, but I think this is bad because I expect quite a bit of overlap between candidate sets
               of different stages and we want to avoid looking at tuples more than once
             */

            val knns = Array(predicate.query.size) {
                if (predicate.k == 1) {
                    MinSingleSelection<ComparablePair<Long, DoubleValue>>()
                } else {
                    MinHeapSelection<ComparablePair<Long, DoubleValue>>(predicate.k)
                }
            }

            /* Generate record set .*/

            /* now find identical signatures that can be treated the same */
            val queryIndicesPerBucketSignature = HashMap<List<Int>, MutableList<Int>>()
            // we need to store the hash as list because lists are compared structurally, not by reference as arrays are
            predicate.query.forEachIndexed {queryIndex, query ->
                queryIndicesPerBucketSignature.getOrPut(lsh.hash(query).toList()) { mutableListOf() }.add(queryIndex) }

            LOGGER.debug("Processing unique bucket signatures (${queryIndicesPerBucketSignature.size}) for ${predicate.query.size} queries.")
            queryIndicesPerBucketSignature.toList().parallelStream().forEach { (bucketSignature, queryIndexes) ->
                LOGGER.debug("Building TIDs for bucketSignature")
                LOGGER.trace("bucketSignature ${bucketSignature} with ${queryIndexes.size} queries")
                val tupleIds = HashSet<Long>()
                bucketSignature.forEachIndexed { stage, bucket ->
                    tupleIds.addAll(this.maps[stage][bucket]!!.toList())
                }
                // building tids takes 15+s for 512 buckets and 4096 query vectors if done before for all queries
                // not sure if this is the right way... We know already from the bucket list which queries have common
                // tids. So we could iterate per stage over the buckets as I did before, but keep in a hash set for each
                // query the tids that were already visited to avoid comparing again, but this can potentially be a huge
                // set and parallel access could be important. I will first do an implementation for a single query vector
                // and then consider parallelization later
                // I expect that with more stages, the overlap between stages becomes smaller and queries will have a
                // more disjoint candidate set. But there's also the fact that you can only add to the candidate set
                // with further stages. Overlap over the entire query set grows, i.e. pairs of query vectors with common
                // candidates will grow and as a consequence we'll be loading tuples multiple times...
                // on the other hand, if we do it stage-wise, we still have the problem of loading multiple times
                // but we can more easily pool queries that need the same data at that stage
                // parallelization is probably easiest by partitioning the data in the db, right? Con is load-imbalance
                if (LOGGER.isTraceEnabled) LOGGER.trace("bucketSignature ${bucketSignature} has ${tupleIds.size} tIds")
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
//            selectToRecordset(this.columns.first(), knns.toList())
            for (knn in knns) {
                for (j in 0 until knn.size) {
                    recordset.addRowUnsafe(knn[j].first, arrayOf(knn[j].second))
                }
            }
            LOGGER.debug("Done.")
            return recordset
        } else {
            throw QueryException.UnsupportedPredicateException("Index '${this.name}' (LSH Index) does not support predicates of type '${predicate::class.simpleName}'.")
        }
    }

    /** (Re-)builds the [SuperBitLSHIndex]. */
    override fun rebuild(tx: Entity.Tx) {
        LOGGER.debug("Rebuilding ${this.name}")
        /* LSH. */
        val specimen = this.acquireSpecimen(tx) ?: throw DatabaseException("Could not gather specimen to create index.") // todo: find better exception
        val lsh = SuperBitLSH(this.config.get().stages, this.config.get().buckets, this.columns[0].logicalSize, this.config.get().seed, specimen, considerImaginary, samplingMethod)


        /* Locally (Re-)create index entries and sort bucket for each stage to corresponding map. */
        val local = List(this.config.get().stages) {
            MutableList(this.config.get().buckets) { mutableListOf<Long>() }
        }
        /* for every record get bucket-signature, then iterate over stages and add tid to the list of that bucket of that stage */
        tx.forEach {
            val value = it[this.columns[0]] ?: throw DatabaseException("Could not find column for entry in index $this") // todo: what if more columns? This should never happen -> need to change type and sort this out on index creation
            if (value is VectorValue<*>) {
                val buckets = lsh.hash(value)
                (buckets zip local).forEach { (bucket, map) ->
                    map[bucket].add(it.tupleId)
                }
            } else {
                throw DatabaseException("$value is no vector column!")
            }
        }

        /* clear existing maps. */
        if (this.maps.size != local.size) {
            throw IllegalArgumentException("This should never happen")
        }
        (this.maps zip local).forEach { (map, localdata) ->
            map.clear()
            localdata.forEachIndexed { bucket, tIds ->
                map[bucket] = tIds.toLongArray()
            }
        }


        /* Commit local database. */
        this.db.commit()
        LOGGER.debug("Done.")
    }

    /**
     * Updates the [SuperBitLSHIndex] with the provided [Record]. This method determines, whether the [Record] should be added or updated
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