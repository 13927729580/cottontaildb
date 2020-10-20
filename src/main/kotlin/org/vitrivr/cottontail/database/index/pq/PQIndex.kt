package org.vitrivr.cottontail.database.index.pq

import org.mapdb.DBMaker
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.column.Complex32VectorColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.math.knn.selection.Selection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.Complex32VectorValue
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import java.util.*
import kotlin.collections.HashMap

/**
 * author: Gabriel Zihlmann
 * date: 25.8.2020
 *
 * Todo: * signatures: Ints are convenient but wasting a lot of space...
 *         we should move towards only using as many bits as necessary...
 *       * avoid copying
 *       * generalize to other datatypes than Complex32VV
 *
 * changes 13.10.2020:
 * * permutation of dimensions will no longer be applied. PQ is 5-10% more accurate without it!
 * * quantizing complex vectors directly is possible and about as accurate as real vectors. This has been changed
 * * in this class now. Performance implications need to be assessed
 */
@ExperimentalUnsignedTypes
class PQIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>,
              config: PQIndexConfig?= null): Index() {
    companion object {
        val CONFIG_NAME = "pq_config"
        val PQ_NAME = "pq_cb"
        val SIG_NAME = "pq_sig"
        val LOGGER = LoggerFactory.getLogger(PQIndex::class.java)

        /**
         * The index of the permutation array is the index in the unpermuted space
         * The value at that index is to which dimension it was permuted in the permuted space
         */
        fun generateRandomPermutation(size: Int, rng: SplittableRandom): Pair<IntArray, IntArray> {
            // init permutation of dimensions as identity permutation
            val permutation = IntArray(size) { it }
            // fisher-yates shuffle
            for (i in 0 until size - 1) {
                val toSwap = i + rng.nextInt(size - i)
                val h = permutation[toSwap]
                permutation[toSwap] = permutation[i]
                permutation[i] = h
            }
            val reversePermutation = IntArray(permutation.size) {
                permutation.indexOf(it)
            }
            return permutation to reversePermutation
        }
    }

    /** The [Path] to the [DBO]'s main file OR folder. */
    override val path: Path = this.parent.path.resolve("idx_pq_$name.db")

    /** The [PQIndex] implementation returns exactly the columns that is indexed. */
    override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("distance"), ColumnType.forName("DOUBLE")))

    /** The type of [Index]. */
    override val type = IndexType.PQ


    /** The internal [DB] reference. */
    private val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }
    val config: PQIndexConfig
    val configOnDisk = this.db.atomicVar(CONFIG_NAME, PQIndexConfig.Serializer).createOrOpen()
    val dimensionsPerSubspace: Int
    val rng: SplittableRandom
    lateinit var pq: PQ
    val pqStore = db.atomicVar(PQ_NAME, PQ.Serializer).createOrOpen()
    val signatures = mutableListOf<IntArray>()
    val tIds = mutableListOf<LongArray>()
    val signaturesStore = db.hashMap(SIG_NAME, Serializer.INT_ARRAY, Serializer.LONG_ARRAY).counterEnable().createOrOpen()

    init {
        if (columns.size != 1) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing a single column")
        }
        if (!columns.all { it.type == ColumnType.forName("COMPLEX32_VEC") || it.type == ColumnType.forName("COMPLEX64_VEC") }) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing complex vector columns, not ${columns.first()::class.java}")
        }
        val cod = configOnDisk.get()
        if (config == null) {
            if (cod == null) {
//                throw StoreException("No config supplied but the config from disk was null!")
                LOGGER.warn("No config supplied, but the config from disk was null!! Using a dummy config. Please consider this index invalid!")
                this.config = PQIndexConfig(1, 1, 5e-3, LookupTablePrecision.SINGLE, 100, 1234L, Complex32VectorColumnType)
            } else {
                this.config = cod
            }
        } else {
            configOnDisk.set(config)
            this.config = config
        }
        // some assumptions. Some are for documentation, some are cheap enough to actually keep and check
        require(this.config.numCentroids <= UShort.MAX_VALUE.toInt())
        require(this.config.numSubspaces > 0)
        require(this.config.numCentroids > 0)
        require(columns[0].logicalSize >= this.config.numSubspaces)
        require(columns[0].logicalSize % this.config.numSubspaces == 0)
        dimensionsPerSubspace = columns[0].logicalSize / this.config.numSubspaces
        rng = SplittableRandom(this.config.seed)
        pqStore.get()?.let { pq = it }

        this.db.commit() // this writes config stuff, so that the commit doesn't wait until rebuild()
        // note that due to this (if it works as expected, which is probably not the case),
        // now indexes that are not yet built can exist
        // and there are no indexEntries in the entity that cannot be opened...
        // (e.g if there is a failure during rebuild()...
        loadSignaturesFromDisk()
    }

    private fun loadSignaturesFromDisk() {
        signatures.clear()
        tIds.clear()
        LOGGER.info("Index ${name.simple} loading Signatures from store")
        signaturesStore.forEach { signature, tIds_ ->
            signatures.add(signature!!)
            tIds.add(tIds_)
        }
        LOGGER.info("Done.")
    }

    /**
     * Flag indicating if this [PQIndex] has been closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Closes this [PQIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!closed) {
            db.close()
            closed = true
        }
    }

    /**
     * Checks if this [Index] can process the provided [Predicate] and returns true if so and false otherwise.
     *
     * @param predicate [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate) =
            predicate is KnnPredicate<*>
                && predicate.query.all { it is ComplexVectorValue<*> }
                && predicate.columns.first() == this.columns[0]
                && predicate.distance is AbsoluteInnerProductDistance

    /**
     * Calculates the cost estimate if this [Index] processing the provided [Predicate].
     *
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate) = Cost.ZERO // todo...

    /**
     * Returns true, if the [Index] supports incremental updates, and false otherwise.
     *
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate(): Boolean {
        return false // todo implement...
    }

    override fun rebuild(tx: Entity.Tx) {
        //todo: don't copy data
        LOGGER.info("Rebuilding PQIndex.")
        val signaturesTidsLoc = HashMap<List<Int>, MutableList<Long>>()
        LOGGER.info("Preparing data.")
        // because tx doesn't have a simple .filter method where we can specify any old boolean, we need to
        // roll our own...
        // this filters the elements randomly based on the learning fraction and permutes them
        val (learningData, learningTIds) =
                mutableListOf<Pair<VectorValue<*>, Long>>().apply {
            tx.forEach { r ->
                if (rng.nextDouble() > config.learningDataFraction) {
                    return@forEach
                }
                this.add(Pair(r[columns[0]] as VectorValue<*>, r.tupleId)) // todo: get rid of intermediary pairs... We need to know the number of records for that (which we can do now because we need to filter manually...)
            }
        }.unzip()
        LOGGER.info("Learning with ${learningTIds.size} vectors...")
        val (pq, signatures) = PQ.fromPermutedData(config.numSubspaces, config.numCentroids, learningData.toTypedArray(), config.type)
        this.pq = pq
        pqStore.set(pq)
        (signatures zip learningTIds).forEach { (sig, tid) ->
            signaturesTidsLoc.getOrPut(sig.toList()) { mutableListOf() }.add(tid)
        }
        LOGGER.info("Learning done.")
        // now get and add signatures for elements not in learning set
        LOGGER.info("Generating signatures for all vectors...")
        val learningTIdsSet = learningTIds.toHashSet() // convert to hash set for O(1) lookup
        var done = 0
        tx.forEach {  r ->
            if (done % 1000000 == 0) LOGGER.info("$done elements processed")
            done++
            if (learningTIdsSet.contains(r.tupleId))
                return@forEach
            val sig = this.pq.getSignature(r[columns[0]] as VectorValue<*>).toList()
            signaturesTidsLoc.getOrPut(sig) { mutableListOf() }.add(r.tupleId)
        }
        LOGGER.info("Done. Storing signatures.")
        // map to intArray for storing. We need to use List<Int> in kotlin code to compare signatures
        // structurally (IntArray is only compared by ref)
        signaturesStore.clear()
        signaturesStore.putAll(signaturesTidsLoc.map { (k, v) -> k.toIntArray() to v.toLongArray()}.toMap())
        LOGGER.info("Done generating and storing signatures. Committing.")
        db.commit()
        LOGGER.info("Loading signatures from disk.")
        loadSignaturesFromDisk()
        LOGGER.info("Done.")
        LOGGER.info("PQIndex rebuild done.")
    }

    private fun permuteSplitComplexRecord(r: Record): Pair<DoubleArray, DoubleArray> {
        val v = (r[columns[0]] as ComplexVectorValue<*>)
        val re = DoubleArray(dimensionsPerSubspace * config.numSubspaces)
        val im = DoubleArray(dimensionsPerSubspace * config.numSubspaces)
        (0 until config.numSubspaces * dimensionsPerSubspace).forEach {
            val complexValue = v[it]
            re[it] = complexValue.real.value.toDouble()
            im[it] = complexValue.imaginary.value.toDouble()
        }
        return Pair(re, im)
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
     * @param p The [Predicate] to perform the lookup.
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @return The resulting [Recordset].
     *
     * @throws QueryException.UnsupportedPredicateException If predicate is not supported by [Index].
     */
    @ExperimentalUnsignedTypes
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        require(canProcess(predicate)) {"The supplied predicate $predicate cannot be processed by index ${this.name}"}
        LOGGER.info("${this.name} Filtering")
        val p = predicate as KnnPredicate<*>

        val approxK = config.kApproxScan
        val knns = scanQueriesSign(p, this.pq, k = approxK)
//        val knns = scanSignQueries(p, pqReal, pqImag)

        // get exact distances...
        LOGGER.info("Re-ranking $approxK signature matches with exact distances.")
        val knnsExact = knns.mapIndexed { i, knn ->
            var numTids = 0
            val knnNew = if (p.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection<ComparablePair<Long, DoubleValue>>(p.k)
            (0 until knn.size).forEach {
                val tIds = knn[it].first
                tIds.forEach { tid ->
                    numTids++
                    val exact = tx.read(tid)[columns[0]]!! as ComplexVectorValue<*>
                    knnNew.offer(ComparablePair(tid, p.distance(exact, p.query[i])))
                }
            }
            LOGGER.trace("Considered $numTids after approximation scan for query $i")
            knnNew
        }
        LOGGER.info("Done")
        return KnnUtilities.selectToRecordset(this.produces.first(), knnsExact.toList())
    }


    @ExperimentalUnsignedTypes
    private fun scanQueriesSign(p: KnnPredicate<*>, pqReal: PQ, k: Int = p.k): List<Selection<ComparablePair<LongArray, Float>>> {
        LOGGER.debug("Converting signature array")
        val sigLength = config.numSubspaces
        val sigReIm = UShortArray(signatures.size * sigLength) {
            signatures[it / (sigLength)][it % (sigLength)].toUShort()
        }
        LOGGER.debug("Done.")
        val knnQueries = p.query.mapIndexed { i, q_ ->
            val q = q_ as Complex32VectorValue
            (if (k == 1) MinSingleSelection<ComparablePair<LongArray, Float>>() else MinHeapSelection<ComparablePair<LongArray, Float>>(k)) to q
        }
        val chunksize = 1
        if (chunksize > 1) {
            knnQueries.chunked(chunksize).parallelStream().forEach { knnQueriesChunk ->
//            LOGGER.info("Processing query ${i + 1} of ${p.query.size}")
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIP = Array(knnQueriesChunk.size) { pqReal.precomputeCentroidQueryIPComplexVectorValue(knnQueriesChunk[it].second) }
                LOGGER.info("Scanning signatures")
                signatures.indices.forEach {
                    knnQueriesChunk.indices.forEach { i ->
                        val sigOffset = it * sigLength // offset into sign array.
                        val tIdsOfSig = tIds[it]
                        val absIPSqApprox =
                                queryCentroidIP[i].approximateIP(sigReIm, sigOffset, sigLength).abs().value
//                if (knn.added < knn.k || knn.peek()!!.second > -absIPSqApprox) // do we really need to create a new pair every single time?
                        // we don't, but keep it for comparability with brute-force
                        knnQueriesChunk[i].first.offer(ComparablePair(tIdsOfSig, -absIPSqApprox))
                    }
                }
            }
        }
        else {
            knnQueries.parallelStream().forEach { (knn, q) ->
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIPRealReal =  pqReal.precomputeCentroidQueryIPComplexVectorValue(q)
                if (LOGGER.isTraceEnabled) LOGGER.trace("Scanning signatures")
                signatures.indices.forEach {
                    val sigOffset = it * sigLength // offset into sign array
                    val tidOfSig = tIds[it]
                    val absIPSqApprox =
                            queryCentroidIPRealReal.approximateIP(sigReIm, sigOffset, sigLength).abs().value
//                if (knn.added < knn.k || knn.peek()!!.second > -absIPSqApprox) // do we really need to create a new pair every single time?
                    knn.offer(ComparablePair(tidOfSig, -absIPSqApprox))
                }
            }
        }
        return knnQueries.unzip().first
    }
}
