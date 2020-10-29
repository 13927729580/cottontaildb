package org.vitrivr.cottontail.database.index.pq

import org.mapdb.DBMaker
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.*
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
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.math.pow

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
 *
 * 22.10.2020:
 * * deduplication is implemented. Brings significant speedups for lower subspace and centroid counts (2-3x)
 * * beneficial to go to higher k in approximate scan. This requires more effort on re-ranking the approx
 *   matches (can become dominant and take a multiple of the time to scan without parallelization).
 *   This has been parallelized now.
 */
@ExperimentalUnsignedTypes
class PQIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>,
              config: PQIndexConfig?= null): Index() {
    companion object {
        const val CONFIG_NAME = "pq_config"
        const val PQ_NAME = "pq_cb"
        const val SIG_NAME = "pq_sig"
        val LOGGER = LoggerFactory.getLogger(PQIndex::class.java)!!

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
    private val configOnDisk = this.db.atomicVar(CONFIG_NAME, PQIndexConfig.Serializer).createOrOpen()
    val dimensionsPerSubspace: Int
    val rng: SplittableRandom
    lateinit var pq: PQ
    lateinit var pqImag: PQ
    val pqStore = db.atomicVar(PQ_NAME, PQ.Serializer).createOrOpen()
    val pqStoreImag = db.atomicVar(PQ_NAME + "_imag", PQ.Serializer).createOrOpen()
    val signatures = mutableListOf<IntArray>()
    val tIds = mutableListOf<LongArray>()
    private val signaturesStore = db.hashMap(SIG_NAME, Serializer.INT_ARRAY, Serializer.LONG_ARRAY).counterEnable().createOrOpen()

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
                this.config = PQIndexConfig(1, 1, 5e-3, PQIndexConfig.Precision.SINGLE, 100, 1234L, PQIndexConfig.ComplexStrategy.DIRECT)
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
        if (this.config.complexStrategy == PQIndexConfig.ComplexStrategy.SPLIT) {
            pqStoreImag.get()?.let { pqImag = it }
        }

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
        LOGGER.debug("Index ${name.simple} loading Signatures from store")
        val expectedSignatureSize = config.numSubspaces * when(config.complexStrategy) { PQIndexConfig.ComplexStrategy.DIRECT -> 1; PQIndexConfig.ComplexStrategy.SPLIT -> 2}
        signaturesStore.forEach { (signature, tIds_) ->
            check(signature.size == expectedSignatureSize) { "Loaded a signature [${signature.joinToString(separator = ",")}] with unexpected signature size (${signature.size} instead of $expectedSignatureSize)."}
            signatures.add(signature!!)
            tIds.add(tIds_)
        }
        LOGGER.debug("Done loading.")
        if (signatures.size > 0) {
            logSignatureStatistics()
        }
    }

    private fun logSignatureStatistics() {
        // some statistics:
        val tIdsPerSignatureDistribution = mutableMapOf<Int, Int>()
        tIds.forEach { tIds1 ->
            tIdsPerSignatureDistribution[tIds1.size] = tIdsPerSignatureDistribution.getOrDefault(tIds1.size, 0) + 1
//            tIdsPerSignatureDistribution.getOrPut(tIds.size) { 0 }.inc() // would this work?
        }
        LOGGER.debug("${signatures.size} unique signatures.")
        LOGGER.debug("Most Common Signature has ${tIds.maxOf { it.size }} tIds.")
        LOGGER.debug("Least Common Signature has ${tIds.minOf { it.size }} tIds.")
        val largestFirst = tIdsPerSignatureDistribution.toSortedMap(compareBy { -it })
        LOGGER.debug("Complete Size distribution (numTIdsPerSignature:numSignatures): ${largestFirst.map { (k, v) -> "$k:$v" }.joinToString()}")
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
        LOGGER.debug("Preparing data.")
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
        val (preProcessedLearningData, preProcessedLearningDataImag, type) = when (config.complexStrategy) {
            PQIndexConfig.ComplexStrategy.SPLIT -> {
                val (preProcessedLearningData, preProcessedLearningDataImag) = learningData.map {
                    it as ComplexVectorValue<*>
                    it.real() to it.imaginary()
                }.unzip()
                val type = when (config.precision) {
                    PQIndexConfig.Precision.SINGLE -> {
                        FloatVectorColumnType
                    }
                    PQIndexConfig.Precision.DOUBLE -> {
                        DoubleVectorColumnType
                    }
                }
                Triple(preProcessedLearningData, preProcessedLearningDataImag, type)
            }
            PQIndexConfig.ComplexStrategy.DIRECT -> {
                Triple(learningData, null, when (config.precision) {
                    PQIndexConfig.Precision.SINGLE -> Complex32VectorColumnType; PQIndexConfig.Precision.DOUBLE -> Complex64VectorColumnType
                })
            }
        }

        LOGGER.debug("Learning from ${learningTIds.size} vectors...")
        val (pq, signatures) = PQ.fromPermutedData(config.numSubspaces, config.numCentroids, preProcessedLearningData.toTypedArray(), type)
        this.pq = pq
        pqStore.set(pq)
        val signaturesTidsLoc = HashMap<List<Int>, MutableList<Long>>()
        when (config.complexStrategy) {
            PQIndexConfig.ComplexStrategy.DIRECT -> {
                (signatures zip learningTIds).forEach { (sig, tid) ->
                    signaturesTidsLoc.getOrPut(sig.toList()) { mutableListOf() }.add(tid)
                }
            }
            PQIndexConfig.ComplexStrategy.SPLIT -> {
                LOGGER.debug("Learning imaginary part from ${learningTIds.size} vectors...")
                val (pqImag, signaturesImag) = PQ.fromPermutedData(config.numSubspaces, config.numCentroids, preProcessedLearningDataImag!!.toTypedArray(), type)
                this.pqImag = pqImag
                pqStoreImag.set(pqImag)
                ((signatures zip signaturesImag) zip learningTIds).forEach { (sigRealImag, tid) ->
                    val (sigReal, sigImag) = sigRealImag
                    signaturesTidsLoc.getOrPut(sigReal.toList() + sigImag.toList()) { mutableListOf() }.add(tid)
                }
            }
        }
        LOGGER.debug("Learning done.")
        // now get and add signatures for elements not in learning set
        LOGGER.debug("Generating signatures for all vectors...")
        val learningTIdsSet = learningTIds.toHashSet() // convert to hash set for O(1) lookup
        val tidsSignatures = findSignaturesParallel(tx, learningTIdsSet)
        LOGGER.trace("Done generating signatures for all vectors. Adding to map.")
        tidsSignatures.forEach { (tid, sig) ->
            signaturesTidsLoc.getOrPut(sig) { mutableListOf() }.add(tid)
        }
        LOGGER.debug("Done. Storing signatures.")
        // map to intArray for storing. We need to use List<Int> in kotlin code to compare signatures
        // structurally (IntArray is only compared by ref)
        signaturesStore.clear()
        signaturesStore.putAll(signaturesTidsLoc.map { (k, v) -> k.toIntArray() to v.toLongArray()}.toMap())
        LOGGER.debug("Done generating and storing signatures. Committing.")
        db.commit()
        LOGGER.debug("Loading signatures from disk.")
        loadSignaturesFromDisk()
        LOGGER.info("PQ rebuild Done.")
        // todo: check if we're really done porting this function to work in SPLIT mode...
    }

    private fun findSignaturesParallel(tx: Entity.Tx, ignoreTids: HashSet<Long>): List<Pair<Long, List<Int>>> {
        fun findSignaturesDirect(tx: Entity.Tx, startTid: Long, endTidInclusive: Long, ignoreTids: HashSet<Long>): List<Pair<Long, List<Int>>> {
            LOGGER.trace("Finding signatures to tids from $startTid to $endTidInclusive (inclusive).")
            val res = mutableListOf<Pair<Long, List<Int>>>()
            tx.forEach(startTid, endTidInclusive) { r ->
                if (!ignoreTids.contains(r.tupleId)) {
                    val signature = this.pq.getSignature(r[columns[0]] as VectorValue<*>).toList()
                    res.add(r.tupleId to signature)
                }
            }
            LOGGER.trace("Done.")
            return res.toList()
        }
        fun findSignaturesSplit(tx: Entity.Tx, startTid: Long, endTidInclusive: Long, ignoreTids: HashSet<Long>): List<Pair<Long, List<Int>>> {
            LOGGER.trace("Finding signatures to tids from $startTid to $endTidInclusive (inclusive).")
            val res = mutableListOf<Pair<Long, List<Int>>>()
            tx.forEach(startTid, endTidInclusive) { r ->
                if (!ignoreTids.contains(r.tupleId)) {
                    val v = r[columns[0]] as ComplexVectorValue<*>
                    val signature = this.pq.getSignature(v.real()).toList() + this.pqImag.getSignature(v.imaginary()).toList()
                    res.add(r.tupleId to signature)
                }
            }
            LOGGER.trace("Done.")
            return res.toList()
        }
        val numThreads = Runtime.getRuntime().availableProcessors()
        // i guess we start counting at 0 -> we could have up to maxTupleId + 1 elements...
        val maxTupleId = tx.maxTupleId()
        val elemsPerThread = (maxTupleId + 1) / numThreads
        val remaining = (maxTupleId + 1) % numThreads
        val exec = Executors.newFixedThreadPool(numThreads)
        val tasks = (0 until numThreads).map { thread ->
            val f = when (config.complexStrategy) {
                PQIndexConfig.ComplexStrategy.DIRECT -> ::findSignaturesDirect
                PQIndexConfig.ComplexStrategy.SPLIT -> ::findSignaturesSplit
            }
            Callable {
                f(tx,
                        thread * elemsPerThread,
                        thread * elemsPerThread + elemsPerThread - 1 + if (thread == numThreads - 1) remaining else 0,
                                ignoreTids)}
        }
        val fresults = exec.invokeAll(tasks)
        val res = fresults.map { it.get()}
        exec.shutdownNow()
        return res.flatten()
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
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @return The resulting [Recordset].
     *
     * @throws QueryException.UnsupportedPredicateException If predicate is not supported by [Index].
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        require(canProcess(predicate)) {"The supplied predicate $predicate cannot be processed by index ${this.name}"}
        LOGGER.info("Index '${this.name}' Filtering")
        val p = predicate as KnnPredicate<*>

        LOGGER.debug("Converting signature array")
        val sigLength = signatures[0].size
        // todo:
        //  * this does not necessarily have to happen at query time! -> move to before
        //  * use adaptable size (UByte for nc <= 128? Custom via Long plus pad?
        //  * For nc 2048-4096 (11-12bits) we can fit 5 points into one long (64 bits). Savings compared to UShort: 20%)
        //  * For 10 bits, we can fit 6 -> 33% saving
        val sigReIm = UShortArray(signatures.size * sigLength) {
            signatures[it / (sigLength)][it % (sigLength)].toUShort()
        }
        LOGGER.debug("Done.")
        val approxK = config.kApproxScan

        val knns = when (config.complexStrategy) {
            PQIndexConfig.ComplexStrategy.DIRECT -> {
                scan(p, pq, sigReIm, sigLength, k = approxK)
            }
            PQIndexConfig.ComplexStrategy.SPLIT -> {
                scanSplit(p, pq, pqImag, sigReIm, sigLength, k = approxK)
            }
        }
        // get exact distances...
        // todo: consider merging scan and exact re-ranking.
        //       * we would not accumulate large knn objects that could use a lot of memory when going to large ks
        LOGGER.debug("Re-ranking $approxK signature matches with exact distances.")
        val knnsExact = knns.map {
            val knnNew = if (p.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection(p.k)
            knnNew
        }
        knnsExact.indices.toList().parallelStream().forEach { i ->
            var numTids = 0
            val knnNew = knnsExact[i]
            val knn = knns[i]
            (0 until knn.size).forEach {
                val tIds = knn[it].first
                tIds.forEach { tid ->
                    numTids++
                    val exact = tx.read(tid)[columns[0]]!! as ComplexVectorValue<*>
                    knnNew.offer(ComparablePair(tid, p.distance(exact, p.query[i])))
                }
            }
            if (LOGGER.isTraceEnabled) {
                LOGGER.trace("Considered $numTids after approximation scan for query $i")
            }
        }
        LOGGER.info("Done filtering")
        return KnnUtilities.selectToRecordset(this.produces.first(), knnsExact)
    }


    @ExperimentalUnsignedTypes
    private fun scan(p: KnnPredicate<*>, pq1: PQ, sigReIm: UShortArray, sigLength: Int, k: Int = p.k): List<Selection<ComparablePair<LongArray, Float>>> {
        LOGGER.debug("Scanning in DIRECT mode.")
        val knnQueries = p.query.mapIndexed { _, q_ ->
            val q = q_ as Complex32VectorValue
            (if (k == 1) MinSingleSelection<ComparablePair<LongArray, Float>>() else MinHeapSelection(k)) to q
        }
        val chunksize = 1
        if (chunksize > 1) {
            knnQueries.chunked(chunksize).parallelStream().forEach { knnQueriesChunk ->
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIP = Array(knnQueriesChunk.size) { pq1.precomputeCentroidQueryIPComplexVectorValue(knnQueriesChunk[it].second) }
                LOGGER.trace("Scanning signatures")
                signatures.indices.forEach {
                    knnQueriesChunk.indices.forEach { i ->
                        processSignature(it, sigReIm, sigLength, queryCentroidIP[i], knnQueriesChunk[i].first)
                    }
                }
            }
        }
        else {
            knnQueries.parallelStream().forEach { (knn, q) ->
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIP =  pq1.precomputeCentroidQueryIPComplexVectorValue(q)
                if (LOGGER.isTraceEnabled) LOGGER.trace("Scanning signatures")
                signatures.indices.forEach {
                    processSignature(it, sigReIm, sigLength, queryCentroidIP, knn)
                }
            }
        }
        return knnQueries.unzip().first
    }
    //todo: move to local method as soon as kotlin supports inline local funcs
    //      but keep outside for now because of overhead (local funcs are objects which are instantiated
    //      whenever the parent func is called!)
    private inline fun processSignature(signatureIndex: Int, sigReIm: UShortArray, sigLength: Int, queryCentroidIP: PQCentroidQueryIPComplexVectorValue, knn: Selection<ComparablePair<LongArray, Float>>) {
        val sigOffset = signatureIndex * sigLength // offset into sign array
        val tidOfSig = tIds[signatureIndex]
        val absIPSqApprox =
                queryCentroidIP.approximateIP(sigReIm, sigOffset, sigLength).abs().value
//                if (knn.added < knn.k || knn.peek()!!.second > -absIPSqApprox) // do we really need to create a new pair every single time?
        knn.offer(ComparablePair(tidOfSig, -absIPSqApprox))
    }


    private fun scanSplit(p: KnnPredicate<*>, pqReal: PQ, pqImag: PQ, sigReIm: UShortArray, sigLength: Int, k: Int = p.k): List<Selection<ComparablePair<LongArray, Float>>> {
        LOGGER.debug("Scanning in SPLIT mode.")
        val knnQueries = p.query.mapIndexed { _, q_ ->
            val q = q_ as Complex32VectorValue
            (if (k == 1) MinSingleSelection<ComparablePair<LongArray, Float>>() else MinHeapSelection(k)) to q
        }
        val chunksize = 1
        if (chunksize > 1) {
            knnQueries.chunked(chunksize).parallelStream().forEach { knnQueriesChunk ->
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIP = Array(knnQueriesChunk.size) {
                    val q = knnQueriesChunk[it].second
                    listOf(
                      pqReal.precomputeCentroidQueryRealIPFloat(q),
                      pqImag.precomputeCentroidQueryImagIPFloat(q),
                      pqReal.precomputeCentroidQueryImagIPFloat(q),
                      pqImag.precomputeCentroidQueryRealIPFloat(q)
                    ).toTypedArray()
                }
                if (LOGGER.isTraceEnabled) LOGGER.trace("Scanning signatures")
                signatures.indices.forEach {
                    knnQueriesChunk.indices.forEach { i ->
                        processSignatureSplit(it, sigReIm, sigLength,
                                queryCentroidIP[i][0],
                                queryCentroidIP[i][1],
                                queryCentroidIP[i][2],
                                queryCentroidIP[i][3],
                                knnQueriesChunk[i].first)
                    }
                }
            }
        }
        else {
            knnQueries.parallelStream().forEach { (knn, q) ->
                if (LOGGER.isTraceEnabled) LOGGER.trace("Precomputing IPs between query and centroids")
                val queryCentroidIPRealReal =  pqReal.precomputeCentroidQueryRealIPFloat(q)
                val queryCentroidIPImagImag =  pqImag.precomputeCentroidQueryImagIPFloat(q)
                val queryCentroidIPRealImag =  pqReal.precomputeCentroidQueryImagIPFloat(q)
                val queryCentroidIPImagReal =  pqImag.precomputeCentroidQueryRealIPFloat(q)
                if (LOGGER.isTraceEnabled) LOGGER.trace("Scanning signatures")
                signatures.indices.forEach {
                    processSignatureSplit(it, sigReIm, sigLength,
                            queryCentroidIPRealReal,
                            queryCentroidIPImagImag,
                            queryCentroidIPRealImag,
                            queryCentroidIPImagReal,
                            knn)
                }
            }
        }
        return knnQueries.unzip().first
    }
    //todo: move to local method as soon as kotlin supports inline local funcs
    private inline fun processSignatureSplit(signatureIndex: Int,
                                             sigReIm: UShortArray,
                                             sigLength: Int,
                                             queryCentroidIPRealReal: PQCentroidQueryIPFloat,
                                             queryCentroidIPImagImag: PQCentroidQueryIPFloat,
                                             queryCentroidIPRealImag: PQCentroidQueryIPFloat,
                                             queryCentroidIPImagReal: PQCentroidQueryIPFloat,
                                             knn: Selection<ComparablePair<LongArray, Float>>) {
        val sigOffset = signatureIndex * sigLength // offset into sign array
        val tidOfSig = tIds[signatureIndex]
        val lengthRealOrImag = sigLength / 2
        val absIPSqApprox = ((
                queryCentroidIPRealReal.approximateIP(sigReIm, sigOffset, lengthRealOrImag)
                        + queryCentroidIPImagImag.approximateIP(sigReIm, sigOffset + lengthRealOrImag, lengthRealOrImag)).pow(2)
                + (
                queryCentroidIPImagReal.approximateIP(sigReIm, sigOffset + lengthRealOrImag, lengthRealOrImag)
                        - queryCentroidIPRealImag.approximateIP(sigReIm, sigOffset, lengthRealOrImag)
                ).pow(2)
                )
//                if (knn.added < knn.k || knn.peek()!!.second > -absIPSqApprox) // do we really need to create a new pair every single time?
        knn.offer(ComparablePair(tidOfSig, -absIPSqApprox))
    }
}
