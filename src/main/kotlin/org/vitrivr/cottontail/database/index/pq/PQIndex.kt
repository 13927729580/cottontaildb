package org.vitrivr.cottontail.database.index.pq

import org.mapdb.DBMaker
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
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
import org.vitrivr.cottontail.model.exceptions.StoreException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import java.util.*
import kotlin.math.pow

/**
 * author: Gabriel Zihlmann
 * date: 25.8.2020
 *
 * Todo: * signatures: Ints are convenient but wasting a lot of space...
 *         we should move towards only using as many bits as necessary...
 *       * avoid copying
 *
 */
@ExperimentalUnsignedTypes
class PQIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>,
              config: PQIndexConfig?= null): Index() {
    companion object {
        val CONFIG_NAME = "pq_config"
        val PQ_REAL_NAME = "pq_cb_real"
        val PQ_IMAG_NAME = "pq_cb_imag"
        val SIG_REAL_NAME = "pq_sig_real"
        val SIG_IMAG_NAME = "pq_sig_imag"
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
    lateinit var pqReal: PQ
    lateinit var pqImag: PQ
    val pqStoreReal = db.atomicVar(PQ_REAL_NAME, PQ.Serializer).createOrOpen()
    val pqStoreImag = db.atomicVar(PQ_IMAG_NAME, PQ.Serializer).createOrOpen()
    val reversePermutation: IntArray
    val permutation: IntArray
    val signaturesReal = mutableListOf<IntArray>()
    val signaturesImag = mutableListOf<IntArray>()
    val signaturesTId = mutableListOf<Long>()
    val signaturesRealStore = db.indexTreeList(SIG_REAL_NAME, PQSignature.Serializer).createOrOpen() // todo: move to Htreemap which has pump for faster bulk data manipulation?
    val signaturesImagStore = db.indexTreeList(SIG_IMAG_NAME, PQSignature.Serializer).createOrOpen()

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
                this.config = PQIndexConfig(1, 1, 5e-3, LookupTablePrecision.SINGLE, 100, 1234L)
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
        with(generateRandomPermutation(columns[0].logicalSize, rng)) {
            reversePermutation = first
            permutation = second
        }
        pqStoreReal.get()?.let { pqReal = it }
        pqStoreImag.get()?.let { pqImag = it }
        LOGGER.info("Index ${name.simple} loading Signatures")
        signaturesRealStore.forEach {
            signaturesReal.add(it!!.signature)
            signaturesTId.add(it.tid)
        }
        signaturesImagStore.forEach { signaturesImag.add(it!!.signature) }
        LOGGER.info("Done.")
        this.db.commit() // this writes config stuff, so that the commit doesn't wait until rebuild()
        // note that due to this (if it works as expected, which is probably not the case),
        // now indexes that are not yet built can exist
        // and there are no indexEntries in the entity that cannot be opened...
        // (e.g if there is a failure during rebuild()...
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
        signaturesReal.clear()
        signaturesRealStore.clear()
        signaturesImag.clear()
        signaturesImagStore.clear()
        signaturesTId.clear()
        LOGGER.info("Permuting data.")
        // because tx doesn't have a simple .filter method where we can specify any old boolean, we need to
        // roll our own...
        // this filters the elements randomly based on the learning fraction and permutes them
        val (permutedLearningDataRealImag, learningTIds) =
                mutableListOf<Pair<Pair<DoubleArray, DoubleArray>, Long>>().apply {
            tx.forEach { r ->
                if (rng.nextDouble() > config.learningDataFraction) {
                    return@forEach
                }
                val reIm = permuteSplitComplexRecord(r)
                this.add(Pair(reIm, r.tupleId)) // todo: get rid of intermediary pairs... We need to know the number of records for that (which we can do now because we need to filter manually...)
            }
        }.unzip()
        val (permutedLearningDataReal, permutedLearningDataImag) = permutedLearningDataRealImag.unzip()
        LOGGER.info("Learning with ${learningTIds.size} vectors...")
        val (cbSigReal, cbSigImag) = listOf(permutedLearningDataReal, permutedLearningDataImag).map { d ->
            val data = d.toTypedArray()
            PQ.fromPermutedData(config.numSubspaces, config.numCentroids, data)
        }
        with(cbSigReal) {
            pqReal = first
            pqStoreReal.set(pqReal)
            signaturesReal.addAll(second)
            signaturesRealStore.addAll(second.mapIndexed { i, sign ->
                PQSignature(learningTIds[i], sign)
            })
        }
        with(cbSigImag) {
            pqImag = first
            pqStoreImag.set(pqImag)
            signaturesImag.addAll(second)
            signaturesImagStore.addAll(second.mapIndexed { i, sign ->
                PQSignature(learningTIds[i], sign)
            })
        }
        signaturesTId.addAll(learningTIds)
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
            val reIm = permuteSplitComplexRecord(r)
            val sigReal = pqReal.getSignature(reIm.first)
            signaturesReal.add(sigReal)
            signaturesRealStore.add(PQSignature(r.tupleId, sigReal))
            val sigImag = pqImag.getSignature(reIm.second)
            signaturesImag.add(sigImag)
            signaturesImagStore.add(PQSignature(r.tupleId, sigImag))
            signaturesTId.add(r.tupleId)
        }
        LOGGER.info("Done generating and storing signatures. Committing.")
        db.commit()
        LOGGER.info("PQIndex rebuild done.")
    }

    private fun permuteSplitComplexRecord(r: Record): Pair<DoubleArray, DoubleArray> {
        val v = (r[columns[0]] as ComplexVectorValue<*>)
        val re = DoubleArray(dimensionsPerSubspace * config.numSubspaces)
        val im = DoubleArray(dimensionsPerSubspace * config.numSubspaces)
        (0 until config.numSubspaces * dimensionsPerSubspace).forEach {
            val complexValue = v[it]
            re[permutation[it]] = complexValue.real.value.toDouble()
            im[permutation[it]] = complexValue.imaginary.value.toDouble()
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
        val knns = scanQueriesSign(p, this.pqReal, this.pqImag, k = approxK)
//        val knns = scanSignQueries(p, pqReal, pqImag)

        // get exact distances...
        LOGGER.info("Re-ranking $approxK approximative matches with exact distances.")
        val knnsExact = knns.mapIndexed { i, knn ->
            val knnNew = if (p.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection<ComparablePair<Long, DoubleValue>>(p.k)
            (0 until knn.size).forEach {
                val tid = knn[it].first
                val exact = tx.read(tid)[columns[0]]!! as ComplexVectorValue<*>
                knnNew.offer(ComparablePair(tid, p.distance(exact, p.query[i])))
            }
            val distDiff = knn.peek()!!.second - knnNew.peek()!!.second.value.toFloat()
            LOGGER.debug("query $i Distance difference between approximate best match and actual with approxK=$approxK: $distDiff")
            knnNew
        }
        LOGGER.info("Done")
        return KnnUtilities.selectToRecordset(this.produces.first(), knnsExact.toList())
    }


    /**
     * having queries be the slow loop seems to double the speed when running sequentially...
     * (120s vs 250s for 128x1 query on full 9M dict single precision). Using 2 lists of PQSignature
     * down to 108s when simply using 2 lists of IntArray and 1 list of Long
     * compare with parallelized and optimized full-scan of ca 65s
     */
    @ExperimentalUnsignedTypes
    private fun scanQueriesSign(p: KnnPredicate<*>, pqReal: PQ, pqImag: PQ, k: Int = p.k): List<Selection<ComparablePair<Long, Float>>> {
        // how about interleaving the signatures arrays?
        // on 128x1 query, this was about the same speed, despite taking ca 3s to interleave
        // -> might be beneficial for larger queries!
        LOGGER.info("Interleaving signature arrays")
        val sigLength = config.numSubspaces
        // JVM only actually has 1d arrays... -> not contiguous in ram
        val sigReIm = UShortArray(signaturesImag.size * sigLength * 2) {
            val i = it / (sigLength * 2)
            val j = it % (sigLength * 2)
            if (j < sigLength) signaturesReal[i][j].toUShort() else signaturesImag[i][j % sigLength].toUShort()
        }
        LOGGER.info("Done.")
        val knnQueries = p.query.mapIndexed { i, q_ ->
            val q = q_ as ComplexVectorValue<*>
            (if (k == 1) MinSingleSelection<ComparablePair<Long, Float>>() else MinHeapSelection<ComparablePair<Long, Float>>(k)) to q
        }
        knnQueries.parallelStream().forEach { (knn, q) ->
//            LOGGER.info("Processing query ${i + 1} of ${p.query.size}")
            LOGGER.info("Permuting query")
            val permutedQueryReal = DoubleArray(q.logicalSize)
            val permutedQueryImag = DoubleArray(q.logicalSize)
            q.forEachIndexed { j, c -> // directly accessing arrays via cast to ComplexXXVectorValue is not really faster overall
                permutedQueryReal[permutation[j]] = c.real.value.toDouble()
                permutedQueryImag[permutation[j]] = c.imaginary.value.toDouble()
            }
            LOGGER.info("Precomputing IPs between query and centroids")
            val queryCentroidIPRealReal = pqReal.precomputeCentroidQueryIPFloat(permutedQueryReal)
            val queryCentroidIPImagImag = pqImag.precomputeCentroidQueryIPFloat(permutedQueryImag)
            val queryCentroidIPRealImag = pqReal.precomputeCentroidQueryIPFloat(permutedQueryImag)
            val queryCentroidIPImagReal = pqImag.precomputeCentroidQueryIPFloat(permutedQueryReal)
            LOGGER.info("Scanning signatures")
            signaturesReal.indices.forEach {
                val sigOffset = it * sigLength * 2 // offset into sign array. first half of signature is real, other is im
                val tid = signaturesTId[it]
                val absIPSqApprox = ((
                        queryCentroidIPRealReal.approximateIP(sigReIm, sigOffset, sigLength)
                                + queryCentroidIPImagImag.approximateIP(sigReIm, sigOffset + sigLength, sigLength)).pow(2)
                        + (
                        queryCentroidIPImagReal.approximateIP(sigReIm, sigOffset + sigLength, sigLength)
                                - queryCentroidIPRealImag.approximateIP(sigReIm, sigOffset, sigLength)
                        ).pow(2)
                        )
//                if (knn.added < knn.k || knn.peek()!!.second > -absIPSqApprox) // do we really need to create a new pair every single time?
                knn.offer(ComparablePair(tid, -absIPSqApprox))
            }
        }
        return knnQueries.unzip().first
    }

    private fun scanSignQueries(p: KnnPredicate<*>, pqReal: PQ, pqImag: PQ): List<Selection<ComparablePair<Long, Float>>> {
        val knns = p.query.map {
            // todo: integrate kApproxScan and then merge with exact values after...
            if (p.k == 1) MinSingleSelection<ComparablePair<Long, Float>>() else MinHeapSelection<ComparablePair<Long, Float>>(p.k)
        }

        LOGGER.info("Permuting queries")
        val permutedQueryReal = Array(p.query.size) { DoubleArray(p.query[0].logicalSize) }
        val permutedQueryImag = Array(p.query.size) { DoubleArray(p.query[0].logicalSize) }
        p.query.forEachIndexed { i, q ->
            q as ComplexVectorValue<*>
            q.forEachIndexed { j, c ->
                permutedQueryReal[i][permutation[j]] = c.real.value.toDouble()
                permutedQueryImag[i][permutation[j]] = c.imaginary.value.toDouble()
            }
        }

        LOGGER.info("Precomputing IPs between queries and centroids")
        val queryCentroidIPRealReal = Array(p.query.size) {
            pqReal.precomputeCentroidQueryIPFloat(permutedQueryReal[it])
        }
        val queryCentroidIPImagReal = Array(p.query.size) {
            pqImag.precomputeCentroidQueryIPFloat(permutedQueryReal[it])
        }
        val queryCentroidIPImagImag = Array(p.query.size) {
            pqImag.precomputeCentroidQueryIPFloat(permutedQueryImag[it])
        }
        val queryCentroidIPRealImag = Array(p.query.size) {
            pqReal.precomputeCentroidQueryIPFloat(permutedQueryImag[it])
        }

        LOGGER.info("Scanning signatures")
        signaturesReal.indices.forEach {
            if (it % 1000000 == 0) {
                LOGGER.info("$it elements scanned.")
            }
            val re = signaturesReal[it]
            val im = signaturesImag[it]
            val tid = signaturesTId[it]
            p.query.indices.forEach { i ->
                val absIPSqApprox = ((
                        queryCentroidIPRealReal[i].approximateIP(re)
                                + queryCentroidIPImagImag[i].approximateIP(im)).pow(2)
                        + (
                        queryCentroidIPImagReal[i].approximateIP(im)
                                - queryCentroidIPRealImag[i].approximateIP(re)
                        ).pow(2)
                        )
                knns[i].offer(ComparablePair(tid, -absIPSqApprox))
            }
        }
        return knns
    }

    fun getApproximation(tid: Long): DoubleArray {
        TODO()
//        return DoubleArray(numSubspaces * dimensionsPerSubspace) { j ->
//            val k = permutation[j] / numSubspaces
//            codebooks[k].centroids[signatures.signatures[i][k]][permutation[j] % dimensionsPerSubspace]
//        }
    }

    fun getExact(i: Int): DoubleArray {
        TODO()
    }
}
