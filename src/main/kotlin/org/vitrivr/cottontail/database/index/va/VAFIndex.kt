package org.vitrivr.cottontail.database.index.va

import org.mapdb.Atomic
import org.mapdb.DBMaker
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.va.marks.Marks
import org.vitrivr.cottontail.database.index.va.marks.MarksGenerator
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities.selectToRecordset
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.math.knn.selection.Selection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign

class VAFIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>): Index() {

    companion object {
        val SIGNATURE_FIELD_NAME = "vaf_signatures"
        val REAL_MARKS_FIELD_NAME = "vaf_marks_real"
        val IMAG_MARKS_FIELD_NAME = "vaf_marks_imag"
        val MARKS_PER_DIM = 50  // doesn't have too much of an influence on execution time with current implementation -> something is quite inefficient... Influence on filter ratio is visible
        val LOGGER: Logger = LoggerFactory.getLogger(VAFIndex::class.java)
    }
    /** The [Path] to the [DBO]'s main file OR folder. */
    override val path: Path = this.parent.path.resolve("idx_vaf_$name.db")

    /** The [VAFIndex] implementation returns exactly the columns that is indexed. */
    override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("distance"), ColumnType.forName("DOUBLE")))

    /** The type of [Index]. */
    override val type = IndexType.VAF

    /** The internal [DB] reference. */
    private val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }
    /** Store for the marks. */
    private val marksRealStore: Atomic.Var<Marks> = db.atomicVar(REAL_MARKS_FIELD_NAME, Marks.MarksSerializer).createOrOpen()
    private val marksImagStore: Atomic.Var<Marks> = db.atomicVar(IMAG_MARKS_FIELD_NAME, Marks.MarksSerializer).createOrOpen()

    private var marksReal: Marks
    private var marksImag: Marks

    /** Store for the signatures. */
    private val signatures = this.db.indexTreeList(SIGNATURE_FIELD_NAME, VectorApproximationSignatureSerializer).createOrOpen()

    private lateinit var signaturesReal: Array<VectorApproximationSignature>
    private lateinit var signaturesImag: Array<VectorApproximationSignature>

    init {
        if (columns.size != 1) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing a single column")
        }
        if (!columns.all { it.type == ColumnType.forName("COMPLEX32_VEC") || it.type == ColumnType.forName("COMPLEX64_VEC") }) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing complex vector columns, not ${columns.first()::class.java}")
        }
        LOGGER.debug("Loading marks")
        marksReal = marksRealStore.get() ?: Marks(Array(1) { DoubleArray(1) { 0.0 } }) // todo: make clean...
        marksImag = marksImagStore.get() ?: Marks(Array(1) { DoubleArray(1) { 0.0 } })
        LOGGER.debug("Loading all signatures")
        updateSignaturesFromStore()
        LOGGER.debug("Done")
    }

    private fun updateSignaturesFromStore() {
        val signaturesReal = mutableListOf<VectorApproximationSignature>()
        val signaturesImag = mutableListOf<VectorApproximationSignature>()
        signatures.forEachIndexed { index, vectorApproximationSignature ->
            if (index % 2 == 0) {
                signaturesReal.add(vectorApproximationSignature!!)
            } else {
                signaturesImag.add(vectorApproximationSignature!!)
            }
        }
        this.signaturesReal = signaturesReal.toTypedArray()
        this.signaturesImag = signaturesImag.toTypedArray()
    }

    /**
     * Flag indicating if this [VAFIndex] has been closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Closes this [VAFIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!closed) {
            db.close()
            closed = true
        }
    }

    /**
     * Calculates the cost estimate if this [Index] processing the provided [Predicate].
     * todo: get real cost estimate
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate) = Cost.ZERO

    override fun canProcess(predicate: Predicate) =
            predicate is KnnPredicate<*>
                    && predicate.query.all { it is ComplexVectorValue<*> }
                    && predicate.columns.first() == this.columns[0]
                    && predicate.distance is AbsoluteInnerProductDistance

    /**
     * Returns true, if the [Index] supports incremental updates, and false otherwise.
     * false for now
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate() = false

    /**
     * (Re-)builds the [Index]. Invoking this method should rebuild the [Index] immediately, without the
     * need to commit (i.e. commit actions must take place inside).
     *
     * This is an internal method! External invocation is only possible through a [Index.Tx] object.
     *
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun rebuild(tx: Entity.Tx) {
        LOGGER.info("Rebuilding $name")
        val specimen = tx.read(tx.maxTupleId())[columns[0]] as ComplexVectorValue<*>
        // find min max for each dim to get marks
        val minReal = DoubleArray(specimen.logicalSize)
        val minImag = DoubleArray(specimen.logicalSize)
        val maxReal = DoubleArray(specimen.logicalSize)
        val maxImag = DoubleArray(specimen.logicalSize)
        LOGGER.debug("Finding min and max for each dim")
        tx.forEach { r ->
            (r[columns.first()] as ComplexVectorValue<*>).forEachIndexed { i, v ->
                minReal[i] = min(minReal[i], v.real.value.toDouble())
                minImag[i] = min(minImag[i], v.imaginary.value.toDouble())
                maxReal[i] = max(maxReal[i], v.real.value.toDouble())
                maxImag[i] = max(maxImag[i], v.imaginary.value.toDouble())
            }
        }
        val marksReal = MarksGenerator.getEquidistantMarks(minReal, maxReal, IntArray(specimen.logicalSize) { MARKS_PER_DIM })
        marksRealStore.set(marksReal)
        this.marksReal = marksReal
        val marksImag = MarksGenerator.getEquidistantMarks(minImag, maxImag, IntArray(specimen.logicalSize) { MARKS_PER_DIM })
        marksImagStore.set(marksImag)
        this.marksImag = marksImag

        LOGGER.debug("Generating signatures for vectors")
        tx.forEach {
            val (valueReal, valueImag) = (it[columns.first()] as ComplexVectorValue<*>).map { it.real.value.toDouble() to it.imaginary.value.toDouble() }.unzip()

            signatures.add(VectorApproximationSignature(it.tupleId, marksReal.getCells(valueReal.toDoubleArray())))
            signatures.add(VectorApproximationSignature(it.tupleId, marksImag.getCells(valueImag.toDoubleArray())))
        }
        db.commit()
        updateSignaturesFromStore()
        LOGGER.info("Done.")
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
        require(canProcess(predicate)) { "The supplied predicate $predicate is not supported by the index" }
        LOGGER.info("filtering")
        predicate as KnnPredicate<*>
        // need as array of primitives for performance
        val queriesSplit = Array(predicate.query.size) {i ->
            DoubleArray(predicate.query[i].logicalSize) {j ->
                (predicate.query[i] as ComplexVectorValue<*>).real(j).value.toDouble()
            } to DoubleArray(predicate.query[i].logicalSize) { j ->
                (predicate.query[i] as ComplexVectorValue<*>).imaginary(j).value.toDouble()
            }
        }
        LOGGER.debug("Precomputing all products of queries and marks")
        /* each query gets a 4-array with product of query and marks for real-real, imag-imag, real-imag, imag-real
           (query componentpart - marks)
         */
        val queriesMarksProducts = Array(queriesSplit.size) {i ->
            arrayOf(
                QueryMarkProducts(queriesSplit[i].first, marksReal),
                QueryMarkProducts(queriesSplit[i].second, marksImag),
                QueryMarkProducts(queriesSplit[i].first, marksImag),
                QueryMarkProducts(queriesSplit[i].second, marksReal)
            )
        }
        LOGGER.debug("Done")
        var countCandidates = 0
        var countRejected = 0
        LOGGER.debug("scanning records")
//        val triple = filterPartSimpleNoParallel(0, signaturesReal.lastIndex, predicate, queriesMarksProducts, tx)
        val triple = filterParallel(predicate, queriesMarksProducts, tx)
        val knns = triple.first
        countCandidates = triple.second
        countRejected = triple.third
        LOGGER.info("Done. Considered candidates: $countCandidates, rejected candidates: $countRejected (${countRejected.toDouble() / (countRejected + countCandidates) * 100} %)")

        return selectToRecordset(this.produces.first(), knns.toList())
    }

    private fun filterPartSimpleNoParallel(start: Int, endInclusive: Int, predicate: KnnPredicate<*>, queriesMarksProducts: Array<Array<QueryMarkProducts>>, tx: Entity.Tx): Triple<List<Selection<ComparablePair<Long,DoubleValue>>>, Int, Int> {
        LOGGER.debug("filtering from $start to $endInclusive")
        val knns = predicate.query.map {
            if (predicate.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection<ComparablePair<Long, DoubleValue>>(predicate.k)
        }
        var countCandidates1 = 0
        var countRejected1 = 0
        (start .. endInclusive).forEach {
            val sigReal = signaturesReal[it]
            val sigImag = signaturesImag[it]
            predicate.query.forEachIndexed { i, query ->
                val absIPDistLB = 1.0 - absoluteComplexInnerProductSqUpperBoundCached2(sigReal.signature, sigImag.signature, queriesMarksProducts[i][0], queriesMarksProducts[i][1], queriesMarksProducts[i][2], queriesMarksProducts[i][3]).pow(0.5)
                if (knns[i].size < predicate.k || knns[i].peek()!!.second > absIPDistLB) {
                    countCandidates1++
                    val tid = sigReal.tupleId
                    knns[i].offer(ComparablePair(tid, predicate.distance((tx.read(tid)[columns.first()] as ComplexVectorValue<*>), query)))
                } else {
                    countRejected1++
                }
            }
        }
        LOGGER.debug("done filtering from $start to $endInclusive")
        return Triple(knns, countCandidates1, countRejected1)
    }

    private fun filterParallel(predicate: KnnPredicate<*>, queriesMarksProducts: Array<Array<QueryMarkProducts>>, tx: Entity.Tx): Triple<List<Selection<ComparablePair<Long,DoubleValue>>>, Int, Int> {
        //split signatures to threads
        val numThreads = 2
        val elemsPerThread = signaturesReal.size / numThreads
        val remaining = signaturesReal.size % numThreads
        val exec = Executors.newFixedThreadPool(numThreads)
        val tasks = (0 until numThreads).map {
            Callable { filterPartSimpleNoParallel(it * elemsPerThread,
                    it * elemsPerThread + elemsPerThread - 1 + if (it == numThreads - 1) remaining else 0,
            predicate, queriesMarksProducts, tx)}
        }
        val fresults = exec.invokeAll(tasks)
        val res = fresults.map { it.get()}
        exec.shutdownNow()
        // merge
        LOGGER.debug("Merging results")
        return res.reduce { acc, perThread ->
            Triple((perThread.first zip acc.first).map { (knnPerThread, knnAcc) ->
                knnAcc.apply {
                    for (i in 0 until knnPerThread.size) offer(knnPerThread[i])
                }
            }, perThread.second + acc.second, perThread.third + acc.third)
        }
    }
}

/*
Blott & Weber 1997 p.8 top
 */
fun upperBoundComponentDifferences(cellsVec: IntArray, query: DoubleArray, marks: Marks): List<Double> {
    val cellsQuery = marks.getCells(query)
    return (cellsVec zip cellsQuery).mapIndexed { i, (v, cq) ->
        val a = { query[i] - marks.marks[i][v] }
        val b = { marks.marks[i][v + 1] - query[i] }
        when {
            v < cq -> {
                a()
            }
            v == cq -> {
                max(a(), b())
            }
            else -> {
                b()
            }
        }
    }
}

/*
Blott & Weber 1997 p.8 top
 */
fun lowerBoundComponentDifferences(cellsVec: IntArray, query: DoubleArray, marks: Marks): List<Double> {
    val cellsQuery = marks.getCells(query)
    return (cellsVec zip cellsQuery).mapIndexed { i, (v, q) ->
        when {
            v < q -> {
                query[i] - marks.marks[i][v + 1]
            }
            v == q -> {
                0.0
            }
            else -> {
                marks.marks[i][v] - query[i]
            }
        }
    }
}

/*
Takes cells (approximation from a DB vector) and another Vector (query) and builds the lower-bound of the
sum of element-by-element products (i.e. it returns a lower bound on the real dot product)
 */
fun lowerBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Marks): Double {
    return cellsVec.mapIndexed { i, cv ->
        if (query[i] < 0) {
            marks.marks[i][cv + 1] * query[i]
        } else {
            marks.marks[i][cv] * query[i]
        }
    }.sum()
}

fun lowerBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts): Double {
    return cellsVec.mapIndexed { i, cv ->
        min(componentProducts.value[i][cv], componentProducts.value[i][cv + 1])
    }.sum()
}

/*
Takes cells (approximation from a real DB vector) and another real Vector (query) and builds the upper-bound of the
sum of element-by-element products (i.e. it returns an upper bound on the real dot product).
Real vector means that this can be a vector of real parts or a vector of imaginary parts.
 */
fun upperBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Marks): Double {
    return cellsVec.mapIndexed { i, cv ->
        if (query[i] < 0) {
            marks.marks[i][cv] * query[i]
        } else {
            marks.marks[i][cv + 1] * query[i]
        }
    }.sum()
}

fun upperBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts): Double {
    return cellsVec.mapIndexed { i, cv ->
        max(componentProducts.value[i][cv], componentProducts.value[i][cv + 1])
    }.sum()
}

/**
 * is private to remove notnull checks in func...
 */
private inline fun lowerUpperBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts, outArray: DoubleArray) {
    var a = 0.0
    var b = 0.0
    cellsVec.forEachIndexed { i, cv ->
        val c = componentProducts.value[i][cv]
        val d = componentProducts.value[i][cv + 1]
        if (c < d) {
            a += c
            b += d
        } else {
            a += d
            b += c
        }
    }
    outArray[0] = a
    outArray[1] = b
}

fun realDotProductBounds(cellsVec: IntArray, query: DoubleArray, marks: Marks): Pair<Double, Double> =
        lowerBoundComponentProductsSum(cellsVec, query, marks) to upperBoundComponentProductsSum(cellsVec, query, marks)

fun ubComplexInnerProductImag(cellsVecImag: IntArray, queryReal: DoubleArray, cellsVecReal: IntArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks) =
        upperBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - lowerBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)

fun ubComplexInnerProductReal(cellsVecReal: IntArray, queryReal: DoubleArray, cellsVecImag: IntArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks) =
        upperBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + upperBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)

fun lbComplexInnerProductImag(cellsVecImag: IntArray, queryReal: DoubleArray, cellsVecReal: IntArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks) =
        lowerBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - upperBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)

fun lbComplexInnerProductReal(cellsVecReal: IntArray, queryReal: DoubleArray, cellsVecImag: IntArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks) =
        lowerBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + lowerBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)

fun ubAbsoluteComplexInnerProductSq(lbIPReal: Double, ubIPReal: Double, lbIPImag: Double, ubIPImag: Double) =
        max(lbIPReal.pow(2), ubIPReal.pow(2)) + max(lbIPImag.pow(2), ubIPImag.pow(2))


fun lbAbsoluteComplexInnerProductSq(lbIPReal: Double, ubIPReal: Double, lbIPImag: Double, ubIPImag: Double) =
        (if (lbIPReal.sign != ubIPReal.sign) {
            0.0
        } else {
            min(lbIPReal.pow(2), ubIPReal.pow(2))
        }
                +
                if (lbIPImag.sign != ubIPImag.sign) {
                    0.0
                } else {
                    min(lbIPImag.pow(2), ubIPImag.pow(2))
                })

fun absoluteComplexInnerProductSqBounds(cellsVecReal: IntArray, cellsVecImag: IntArray, queryReal: DoubleArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks): Pair<Double, Double> {
    val lbDPReal = lbComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
    val lbDPImag = lbComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
    val ubDPReal = ubComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
    val ubDPImag = ubComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
    return lbAbsoluteComplexInnerProductSq(lbDPReal, ubDPReal, lbDPImag, ubDPImag) to ubAbsoluteComplexInnerProductSq(lbDPReal, ubDPReal, lbDPImag, ubDPImag)
}

fun absoluteComplexInnerProductSqUpperBound(cellsVecReal: IntArray, cellsVecImag: IntArray, queryReal: DoubleArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks): Double {
    val lbDPReal = lbComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
    val lbDPImag = lbComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
    val ubDPReal = ubComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
    val ubDPImag = ubComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
    return ubAbsoluteComplexInnerProductSq(lbDPReal, ubDPReal, lbDPImag, ubDPImag)
}
fun absoluteComplexInnerProductSqUpperBoundInlined(cellsVecReal: IntArray, cellsVecImag: IntArray, queryReal: DoubleArray, queryImag: DoubleArray, marksReal: Marks, marksImag: Marks): Double {
    val lbIPReal = lowerBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + lowerBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)
    val lbIPImag = lowerBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - upperBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)
    val ubIPReal = upperBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + upperBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)
    val ubIPImag = upperBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - lowerBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)
    return max(lbIPReal.pow(2), ubIPReal.pow(2)) + max(lbIPImag.pow(2), ubIPImag.pow(2))
}

inline fun absoluteComplexInnerProductSqUpperBoundCached(cellsVecReal: IntArray,
                                                  cellsVecImag: IntArray,
                                                  queryMarkProductsRealReal: QueryMarkProducts,
                                                  queryMarkProductsImagImag: QueryMarkProducts,
                                                  queryMarkProductsRealImag: QueryMarkProducts,
                                                  queryMarkProductsImagReal: QueryMarkProducts): Double {
    val lbIPReal = lowerBoundComponentProductsSum(cellsVecReal, queryMarkProductsRealReal) + lowerBoundComponentProductsSum(cellsVecImag, queryMarkProductsImagImag)
    val lbIPImag = lowerBoundComponentProductsSum(cellsVecImag, queryMarkProductsRealImag) - upperBoundComponentProductsSum(cellsVecReal, queryMarkProductsImagReal)
    val ubIPReal = upperBoundComponentProductsSum(cellsVecReal, queryMarkProductsRealReal) + upperBoundComponentProductsSum(cellsVecImag, queryMarkProductsImagImag)
    val ubIPImag = upperBoundComponentProductsSum(cellsVecImag, queryMarkProductsRealImag) - lowerBoundComponentProductsSum(cellsVecReal, queryMarkProductsImagReal)
    return max(lbIPReal.pow(2), ubIPReal.pow(2)) + max(lbIPImag.pow(2), ubIPImag.pow(2))
}

private inline fun absoluteComplexInnerProductSqUpperBoundCached2(cellsVecReal: IntArray,
                                                   cellsVecImag: IntArray,
                                                   queryMarkProductsRealReal: QueryMarkProducts,
                                                   queryMarkProductsImagImag: QueryMarkProducts,
                                                   queryMarkProductsRealImag: QueryMarkProducts,
                                                   queryMarkProductsImagReal: QueryMarkProducts): Double {
    val realRealRealBounds = DoubleArray(2)
    lowerUpperBoundComponentProductsSum(cellsVecReal, queryMarkProductsRealReal, realRealRealBounds)
    val imagImagImagBounds = DoubleArray(2)
    lowerUpperBoundComponentProductsSum(cellsVecImag, queryMarkProductsImagImag, imagImagImagBounds)
    val imagRealImagBounds = DoubleArray(2)
    lowerUpperBoundComponentProductsSum(cellsVecImag, queryMarkProductsRealImag, imagRealImagBounds)
    val realImagRealBounds = DoubleArray(2)
    lowerUpperBoundComponentProductsSum(cellsVecReal, queryMarkProductsImagReal, realImagRealBounds)
    return max((realRealRealBounds[0] + imagImagImagBounds[0]).pow(2), (realRealRealBounds[1] + imagImagImagBounds[1]).pow(2)) + max((imagRealImagBounds[0] - realImagRealBounds[1]).pow(2), (imagRealImagBounds[1] - realImagRealBounds[0]).pow(2))
}

fun absoluteComplexInnerProductSqUpperBoundCached2Public(cellsVecReal: IntArray,
                                                           cellsVecImag: IntArray,
                                                           queryMarkProductsRealReal: QueryMarkProducts,
                                                           queryMarkProductsImagImag: QueryMarkProducts,
                                                           queryMarkProductsRealImag: QueryMarkProducts,
                                                           queryMarkProductsImagReal: QueryMarkProducts): Double {
    return absoluteComplexInnerProductSqUpperBoundCached2(cellsVecReal,
            cellsVecImag,
            queryMarkProductsRealReal,
            queryMarkProductsImagImag,
            queryMarkProductsRealImag,
            queryMarkProductsImagReal)
}