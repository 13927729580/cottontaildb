package org.vitrivr.cottontail.database.index.vaplus

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.vitrivr.cottontail.model.values.Complex64Value
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import org.vitrivr.cottontail.testutils.getComplexVectorsFromFile

import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign
import kotlin.reflect.full.createType
import kotlin.reflect.full.declaredMemberProperties

internal class VAPlusTest {
    val vap = VAPlus()
    val random = Random(1234)
    val numVecs = 100
    val numDim = 20
    val marksPerDim = 100
    val realdata = Array(numVecs) {
        DoubleArray(numDim) { random.nextGaussian() }
    }
    val imaginarydata = Array(numVecs) { // imaginary parts
        DoubleArray(numDim) { random.nextGaussian() }
    }
    val realmarks = MarksGenerator.getEquidistantMarks(realdata, IntArray(realdata.first().size) { marksPerDim })
    val imaginarymarks = MarksGenerator.getEquidistantMarks(imaginarydata, IntArray(imaginarydata.first().size) { marksPerDim })

    data class BoundTighness(val lbDistReal: Double,
                             val ubDistReal: Double,
                             val lbDistImag: Double,
                             val ubDistImag: Double,
                             val lbDistAbs: Double,
                             val ubDistAbs: Double
    )
    @Test
    fun computeBounds() {
        val vector = realdata.first()
        val bounds = vap.computeBounds(vector, realmarks)
        println("vector")
        println(vector.joinToString())
        println("bounds first (lbounds)")
        bounds.first.forEach { println(it.joinToString()) }
        println("bounds second (ubounds)")
        bounds.second.forEach { println(it.joinToString()) }
    }

    @Test
    fun getCells() {
        val cells = vap.getCells(realdata.first(), realmarks)
        println("marks")
        for (m in realmarks) {
            println(m.joinToString())
        }
        println("cells")
        println(cells.joinToString())
    }

    @Test
    fun boundL2Real() {
        // this is a test implementing what's on p8 top of blott&weber 1997
        for (vector in realdata) {
            /*
            todo: there will be an indexOutOfBounds error if the vector component is not contained in the marks!
                  this can happen if the marks are not generated based on the entire data in the DB, but just sampled
                  from there! We could artificially add the Double.Max_VALUE and MIN_VALUE to the marks to catch this
                  see MarksGenerator
            */
            val query = DoubleArray(vector.size) { random.nextGaussian() }
            val cellsVec = vap.getCells(vector, realmarks)
            val cellsQuery = vap.getCells(query, realmarks)
            val realmarks = realmarks
            val lbl2sq = lowerBoundComponentDifferences(cellsVec, query, realmarks).map { it.pow(2) }.sum()
            val ubl2sq = upperBoundComponentDifferences(cellsVec, query, realmarks).map { it.pow(2) }.sum()
            val l2sq = DoubleVectorValue(vector).l2(DoubleVectorValue(query)).pow(2).value
            println("actual squared l2 norm $l2sq")
            println("ub of squared l2 norm $ubl2sq")
            println("lb of squared l2 norm $lbl2sq")
            assertTrue(l2sq <= ubl2sq, "actual l2 larger than upper bound!!")
            assertTrue(l2sq >= lbl2sq, "actual l2 smaller than lower bound!!")
        }
    }

    @Test
    fun boundL2Complex() {
        (realdata zip imaginarydata).forEach { (vectorReal, vectorImag) ->
            /*
            todo: there will be an indexOutOfBounds error if the vector component is not contained in the marks!
                  this can happen if the marks are not generated based on the entire data in the DB, but just sampled
                  from there! We could artificially add the Double.Max_VALUE and MIN_VALUE to the marks to catch this
                  see MarksGenerator
            */
            val queryReal = DoubleArray(vectorReal.size) { random.nextGaussian() }
            val queryImag = DoubleArray(vectorImag.size) { random.nextGaussian() }
            val cellsVecReal = vap.getCells(vectorReal, realmarks)
            val cellsVecImag = vap.getCells(vectorImag, imaginarymarks)
            val lbl2sqReal = lowerBoundComponentDifferences(cellsVecReal, queryReal, realmarks).map { it.pow(2) }.sum()
            val lbl2sqImag = lowerBoundComponentDifferences(cellsVecImag, queryImag, imaginarymarks).map { it.pow(2) }.sum()
            val lbl2sq = lbl2sqReal + lbl2sqImag
            val ubl2sqReal = upperBoundComponentDifferences(cellsVecReal, queryReal, realmarks).map { it.pow(2) }.sum()
            val ubl2sqImag = upperBoundComponentDifferences(cellsVecImag, queryImag, imaginarymarks).map { it.pow(2) }.sum()
            val ubl2sq = ubl2sqReal + ubl2sqImag
            val l2sq = Complex64VectorValue((vectorReal zip vectorImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray())
                    .l2(Complex64VectorValue((queryReal zip queryImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray()))
                    .pow(2).value
            println("actual squared l2 norm $l2sq")
            println("ub of squared l2 norm $ubl2sq")
            println("lb of squared l2 norm $lbl2sq")
            assertTrue(l2sq <= ubl2sq, "actual l2 larger than upper bound!!")
            assertTrue(l2sq >= lbl2sq, "actual l2 smaller than lower bound!!")
        }
    }

    @Test
    fun boundRealDotProduct() {
        realdata.forEach { vector ->
            val query = DoubleArray(vector.size) { random.nextGaussian() }
            val cellsVec = vap.getCells(vector, realmarks)
            val lbDP = lowerBoundComponentProductsSum(cellsVec, query, realmarks)
            val ubDP = upperBoundComponentProductsSum(cellsVec, query, realmarks)
            val DP = DoubleVectorValue(vector).dot(DoubleVectorValue(query))
            println("actual dot product $DP")
            println("lb of DP $lbDP")
            println("ub of DP $ubDP")
            assertTrue(DP >= lbDP, "actual DP smaller than lower bound!")
            assertTrue(DP <= ubDP, "actual DP greater than upper bound!")
        }
    }

    /*
    Tests whether the upper and lower bounds on the real and imaginary
     */
    @Test
    fun boundComplexDotProduct() {
        val boundTightnesses = (realdata zip imaginarydata).map {
            testComplexInnerProductBounds(
                    it.first,
                    it.second,
                    DoubleArray(it.first.size) { random.nextGaussian() },
                    DoubleArray(it.first.size) { random.nextGaussian() },
                    realmarks,
                    imaginarymarks
            )
        }
    }

    private fun testComplexInnerProductBounds(realParts: DoubleArray, imaginaryParts: DoubleArray, queryReal: DoubleArray, queryImag: DoubleArray, marksReal: Array<DoubleArray>, marksImag: Array<DoubleArray>): BoundTighness {
        val vecReal = realParts
        val vecImag = imaginaryParts
        val cellsVecReal = vap.getCells(vecReal, marksReal)
        val cellsVecImag = vap.getCells(vecImag, marksImag)
        val lbDPReal = lbComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
        val lbDPImag = lbComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
        val ubDPReal = ubComplexInnerProductReal(cellsVecReal, queryReal, cellsVecImag, queryImag, marksReal, marksImag)
        val ubDPImag = ubComplexInnerProductImag(cellsVecImag, queryReal, cellsVecReal, queryImag, marksReal, marksImag)
        val dot = Complex64VectorValue((vecReal zip vecImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray()).dot(Complex64VectorValue((queryReal zip queryImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray()))
        val DPReal = dot.real.value
        val DPImag = dot.imaginary.value
        println("actual real part of dot product $DPReal")
        println("lb of DPReal $lbDPReal")
        println("ub of DPReal $ubDPReal")
        println("actual imaginary part of dot product $DPImag")
        println("lb of DPImag $lbDPImag")
        println("ub of DPImag $ubDPImag")
        assertTrue(DPReal >= lbDPReal, "actual DPReal smaller than lower bound!")
        assertTrue(DPImag >= lbDPImag, "actual DPImag smaller than lower bound!")
        assertTrue(DPReal <= ubDPReal, "actual DPReal greater than upper bound!")
        assertTrue(DPImag <= ubDPImag, "actual DPImag greater than upper bound!")

        val lbDPabs = lbAbsoluteComplexInnerProduct(lbDPReal, ubDPReal, lbDPImag, ubDPImag)
        val ubDPabs = ubAbsoluteComplexInnerProduct(lbDPReal, ubDPReal, lbDPImag, ubDPImag)
        val dpabs = dot.abs().value
        println("actual magnitude of dot product $dpabs")
        println("lb of dpabs $lbDPabs")
        println("ub of dpabs $ubDPabs")
        assertTrue(dpabs >= lbDPabs, "actual DPabs smaller than lower bound!")
        assertTrue(dpabs <= ubDPabs, "actual DPabs greater than upper bound!")
        return BoundTighness(DPReal - lbDPReal, ubDPReal - DPReal, DPImag - lbDPImag, ubDPImag - DPImag, dpabs - lbDPabs, ubDPabs - dpabs)
    }

    private fun ubAbsoluteComplexInnerProduct(lbDPReal: Double, ubDPReal: Double, lbDPImag: Double, ubDPImag: Double) =
            (max(lbDPReal.pow(2), ubDPReal.pow(2)) + max(lbDPImag.pow(2), ubDPImag.pow(2))).pow(0.5)

    private fun lbAbsoluteComplexInnerProduct(lbDPReal: Double, ubDPReal: Double, lbDPImag: Double, ubDPImag: Double) =
            (if (lbDPReal.sign != ubDPReal.sign) {
                0.0
            } else {
                min(lbDPReal.pow(2), ubDPReal.pow(2))
            }
            +
            (if (lbDPImag.sign != ubDPImag.sign) {
                0.0
            } else {
                min(lbDPImag.pow(2), ubDPImag.pow(2))
            })
            ).pow(0.5)

    private fun ubComplexInnerProductImag(cellsVecImag: IntArray, queryReal: DoubleArray, cellsVecReal: IntArray, queryImag: DoubleArray, marksReal: Array<DoubleArray>, marksImag: Array<DoubleArray>) =
            upperBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - lowerBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)

    private fun ubComplexInnerProductReal(cellsVecReal: IntArray, queryReal: DoubleArray, cellsVecImag: IntArray, queryImag: DoubleArray, marksReal: Array<DoubleArray>, marksImag: Array<DoubleArray>) =
            upperBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + upperBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)

    private fun lbComplexInnerProductImag(cellsVecImag: IntArray, queryReal: DoubleArray, cellsVecReal: IntArray, queryImag: DoubleArray, marksReal: Array<DoubleArray>, marksImag: Array<DoubleArray>) =
            lowerBoundComponentProductsSum(cellsVecImag, queryReal, marksImag) - upperBoundComponentProductsSum(cellsVecReal, queryImag, marksReal)

    private fun lbComplexInnerProductReal(cellsVecReal: IntArray, queryReal: DoubleArray, cellsVecImag: IntArray, queryImag: DoubleArray, marksReal: Array<DoubleArray>, marksImag: Array<DoubleArray>) =
            lowerBoundComponentProductsSum(cellsVecReal, queryReal, marksReal) + lowerBoundComponentProductsSum(cellsVecImag, queryImag, marksImag)

    @ExperimentalStdlibApi
    @Test
    fun boundComplexDotProductFromFile() {
        val data = getComplexVectorsFromFile("src/test/resources/sampledVectors.csv/")
        val dataReal = data.map { vec -> vec.map { it.real.value }.toDoubleArray() }.toTypedArray()
        val dataImag = data.map { vec -> vec.map { it.imaginary.value }.toDoubleArray() }.toTypedArray()
        val marksReal = MarksGenerator.getEquidistantMarks(dataReal, IntArray(numDim) { marksPerDim })
        val marksImag = MarksGenerator.getEquidistantMarks(dataImag, IntArray(numDim) { marksPerDim })
        val marksRealNonUniform = MarksGenerator.getNonUniformMarks(dataReal, IntArray(numDim) { marksPerDim })
        val marksImagNonUniform = MarksGenerator.getNonUniformMarks(dataImag, IntArray(numDim) { marksPerDim })
        val marksRealEquallyPopulated = MarksGenerator.getEquallyPopulatedMarks(dataReal, IntArray(numDim) { marksPerDim })
        val marksImagEquallyPopulated = MarksGenerator.getEquallyPopulatedMarks(dataImag, IntArray(numDim) { marksPerDim })

        val boundTightnessesUniform = data.indices.flatMap { i ->
            data.indices.map { j ->
                testComplexInnerProductBounds(dataReal[i], dataImag[i], dataReal[j], dataImag[j], marksReal, marksImag)
            }
        }
        val boundTightnessesNonUniform = data.indices.flatMap { i ->
            data.indices.map { j ->
                testComplexInnerProductBounds(dataReal[i], dataImag[i], dataReal[j], dataImag[j], marksRealNonUniform, marksImagNonUniform)
            }
        }
        // todo: investigate why this gives worse average results...
        val boundTightnessesEquallyPopulated = data.indices.flatMap { i ->
            data.indices.map { j ->
                testComplexInnerProductBounds(dataReal[i], dataImag[i], dataReal[j], dataImag[j], marksRealEquallyPopulated, marksImagEquallyPopulated)
            }
        }
        BoundTighness::class.declaredMemberProperties.filter {p -> p.returnType == Double::class.createType()}.forEach { p ->
            listOf(boundTightnessesUniform to "uniform",
                    boundTightnessesNonUniform to "nonUniform",
                    boundTightnessesEquallyPopulated to "equallyPopulated").forEach {
                println(it.second)
                val values = it.first.map(p).map { it as Double }
                val sum = values.sum()
                val avg = sum / values.size
                val max = values.max()!!
                val min = values.min()!!
                println(p.name)
                println("min: $min, avg: $avg, max: $max")
            }
        }
    }

    /*
    Blott & Weber 1997 p.8 top
    todo: move out of test to actual code
     */
    private fun upperBoundComponentDifferences(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): List<Double> {
        val cellsQuery = vap.getCells(query, marks)
        return (cellsVec zip cellsQuery).mapIndexed { i, (v, cq) ->
            val a = { query[i] - marks[i][v] }
            val b = { marks[i][v + 1] - query[i] }
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
    todo: move out of test to actual code
     */
    private fun lowerBoundComponentDifferences(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): List<Double> {
        val cellsQuery = vap.getCells(query, marks)
        return (cellsVec zip cellsQuery).mapIndexed { i, (v, q) ->
            when {
                v < q -> {
                    query[i] - marks[i][v + 1]
                }
                v == q -> {
                    0.0
                }
                else -> {
                    marks[i][v] - query[i]
                }
            }
        }
    }

    /*
    Takes cells (approximation from a DB vector) and another Vector (query) and builds the lower-bound of the
    sum of element-by-element products (i.e. it returns a lower bound on the real dot product)
    todo: move out of test to actual code
     */
    private fun lowerBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): Double {
        return cellsVec.mapIndexed { i, cv ->
            if (query[i] < 0) {
                marks[i][cv + 1] * query[i]
            } else {
                marks[i][cv] * query[i]
            }
        }.sum()
    }

    /*
    Takes cells (approximation from a real DB vector) and another real Vector (query) and builds the upper-bound of the
    sum of element-by-element products (i.e. it returns an upper bound on the real dot product).
    Real vector means that this can be a vector of real parts or a vector of imaginary parts.
    todo: move out of test to actual code
     */
    private fun upperBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): Double {
        return cellsVec.mapIndexed { i, cv ->
            if (query[i] < 0) {
                marks[i][cv] * query[i]
            } else {
                marks[i][cv + 1] * query[i]
            }
        }.sum()
    }
}