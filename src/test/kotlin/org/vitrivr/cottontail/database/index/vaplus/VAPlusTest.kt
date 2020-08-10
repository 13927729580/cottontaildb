package org.vitrivr.cottontail.database.index.vaplus

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.vitrivr.cottontail.model.values.Complex64Value
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue

import java.util.*
import kotlin.math.max
import kotlin.math.pow

internal class VAPlusTest {
    val vap = VAPlus()
    val random = Random(1234)
    val realdata = Array(10) {
        DoubleArray(20) { random.nextGaussian() }
    }
    val imaginarydata = Array(10) { // imaginary parts
        DoubleArray(20) { random.nextGaussian() }
    }
    val realmarks = MarksGenerator.getEquidistantMarks(realdata, IntArray(realdata.first().size) { 100 })
    val imaginarymarks = MarksGenerator.getEquidistantMarks(imaginarydata, IntArray(imaginarydata.first().size) { 100 })

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
        (realdata zip imaginarydata).forEach {(vectorReal, vectorImag) ->
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
                    .l2(Complex64VectorValue((queryReal zip queryImag).map { (a, b) -> Complex64Value(a, b)}.toTypedArray()))
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
        realdata.forEach {vector ->
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
    we can reliably get bounds on the real or imaginary part of the product, but not on the magnitude!
    this is because of the non-monotonicity of abs()...
    large negative values will become large positive values...
    we can't make any assumption on the sign of the vector components...
     */
    @Test
    fun boundComplexDotProduct() {
        (realdata zip imaginarydata).forEach {(vecReal, vecImag) ->
            val queryReal = DoubleArray(vecReal.size) { random.nextGaussian() }
            val queryImag = DoubleArray(vecImag.size) { random.nextGaussian() }
            val cellsVecReal = vap.getCells(vecReal, realmarks)
            val cellsVecImag = vap.getCells(vecImag, imaginarymarks)
            val lbDPReal = lowerBoundComponentProductsSum(cellsVecReal, queryReal, realmarks) + lowerBoundComponentProductsSum(cellsVecImag, queryImag, imaginarymarks)
            val lbDPImag = lowerBoundComponentProductsSum(cellsVecImag, queryReal, imaginarymarks) - upperBoundComponentProductsSum(cellsVecReal, queryImag, realmarks)
            val ubDPReal = upperBoundComponentProductsSum(cellsVecReal, queryReal, realmarks) + upperBoundComponentProductsSum(cellsVecImag, queryImag, imaginarymarks)
            val ubDPImag = upperBoundComponentProductsSum(cellsVecImag, queryReal, imaginarymarks) - lowerBoundComponentProductsSum(cellsVecReal, queryImag, realmarks)
            val dot = Complex64VectorValue((vecReal zip vecImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray()).dot(Complex64VectorValue((queryReal zip queryImag).map { (a, b) -> Complex64Value(a, b) }.toTypedArray()))
            val DPReal = dot.real.value
            val DPImag = dot.imaginary.value
            println("actual real part of dot product $DPReal")
            println("actual imaginary part of dot product $DPImag")
            println("lb of DPReal $lbDPReal")
            println("lb of DPImag $lbDPImag")
            println("ub of DPReal $ubDPReal")
            println("ub of DPImag $ubDPImag")
            assertTrue(DPReal >= lbDPReal, "actual DPReal smaller than lower bound!")
            assertTrue(DPImag >= lbDPImag, "actual DPImag smaller than lower bound!")
            assertTrue(DPReal <= ubDPReal, "actual DPReal greater than upper bound!")
            assertTrue(DPImag <= ubDPImag, "actual DPImag greater than upper bound!")
        }
    }

    /*
    Blott & Weber 1997 p.8 top
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

     */
    private fun lowerBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): Double {
        return cellsVec.mapIndexed { i, cv ->
            if (query[i] < 0) {
                    marks[i][cv + 1] * query[i]
                }
            else {
                marks[i][cv] * query[i]
            }
        }.sum()
    }

    /*
    Takes cells (approximation from a real DB vector) and another real Vector (query) and builds the upper-bound of the
    sum of element-by-element products (i.e. it returns an upper bound on the real dot product).
    Real vector means that this can be a vector of real parts or a vector of imaginary parts.
     */
    private fun upperBoundComponentProductsSum(cellsVec: IntArray, query: DoubleArray, marks: Array<DoubleArray>): Double {
        return cellsVec.mapIndexed { i, cv ->
            if (query[i] < 0) {
                marks[i][cv] * query[i]
            }
            else {
                marks[i][cv + 1] * query[i]
            }
        }.sum()
    }
}