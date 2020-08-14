package org.vitrivr.cottontail.database.index.va

import org.vitrivr.cottontail.database.index.va.marks.Marks
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign

/*
 * todo: think about putting these functinos into a class that can precompute and cache the differences between
 *       marks and query points, etc...
 *
 */

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

private fun lowerBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts): Double {
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

private fun upperBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts): Double {
    return cellsVec.mapIndexed { i, cv ->
        max(componentProducts.value[i][cv], componentProducts.value[i][cv + 1])
    }.sum()
}

/**
 * is private to remove notnull checks in func...
 */
private fun lowerUpperBoundComponentProductsSum(cellsVec: IntArray, componentProducts: QueryMarkProducts, outArray: DoubleArray) {
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

fun absoluteComplexInnerProductSqUpperBoundCached(cellsVecReal: IntArray,
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

/**
 * consider putting this into a class where it's called and make private
 * to avoid Intrinsics.checkParameterNotNull calls...
 */
fun absoluteComplexInnerProductSqUpperBoundCached2(cellsVecReal: IntArray,
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
