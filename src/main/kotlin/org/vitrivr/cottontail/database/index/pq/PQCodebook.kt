package org.vitrivr.cottontail.database.index.pq

import org.apache.commons.math3.linear.*
import org.apache.commons.math3.linear.MatrixUtils.*
import org.apache.commons.math3.ml.clustering.CentroidCluster
import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.stat.correlation.Covariance
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.index.pq.clustering.KMeansClustererComplex
import org.vitrivr.cottontail.model.values.*
import org.vitrivr.cottontail.model.values.types.ComplexValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.model.values.types.RealVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.util.*
import kotlin.IllegalArgumentException
import kotlin.math.absoluteValue

/**
 * Class representing a codebook for a single subspace for Product Quantization
 * The codebook contains the centroids (real valued vectors) for the subspace
 * @property inverseDataCovarianceMatrix is the covariance matrix that was used for learning the codebook. Entries
 * in the array are column-vectors of that matrix
 */
class PQCodebook<T: VectorValue<*>> (val centroids: Array<T>, val inverseDataCovarianceMatrix: Array<T>) {
    companion object {
        private val LOGGER: Logger = LoggerFactory.getLogger(PQCodebook::class.java)

        /**
         * @param subspaceData entries in this array are subvectors of the data to index
         */
        fun learnFromRealData(subspaceData: Array<DoubleArray>, inverseDataCovMatrix: RealMatrix, numCentroids: Int, maxIterations: Int): Pair<PQCodebook<DoubleVectorValue>, IntArray> {
            val (centroidClusters, signatures) = clusterRealData(subspaceData, inverseDataCovMatrix, numCentroids, maxIterations)
            return PQCodebook(Array(numCentroids) {
                DoubleVectorValue(centroidClusters[it].center.point)
            }, inverseDataCovMatrix.data.map { DoubleVectorValue(it) }.toTypedArray()) to signatures
        }

        /**
         * @param subspaceData entries in this array are subvectors of the data to index
         */
        fun learnFromRealData(subspaceData: Array<DoubleArray>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook<DoubleVectorValue>, IntArray> {
            val inverseDataCovMatrix = inverse(Covariance(subspaceData, false).covarianceMatrix)
            return learnFromRealData(subspaceData, inverseDataCovMatrix, numCentroids, maxIterations)
        }

        /**
         * Internally, for real valued data, the clustering is done with apache commons k-means++ in double precision
         * but the returned codebook contains centroids of the same datatype as was supplied
         * @param subspaceData entries in this array are subvectors of the data to index
         */
        fun learnFromRealData(subspaceData: Array<out RealVectorValue<*>>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook<out RealVectorValue<*>>, IntArray> {
            LOGGER.info("Calculating inverse data covariance matrix from supplied data.")
            require(subspaceData.all { it::class.java == subspaceData[0]::class.java })
            return when(subspaceData[0]) {
                is DoubleVectorValue -> {
                    val data = Array(subspaceData.size) {
                        (subspaceData[it] as DoubleVectorValue).data
                    }
                    learnFromRealData(data, numCentroids, maxIterations)
                }
                is FloatVectorValue -> {
                    val array = Array(subspaceData.size) {
                        (subspaceData[it] as FloatVectorValue).data.map { j -> j.toDouble() }.toDoubleArray()
                    }
                    val inverseDataCovMatrix = inverse(Covariance(array, false).covarianceMatrix)
                    val (centroidClusters, signatures) = clusterRealData(array, inverseDataCovMatrix, numCentroids, maxIterations)
                    PQCodebook(Array(numCentroids) {
                        FloatVectorValue(centroidClusters[it].center.point.map { v ->
                            v.toFloat()
                        }.toFloatArray())
                    }, inverseDataCovMatrix.data.map {
                        FloatVectorValue(it.map {
                            v -> v.toFloat()
                        }.toFloatArray()) }.toTypedArray()) to signatures
                }
                else -> throw IllegalArgumentException("Other RealVectorValue types not implemented for PQ")
            }
        }

        private fun clusterRealData(subspaceData: Array<DoubleArray>, inverseDataCovMatrix: RealMatrix, numCentroids: Int, maxIterations: Int): Pair<MutableList<CentroidCluster<Vector>>, IntArray> {
            val clusterer = KMeansPlusPlusClusterer<Vector>(numCentroids, maxIterations) { a, b ->
                mahalanobisSqOpt(a, 0, a.size, b, 0, inverseDataCovMatrix)
            }
            LOGGER.info("Learning...")
            val centroidClusters = clusterer.cluster(subspaceData.mapIndexed { i, value ->
                Vector(value, i)
            })
            LOGGER.info("Done learning.")
            LOGGER.info("Building codebook and signatures from commons math result")
            val signatures = IntArray(subspaceData.size)
            centroidClusters.forEachIndexed { i, cluster ->
                cluster.points.forEach {
                    signatures[it.index] = i
                }
            }
            return centroidClusters to signatures
        }

        /** todo: apache commons clustering doesn't work with complex out of box, so need to probably roll our own...
        *         we could re-interpret the doubles in complex way for distance calculation which would enable
        *         us to use the commons clusterer
        */
        fun learnFromComplexData(subspaceData: Array<out ComplexVectorValue<out Number>>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook<out ComplexVectorValue<*>>, IntArray> {
            val cov = complexCovarianceMatrix(subspaceData)
            val inverseDataCovMatrixCommons = invertComplexMatrix(cov)
            val inverseDataCovMatrix = fieldMatrixToVectorArray(inverseDataCovMatrixCommons, subspaceData[0]::class)
//            val (signatures, centroids) = clusterComplexData(numCentroids, inverseDataCovMatrix, subspaceData, maxIterations)
            val (signatures, centroids) = clusterComplexDataPlusPlus(numCentroids, inverseDataCovMatrix, subspaceData, maxIterations)
            return when (subspaceData[0]) {
                is Complex32VectorValue -> {
                    PQCodebook(centroids.map{ it as Complex32VectorValue}.toTypedArray(), inverseDataCovMatrix.map{ it as Complex32VectorValue}.toTypedArray()) to signatures
                }
                is Complex64VectorValue -> {
                    PQCodebook(centroids.map { it as Complex64VectorValue }.toTypedArray(), inverseDataCovMatrix.map { it as Complex64VectorValue }.toTypedArray()) to signatures
                }
                else -> error("Unsupported type ${subspaceData[0]::class}")
            }
        }

        /**
         * @param inverseDataCovMatrix is an array of column-vectors of the non-centered data covariance matrix
         */
        private fun clusterComplexData(numCentroids: Int, inverseDataCovMatrix: Array<out ComplexVectorValue<*>>, subspaceData: Array<out ComplexVectorValue<out Number>>, maxIterations: Int): Pair<IntArray, Array<out ComplexVectorValue<*>>> {
            val c = KMeansClustererComplex<ComplexVectorValue<*>>(numCentroids, SplittableRandom(1234L)) { a, b ->
                mahalanobisSqOpt(a, 0, a.logicalSize, b, 0, inverseDataCovMatrix)
            }
            val clusterResults = c.cluster(subspaceData, maxIterations)
            val signatures = IntArray(subspaceData.size)
            val centroids = clusterResults.mapIndexed { i, clusterCenter ->
                clusterCenter.clusterPointIndices.forEach { clusterPointIndex ->
                    signatures[clusterPointIndex] = i
                }
                clusterCenter.center
            }.toTypedArray()
            return Pair(signatures, centroids)
        }

        /**
         * @param inverseDataCovMatrix is an array of column-vectors of the non-centered data covariance matrix
         */
        private fun clusterComplexDataPlusPlus(numCentroids: Int, inverseDataCovMatrix: Array<out ComplexVectorValue<*>>, subspaceData: Array<out ComplexVectorValue<*>>, maxIterations: Int): Pair<IntArray, Array<out ComplexVectorValue<*>>> {
            // to decouple from the implementation of [Complex64VectorValue] or [Complex32VectorValue] we need copy the
            // data to a DoubleArray with a layout that we decide...
            val subspaceDataDoubles = subspaceData.map { v ->
                DoubleArray(v.logicalSize shl 1) { i ->
                    if (i % 2 == 0) v[i / 2].real.value.toDouble() else v[i / 2].imaginary.value.toDouble()
                }
            }.toTypedArray()
            // do same for matrix. But it will be more efficient later on if we have it as row vectors
//            val inverseDataCovMatrixDoubles = inverseDataCovMatrix.map { v ->
//                DoubleArray(v.logicalSize shl 1) { i ->
//                    if (i % 2 == 0) v[i / 2].real.value.toDouble() else v[i / 2].imaginary.value.toDouble()
//                }
//            }.toTypedArray()
            val inverseDataCovMatrixRowVectors = Array(inverseDataCovMatrix[0].logicalSize) { row ->
                DoubleArray(inverseDataCovMatrix.size shl 1) { colComponent ->
                    if (colComponent % 2 == 0) inverseDataCovMatrix[colComponent / 2][row].real.value.toDouble() else inverseDataCovMatrix[colComponent / 2][row].imaginary.value.toDouble()
                }
            }
            val dist: (a: DoubleArray, b: DoubleArray) -> Double = { a, b ->
                require(a.size == b.size)
                // d = (a-b)^T*cov^(-1)*(a-b)
                // or is it d = (a-b)^H*cov^(-1)*(a-b)??
                val diff = DoubleArray(a.size) {
                    a[it] - b[it]
                }
                var dReal = 0.0
                var dImag = 0.0
                for (mRow in inverseDataCovMatrixRowVectors.indices) {
                    var hReal = 0.0
                    var hImag = 0.0
                    for(vCol in 0 until diff.size / 2) { // iterate vector components (1/2 because each component is 2 elements (real + imag)
                        // now do basically complex version of what's happening in mahalanobisSqOpt()
//                        hReal += inverseDataCovMatrixRowVectors[mRow][vCol * 2] * diff[vCol * 2] - inverseDataCovMatrixRowVectors[mRow][vCol * 2 + 1] * diff[vCol * 2 + 1] // case without conjugation
//                        hImag += inverseDataCovMatrixRowVectors[mRow][vCol * 2 + 1] * diff[vCol * 2] + inverseDataCovMatrixRowVectors[mRow][vCol * 2] * diff[vCol * 2 + 1]
                        hReal += inverseDataCovMatrixRowVectors[mRow][vCol * 2] * diff[vCol * 2] + inverseDataCovMatrixRowVectors[mRow][vCol * 2 + 1] * diff[vCol * 2 + 1] // case with conjugation. Without it, dist is not real! todo: figure
                        hImag += inverseDataCovMatrixRowVectors[mRow][vCol * 2 + 1] * diff[vCol * 2] - inverseDataCovMatrixRowVectors[mRow][vCol * 2] * diff[vCol * 2 + 1]
                    }
                    dReal += diff[mRow * 2] * hReal - diff[mRow * 2 + 1] * hImag
                    dImag += diff[mRow * 2 + 1] * hReal + diff[mRow * 2] * hImag // todo: once we're confident, we can drop this...
                }
                check(dImag.absoluteValue < 1e-5) {"Distance should be real but imaginary part was $dImag"}
                check(dReal >= 0) {"Distance must be >= 0 but was $dReal"}
                dReal
            }
            val distAbsIP: (a: DoubleArray, b: DoubleArray) -> Double = { a, b ->
                //make a distance that returns 1 - abs(a.dot(b)) (a, b are normalized)
                val a_ = Complex64VectorValue(Array(a.size / 2) {
                    Complex64Value(a[it * 2], a[it * 2 + 1])
                })
                val b_ = Complex64VectorValue(Array(b.size / 2) {
                    Complex64Value(b[it * 2], b[it * 2 + 1])
                })
                val d = 1.0 - a_.dot(b_).abs().value
                check(d >= -1e-5) {"Distance must be >= 0 but was $d"}
                d.coerceAtLeast(0.0)
            }
            val c = KMeansPlusPlusClusterer<Vector>(numCentroids, maxIterations, dist)
//            val c = KMeansPlusPlusClusterer<Vector>(numCentroids, maxIterations, distAbsIP) // this guy doesn't converge...
            val clusterResults = c.cluster(subspaceDataDoubles.mapIndexed { i, v -> Vector(v, i) })
            val signatures = IntArray(subspaceData.size)
            clusterResults.forEachIndexed { i, clusterCenter ->
                clusterCenter.points.forEach { v ->
                    signatures[v.index] = i
                }
            }
            val centers = when(subspaceData[0]) {
                is Complex32VectorValue -> {
                    clusterResults.map { cCenter ->
                        Complex32VectorValue(Array (cCenter.center.point.size / 2) {
                            Complex32Value(cCenter.center.point[it * 2], cCenter.center.point[it * 2 + 1])
                            }
                        )
                    }.toTypedArray()
                }
                is Complex64VectorValue -> {
                    clusterResults.map { cCenter ->
                        Complex64VectorValue(Array (cCenter.center.point.size / 2) {
                            Complex64Value(cCenter.center.point[it * 2], cCenter.center.point[it * 2 + 1])
                            }
                        )
                    }.toTypedArray()
                }
                else -> throw IllegalArgumentException("Unsupported type ${subspaceData[0]::class}")
            }
            return signatures to centers
        }

        inline fun mahalanobisSqOpt(a: DoubleArray, aStart: Int, length: Int, b: DoubleArray, bStart: Int, inverseDataCovMatrix: RealMatrix): Double {
            require(inverseDataCovMatrix.columnDimension == length)
            require(inverseDataCovMatrix.rowDimension == length)
            var dist = 0.0
            val diff = DoubleArray(length) {
                a[aStart + it] - b[bStart + it]
            }
            for (i in 0 until inverseDataCovMatrix.columnDimension) {
                var h = 0.0
                for (j in diff.indices) {
                    h += diff[j] * inverseDataCovMatrix.getEntry(i, j)
                }
                dist += h * diff[i]
            }
            return dist
        }

        /**
         * todo: check efficiency of this. It will probably be used A LOT
         *       is this correct for complex case? dot includes a conjugate. Normal matrix multiplication not
         * @param inverseDataCovMatrix is a collection of column-vectors
         */
        inline fun mahalanobisSqOpt(a: VectorValue<*>, aStart: Int, length: Int, b: VectorValue<*>, bStart: Int, inverseDataCovMatrix: Array<out VectorValue<*>>): Double {
            require(inverseDataCovMatrix.size == length)
            require(inverseDataCovMatrix[0].logicalSize == length)
            var dist: ComplexValue<*> = when (val t = a::class.java) {
                Complex32VectorValue::class.java -> Complex32Value.ZERO
                Complex64VectorValue::class.java -> Complex64Value.ZERO
                else -> error("Unknown type $t")
            }

            val diff = a.minus(b, aStart, bStart, length)
            for (i in inverseDataCovMatrix.indices) {
                val ip = inverseDataCovMatrix[i].dot(diff) // todo: should we use dot here (because of conj)?
                val v = diff[i] * ip
                dist += v
            }
            check(dist.imaginary.abs().value.toDouble() < 1e-5)
            return dist.real.value.toDouble()
        }

    }

    init {
        require(centroids.all { c -> c.logicalSize == centroids[0].logicalSize && centroids[0]::class.java == c::class.java})
        require(centroids[0].logicalSize == inverseDataCovarianceMatrix.size)
        require(inverseDataCovarianceMatrix[0].logicalSize == inverseDataCovarianceMatrix.size
                && inverseDataCovarianceMatrix.all {
                    it.logicalSize == inverseDataCovarianceMatrix[0].logicalSize
                    && it::class.java == inverseDataCovarianceMatrix[0]::class.java
                }){ "The inverted data covariance matrix must be square and all entries must be of the" +
                    "same type!" }
        require(inverseDataCovarianceMatrix.indices.all { inverseDataCovarianceMatrix[it][it].imaginary.value.toDouble() < 1.0e-5 })
            { "Inverse of data covariance matrix does not have a real diagonal!" }
        // todo: do better test to actually test for being hermitian, not just whether it's square...
//        require(isSymmetric(inverseDataCovarianceMatrix, 1e-5))
    }

    /**
     * returns the centroid index to which the supplied vector is quantized.
     * supplied vector v can be the full vector covering multiple subspaces
     * in this case start and length should be specified to indicate the range
     * of the subspace of this [PQCodebook]
     */
    fun quantizeVector(v: VectorValue<*>, start: Int = 0, length: Int = v.logicalSize): Int {
        return smallestMahalanobis(v, start, length)
    }


    /**
     * Will return the centroid index in the codebook to which the supplied vector has the smallest
     * mahalanobis distance
     */
    private fun smallestMahalanobis(v: VectorValue<*>, start: Int, length: Int): Int {
        require(length == centroids[0].logicalSize)
        var mahIndex = 0
        var mah = Double.POSITIVE_INFINITY
        centroids.forEachIndexed { i, c ->
            val m = mahalanobisSqOpt(c, 0, length, v, start, inverseDataCovarianceMatrix)
            if (m < mah) {
                mah = m
                mahIndex = i
            }
        }
        return mahIndex

    }

    /**
     * returns a list of lists of strings. First element is header of data as list of strings,
     * following elements are the centroid data
     */
    fun centroidsToString(): List<List<String>> {
        return when (val firstC = centroids[0]) {
            is RealVectorValue<*> -> {
                val header = listOf("Index") + firstC.indices.map { "component$it" }
                val data = centroids.mapIndexed { i, c ->
                    listOf("$i") + (c as RealVectorValue<*>).map { it.value.toDouble().toString() }
                }
                check(header.size == data[0].size)
                listOf(header) + data
            }
            is ComplexVectorValue<*> -> {
                val header = listOf("Index") + firstC.indices.flatMap { listOf("component${it}Real", "component${it}Imag") }
                val data = centroids.mapIndexed { i, c ->
                    listOf("$i") + (c as ComplexVectorValue<*>).flatMap { component ->
                        listOf(component.real, component.imaginary).map { it.value.toDouble().toString() }
                    }
                }
                check(header.size == data[0].size)
                listOf(header) + data
            }
            else -> {
                error("unexpected type ${firstC::class}")
            }
        }
    }
}

private class Vector(val data: DoubleArray, val index: Int): Clusterable {
    override fun getPoint() = data
}
