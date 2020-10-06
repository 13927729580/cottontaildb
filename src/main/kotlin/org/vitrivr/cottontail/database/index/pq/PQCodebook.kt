package org.vitrivr.cottontail.database.index.pq

import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.linear.*
import org.apache.commons.math3.linear.MatrixUtils.*
import org.apache.commons.math3.ml.clustering.CentroidCluster
import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.stat.correlation.Covariance
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.model.values.*
import org.vitrivr.cottontail.model.values.types.ComplexValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.model.values.types.RealVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.lang.IllegalArgumentException
import java.util.*
import kotlin.reflect.KClass

/**
 * Class representing a codebook for a single subspace for Product Quantization
 * The codebook contains the centroids (real valued vectors) for the subspace
 *
 */
class PQCodebook (val centroids: Array<out VectorValue<*>>, val inverseDataCovarianceMatrix: Array<out VectorValue<*>>) {
    companion object {
        private val LOGGER: Logger = LoggerFactory.getLogger(PQCodebook::class.java)
        /**
         * Internally, for real valued data, the clustering is done with apache commons k-means++ in double precision
         * but the returned codebook contains centroids of the same datatype as was supplied
         */
        fun learnFromData(subspaceData: Array<RealVectorValue<*>>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            LOGGER.info("Calculating inverse data covariance matrix from supplied data.")
            require(subspaceData.all { it::class.java == subspaceData[0]::class.java })
            val (castSubspaceData, inverseDataCovMatrix) =
                    when(subspaceData[0]) {
                        is DoubleVectorValue -> {
                            val array = Array(subspaceData.size) {
                                (subspaceData[it] as DoubleVectorValue).data
                            }
                            array to inverse(Covariance(array, false).covarianceMatrix)
                        }
                        is FloatVectorValue -> {
                            val array = Array(subspaceData.size) {
                                (subspaceData[it] as FloatVectorValue).data.map { j -> j.toDouble() }.toDoubleArray()
                            }
                            array to inverse(Covariance(array, false).covarianceMatrix)
                        }
                        else -> {
                            TODO("Other types not yet implemented for PQ")
                        }
                    }
            return learnFromData(castSubspaceData, inverseDataCovMatrix, numCentroids, maxIterations, subspaceData[0].javaClass)
        }

        fun learnFromData(subspaceData: Array<DoubleArray>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            val inverseDataCovMatrix = inverse(Covariance(subspaceData, false).covarianceMatrix)
            return learnFromData(subspaceData, inverseDataCovMatrix, numCentroids, maxIterations, DoubleVectorValue::class.java)
        }

        fun learnFromData(subspaceData: Array<DoubleArray>, inverseDataCovMatrix: RealMatrix, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            return learnFromData(subspaceData, inverseDataCovMatrix, numCentroids, maxIterations, DoubleVectorValue::class.java)
        }

        fun learnFromData(subspaceData: Array<DoubleArray>, inverseDataCovMatrix: RealMatrix, numCentroids: Int, maxIterations: Int, type: Class<*>): Pair<PQCodebook, IntArray> {
            val (signatures, centroidClusters) = clusterData(subspaceData, numCentroids, maxIterations, inverseDataCovMatrix)
            LOGGER.info("Building codebook and signatures from commons math result")
            val (learnedCentroids, castInvDataCovMatrix) =
                when(type) {
                     DoubleVectorValue::class.java -> {
                         Pair(Array(numCentroids) {
                             DoubleVectorValue(centroidClusters[it].center.point)
                         }, inverseDataCovMatrix.data.map { DoubleVectorValue(it) }.toTypedArray())
                    }
                    FloatVectorValue::class.java -> {
                        Pair(Array(numCentroids) {
                            FloatVectorValue(centroidClusters[it].center.point.map { v -> v.toFloat() }.toFloatArray())
                        }, inverseDataCovMatrix.data.map { FloatVectorValue(it.map { v -> v.toFloat() }.toFloatArray()) }.toTypedArray())
                    }
                    else -> throw IllegalArgumentException("Unsupported type ${type}")
                }
            centroidClusters.forEachIndexed { i, cluster ->
                cluster.points.forEach {
                    signatures[it.index] = i
                }
            }
            return PQCodebook(learnedCentroids, castInvDataCovMatrix) to signatures
        }

        private fun clusterData(subspaceData: Array<DoubleArray>, numCentroids: Int, maxIterations: Int, inverseDataCovMatrix: RealMatrix): Pair<IntArray, MutableList<CentroidCluster<Vector>>> {
            val signatures = IntArray(subspaceData.size)
            // kmeans clusterer with mahalanobis distance
            val clusterer = KMeansPlusPlusClusterer<Vector>(numCentroids, maxIterations) { a, b ->
                mahalanobisSqOpt(a, 0, a.size, b, 0, b.size, inverseDataCovMatrix)
            }
            LOGGER.info("Learning...")
            val centroidClusters = clusterer.cluster(subspaceData.mapIndexed { i, value ->
                Vector(value, i)
            })
            LOGGER.info("Done learning.")
            return Pair(signatures, centroidClusters)
        }

        /** todo: make learning work with complex data...
        *         apache commons clustering doesn't work with complex out of box, so need to probably roll our own...
        *         we could re-interpret the doubles in complex way for distance calculation which would enable
        *         us to use the commons clusterer
        */
        fun learnFromData(subspaceData: Array<out ComplexVectorValue<out Number>>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            val cov = complexCovarianceMatrix(subspaceData)
            val inverseDataCovMatrixCommons = invertComplexMatrix(cov)
            val inverseDataCovMatrix = fieldMatrixToVectorArray(inverseDataCovMatrixCommons, subspaceData[0]::class)
            val c = KMeansClustererComplex<ComplexVectorValue<*>>(subspaceData, SplittableRandom(1234L)) { a, b ->
                mahalanobisSqOpt(a, 0, a.logicalSize, b, 0, b.logicalSize, inverseDataCovMatrix)
            }
            val clusterResults = c.cluster(numCentroids, maxIterations)
            val signatures = IntArray(subspaceData.size)
            val centroids = clusterResults.mapIndexed { i, clusterCenter ->
                clusterCenter.clusterPointIndices.forEach { clusterPointIndex ->
                        signatures[clusterPointIndex] = i
                    }
                clusterCenter.center
            }.toTypedArray()
            return PQCodebook(centroids, inverseDataCovMatrix.map { it }.toTypedArray()) to signatures
        }

        /**
         * todo: I'm sure this when construct can be made prettier instead of explicit type checking. Of course also applies to code above
         */
        private fun fieldMatrixToVectorArray(matrixCommons: FieldMatrix<Complex>, type: KClass<out ComplexVectorValue<*>>): Array<ComplexVectorValue<*>> {
            return Array(matrixCommons.data.size) { i ->
                when (type) {
                    Complex32VectorValue::class -> {
                        Complex32VectorValue(matrixCommons.data[i].map { Complex32Value(it.real.toFloat(), it.imaginary.toFloat()) }.toTypedArray())
                    }
                    Complex64VectorValue::class -> {
                        Complex32VectorValue(matrixCommons.data[i].map { Complex64Value(it.real, it.imaginary) }.toTypedArray())
                    }
                    else -> {
                        error("Unsupported type $type")
                    }
                }
            }
        }

        private fun invertComplexMatrix(matrix: FieldMatrix<Complex>): FieldMatrix<Complex> {
            val lud = FieldLUDecomposition<Complex>(matrix)
            return lud.solver.inverse
        }

        /**
         * estimate cov matrix in real case via Q_XX = M_X * M_X^T
         * we estimate cov matrix in complex case as Q_XX = 1/n * M_X * M_X^H where ()^H means conjugate transpose
         */
        private fun complexCovarianceMatrix(data: Array<out ComplexVectorValue<*>>): FieldMatrix<Complex> {
            val fieldDataMatrix = BlockFieldMatrix<Complex>(Array(data[0].logicalSize) { row ->
                Array(data.size) { col ->
                    Complex(data[col][row].real.value.toDouble(), data[col][row].imaginary.value.toDouble())
                }
            })
            val fieldDataMatrixH = fieldDataMatrix.transpose()
            fieldDataMatrixH.walkInOptimizedOrder(object : FieldMatrixChangingVisitor<Complex> {
                override fun end() : Complex {return Complex.ZERO}
                override fun start(rows: Int, columns: Int, startRow: Int, endRow: Int, startColumn: Int, endColumn: Int) { return }
                override fun visit(row: Int, column: Int, value: Complex) = value.conjugate()
            })
            return fieldDataMatrix.multiply(fieldDataMatrixH).scalarMultiply(Complex(1.0 / data.size.toDouble(), 0.0))
        }

        /**
         * todo: only use a single length parameter, 2 make no sense...
         */
        inline fun mahalanobisSqOpt(a: DoubleArray, aStart: Int, aLength: Int, b: DoubleArray, bStart: Int, bLength: Int, inverseDataCovMatrix: RealMatrix): Double {
            require(aLength == bLength)
            require(inverseDataCovMatrix.columnDimension == aLength)
            require(inverseDataCovMatrix.rowDimension == aLength)
            var dist = 0.0
            val diff = DoubleArray(aLength) {
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
         * todo: only use a single length parameter, 2 make no sense...
         */
        inline fun mahalanobisSqOpt(a: VectorValue<*>, aStart: Int, aLength: Int, b: VectorValue<*>, bStart: Int, bLength: Int, inverseDataCovMatrix: Array<out VectorValue<*>>): Double {
            require(aLength == bLength)
            require(inverseDataCovMatrix.size == aLength)
            require(inverseDataCovMatrix[0].logicalSize == aLength)
            var dist: ComplexValue<*> = when (val t = a::class.java) {
                Complex32VectorValue::class.java -> Complex32Value.ZERO
                Complex64VectorValue::class.java -> Complex64Value.ZERO
                else -> error("Unknown type $t")
            }

            val diff = if (aLength != a.logicalSize || bLength != b.logicalSize) {
                (a.get(aStart, aLength) - b.get(bStart, bLength))
            }
            else {
                (a - b)
            }
            for (i in inverseDataCovMatrix.indices) {
                val ip = inverseDataCovMatrix[i].dot(diff)
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
        require(v.logicalSize == centroids[0].logicalSize)
        var mahIndex = 0
        var mah = Double.POSITIVE_INFINITY
        centroids.forEachIndexed { i, c ->
            val m = mahalanobisSqOpt(c, 0, c.logicalSize, v, start, length, inverseDataCovarianceMatrix)
            if (m < mah) {
                mah = m
                mahIndex = i
            }
        }
        return mahIndex

    }

}

private class Vector(val data: DoubleArray, val index: Int): Clusterable {
    override fun getPoint() = data
}
