package org.vitrivr.cottontail.database.index.pq

import org.apache.commons.math3.linear.MatrixUtils.*
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.stat.correlation.Covariance
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**
 * Class representing a codebook for a single subspace for Product Quantization
 * The codebook contains the centroids (real valued vectors) for the subspace
 */
class PQCodebook (val centroids: Array<DoubleArray>, val inverseDataCovarianceMatrix: RealMatrix) {
    companion object {
        private val LOGGER: Logger = LoggerFactory.getLogger(PQCodebook::class.java)
        fun learnFromData(subspaceData: Array<DoubleArray>, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            LOGGER.info("Calculating inverse data covariance matrix from supplied data.")
            val inverseDataCovMatrix = inverse(Covariance(subspaceData, false).covarianceMatrix)
            return learnFromData(subspaceData, inverseDataCovMatrix, numCentroids, maxIterations)
        }

        fun learnFromData(subspaceData: Array<DoubleArray>, inverseDataCovMatrix: RealMatrix, numCentroids: Int, maxIterations: Int): Pair<PQCodebook, IntArray> {
            val learnedCentroids = Array(numCentroids) { DoubleArray(subspaceData[0].size) }
            val signatures = IntArray(subspaceData.size)
            // kmeans clusterer with mahalanobis distance
            val clusterer = KMeansPlusPlusClusterer<Vector>(numCentroids, maxIterations) { a, b ->
                mahalanobisSqOpt(a, 0, a.size, b, 0, b.size, inverseDataCovMatrix)
            }
            LOGGER.info("Learning...")
            val centroidClusters = clusterer.cluster(subspaceData.mapIndexed { i, value ->
                Vector(value, i) })
            LOGGER.info("Done learning.")
            LOGGER.info("Building codebook and signatures from commons math result")
            centroidClusters.forEachIndexed { i, cluster ->
                learnedCentroids[i] = cluster.center.point
                cluster.points.forEach {
                    signatures[it.index] = i
                }
            }
            return PQCodebook(learnedCentroids, inverseDataCovMatrix) to signatures
        }

        inline fun mahalanobisSqNaive(a: DoubleArray, b: DoubleArray, inverseDataCovMatrix: RealMatrix): Double {
            /**
             * todo: speed this up!  This is 99% of CPU time... Mostly spent for the commons math operations
             */
            require(a.size == b.size)
            val diff = createColumnRealMatrix(DoubleArray(a.size) {
                a[it] - b[it]
            })
            val res = diff.transpose().multiply(inverseDataCovMatrix.multiply(diff))
            check(res.columnDimension == 1)
            check(res.rowDimension == 1)
            return res.data[0][0]
        }

        inline fun mahalanobisSqOpt(a: DoubleArray, aStart: Int, aLength: Int, b: DoubleArray, bStart: Int, bLength: Int, inverseDataCovMatrix: RealMatrix): Double {
            /**
             * faster :)
             */
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
    }

    init {
        require(isSymmetric(inverseDataCovarianceMatrix, 1e-5))
    }

    /**
     * returns the centroid index to which the supplied vector is quantized.
     * supplied vector v can be the full vector covering multiple subspaces
     * in this case start and length should be specified to indicate the range
     * of the subspace of this [PQCodebook]
     */
    fun quantizeVector(v: DoubleArray, start: Int = 0, length: Int = v.size): Int {
        return smallestMahalanobis(v, start, length)
    }

    /**
     * Will return the centroid index in the codebook to which the supplied vector has the largest inner product
     */
    private fun mips(v: DoubleArray, start: Int, length: Int): Int {
        var mipIndex = 0
        var mip = Double.NEGATIVE_INFINITY
        centroids.forEachIndexed { i, c ->
            var m =  0.0
            for (j in 0 until length) {
                m += v[start + j] * c[j]
            }
            if (m > mip) {
                mip = m
                mipIndex = i
            }
        }
        return mipIndex
    }

    /**
     * Will return the centroid index in the codebook to which the supplied vector hast the smallest
     * mahalanobis distance
     */
    private fun smallestMahalanobis(v: DoubleArray, start: Int, length: Int): Int {
        var mahIndex = 0
        var mah = Double.POSITIVE_INFINITY
        centroids.forEachIndexed { i, c ->
            val m = mahalanobisSqOpt(c, 0, c.size, v, start, length, inverseDataCovarianceMatrix)
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
