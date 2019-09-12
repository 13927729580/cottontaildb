package ch.unibas.dmi.dbis.cottontail.math.knn.metrics

import ch.unibas.dmi.dbis.cottontail.utilities.extensions.minus
import ch.unibas.dmi.dbis.cottontail.utilities.extensions.plus
import ch.unibas.dmi.dbis.cottontail.utilities.extensions.times
import ch.unibas.dmi.dbis.cottontail.utilities.extensions.toDouble
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.reduce3.*
import org.nd4j.linalg.factory.Nd4j
import java.util.*
import kotlin.math.*

enum class Distance : DistanceFunction {
    /**
     * L1 or Manhattan distance between two vectors. Vectors must be of the same size!
     */
    L1 {
        override val operations: Int = 1
        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = Nd4j.getExecutioner().exec(ManhattanDistance(a,b, 0)).getDouble(0)
        override fun invoke(a: INDArray, b: INDArray): Double = Nd4j.getExecutioner().exec(ManhattanDistance(a,b, 0)).getDouble(0)
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                sum += when {
                    !a[i] && b[i] -> 1
                    a[i] && !b[i] -> -1
                    else -> 0
                } * weights[i]
            }
            return sum
        }
        override fun invoke(a: BitSet, b: BitSet): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                sum += when {
                    !a[i] && b[i] -> 1
                    a[i] && !b[i] -> -1
                    else -> 0
                }
            }
            return sum
        }
    },

    /**
     * L2 or Euclidian distance between two vectors. Vectors must be of the same size!
     */
    L2 {
        override val operations: Int = 2
        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = Nd4j.getExecutioner().exec(EuclideanDistance(a,b, 0)).getDouble(0)
        override fun invoke(a: INDArray, b: INDArray): Double = Nd4j.getExecutioner().exec(EuclideanDistance(a,b, 0)).getDouble(0)
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double = sqrt(L2SQUARED(a, b, weights))
        override fun invoke(a: BitSet, b: BitSet): Double = sqrt(L2SQUARED(a, b))
    },

    /**
     * Squared L2 or Euclidian distance between two vectors. Vectors must be of the same size!
     */
    L2SQUARED {
        override val operations: Int = 2
        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = Nd4j.getExecutioner().exec(EuclideanDistance(a,b, 0)).getDouble(0).pow(2)
        override fun invoke(a: INDArray, b: INDArray): Double = Nd4j.getExecutioner().exec(EuclideanDistance(a,b, 0)).getDouble(0).pow(2)
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                sum += (b[i] - a[i]) * (b[i] - a[i]) * weights[i]
            }
            return sum
        }

        override fun invoke(a: BitSet, b: BitSet): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                sum += (b[i] - a[i]) * (b[i] - a[i])
            }
            return sum
        }
    },

    /**
     * Chi Squared distance between two vectors. Vectors must be of the same size!
     */
    CHISQUARED {
        override val operations: Int = 3

        override fun invoke(a: INDArray, b: INDArray): Double {
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }

        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double {
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }

        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                if (abs(a[i] + b[i]) > 0) {
                    sum += ((b[i] - a[i]) * (b[i] - a[i])) / (b[i] + a[i]) * weights[i]
                }
            }
            return sum
        }

        override fun invoke(a: BitSet, b: BitSet): Double {
            var sum = 0.0
            for (i in 0 until b.size()) {
                if (abs(a[i] + b[i]) > 0) {
                    sum += ((b[i] - a[i]) * (b[i] - a[i])) / (b[i] + a[i])
                }
            }
            return sum
        }
    },

    COSINE {

        override val operations: Int = 3
        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = Nd4j.getExecutioner().exec(CosineDistance(a,b)).getDouble(0)
        override fun invoke(a: INDArray, b: INDArray): Double = Nd4j.getExecutioner().exec(CosineDistance(a,b)).getDouble(0)
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double {
            var dot = 0.0
            var c = 0.0
            var d = 0.0

            for (i in 0 until b.size()) {
                dot += a[i] * b[i] * weights[i]
                c += a[i] * a[i] * weights[i]
                d += b[i] * b[i] * weights[i]
            }
            val div = sqrt(c) * sqrt(d)

            return if (div < 1e-6 || div.isNaN()) {
                1.0
            } else {
                1.0 - dot / div
            }
        }
        override fun invoke(a: BitSet, b: BitSet): Double {
            var dot = 0L
            var c = 0L
            var d = 0L

            for (i in 0 until b.size()) {
                dot += a[i] * b[i]
                c += a[i] * a[i]
                d += b[i] * b[i]
            }
            val div = sqrt(c.toDouble()) * sqrt(d.toDouble())

            return if (div < 1e-6 || div.isNaN()) {
                1.0
            } else {
                1.0 - dot / div
            }
        }
    },

    /**
     * Hamming distance: Makes an element wise comparison of the two arrays and increases the distance by 1, everytime two corresponding elements don't match.
     */
    HAMMING {
        override val operations: Int = 1
        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = Nd4j.getExecutioner().exec(HammingDistance(a,b)).getDouble(0)
        override fun invoke(a: INDArray, b: INDArray): Double = Nd4j.getExecutioner().exec(HammingDistance(a,b)).getDouble(0)
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double = (0 until b.size()).mapIndexed { i, _ -> if (b[i] == a[i]) 0.0f else weights[i] }.sum().toDouble()
        override fun invoke(a: BitSet, b: BitSet): Double = (0 until b.size()).mapIndexed { i, _ -> if (b[i] == a[i]) 0.0 else 1.0 }.sum()
    },

    /**
     * Haversine distance only applicable for two spherical (= earth) coordinates in degrees. Hence the arrays <b>have</b> to be of size two each
     */
    HAVERSINE {

        override val operations: Int = 1 // Single calculation as this is fixed.

        /**
         * A constant for the approx. earth radius in meters
         */
        private val EARTH_RADIUS = 6371E3 // In meters

        override fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double = this.haversine(a.getDouble(0), a.getDouble(1), b.getDouble(0), b.getDouble(1))
        override fun invoke(a: INDArray, b: INDArray): Double = this.haversine(a.getDouble(0), a.getDouble(1), b.getDouble(0), b.getDouble(1))
        override fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double = this.haversine(a[0].toDouble(), a[1].toDouble(), b[0].toDouble(), b[1].toDouble())
        override fun invoke(a: BitSet, b: BitSet): Double = this.haversine(a[0].toDouble(), a[1].toDouble(), b[0].toDouble(), b[1].toDouble())

        /**
         * Calculates the haversine distance of two spherical coordinates in degrees.
         *
         * @param a_lat Start coordinate (latitude) in degrees.
         * @param a_lon Start coordinate (longitude) in degrees.
         * @param b_lat End coordinate (latitude) in degrees.
         * @param b_lon End coordinate (longitude) in degrees.

         * @return The haversine distance between the two points
         */
        private fun haversine(a_lat: Double, a_lon: Double, b_lat: Double, b_lon: Double): Double {
            val phi1 = StrictMath.toRadians(a_lat)
            val phi2 = StrictMath.toRadians(b_lat)
            val deltaPhi = StrictMath.toRadians(b_lat - a_lat)
            val deltaLambda = StrictMath.toRadians(b_lon - a_lon)
            val c = sin(deltaPhi / 2.0) * sin(deltaPhi / 2.0) + cos(phi1) * cos(phi2) * sin(deltaLambda / 2.0) * sin(
                    deltaLambda / 2.0
            )
            val d = 2.0 * atan2(sqrt(c), sqrt(1 - c))
            return EARTH_RADIUS * d
        }
    }
}
