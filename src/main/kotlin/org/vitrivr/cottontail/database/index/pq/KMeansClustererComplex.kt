package org.vitrivr.cottontail.database.index.pq

import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import java.util.*

/**
 * A class that does KMeans clustering on complex Vectors
 * @author: Gabriel Zihlmann, 3.10.2020
 */
class KMeansClustererComplex<T: ComplexVectorValue<*>> (val data: Array<T>, val rng: SplittableRandom, val distance: (a: T, b: T) -> Double) {
    data class ClusterCenter<T> (val center: T, val clusterPointIndices: IntArray) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as ClusterCenter<*>

            if (center != other.center) return false
            if (!clusterPointIndices.contentEquals(other.clusterPointIndices)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = center?.hashCode() ?: 0
            result = 31 * result + clusterPointIndices.contentHashCode()
            return result
        }
    }

    fun cluster(k: Int, eps: Double, maxIterations: Int): Array<ClusterCenter<T>> {
        // initialize by choosing a random datapoint as cluster centers
        var centroids = List(k) {
            data[rng.nextInt(data.size)].copy() as T // unchecked is ok, since we copy from T type...
        }

        val assignedVectors = Array(k) { mutableListOf<Int>() }
        var iter = 0
        while (iter < maxIterations) {
            iter++
            // assign new centroids
            assignedVectors.forEach { it.clear() }
            data.forEachIndexed{ i, v ->
                var minIndex = -1
                var minDist = Double.MAX_VALUE
                centroids.forEachIndexed { j, c ->
                    val d = distance(c, v)
                    if (d < minDist) {
                        minIndex = j
                        minDist = d
                    }
                }
                assignedVectors[minIndex].add(i)
            }
            // new centers
            var meanMovement = 0.0
            centroids = assignedVectors.mapIndexed { i, vecIndexes ->
                var mean: T = data[vecIndexes[0]]
                vecIndexes.subList(1, vecIndexes.size).forEach {
                    mean = (mean + data[it]) as T
                } // sum up all vectors, then divide by their count
                mean = (mean / DoubleValue(vecIndexes.size.toDouble())) as T
                meanMovement += (centroids[i] - mean).norm2().value.toDouble()
                mean
            }
            if (meanMovement < eps) {
                break
            }
        }
        return (centroids zip assignedVectors).map { (centroid, vectors) ->
            ClusterCenter(centroid, vectors.toIntArray())
        }.toTypedArray()
    }

}
