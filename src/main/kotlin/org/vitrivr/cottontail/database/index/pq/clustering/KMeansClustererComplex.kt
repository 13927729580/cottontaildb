package org.vitrivr.cottontail.database.index.pq.clustering

import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import java.util.*

/**
 * A class that does KMeans clustering on complex Vectors
 * @author: Gabriel Zihlmann, 3.10.2020
 */
class KMeansClustererComplex<T: ComplexVectorValue<out Number>> (val k: Int, val rng: SplittableRandom, val distance: (a: T, b: T) -> Double) {
    companion object {
        private val LOGGER = LoggerFactory.getLogger(KMeansClustererComplex::class.java)
    }
    /**
     * Cluster Center contains the centroid vector and an array with the indexes (in the original data
     * of this Clusterer object) that were assigned to this cluster center
     */
    data class ClusterCenter<out T> (val center: T, val clusterPointIndices: IntArray) {
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

    /**
     * returns an array of [ClusterCenter]
     */
    fun cluster(data: Array<out T>, maxIterations: Int): Array<ClusterCenter<T>> {
        // initialize by choosing a random datapoint as cluster centers
        var centroids = List(k) {
            data[rng.nextInt(data.size)].copy() as T // unchecked is ok, since we copy from T type...
        }

        val assignedVectors = Array(k) { mutableListOf<Int>() }
        val assignedCenter = IntArray(data.size)
        var iter = 0
        while (iter < maxIterations) {
            iter++
            // assign new centroids
            var assignmentChanged = 0
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
                if (assignedCenter[i] != minIndex) {
                    assignmentChanged++
                    assignedCenter[i] = minIndex
                }
                assignedVectors[minIndex].add(i)
            }
            // new centers
            centroids = assignedVectors.mapIndexed { i, vecIndexes ->
                var mean: T = data[vecIndexes[0]]
                vecIndexes.subList(1, vecIndexes.size).forEach {
                    mean = (mean + data[it]) as T
                } // sum up all vectors, then divide by their count
                mean = (mean / DoubleValue(vecIndexes.size.toDouble())) as T
                mean
            }
            LOGGER.trace("Iteration $iter AssignmentsChanged $assignmentChanged")
            if (assignmentChanged == 0) {
                break
            }
        }
        LOGGER.info("Clusterer stopped after $iter iterations (maxIterations=$maxIterations)")
        if (iter >= maxIterations) LOGGER.warn("Clusterer reached maximum iterations!")
        return (centroids zip assignedVectors).map { (centroid, vectors) ->
            ClusterCenter(centroid, vectors.toIntArray())
        }.toTypedArray()
    }

}
