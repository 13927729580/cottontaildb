package org.vitrivr.cottontail.database.index.pq

import org.apache.commons.math3.linear.MatrixUtils.createRealMatrix
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**
 * Product Quantizer that minimizes inner product error. input data should be permuted for better results!
 * author: Gabriel Zihlmann
 * date: 25.8.2020
 */
class PQ(val codebooks: Array<PQCodebook>) {
    companion object Serializer : org.mapdb.Serializer<PQ> {
        private val LOGGER: Logger = LoggerFactory.getLogger(PQ::class.java)
        private val doubleArraySerializer = org.mapdb.Serializer.DOUBLE_ARRAY!!
        fun fromPermutedData(numSubspaces: Int, numCentroids: Int, permutedData: Array<DoubleArray>, permutedExampleQueryData: Array<DoubleArray>): Pair<PQ, Array<IntArray>> {
            return fromPermutedData(numSubspaces, numCentroids, permutedData)
        }

        fun fromPermutedData(numSubspaces: Int, numCentroids: Int, permutedData: Array<DoubleArray>): Pair<PQ, Array<IntArray>> {
            LOGGER.info("Initializing PQ from initial data.")
            // some assumptions. Some are for documentation, some are cheap enough to actually keep and check
            require(permutedData.all { it.size == permutedData[0].size })
            require(numSubspaces > 0)
            require(numCentroids > 0)
            require(permutedData[0].size >= numSubspaces)
            require(permutedData[0].size % numSubspaces == 0)
            val dimensionsPerSubspace = permutedData[0].size / numSubspaces
            val subspaceSignatures = Array(permutedData.size) { IntArray(numSubspaces) }
            LOGGER.info("Creating subspace data")
            // wasteful copy...
            val permutedSubspaceData = (0 until numSubspaces).map { k ->
                k to Array(permutedData.size) { i ->
                    DoubleArray(dimensionsPerSubspace) { j ->
                        permutedData[i][k * dimensionsPerSubspace + j]
                    }
                }
            }
            LOGGER.info("Learning centroids")
            val codebooks = Array<PQCodebook?>(numSubspaces) { null }
            permutedSubspaceData.toList().parallelStream().forEach { (k, permutedData) ->
                LOGGER.info("Processing subspace ${k + 1} of $numSubspaces")
                val (codebook, signatures) = PQCodebook.learnFromData(permutedData, numCentroids, 1000)
                signatures.forEachIndexed { i, c ->
                    subspaceSignatures[i][k] = c
                }
                codebooks[k] = codebook
                LOGGER.info("Done processing subspace ${k + 1} of $numSubspaces")
            }
            LOGGER.info("PQ initialization done.")
            return PQ(codebooks.map { it!! }.toTypedArray()) to subspaceSignatures
        }

        /**
         * Serializes the content of the given value into the given
         * [DataOutput2].
         *
         * @param out DataOutput2 to save object into
         * @param value Object to serialize
         *
         * @throws IOException in case of an I/O error
         */
        override fun serialize(out: DataOutput2, value: PQ) {
            out.packInt(value.numSubspaces)
            out.packInt(value.numCentroids)
            out.packInt(value.dimensionsPerSubspace)
            value.codebooks.forEach {
                it.centroids.forEach { c ->
                    doubleArraySerializer.serialize(out, c)
                }
                val cov = it.inverseDataCovarianceMatrix.data // first dim are rows
                for (i in 0 until value.dimensionsPerSubspace) {
                    doubleArraySerializer.serialize(out, cov[i])
                }
            }
        }

        /**
         * Deserializes and returns the content of the given [DataInput2].
         *
         * @param input DataInput2 to de-serialize data from
         * @param available how many bytes that are available in the DataInput2 for
         * reading, may be -1 (in streams) or 0 (null).
         *
         * @return the de-serialized content of the given [DataInput2]
         * @throws IOException in case of an I/O error
         */
        override fun deserialize(input: DataInput2, available: Int): PQ {
            val numSubspaces = input.unpackInt()
            val numCentroids = input.unpackInt()
            val dimensionsPerSubspace = input.unpackInt()
            return PQ(Array(numSubspaces) {
                PQCodebook(Array(numCentroids) {
                    // todo: check if available - pos is actually correct...
                    doubleArraySerializer.deserialize(input, available - input.pos)
                },
                createRealMatrix(Array(dimensionsPerSubspace) {
                    doubleArraySerializer.deserialize(input, available - input.pos)
                }))
            })
        }
    }

    val numSubspaces = codebooks.size
    val numCentroids: Int
    val dimensionsPerSubspace: Int

    init {
        require(codebooks.all {
            it.centroids.size == codebooks[0].centroids.size
            it.centroids.all { c -> c.size == it.centroids[0].size }
        }) // ideally should check this in codebook class, but inline classes can't have init block...
        dimensionsPerSubspace = codebooks[0].centroids[0].size
        numCentroids = codebooks[0].centroids.size
    }

    /**
     * Calculates the IP between
     * the two approximations specified
     */
    fun approximateSymmetricIP(sigi: IntArray, sigj: IntArray): Double {
        var res = 0.0
        for (k in 0 until numSubspaces) {
            val centi = codebooks[k].centroids[sigi[k]]
            val centj = codebooks[k].centroids[sigj[k]]
            for (l in 0 until dimensionsPerSubspace) {
                res += centi[l] * centj[l]
            }
        }
        return res
    }

    /**
     * Calculates the IP between
     * the approximation specified with the index and the supplied vector which was permuted with the same permutation
     * that was applied to the data when creating this [PQ] object.
     * This is more accurate than the symmetric (approximateSymmetricIP)
     */
    fun approximateAsymmetricIP(sigi: IntArray, v: DoubleArray): Double {
//        require(v.size == numSubspaces * dimensionsPerSubspace)
        var res = 0.0
        for (k in 0 until numSubspaces) {
            val centi = codebooks[k].centroids[sigi[k]]
            for (l in 0 until dimensionsPerSubspace) {
                res += centi[l] * v[k * dimensionsPerSubspace + l]
            }
        }
        return res
    }

    /**
     * Finds for each subspace the centroid with the largest inner product to the specified vector (needs
     * to be permuted the same way that this [PQ] object was built!
     * todo: no-copy...
     */
    fun getSignature(v: DoubleArray): IntArray {
        require(v.size == numSubspaces * dimensionsPerSubspace)
        return IntArray(numSubspaces) { k ->
            codebooks[k].quantizeVector(v, k * dimensionsPerSubspace, dimensionsPerSubspace)
        }
    }

    fun precomputeCentroidQueryIP(q: DoubleArray): PQCentroidQueryIP {
        return PQCentroidQueryIP(Array(numSubspaces) { k ->
            DoubleArray(numCentroids) { i ->
                var ip = 0.0
                for (j in 0 until dimensionsPerSubspace) {
                    ip += q[k * dimensionsPerSubspace + j] * codebooks[k].centroids[i][j]
                }
                ip
            }
        }
        )
    }

    fun precomputeCentroidQueryIPFloat(q: DoubleArray): PQCentroidQueryIPFloat {
        return PQCentroidQueryIPFloat(Array(numSubspaces) { k ->
            FloatArray(numCentroids) { i ->
                var ip = 0.0F
                for (j in 0 until dimensionsPerSubspace) {
                    ip += (q[k * dimensionsPerSubspace + j] * codebooks[k].centroids[i][j]).toFloat()
                }
                ip
            }
        }
        )
    }
}