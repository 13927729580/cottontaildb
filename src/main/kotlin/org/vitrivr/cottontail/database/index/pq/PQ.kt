package org.vitrivr.cottontail.database.index.pq

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.MatrixUtils.createRealMatrix
import org.apache.commons.math3.stat.correlation.Covariance
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.serializers.FixedDoubleVectorSerializer
import org.vitrivr.cottontail.model.values.Complex32VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.model.values.types.NumericValue
import org.vitrivr.cottontail.model.values.types.RealVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue

/**
 * Product Quantizer that minimizes inner product error. input data should be permuted for better results!
 * author: Gabriel Zihlmann
 * date: 25.8.2020
 * Roughly following Guo et al. 2015 - Quantization based Fast Inner Product Search
 */
class PQ(val codebooks: Array<PQCodebook>) {
    companion object Serializer : org.mapdb.Serializer<PQ> {
        private val LOGGER: Logger = LoggerFactory.getLogger(PQ::class.java)
        fun fromPermutedData(numSubspaces: Int, numCentroids: Int, permutedData: Array<out VectorValue<*>>): Pair<PQ, Array<IntArray>> {
            LOGGER.info("Initializing PQ from initial data.")
            // some assumptions. Some are for documentation, some are cheap enough to actually keep and check
            require(permutedData.all { it.logicalSize == permutedData[0].logicalSize && it::class.java == permutedData[0]::class.java })
            require(numSubspaces > 0)
            require(numCentroids > 0)
            require(permutedData[0].logicalSize >= numSubspaces)
            require(permutedData[0].logicalSize % numSubspaces == 0)

            val dimensionsPerSubspace = permutedData[0].logicalSize / numSubspaces
            val subspaceSignatures = Array(permutedData.size) { IntArray(numSubspaces) }
            LOGGER.info("Creating subspace data")
            // wasteful copy...
            val permutedSubspaceData = (0 until numSubspaces).map { k ->
                k to permutedData.map { v -> v.get(k * dimensionsPerSubspace, dimensionsPerSubspace) }.toTypedArray()
            }
            LOGGER.info("Learning centroids")
            val codebooks = Array<PQCodebook?>(numSubspaces) { null }
            permutedSubspaceData.parallelStream().forEach { (k, permutedData) ->
                LOGGER.info("Processing subspace ${k + 1}")
                val (codebook, signatures) = when (permutedData[0]) {
                    is RealVectorValue<*> -> {
                        PQCodebook.learnFromData(permutedData.map { it as RealVectorValue<*> }.toTypedArray(), numCentroids, 1000) // todo: check maxIterations parameter if relevant...
                    }
                    is ComplexVectorValue<*> -> {
                        PQCodebook.learnFromData(permutedData.map { it as ComplexVectorValue<*> }.toTypedArray(), numCentroids, 1000) // todo: check maxIterations parameter if relevant...
                    }
                    else -> {
                        error("Unknown type")
                    }
                }
                signatures.forEachIndexed { i, c ->
                    subspaceSignatures[i][k] = c
                }
                codebooks[k] = codebook
                LOGGER.info("Done processing subspace ${k + 1} of $numSubspaces")
            }
            LOGGER.info("PQ initialization done.")
            return PQ(codebooks.map { it!! }.toTypedArray()) to subspaceSignatures
        }

        fun fromPermutedData(numSubspaces: Int, numCentroids: Int, permutedData: Array<DoubleArray>, permutedExampleQueryData: Array<DoubleArray>? = null): Pair<PQ, Array<IntArray>> {
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
            val permutedSubspaceData = splitDataIntoSubspaces(numSubspaces, permutedData, dimensionsPerSubspace).mapIndexed { k, ssData -> k to ssData }
            val permutedSubspaceExampleData = permutedExampleQueryData?.let { splitDataIntoSubspaces(numSubspaces, it, dimensionsPerSubspace) }
            LOGGER.info("Learning centroids")
            val codebooks = Array<PQCodebook?>(numSubspaces) { null }
            permutedSubspaceData.parallelStream().forEach { (k, permutedData) ->
                LOGGER.info("Processing subspace ${k + 1}")
                val (codebook, signatures) = if (permutedSubspaceExampleData != null) {
                    val inverseQCovMatrix = MatrixUtils.inverse(Covariance(permutedSubspaceExampleData[k], false).covarianceMatrix)
                    PQCodebook.learnFromData(permutedData, inverseQCovMatrix, numCentroids, 1000) // todo: check maxIterations parameter if relevant...
                } else {
                    PQCodebook.learnFromData(permutedData, numCentroids, 1000)
                }
                signatures.forEachIndexed { i, c ->
                    subspaceSignatures[i][k] = c
                }
                codebooks[k] = codebook
                LOGGER.info("Done processing subspace ${k + 1} of $numSubspaces")
            }
            LOGGER.info("PQ initialization done.")
            return PQ(codebooks.map { it!! }.toTypedArray()) to subspaceSignatures
        }

        private fun splitDataIntoSubspaces(numSubspaces: Int, permutedData: Array<DoubleArray>, dimensionsPerSubspace: Int): List<Array<DoubleArray>> {
            return (0 until numSubspaces).map { k ->
                Array(permutedData.size) { i ->
                    DoubleArray(dimensionsPerSubspace) { j ->
                        permutedData[i][k * dimensionsPerSubspace + j]
                    }
                }
            }
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
            TODO()
        }
//            out.packInt(value.numSubspaces)
//            out.packInt(value.numCentroids)
//            out.packInt(value.dimensionsPerSubspace)
//            value.codebooks.forEach {
//                it.centroids.forEach { c ->
//                    FixedDoubleVectorSerializer.serialize(out, c)
//                }
//                val cov = it.inverseDataCovarianceMatrix.data // first dim are rows
//                for (i in 0 until value.dimensionsPerSubspace) {
//                    doubleArraySerializer.serialize(out, cov[i])
//                }
//            }
//        }

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
            TODO()
//            val numSubspaces = input.unpackInt()
//            val numCentroids = input.unpackInt()
//            val dimensionsPerSubspace = input.unpackInt()
//            return PQ(Array(numSubspaces) {
//                PQCodebook(Array(numCentroids) {
//                    // todo: check if available - pos is actually correct...
//                    doubleArraySerializer.deserialize(input, available - input.pos)
//                },
//                createRealMatrix(Array(dimensionsPerSubspace) {
//                    doubleArraySerializer.deserialize(input, available - input.pos)
//                }))
//            })
        }
    }

    val numSubspaces = codebooks.size
    val numCentroids: Int
    val dimensionsPerSubspace: Int

    init {
        require(codebooks.all {
            it.centroids.size == codebooks[0].centroids.size
        })
        dimensionsPerSubspace = codebooks[0].centroids[0].logicalSize
        numCentroids = codebooks[0].centroids.size
    }


    /**
     * Calculates the IP between
     * the approximation specified with the index and the supplied vector which was permuted with the same permutation
     * that was applied to the data when creating this [PQ] object.
     * todo: this always copies! baaad
     */
    fun approximateAsymmetricIP(sigi: IntArray, v: DoubleArray): Double {
//        require(v.size == numSubspaces * dimensionsPerSubspace)
        var res = codebooks[0].centroids[sigi[0]].dot(DoubleVectorValue(v.slice(0 until dimensionsPerSubspace)))
        for (k in 1 until numSubspaces) {
            val centi = codebooks[k].centroids[sigi[k]]
            res += centi.dot(DoubleVectorValue(v.slice(k * dimensionsPerSubspace until (k + 1) * dimensionsPerSubspace)))
        }
        check(res.imaginary.value.toDouble() < 1e-5)
        return res.real.value.toDouble()
    }

    /**
     * Calculates the IP between
     * the approximation specified with the index and the supplied vector which was permuted with the same permutation
     * that was applied to the data when creating this [PQ] object.
     * todo: this always copies! baaad
     */
    fun approximateAsymmetricIP(sigi: IntArray, v: VectorValue<*>): NumericValue<*> {
//        require(v.size == numSubspaces * dimensionsPerSubspace)
        var res = codebooks[0].centroids[sigi[0]].dot(v.get(0, dimensionsPerSubspace))
        for (k in 1 until numSubspaces) {
            val centi = codebooks[k].centroids[sigi[k]]
            res += centi.dot(v.get(k * dimensionsPerSubspace, dimensionsPerSubspace))
        }
        return res
    }

    /**
     * Finds for each subspace the centroid with the largest inner product to the specified vector (needs
     * to be permuted the same way that this [PQ] object was built!
     * todo: no-copy...
     */
    fun getSignature(v: DoubleArray): IntArray {
        TODO()
//        require(v.size == numSubspaces * dimensionsPerSubspace)
//        return IntArray(numSubspaces) { k ->
//            codebooks[k].quantizeVector(v, k * dimensionsPerSubspace, dimensionsPerSubspace)
//        }
    }

    fun precomputeCentroidQueryIP(permutedQuery: DoubleArray): PQCentroidQueryIP {
        TODO()
//        return PQCentroidQueryIP(Array(numSubspaces) { k ->
//            DoubleArray(numCentroids) { i ->
//                var ip = 0.0
//                for (j in 0 until dimensionsPerSubspace) {
//                    ip += permutedQuery[k * dimensionsPerSubspace + j] * codebooks[k].centroids[i][j]
//                }
//                ip
//            }
//        }
//        )
    }

    fun precomputeCentroidQueryIPFloat(permutedQuery: DoubleArray): PQCentroidQueryIPFloat {
        TODO()
//        return PQCentroidQueryIPFloat(Array(numSubspaces) { k ->
//            FloatArray(numCentroids) { i ->
//                var ip = 0.0F
//                for (j in 0 until dimensionsPerSubspace) {
//                    ip += (permutedQuery[k * dimensionsPerSubspace + j] * codebooks[k].centroids[i][j]).toFloat()
//                }
//                ip
//            }
//        }
//        )
    }

    /*
    reversePermutation: intArray holding at index i the index of the dimension in the original space
    so: i is in "permuted space", the value in the array is where it was in the unpermuted space -> call it
    reversePermutation
     */
    fun precomputeCentroidQueryRealIPFloat(unPermutedQuery: Complex32VectorValue, reversePermutation: IntArray): PQCentroidQueryIPFloat {
        TODO()
//        return PQCentroidQueryIPFloat(Array(numSubspaces) { k ->
//            FloatArray(numCentroids) { i ->
//                var ip = 0.0F
//                for (j in 0 until dimensionsPerSubspace) {
//                    ip += (unPermutedQuery[reversePermutation[k * dimensionsPerSubspace + j]].real.value * codebooks[k].centroids[i][j]).toFloat()
//                }
//                ip
//            }
//        }
//        )
    }

    /*
    reversePermutation: intArray holding at index i the index of the dimension in the original space
    so: i is in "permuted space", the value in the array is where it was in the unpermuted space -> call it
    reversePermutation
     */
    fun precomputeCentroidQueryImagIPFloat(unPermutedQuery: Complex32VectorValue, reversePermutation: IntArray): PQCentroidQueryIPFloat {
        TODO()
//        return PQCentroidQueryIPFloat(Array(numSubspaces) { k ->
//            FloatArray(numCentroids) { i ->
//                var ip = 0.0F
//                for (j in 0 until dimensionsPerSubspace) {
//                    ip += (unPermutedQuery[reversePermutation[k * dimensionsPerSubspace + j]].imaginary.value * codebooks[k].centroids[i][j]).toFloat()
//                }
//                ip
//            }
//        }
//        )
    }
}
