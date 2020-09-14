package org.vitrivr.cottontail.database.index.pq

/**
 * Data structure to hold inner products between a query vector and all centroids for
 * all subspaces.
 * This is intended as a lookup table for running PQ based queries
 * todo maybe: use 1d array. Need to store length of a row somewhere which is tricky for
 *       inline class: use first entry?
 *       second entry could be how many rows there are?
 */
inline class PQCentroidQueryIPFloat(val data: Array<FloatArray>) {
    inline fun approximateIP(signature: IntArray): Float {
        var ip = 0.0F
        signature.indices.forEach {
            ip += data[it][signature[it]]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    inline fun approximateIP(signature: IntArray, start: Int, length: Int): Float {
        var ip = 0.0F
        (0 until length).forEach {
            ip += data[it][signature[it + start]]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UShortArray, start: Int, length: Int): Float {
        var ip = 0.0F
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UByteArray, start: Int, length: Int): Float {
        var ip = 0.0F
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }
}