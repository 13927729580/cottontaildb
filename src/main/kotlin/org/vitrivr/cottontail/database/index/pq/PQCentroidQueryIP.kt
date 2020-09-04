package org.vitrivr.cottontail.database.index.pq

/**
 * Data structure to hold inner products between a query vector and all centroids for
 * all subspaces.
 * This is intended as a lookup table for running PQ based queries
 */
inline class PQCentroidQueryIP(val data: Array<DoubleArray>) {
    inline fun approximateIP(signature: IntArray): Double {
        var ip = 0.0
        signature.indices.forEach {
            ip += data[it][signature[it]]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    inline fun approximateIP(signature: IntArray, start: Int, length: Int): Double {
        var ip = 0.0
        (0 until length).forEach {
            ip += data[it][signature[it + start]]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UShortArray, start: Int, length: Int): Double {
        var ip = 0.0
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UByteArray, start: Int, length: Int): Double {
        var ip = 0.0
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }
}