package org.vitrivr.cottontail.database.index.pq

import org.vitrivr.cottontail.model.values.FloatValue
import org.vitrivr.cottontail.model.values.types.NumericValue
import org.vitrivr.cottontail.model.values.types.VectorValue

/**
 * Data structure to hold inner products between a query vector and all centroids for
 * all subspaces.
 * This is intended as a lookup table for running PQ based queries
 * todo maybe: use 1d array. Need to store length of a row somewhere which is tricky for
 *       inline class: use first entry?
 *       second entry could be how many rows there are?
 * todo: assess performance penalty of cottontail types over DoubleArray
 */
inline class PQCentroidQueryIPVectorValue(val data: Array<VectorValue<*>>) {
    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UShortArray, start: Int, length: Int): NumericValue<*> {
        var ip = FloatValue.ZERO
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }

    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UByteArray, start: Int, length: Int): NumericValue<*> {
        var ip = FloatValue.ZERO
        (0 until length).forEach {
            ip += data[it][signature[it + start].toInt()]
        }
        return ip
    }
}