package org.vitrivr.cottontail.database.index.pq

import org.vitrivr.cottontail.model.values.Complex32Value
import org.vitrivr.cottontail.model.values.Complex32VectorValue

/**
 * Data structure to hold inner products between a query vector and all centroids for
 * all subspaces.
 * This is intended as a lookup table for running PQ based queries
 * todo maybe: use 1d array. Need to store length of a row somewhere which is tricky for
 *       inline class: use first entry?
 *       second entry could be how many rows there are?
 * todo: assess performance penalty of cottontail types over DoubleArray
 */
inline class PQCentroidQueryIPComplexVectorValue(val data: Array<Complex32VectorValue>) {
    /**
     * for a larger array signature with an offset
     */
    @ExperimentalUnsignedTypes
    inline fun approximateIP(signature: UShortArray, start: Int, length: Int): Complex32Value {
        val ip = Complex32Value.ZERO
        (0 until length).forEach {
            ip.data[0] += data[it].data[signature[it + start].toInt() shl 1]
            ip.data[1] += data[it].data[(signature[it + start].toInt() shl 1) + 1]
        }
        return ip
    }
}