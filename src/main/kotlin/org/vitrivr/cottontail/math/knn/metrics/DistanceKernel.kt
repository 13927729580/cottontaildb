package org.vitrivr.cottontail.math.knn.metrics

import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.VectorValue

/**
 * A kernel function used for distance calculations between two [VectorValue]s. Whenever possible,
 * these calculations should be using 'native' operations of the respective [VectorValue] implementation
 *
 * @author Ralph Gasser
 * @version 1.1
 */
interface DistanceKernel {
    /** Estimate of the cost required per vector component. */
    val cost: Double

    /**
     * Calculates the distance between two [VectorValue]s.
     *
     * @param a First [VectorValue]
     * @param b Second [VectorValue]
     * @return Distance between a and b.
     */
    operator fun invoke(a: VectorValue<*>, b: VectorValue<*>): DoubleValue

    /**
     * Calculates the weighted distance between two [VectorValue]s.
     *
     * @param a First [VectorValue]
     * @param b Second [VectorValue]
     * @param weights The per component weight (usually between 0.0 and 1.0)
     * @return Distance between a and b.
     */
    operator fun invoke(a: VectorValue<*>, b: VectorValue<*>, weights: VectorValue<*>): DoubleValue
}