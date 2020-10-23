package org.vitrivr.cottontail.model.values.types

/**
 * Represents a vector value of any type, i.e. a value that consists only more than one entry. Vector
 * values are always numeric! This  is an abstraction over the existing primitive array types provided
 * by Kotlin. It allows for the advanced type system implemented by Cottontail DB.
 *
 * @version 1.2
 * @author Ralph Gasser
 */
interface VectorValue<T: Number> : Value {
    /**
     * Returns the i-th entry of  this [VectorValue].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    operator fun get(i: Int): NumericValue<T>

    /**
     * Returns the subvector of length [length] starting from [start] of this [VectorValue].
     *
     * @param start Index of the first entry of the returned vector.
     * @param length how many elements, including start, to return
     * @return The subvector starting at index start containing length elements.
     */
    fun get(start: Int, length: Int): VectorValue<T>

    /**
     * Returns the i-th entry of  this [VectorValue] as [Boolean].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    fun getAsBool(i: Int): Boolean

    /**
     * Returns true, if this [VectorValue] consists of all zeroes, i.e. [0, 0, ... 0]
     *
     * @return True, if this [VectorValue] consists of all zeroes
     */
    fun allZeros(): Boolean

    /**
     * Returns true, if this [VectorValue] consists of all ones, i.e. [1, 1, ... 1]
     *
     * @return True, if this [VectorValue] consists of all ones
     */
    fun allOnes(): Boolean

    /**
     * Returns the indices of this [VectorValue].
     *
     * @return The indices of this [VectorValue]
     */
    val indices: IntRange

    /**
     * Creates and returns an exact copy of this [VectorValue].
     *
     * @return Exact copy of this [VectorValue].
     */
    fun copy(): VectorValue<T>

    /**
     * Calculates the element-wise sum of this and the other [VectorValue].
     *
     * @param other The [VectorValue] to add to this [VectorValue].
     * @return [VectorValue] that contains the element-wise sum of the two input [VectorValue]s
     */
    operator fun plus(other: VectorValue<*>): VectorValue<T>

    /**
     * Calculates the element-wise difference of this and the other [VectorValue].
     *
     * @param other The [VectorValue] to subtract from this [VectorValue].
     * @return [VectorValue] that contains the element-wise difference of the two input [VectorValue]s
     */
    operator fun minus(other: VectorValue<*>) = if (this.logicalSize == other.logicalSize)
        minus(other, 0, 0, logicalSize)
    else throw IllegalArgumentException("Dimensions ${this.logicalSize} and ${other.logicalSize} don't agree!")

    /**
     * Calculates the element-wise difference of this and the other [VectorValue]. Subvectors can be defined by the
     * [start] [startOther] and [length] parameters.
     *
     * @param other The [VectorValue] to subtract from this [VectorValue].
     * @param start the index of the subvector of this to start from
     * @param otherStart the index of the subvector of other to start from
     * @param length the number of elements to build the dot product with from the respective starts
     * @return [VectorValue] that contains the element-wise difference of the two input [VectorValue]s
     */
    fun minus(other: VectorValue<*>, start: Int, otherStart: Int, length: Int): VectorValue<T>

    /**
     * Calculates the element-wise product of this and the other [VectorValue].
     *
     * @param other The [VectorValue] to multiply this [VectorValue] with.
     * @return [VectorValue] that contains the element-wise product of the two input [VectorValue]s
     */
    operator fun times(other: VectorValue<*>): VectorValue<T>

    /**
     * Calculates the element-wise quotient of this and the other [VectorValue].
     *
     * @param other The [VectorValue] to divide this [VectorValue] by.
     * @return [VectorValue] that contains the element-wise quotient of the two input [VectorValue]s
     */
    operator fun div(other: VectorValue<*>): VectorValue<T>

    operator fun plus(other: NumericValue<*>): VectorValue<T>
    operator fun minus(other: NumericValue<*>): VectorValue<T>
    operator fun times(other: NumericValue<*>): VectorValue<T>
    operator fun div(other: NumericValue<*>): VectorValue<T>

    /**
     * Creates a new [VectorValue] that contains the absolute values this [VectorValue]'s elements.
     *
     * @return [VectorValue] with the element-wise absolute values.
     */
    fun abs(): RealVectorValue<T>

    /**
     * Creates a new [VectorValue] that contains the values this [VectorValue]'s elements raised to the power of x.
     *
     * @param x The exponent for the operation.
     * @return [VectorValue] with the element-wise values raised to the power of x.
     */
    fun pow(x: Int): VectorValue<*>

    /**
     * Creates a new [VectorValue] that contains the square root values this [VectorValue]'s elements.
     *
     * @return [VectorValue] with the element-wise square root values.
     */
    fun sqrt(): VectorValue<*>

    /**
     * Builds the sum of the elements of this [VectorValue].
     *
     * <strong>Warning:</string> Since the value generated by this function might not go into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @return Sum of the elements of this [VectorValue].
     */
    fun sum(): NumericValue<*>

    /**
     * Calculates the magnitude of this [VectorValue] with respect to the L2 / Euclidean distance.
     */
    fun norm2(): RealValue<*>

    /**
     * Builds the dot product between this and the other [VectorValue].
     *
     * <strong>Warning:</string> Since the value generated by this function might not fit into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @return Sum of the elements of this [VectorValue].
     */

    infix fun dot(other: VectorValue<*>) = if (other.logicalSize == this.logicalSize)
        dot(other, 0, 0, logicalSize)
    else throw IllegalArgumentException("Dimensions ${this.logicalSize} and ${other.logicalSize} don't agree!")


    /**
     * Builds the dot product between this and the other [VectorValue]. Subvectors can be defined by the
     * [start] [startOther] and [length] parameters.
     *
     * <strong>Warning:</string> Since the value generated by this function might not fit into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @param start the index of the subvector of this to start from
     * @param otherStart the index of the subvector of other to start from
     * @param length the number of elements to build the dot product with from the respective starts
     * @return Sum of the elements of this [VectorValue].
     */
    fun dot(other: VectorValue<*>, start: Int, otherStart: Int, length: Int): NumericValue<*>

    /**
     * Special implementation of the L1 / Manhattan distance. Can be overridden to create optimized versions of it.
     *
     * <strong>Warning:</string> Since the value generated by this function might not fit into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @param other The [VectorValue] to calculate the distance to.
     */
    infix fun l1(other: VectorValue<*>): RealValue<*> = ((this - other).abs()).sum().real

    /**
     * Special implementation of the L2 / Euclidean distance. Can be overridden to create optimized versions of it.
     *
     * <strong>Warning:</string> Since the value generated by this function might not go into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @param other The [VectorValue] to calculate the distance to.
     */
    infix fun l2(other: VectorValue<*>): RealValue<*> = ((this - other).abs().pow(2)).sum().sqrt().real
    // todo: abs() will by definition return a vector with components >= 0 and pow() of that can only give reals...

    /**
     * Special implementation of the LP / Minkowski distance. Can be overridden to create optimized versions of it.
     *
     * <strong>Warning:</string> Since the value generated by this function might not go into the
     * type held by this [VectorValue], the [NumericValue] returned by this function might differ.
     *
     * @param other The [VectorValue] to calculate the distance to.
     */
    fun lp(other: VectorValue<*>, p: Int): RealValue<*> = ((this - other).abs().pow(p)).sum().pow(1.0 / p).real
}