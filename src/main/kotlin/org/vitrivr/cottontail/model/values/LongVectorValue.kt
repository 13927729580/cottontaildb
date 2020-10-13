package org.vitrivr.cottontail.model.values

import org.vitrivr.cottontail.model.values.types.NumericValue
import org.vitrivr.cottontail.model.values.types.RealVectorValue
import org.vitrivr.cottontail.model.values.types.Value
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.pow

/**
 * This is an abstraction over a [FloatArray] and it represents a vector of [Float]s.
 *
 * @author Ralph Gasser
 * @version 1.1
 */
inline class LongVectorValue(val data: LongArray) : RealVectorValue<Long> {

    companion object {
        /**
         * Generates a [LongVectorValue] of the given size initialized with random numbers.
         *
         * @param size Size of the new [LongVectorValue]
         * @param rnd A [SplittableRandom] to generate the random numbers.
         */
        fun random(size: Int, rnd: SplittableRandom = SplittableRandom(System.currentTimeMillis())) = LongVectorValue(LongArray(size) { rnd.nextLong() })

        /**
         * Generates a [LongVectorValue] of the given size initialized with ones.
         *
         * @param size Size of the new [LongVectorValue]
         */
        fun one(size: Int) = LongVectorValue(LongArray(size) { 1L })

        /**
         * Generates a [IntVectorValue] of the given size initialized with zeros.
         *
         * @param size Size of the new [LongVectorValue]
         */
        fun zero(size: Int) = LongVectorValue(LongArray(size))
    }

    constructor(input: List<Number>) : this(LongArray(input.size) { input[it].toLong() })
    constructor(input: Array<Number>) : this(LongArray(input.size) { input[it].toLong() })

    override val logicalSize: Int
        get() = data.size

    override fun compareTo(other: Value): Int {
        throw IllegalArgumentException("LongVectorValues can can only be compared for equality.")
    }

    /**
     * Returns the indices of this [LongVectorValue].
     *
     * @return The indices of this [LongVectorValue]
     */
    override val indices: IntRange
        get() = this.data.indices

    /**
     * Returns the i-th entry of  this [LongVectorValue].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    override fun get(i: Int) = LongValue(this.data[i])

    /**
     * Returns the i-th entry of  this [LongVectorValue] as [Boolean].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    override fun getAsBool(i: Int) = this.data[i] != 0L

    /**
     * Returns true, if this [LongVectorValue] consists of all zeroes, i.e. [0, 0, ... 0]
     *
     * @return True, if this [LongVectorValue] consists of all zeroes
     */
    override fun allZeros(): Boolean = this.data.all { it == 0L }

    /**
     * Returns true, if this [LongVectorValue] consists of all ones, i.e. [1, 1, ... 1]
     *
     * @return True, if this [LongVectorValue] consists of all ones
     */
    override fun allOnes(): Boolean = this.data.all { it == 1L }

    /**
     * Creates and returns a copy of this [LongVectorValue].
     *
     * @return Exact copy of this [LongVectorValue].
     */
    override fun copy(): LongVectorValue = LongVectorValue(this.data.copyOf(this.logicalSize))

    override fun plus(other: VectorValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] + other[it].asLong()).value
    })

    override fun minus(other: VectorValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] - other[it].asLong()).value
    })

    override fun times(other: VectorValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] * other[it].asLong()).value
    })

    override fun div(other: VectorValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] / other[it].asLong()).value
    })

    override fun plus(other: NumericValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] + other.asLong()).value
    })

    override fun minus(other: NumericValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] - other.asLong()).value
    })

    override fun times(other: NumericValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] * other.asLong()).value
    })

    override fun div(other: NumericValue<*>): LongVectorValue = LongVectorValue(LongArray(this.logicalSize) {
        (this[it] / other.asLong()).value
    })

    override fun pow(x: Int) = DoubleVectorValue(DoubleArray(this.data.size) {
        this.data[it].toDouble().pow(x)
    })

    override fun sqrt() = DoubleVectorValue(DoubleArray(this.data.size) {
        kotlin.math.sqrt(this.data[it].toDouble())
    })

    override fun abs() = LongVectorValue(LongArray(this.logicalSize) {
        kotlin.math.abs(this.data[it])
    })

    override fun sum(): LongValue = LongValue(this.data.sum())

    override fun norm2(): DoubleValue {
        var sum = 0.0
        for (i in this.indices) {
            sum += this[i].value * this[i].value
        }
        return DoubleValue(kotlin.math.sqrt(sum))
    }

    override fun dot(other: VectorValue<*>): LongValue {
        var sum = 0L
        for (i in this.indices) {
            sum += other[i].value.toLong() * this[i].value
        }
        return LongValue(sum)
    }

    override fun l1(other: VectorValue<*>): LongValue {
        var sum = 0L
        for (i in this.indices) {
            sum += (other[i].value.toLong() - this[i].value).absoluteValue
        }
        return LongValue(sum)
    }

    override fun l2(other: VectorValue<*>): FloatValue {
        var sum = 0.0
        for (i in this.indices) {
            sum += (other[i].value.toDouble() - this[i].value).pow(2)
        }
        return FloatValue(kotlin.math.sqrt(sum))
    }

    override fun lp(other: VectorValue<*>, p: Int): FloatValue {
        var sum = 0.0
        for (i in this.indices) {
            sum += (other[i].value.toDouble() - this[i].value).pow(p)
        }
        return FloatValue(sum.pow(1.0/p))
    }

    /**
     * Returns the subvector of length [length] starting from [start] of this [VectorValue].
     *
     * @param start Index of the first entry of the returned vector.
     * @param length how many elements, including start, to return
     * @return The subvector starting at index start containing length elements.
     */
    override fun get(start: Int, length: Int): VectorValue<Long> {
        TODO("Not yet implemented")
    }

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
    override fun minus(other: VectorValue<*>, start: Int, otherStart: Int, length: Int): VectorValue<Long> {
        TODO("Not yet implemented")
    }

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
    override fun dot(other: VectorValue<*>, start: Int, otherStart: Int, length: Int): NumericValue<*> {
        TODO("Not yet implemented")
    }
}