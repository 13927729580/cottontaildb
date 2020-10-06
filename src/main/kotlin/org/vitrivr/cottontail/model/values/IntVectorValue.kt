package org.vitrivr.cottontail.model.values

import org.vitrivr.cottontail.model.values.types.NumericValue
import org.vitrivr.cottontail.model.values.types.RealVectorValue
import org.vitrivr.cottontail.model.values.types.Value
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.pow

/**
 * This is an abstraction over an [IntArray] and it represents a vector of [Int]s.
 *
 * @author Ralph Gasser
 * @version 1.1
 */
inline class IntVectorValue(val data: IntArray) : RealVectorValue<Int> {

    companion object {
        /**
         * Generates a [IntVectorValue] of the given size initialized with random numbers.
         *
         * @param size Size of the new [IntVectorValue]
         * @param rnd A [SplittableRandom] to generate the random numbers.
         */
        fun random(size: Int, rnd: SplittableRandom = SplittableRandom(System.currentTimeMillis())) = IntVectorValue(IntArray(size) { rnd.nextInt() })

        /**
         * Generates a [IntVectorValue] of the given size initialized with ones.
         *
         * @param size Size of the new [IntVectorValue]
         */
        fun one(size: Int) = IntVectorValue(IntArray(size) { 1 })

        /**
         * Generates a [IntVectorValue] of the given size initialized with zeros.
         *
         * @param size Size of the new [IntVectorValue]
         */
        fun zero(size: Int) = IntVectorValue(IntArray(size))
    }

    constructor(input: List<Number>) : this(IntArray(input.size) { input[it].toInt() })
    constructor(input: Array<Number>) : this(IntArray(input.size) { input[it].toInt() })

    override val logicalSize: Int
        get() = this.data.size

    override fun compareTo(other: Value): Int {
        throw IllegalArgumentException("IntVectorValues can can only be compared for equality.")
    }

    /**
     * Returns the indices of this [IntVectorValue].
     *
     * @return The indices of this [IntVectorValue]
     */
    override val indices: IntRange
        get() = this.data.indices

    /**
     * Returns the i-th entry of  this [IntVectorValue].
     *
     * @param i Index of the entry.
     */
    override fun get(i: Int) = IntValue(this.data[i])

    /**
     * Returns the i-th entry of  this [IntVectorValue] as [Boolean].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    override fun getAsBool(i: Int) = this.data[i] != 0

    /**
     * Returns true, if this [IntVectorValue] consists of all zeroes, i.e. [0, 0, ... 0]
     *
     * @return True, if this [IntVectorValue] consists of all zeroes
     */
    override fun allZeros(): Boolean = this.data.all { it == 0 }

    /**
     * Returns true, if this [IntVectorValue] consists of all ones, i.e. [1, 1, ... 1]
     *
     * @return True, if this [IntVectorValue] consists of all ones
     */
    override fun allOnes(): Boolean = this.data.all { it == 1 }

    /**
     * Creates and returns a copy of this [IntVectorValue].
     *
     * @return Exact copy of this [IntVectorValue].
     */
    override fun copy(): IntVectorValue = IntVectorValue(this.data.copyOf(this.logicalSize))

    override fun plus(other: VectorValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] + other[it].asInt()).value
    })

    override fun minus(other: VectorValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] - other[it].asInt()).value
    })

    override fun times(other: VectorValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] * other[it].asInt()).value
    })

    override fun div(other: VectorValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] / other[it].asInt()).value
    })

    override fun plus(other: NumericValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] + other.asInt()).value
    })

    override fun minus(other: NumericValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] - other.asInt()).value
    })

    override fun times(other: NumericValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] * other.asInt()).value
    })

    override fun div(other: NumericValue<*>) = IntVectorValue(IntArray(this.logicalSize) {
        (this[it] / other.asInt()).value
    })

    override fun pow(x: Int) = DoubleVectorValue(DoubleArray(this.data.size) {
        this.data[it].toDouble().pow(x)
    })

    override fun sqrt() = DoubleVectorValue(DoubleArray(this.data.size) {
        kotlin.math.sqrt(this.data[it].toDouble())
    })

    override fun abs() = IntVectorValue(IntArray(this.data.size) {
        kotlin.math.abs(this.data[it])
    })

    override fun sum(): IntValue = IntValue(this.data.sum())

    override fun norm2(): FloatValue {
        var sum = 0.0f
        for (i in this.indices) {
            sum += this[i].value * this[i].value
        }
        return FloatValue(kotlin.math.sqrt(sum))
    }

    override fun dot(other: VectorValue<*>): IntValue {
        var sum = 0
        for (i in this.indices) {
            sum += other[i].value.toInt() * this[i].value
        }
        return IntValue(sum)
    }

    override fun l1(other: VectorValue<*>): FloatValue = when (other) {
        is LongVectorValue -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other.data[i]).absoluteValue
            }
            FloatValue(sum)
        }
        is IntVectorValue -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other.data[i]).absoluteValue
            }
            FloatValue(sum)
        }
        else -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other[i].value.toFloat()).absoluteValue
            }
            FloatValue(sum)
        }
    }

    override fun l2(other: VectorValue<*>): FloatValue = when (other) {
        is LongVectorValue -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other.data[i]).toFloat().pow(2)
            }
            FloatValue(kotlin.math.sqrt(sum))
        }
        is IntVectorValue -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other.data[i]).toFloat().pow(2)
            }
            FloatValue(kotlin.math.sqrt(sum))
        }
        else -> {
            var sum = 0.0f
            for (i in this.data.indices) {
                sum += (this.data[i] - other[i].value.toFloat()).pow(2)
            }
            FloatValue(kotlin.math.sqrt(sum))
        }
    }

    override fun lp(other: VectorValue<*>, p: Int): FloatValue {
        var sum = 0.0f
        for (i in this.indices) {
            sum += (other[i].value.toFloat() - this[i].value).pow(p)
        }
        return FloatValue(sum.pow(1.0f/p))
    }

    /**
     * Returns the subvector of length [length] starting from [start] of this [VectorValue].
     *
     * @param start Index of the first entry of the returned vector.
     * @param length how many elements, including start, to return
     * @return The subvector starting at index start containing length elements.
     */
    override fun get(start: Int, length: Int): VectorValue<Int> {
        TODO("Not yet implemented")
    }
}