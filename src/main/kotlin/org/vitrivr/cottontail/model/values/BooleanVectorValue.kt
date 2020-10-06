package org.vitrivr.cottontail.model.values

import org.vitrivr.cottontail.model.values.types.*
import org.vitrivr.cottontail.utilities.extensions.init
import org.vitrivr.cottontail.utilities.extensions.toByte
import java.util.*

/**
 * This is an abstraction over a [BitSet] and it represents a vector of [Boolean]s.
 *
 * @author Ralph Gasser
 * @version 1.1
 */
inline class BooleanVectorValue(val value: BitSet) : VectorValue<Byte> {


    constructor(input: List<Number>) : this(BitSet(input.size).init { input[it].toInt() == 1 })
    constructor(input: Array<Number>) : this(BitSet(input.size).init { input[it].toInt() == 1 })
    constructor(input: Array<Boolean>) : this(BitSet(input.size).init { input[it] })

    override val logicalSize: Int
        get() = value.length()

    override fun compareTo(other: Value): Int {
        TODO("Not yet implemented")
    }


    /**
     * Returns the indices of this [BooleanVectorValue].
     *
     * @return The indices of this [BooleanVectorValue]
     */
    override val indices: IntRange
        get() = IntRange(0, this.value.length()-1)

    /**
     * Returns the i-th entry of  this [BooleanVectorValue].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    override fun get(i: Int): ByteValue = ByteValue(this.value[i].toByte())

    /**
     * Returns the i-th entry of  this [BooleanVectorValue] as [Boolean].
     *
     * @param i Index of the entry.
     * @return The value at index i.
     */
    override fun getAsBool(i: Int) = this.value[i]

    /**
     * Returns true, if this [BooleanVectorValue] consists of all zeroes, i.e. [0, 0, ... 0]
     *
     * @return True, if this [BooleanVectorValue] consists of all zeroes
     */
    override fun allZeros(): Boolean = this.indices.all { !this.value[it] }

    /**
     * Returns true, if this [BooleanVectorValue] consists of all ones, i.e. [1, 1, ... 1]
     *
     * @return True, if this [BooleanVectorValue] consists of all ones
     */
    override fun allOnes(): Boolean = this.indices.all { this.value[it] }

    /**
     * Creates and returns a copy of this [BooleanVectorValue].
     *
     * @return Exact copy of this [BooleanVectorValue].
     */
    override fun copy(): BooleanVectorValue = BooleanVectorValue(BitSet(this.logicalSize).init { this.value[it] })

    override fun plus(other: VectorValue<*>): VectorValue<Byte> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun minus(other: VectorValue<*>): VectorValue<Byte> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun times(other: VectorValue<*>): VectorValue<Byte> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun div(other: VectorValue<*>): VectorValue<Byte> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun plus(other: NumericValue<*>): BooleanVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun minus(other: NumericValue<*>): BooleanVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun times(other: NumericValue<*>): BooleanVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun div(other: NumericValue<*>): BooleanVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun pow(x: Int): DoubleVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun sqrt(): DoubleVectorValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun abs(): RealVectorValue<Byte> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun sum(): ByteValue {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun norm2(): RealValue<*> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    override fun dot(other: VectorValue<*>): RealValue<*> {
        throw UnsupportedOperationException("A BooleanVector array cannot be used to perform arithmetic operations!")
    }

    /**
     * Returns the subvector of length [length] starting from [start] of this [VectorValue].
     *
     * @param start Index of the first entry of the returned vector.
     * @param length how many elements, including start, to return
     * @return The subvector starting at index start containing length elements.
     */
    override fun get(start: Int, length: Int): VectorValue<Byte> {
        TODO("Not yet implemented")
    }
}