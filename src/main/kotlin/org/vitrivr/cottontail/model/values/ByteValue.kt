package org.vitrivr.cottontail.model.values

inline class ByteValue(override val value: Byte) : Value<Byte> {
    override val size: Int
        get() = -1

    override val numeric: Boolean
        get() = true

    override fun compareTo(other: Value<*>): Int = when (other) {
        is BooleanValue -> this.value.compareTo(if (other.value) {
            1.toByte()
        } else {
            0.toByte()
        })
        is ByteValue -> this.value.compareTo(other.value)
        is ShortValue -> this.value.compareTo(other.value)
        is IntValue -> this.value.compareTo(other.value)
        is LongValue -> this.value.compareTo(other.value)
        is DoubleValue -> this.value.compareTo(other.value)
        is FloatValue -> this.value.compareTo(other.value)
        else -> throw IllegalArgumentException("ByteValues can only be compared to other numeric values.")
    }
}