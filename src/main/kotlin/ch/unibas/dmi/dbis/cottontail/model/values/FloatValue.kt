package ch.unibas.dmi.dbis.cottontail.model.values

inline class FloatValue(override val value: Float) : Value<Float> {
    override val size: Int
        get() = -1

    override val numeric: Boolean
        get() = true

    override fun compareTo(other: Value<*>): Int = when (other) {
        is BooleanValue -> this.value.compareTo(if (other.value) { 1.0f } else { 0.0f })
        is ByteValue -> this.value.compareTo(other.value)
        is ShortValue -> this.value.compareTo(other.value)
        is IntValue -> this.value.compareTo(other.value)
        is LongValue -> this.value.compareTo(other.value)
        is DoubleValue -> this.value.compareTo(other.value)
        is FloatValue -> this.value.compareTo(other.value)
        else -> throw IllegalArgumentException("FloatValues can only be compared to other numeric values.")
    }
}