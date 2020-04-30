package org.vitrivr.cottontail.database.column

import org.mapdb.Serializer
import org.vitrivr.cottontail.database.serializers.*
import org.vitrivr.cottontail.model.values.*
import org.vitrivr.cottontail.model.values.types.Value
import kotlin.reflect.KClass
import kotlin.reflect.full.safeCast

/**
 * Specifies the type of a Cottontail DB [Column]. This construct allows for some degree of type safety in the eye de-/serialization.
 * The column types are stored as strings and mapped to the respective class using [ColumnType.forName].
 *
 * @see Column
 *
 * @author Ralph Gasser
 * @version 1.1
 */
sealed class ColumnType<T : Value> {
    abstract val name: String
    abstract val type: KClass<T>
    abstract val numeric: Boolean


    companion object {
        /**
         * Returns the [ColumnType] for the provided name.
         *
         * @param name For which to lookup the [ColumnType].
         */
        fun forName(name: String): ColumnType<*> = when (name.toUpperCase()) {
            "BOOLEAN" -> BooleanColumnType()
            "BYTE" -> ByteColumnType()
            "SHORT" -> ShortColumnType()
            "INTEGER" -> IntColumnType()
            "LONG" -> LongColumnType()
            "FLOAT" -> FloatColumnType()
            "DOUBLE" -> DoubleColumnType()
            "STRING" -> StringColumnType()
            "COMPLEX32" -> Complex32ColumnType()
            "COMPLEX64" -> Complex64ColumnType()
            "INT_VEC" -> IntVectorColumnType()
            "LONG_VEC" -> LongVectorColumnType()
            "FLOAT_VEC" -> FloatVectorColumnType()
            "DOUBLE_VEC" -> DoubleVectorColumnType()
            "BOOL_VEC" -> BooleanVectorColumnType()
            "COMPLEX32_VEC" -> Complex32VectorColumnType()
            "COMPLEX64_VEC" -> Complex64VectorColumnType()
            else -> throw java.lang.IllegalArgumentException("The column type $name does not exists!")
        }
    }


    fun cast(value: Value?): T? = this.type.safeCast(value)
    fun compatible(value: Value) = this.type.isInstance(value)

    /**
     * Returns a [Serializer] for this [ColumnType]. Some [ColumnType] require a size attribute
     *
     * @param size The size of the column (e.g. for vectors). Defaults to -1.
     */
    abstract fun serializer(size: Int = -1): Serializer<T>

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ColumnType<*>
        if (name != other.name) return false
        return true
    }

    override fun hashCode(): Int {
        return name.hashCode()
    }

    override fun toString(): String = this.name
}

@Suppress("UNCHECKED_CAST")
class BooleanColumnType : ColumnType<BooleanValue>() {
    override val name = "BOOLEAN"
    override val numeric = true
    override val type: KClass<BooleanValue> = BooleanValue::class
    override fun serializer(size: Int): Serializer<BooleanValue> = BooleanValueSerializer
}

@Suppress("UNCHECKED_CAST")
class ByteColumnType : ColumnType<ByteValue>() {
    override val name = "BYTE"
    override val numeric = true
    override val type: KClass<ByteValue> = ByteValue::class
    override fun serializer(size: Int): Serializer<ByteValue> = ByteValueSerializer
}

@Suppress("UNCHECKED_CAST")
class ShortColumnType : ColumnType<ShortValue>() {
    override val name = "SHORT"
    override val numeric = true
    override val type: KClass<ShortValue> = ShortValue::class
    override fun serializer(size: Int): Serializer<ShortValue> = ShortValueSerializer
}

@Suppress("UNCHECKED_CAST")
class IntColumnType : ColumnType<IntValue>() {
    override val name = "INTEGER"
    override val numeric = true
    override val type: KClass<IntValue> = IntValue::class
    override fun serializer(size: Int): Serializer<IntValue> = IntValueSerializer
}

@Suppress("UNCHECKED_CAST")
class LongColumnType : ColumnType<LongValue>() {
    override val name = "LONG"
    override val numeric = true
    override val type: KClass<LongValue> = LongValue::class
    override fun serializer(size: Int): Serializer<LongValue> = LongValueSerializer
}

@Suppress("UNCHECKED_CAST")
class FloatColumnType : ColumnType<FloatValue>() {
    override val name = "FLOAT"
    override val numeric = true
    override val type: KClass<FloatValue> = FloatValue::class
    override fun serializer(size: Int): Serializer<FloatValue> = FloatValueSerializer
}

@Suppress("UNCHECKED_CAST")
class DoubleColumnType : ColumnType<DoubleValue>() {
    override val name = "DOUBLE"
    override val numeric = true
    override val type: KClass<DoubleValue> = DoubleValue::class
    override fun serializer(size: Int): Serializer<DoubleValue> = DoubleValueSerializer
}

@Suppress("UNCHECKED_CAST")
class StringColumnType : ColumnType<StringValue>() {
    override val name = "STRING"
    override val numeric = false
    override val type: KClass<StringValue> = StringValue::class
    override fun serializer(size: Int): Serializer<StringValue> = StringValueSerializer
}

@Suppress("UNCHECKED_CAST")
class Complex32ColumnType : ColumnType<Complex32Value>() {
    override val name = "COMPLEX32"
    override val numeric = true
    override val type: KClass<Complex32Value> = Complex32Value::class
    override fun serializer(size: Int): Serializer<Complex32Value> = Complex32ValueSerializer
}

@Suppress("UNCHECKED_CAST")
class Complex64ColumnType : ColumnType<Complex64Value>() {
    override val name = "COMPLEX64"
    override val numeric = true
    override val type: KClass<Complex64Value> = Complex64Value::class
    override fun serializer(size: Int): Serializer<Complex64Value> = Complex64ValueSerializer
}

@Suppress("UNCHECKED_CAST")
class IntVectorColumnType : ColumnType<IntVectorValue>() {
    override val name = "INT_VEC"
    override val numeric = false
    override val type: KClass<IntVectorValue> = IntVectorValue::class
    override fun serializer(size: Int): Serializer<IntVectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedIntVectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class LongVectorColumnType : ColumnType<LongVectorValue>() {
    override val name = "LONG_VEC"
    override val numeric = false
    override val type: KClass<LongVectorValue> = LongVectorValue::class
    override fun serializer(size: Int): Serializer<LongVectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedLongVectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class FloatVectorColumnType : ColumnType<FloatVectorValue>() {
    override val name = "FLOAT_VEC"
    override val numeric = false
    override val type: KClass<FloatVectorValue> = FloatVectorValue::class
    override fun serializer(size: Int): Serializer<FloatVectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedFloatVectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class DoubleVectorColumnType : ColumnType<DoubleVectorValue>() {
    override val name = "DOUBLE_VEC"
    override val numeric = false
    override val type: KClass<DoubleVectorValue> = DoubleVectorValue::class
    override fun serializer(size: Int): Serializer<DoubleVectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedDoubleVectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class BooleanVectorColumnType : ColumnType<BooleanVectorValue>() {
    override val name = "BOOL_VEC"
    override val numeric = false
    override val type: KClass<BooleanVectorValue> = BooleanVectorValue::class
    override fun serializer(size: Int): Serializer<BooleanVectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedBooleanVectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class Complex32VectorColumnType : ColumnType<Complex32VectorValue>() {
    override val name = "COMPLEX32_VEC"
    override val numeric = false
    override val type: KClass<Complex32VectorValue> = Complex32VectorValue::class
    override fun serializer(size: Int): Serializer<Complex32VectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedComplex32VectorSerializer(size)
    }
}

@Suppress("UNCHECKED_CAST")
class Complex64VectorColumnType : ColumnType<Complex64VectorValue>() {
    override val name = "COMPLEX64_VEC"
    override val numeric = false
    override val type: KClass<Complex64VectorValue> = Complex64VectorValue::class
    override fun serializer(size: Int): Serializer<Complex64VectorValue> {
        if (size <= 0) throw IllegalArgumentException("Size attribute for a $name type must be > 0 (is $size).")
        return FixedComplex64VectorSerializer(size)
    }
}