package org.vitrivr.cottontail.model.basics

import org.vitrivr.cottontail.database.column.*
import org.vitrivr.cottontail.model.values.*
import org.vitrivr.cottontail.model.values.types.Value


/**
 * A definition class for a Cottontail DB column be it in a DB or in-memory context  Specifies all the properties of such a and facilitates validation.
 *
 * @author Ralph Gasser
 * @version 1.3.1
 */
class ColumnDef<T: Value>(val name: Name.ColumnName, val type: ColumnType<T>, val logicalSize: Int = 1, val nullable: Boolean = true) {

    /** The physical size of this [ColumnDef] in bytes. */
    val physicalSize: Int
        get() = this.logicalSize * this.type.size

    /**
     * Companion object with some convenience methods.
     */
    companion object {
        /**
         * Returns a [ColumnDef] with the provided attributes. The only difference as compared to using the constructor, is that the [ColumnType] can be provided by name.
         *
         * @param name Name of the new [Column]
         * @param type Name of the [ColumnType] of the new [Column]
         * @param size Logical size of the new [Column] (e.g. for vectors), where eligible.
         * @param nullable Whether or not the [Column] should be nullable.
         */
        fun withAttributes(name: Name.ColumnName, type: String, size: Int = -1, nullable: Boolean = true): ColumnDef<*> = ColumnDef(name, ColumnType.forName(type), size, nullable)
    }

    /**
     * Validates a value with regard to this [ColumnDef] return a flag indicating whether validation was passed.
     *
     * @param value The value that should be validated.
     * @return True if value passes validation, false otherwise.
     */
    fun validate(value: Value?): Boolean {
        if (value != null) {
            if (!this.type.compatible(value)) {
                return false
            }
            return when {
                (value is DoubleVectorValue && value.logicalSize != this.logicalSize) ||
                        (value is FloatVectorValue && value.logicalSize != this.logicalSize) ||
                        (value is LongVectorValue && value.logicalSize != this.logicalSize) ||
                        (value is IntVectorValue && value.logicalSize != this.logicalSize) ||
                        (value is Complex32VectorValue && value.logicalSize != this.logicalSize) ||
                        (value is Complex64VectorValue && value.logicalSize != this.logicalSize) -> false
                else -> true
            }
        } else return this.nullable
    }

    /**
     * Returns the default value for this [ColumnDef].
     *
     * @return Default value for this [ColumnDef].
     */
    fun defaultValue(): Value? = when {
        this.nullable -> null
        this.type is StringColumnType -> StringValue("")
        this.type is FloatColumnType -> FloatValue(0.0f)
        this.type is DoubleColumnType -> DoubleValue(0.0)
        this.type is IntColumnType -> IntValue(0)
        this.type is LongColumnType -> LongValue(0L)
        this.type is ShortColumnType -> ShortValue(0.toShort())
        this.type is ByteColumnType -> ByteValue(0.toByte())
        this.type is BooleanColumnType -> BooleanValue(false)
        this.type is Complex32ColumnType -> Complex32Value.ZERO
        this.type is Complex64ColumnType -> Complex64Value.ZERO
        this.type is DoubleVectorColumnType -> DoubleVectorValue(DoubleArray(this.logicalSize))
        this.type is FloatVectorColumnType -> FloatVectorValue(FloatArray(this.logicalSize))
        this.type is LongVectorColumnType -> LongVectorValue(LongArray(this.logicalSize))
        this.type is IntVectorColumnType -> IntVectorValue(IntArray(this.logicalSize))
        this.type is BooleanVectorColumnType -> BooleanVectorValue(BooleanArray(this.logicalSize))
        this.type is Complex32VectorColumnType -> Complex32VectorValue(FloatArray(2 * this.logicalSize) { 0.0f })
        this.type is Complex64VectorColumnType -> Complex64VectorValue(DoubleArray(2 * this.logicalSize) { 0.0 })
        else -> throw RuntimeException("Default value for the specified type $type has not been specified yet!")
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ColumnDef<*>

        if (name != other.name) return false
        if (type != other.type) return false
        if (logicalSize != other.logicalSize) return false

        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + type.hashCode()
        result = 31 * result + logicalSize.hashCode()
        return result
    }

    override fun toString(): String = "$name(type=$type, size=$logicalSize, nullable=$nullable)"
}