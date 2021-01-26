package org.vitrivr.cottontail.utilities.di

import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import org.vitrivr.cottontail.database.column.*
import org.vitrivr.cottontail.grpc.CottontailGrpc
import org.vitrivr.cottontail.model.basics.ColumnDef
import java.nio.file.Files
import java.nio.file.Path

/**
 * A [DataImporter] implementation that can be used to read a JSON file containing an array of entries.
 * Each entry needs to be an object for which each key corresponds to a field in the [schema]. The
 * fields must occur in order of definition.
 *
 * @author Ralph Gasser
 * @version 1.0.0
 */
class JsonDataImporter(override val path: Path, override val schema: Array<ColumnDef<*>>) :
    DataImporter {

    /** The [JsonReader] instance used to read the JSON file. */
    private val reader = JsonReader(Files.newBufferedReader(this.path))

    init {
        this.reader.beginArray()         /* Starts reading the JSON array, which is the expected input. */
    }

    /** The [ImportFormat] handled by this [DataImporter]. */
    override val format: ImportFormat = ImportFormat.JSON

    /** Indicator whether this [DataImporter] has been closed. */
    override var closed: Boolean = false
        private set

    /**
     * Returns the next [CottontailGrpc.InsertMessage] from the data.
     *
     * @return [CottontailGrpc.InsertMessage]
     */
    override fun next(): CottontailGrpc.InsertMessage {
        this.reader.beginObject()
        val message = CottontailGrpc.InsertMessage.newBuilder()
        for (column in this.schema) {
            val name = this.reader.nextName()
            if (column.name.simple == name || column.name.toString() == name) {
                val element = CottontailGrpc.InsertMessage.InsertElement.newBuilder()
                val value = when (column.type) {
                    BooleanColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setBooleanData(this.reader.nextBoolean())
                    ByteColumnType,
                    ShortColumnType,
                    IntColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setIntData(this.reader.nextInt())
                    LongColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setLongData(this.reader.nextLong())
                    FloatColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setFloatData(this.reader.nextDouble().toFloat())
                    DoubleColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setDoubleData(this.reader.nextDouble())
                    StringColumnType -> CottontailGrpc.Literal.newBuilder()
                        .setStringData(this.reader.nextString())
                    Complex32ColumnType -> this.readComplex32Value()
                    Complex64ColumnType -> this.readComplex64Value()
                    IntVectorColumnType -> this.readIntVector(column.logicalSize)
                    LongVectorColumnType -> this.readLongVector(column.logicalSize)
                    FloatVectorColumnType -> this.readFloatVector(column.logicalSize)
                    DoubleVectorColumnType -> this.readDoubleVector(column.logicalSize)
                    BooleanVectorColumnType -> this.readBooleanVector(column.logicalSize)
                    Complex32VectorColumnType -> this.readComplex32Vector(column.logicalSize)
                    Complex64VectorColumnType -> this.readComplex64Vector(column.logicalSize)
                }
                element.setColumn(
                    CottontailGrpc.ColumnName.newBuilder().setName(column.name.simple)
                )
                element.setValue(value)
                message.addInserts(element)
            }
        }
        this.reader.endObject()
        return message.build()
    }

    /**
     * Checks if there is another entry and returns true if so, and false otherwise.
     *
     * @return True if there is another entry, false otherwise.
     */
    override fun hasNext(): Boolean =
        this.reader.hasNext() && this.reader.peek() == JsonToken.BEGIN_OBJECT

    /**
     * Reads a complex 32 value from the JSON file.
     *
     * @return [CottontailGrpc.Literal.Builder] containing the value.
     */
    private fun readComplex32Value(): CottontailGrpc.Literal.Builder {
        val value = CottontailGrpc.Complex32.newBuilder()
        this.reader.beginObject()
        this.reader.nextName()
        value.real = this.reader.nextDouble().toFloat()
        this.reader.nextName()
        value.imaginary = this.reader.nextDouble().toFloat()
        this.reader.endObject()
        return CottontailGrpc.Literal.newBuilder().setComplex32Data(value)
    }

    /**
     * Reads a complex 32 value from the JSON file.
     *
     * @return [CottontailGrpc.Literal.Builder] containing the value.
     */
    private fun readComplex64Value(): CottontailGrpc.Literal.Builder {
        val value = CottontailGrpc.Complex64.newBuilder()
        this.reader.beginObject()
        this.reader.nextName()
        value.real = this.reader.nextDouble()
        this.reader.nextName()
        value.imaginary = this.reader.nextDouble()
        this.reader.endObject()
        return CottontailGrpc.Literal.newBuilder().setComplex64Data(value)
    }

    /**
     * Reads a boolean vector of the given size from the JSON file.
     *
     * @param size The size of the boolean vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readBooleanVector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.BoolVector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.reader.nextBoolean())
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setBoolVector(vector))
    }

    /**
     * Reads a int vector of the given size from the JSON file.
     *
     * @param size The size of the int vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readIntVector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.IntVector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.reader.nextInt())
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setIntVector(vector))
    }

    /**
     * Reads a long vector of the given size from the JSON file.
     *
     * @param size The size of the long vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readLongVector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.LongVector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.reader.nextLong())
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setLongVector(vector))
    }

    /**
     * Reads a float vector of the given size from the JSON file.
     *
     * @param size The size of the float vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readFloatVector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.FloatVector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.reader.nextDouble().toFloat())
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setFloatVector(vector))
    }

    /**
     * Reads a double vector of the given size from the JSON file.
     *
     * @param size The size of the double vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readDoubleVector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.DoubleVector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.reader.nextDouble())
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setDoubleVector(vector))
    }

    /**
     * Reads a complex 32 vector of the given size from the JSON file.
     *
     * @param size The size of the double vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readComplex32Vector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.Complex32Vector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.readComplex32Value().complex32Data)
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setComplex32Vector(vector))
    }

    /**
     * Reads a complex64 vector of the given size from the JSON file.
     *
     * @param size The size of the complex64 vector.
     * @return [CottontailGrpc.Literal.Builder] containing the vector.
     */
    private fun readComplex64Vector(size: Int): CottontailGrpc.Literal.Builder {
        val vector = CottontailGrpc.Complex64Vector.newBuilder()
        this.reader.beginArray()
        for (i in 0 until size) {
            vector.addVector(this.readComplex64Value().complex64Data)
        }
        this.reader.endArray()
        return CottontailGrpc.Literal.newBuilder()
            .setVectorData(CottontailGrpc.Vector.newBuilder().setComplex64Vector(vector))
    }

    /**
     * Closes this [JsonDataImporter]
     */
    override fun close() {
        if (!this.closed) {
            this.reader.close()
            this.closed = true
        }
    }
}