package ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk

import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.basics.Page
import java.nio.ByteBuffer

import kotlin.math.max

/**
 * This is a wrapper for an individual data [DataPage] managed by the HARE storage engine. At their core,
 * [DataPage]s are mere chunks of data  by a [ByteBuffer] with a fixed size= 2^n.
 *
 * @see DiskManager
 *
 * @version 1.1
 * @author Ralph Gasser
 */
inline class DataPage(private val _data: ByteBuffer) : Page {
    /**
     * This is an internal accessor which creates a duplicate view of the [ByteBuffer] backing this [DataPage].
     * It should only be used by the HARE storage engine.
     */
    override val data: ByteBuffer
        get() = this._data.duplicate().rewind()

    override fun getBytes(index: Int, byteBuffer: ByteBuffer): ByteBuffer {
        byteBuffer.put(this.data.position(index).limit(index + byteBuffer.remaining()))
        return byteBuffer
    }
    override fun getBytes(index: Int, bytes: ByteArray) : ByteArray {
        this.data.position(index).get(bytes)
        return bytes
    }
    override fun getBytes(index: Int, limit: Int) : ByteArray = getBytes(index, ByteArray(max(this._data.capacity(), limit-index)))
    override fun getBytes(index: Int) : ByteArray = getBytes(index, this._data.capacity())
    override fun getByte(index: Int): Byte = this._data.get(index)
    override fun getShort(index: Int): Short = this._data.getShort(index)
    override fun getChar(index: Int): Char = this._data.getChar(index)
    override fun getInt(index: Int): Int = this._data.getInt(index)
    override fun getLong(index: Int): Long = this._data.getLong(index)
    override fun getFloat(index: Int): Float = this._data.getFloat(index)
    override fun getDouble(index: Int): Double =  this._data.getDouble(index)

    /**
     * Writes a [Byte] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Byte] value to write.
     * @return This [DataPage]
     */
    override fun putByte(index: Int, value: Byte): DataPage {
        this._data.put(index, value)
        return this
    }

    /**
     * Writes a [ByteArray] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [ByteArray] value to write.
     * @return This [DataPage]
     */
    override fun putBytes(index: Int, value: ByteArray): DataPage {
        this._data.position(index).put(value).rewind()
        return this
    }

    /**
     * Writes a [ByteBuffer] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [ByteArray] value to write.
     * @return This [DataPage]
     */
    override fun putBytes(index: Int, value: ByteBuffer): DataPage {
        this._data.position(index).put(value).rewind()
        return this
    }

    /**
     * Writes a [Short] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Short] value to write.
     * @return This [DataPage]
     */
    override fun putShort(index: Int, value: Short): DataPage {
        this._data.putShort(index, value)
        return this
    }

    /**
     * Writes a [Char] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Char] value to write.
     * @return This [DataPage]
     */
    override fun putChar(index: Int, value: Char): DataPage {
        this._data.putChar(index, value)
        return this
    }

    /**
     * Writes a [Int] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Int] value to write.
     * @return This [DataPage]
     */
    override fun putInt(index: Int, value: Int): DataPage {
        this._data.putInt(index, value)
        return this
    }

    /**
     * Writes a [Long] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Long] value to write.
     * @return This [DataPage]
     */
    override fun putLong(index: Int, value: Long): DataPage {
        this._data.putLong(index, value)
        return this
    }

    /**
     * Writes a [Float] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Float] value to write.
     * @return This [DataPage]
     */
    override fun putFloat(index: Int, value: Float): DataPage {
        this._data.putFloat(index, value)
        return this
    }

    /**
     * Writes a [Double] to the given position.
     *
     * @param index Position to write byte to.
     * @param value New [Double] value to write.
     * @return This [DataPage]
     */
    override fun putDouble(index: Int, value: Double): DataPage {
        this._data.putDouble(index, value)
        return this
    }

    /**
     * Clears the data in this [DataPage] effectively setting it to zero.
     */
    override fun clear(): DataPage {
        for (i in 0 until this._data.capacity()) {
            this._data.put(0, 0)
        }
        return this
    }
}

