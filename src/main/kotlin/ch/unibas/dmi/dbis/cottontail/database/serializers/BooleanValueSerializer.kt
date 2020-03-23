package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.BooleanValue
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.serializer.Serializer
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import java.nio.channels.ReadableByteChannel
import java.nio.channels.WritableByteChannel

object BooleanValueSerializer : Serializer<BooleanValue> {
    override fun deserialize(input: DataInput2, available: Int): BooleanValue = BooleanValue(input.readBoolean())
    override fun serialize(out: DataOutput2, value: BooleanValue) {
        out.writeBoolean(value.value)
    }
    override val physicalSize: Int = Byte.SIZE_BYTES
    override val logicalSize: Int = -1
    override fun serialize(channel: WritableByteChannel, value: BooleanValue) {
        TODO("Not yet implemented")
    }

    override fun deserialize(channel: ReadableByteChannel): BooleanValue {
        TODO("Not yet implemented")
    }
}