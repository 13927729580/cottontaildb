package org.vitrivr.cottontail.database.index.va

import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.mapdb.Serializer

data class VectorApproximationSignature(val tupleId: Long, val signature: IntArray)

object VAPlusSignatureSerializer : Serializer<VectorApproximationSignature> {

    override fun serialize(out: DataOutput2, value: VectorApproximationSignature) {
        //val signatureArray = value.signature.toLongArray()
        val signatureArray = value.signature
        out.writeLong(value.tupleId)
        out.packInt(signatureArray.size)
        //out.packLongArray(signatureArray, 0, signatureArray.size)
        for (signature in signatureArray) {
            out.writeInt(signature)
        }
    }

    override fun deserialize(input: DataInput2, available: Int): VectorApproximationSignature {
        val tupleId = input.readLong()
        //val signature = LongArray(input.unpackInt())
        val signature = IntArray(input.unpackInt()) {
            input.readInt()
        }
        //input.unpackLongArray(signature, 0, signature.size)
        return VectorApproximationSignature(tupleId, signature)
    }

}
