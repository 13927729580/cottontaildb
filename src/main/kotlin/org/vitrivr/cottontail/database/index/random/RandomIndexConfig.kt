package org.vitrivr.cottontail.database.index.random

import org.mapdb.DataInput2
import org.mapdb.DataOutput2

data class RandomIndexConfig(val fraction: Double, val seed: Long) {

    companion object Serializer : org.mapdb.Serializer<RandomIndexConfig> {
        override fun serialize(out: DataOutput2, value: RandomIndexConfig) {
            out.writeDouble(value.fraction)
            out.packLong(value.seed)
        }

        override fun deserialize(input: DataInput2, available: Int): RandomIndexConfig =
                RandomIndexConfig(input.readDouble(),
                        input.unpackLong(),
                )

        fun fromParamMap(params: Map<String, String>) = RandomIndexConfig(
                fraction = params["fraction"]!!.toDouble(),
                seed = params["seed"]!!.toLong(),
        )
    }
}

