package org.vitrivr.cottontail.database.index.lsh.superbit

import org.mapdb.DataInput2
import org.mapdb.DataOutput2

data class SuperBitLSHIndexConfig(val buckets: Int, val stages: Int, val seed: Long, val considerImaginary: Boolean, val samplingMethod: SuperBit.SamplingMethod) {
    companion object Serializer: org.mapdb.Serializer<SuperBitLSHIndexConfig> {
        override fun serialize(out: DataOutput2, value: SuperBitLSHIndexConfig) {
            out.packInt(value.buckets)
            out.packInt(value.stages)
            out.packLong(value.seed)
            out.packInt(if (value.considerImaginary) 1 else 0)
            out.packInt(value.samplingMethod.ordinal)
        }

        override fun deserialize(input: DataInput2, available: Int): SuperBitLSHIndexConfig =
                SuperBitLSHIndexConfig(input.unpackInt(),
                        input.unpackInt(),
                        input.unpackLong(),
                        input.unpackInt() != 0,
                        SuperBit.SamplingMethod.values()[input.unpackInt()])
        fun fromParamMap(params: Map<String, String>) = SuperBitLSHIndexConfig(
            params[SuperBitSLHIndexConfigParamMapKeys.NUM_STAGES.key]!!.toInt(),
            params[SuperBitSLHIndexConfigParamMapKeys.NUM_BUCKETS.key]!!.toInt(),
            params[SuperBitSLHIndexConfigParamMapKeys.SEED.key]!!.toLong(),
            params[SuperBitSLHIndexConfigParamMapKeys.CONSIDER_IMAGINARY.key]!!.toInt() != 0,
            SuperBit.SamplingMethod.valueOf(params[SuperBitSLHIndexConfigParamMapKeys.SAMPLING_METHOD.key]!!))
    }
}

