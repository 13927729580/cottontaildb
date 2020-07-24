package org.vitrivr.cottontail.database.index.lsh.superbit

import org.mapdb.DataInput2
import org.mapdb.DataOutput2

data class NonBucketingSuperBitLSHIndexConfig(val superBitDepth: Int, val superBitsPerStage: Int, val stages: Int, val seed: Long, val considerImaginary: Boolean, val samplingMethod: SuperBit.SamplingMethod) {
    companion object Serializer: org.mapdb.Serializer<NonBucketingSuperBitLSHIndexConfig> {
        override fun serialize(out: DataOutput2, value: NonBucketingSuperBitLSHIndexConfig) {
            out.packInt(value.superBitDepth)
            out.packInt(value.superBitsPerStage)
            out.packInt(value.stages)
            out.packLong(value.seed)
            out.packInt(if (value.considerImaginary) 1 else 0)
            out.packInt(value.samplingMethod.ordinal)
        }

        override fun deserialize(input: DataInput2, available: Int): NonBucketingSuperBitLSHIndexConfig =
                NonBucketingSuperBitLSHIndexConfig(input.unpackInt(),
                        input.unpackInt(),
                        input.unpackInt(),
                        input.unpackLong(),
                        input.unpackInt() != 0,
                        SuperBit.SamplingMethod.values()[input.unpackInt()])
        fun fromParamMap(params: Map<String, String>) = NonBucketingSuperBitLSHIndexConfig(
                params[SuperBitSLHIndexConfigParamMapKeys.SUPERBIT_DEPTH.key]!!.toInt(),
                params[SuperBitSLHIndexConfigParamMapKeys.SUPERBITS_PER_STAGE.key]!!.toInt(),
                params[SuperBitSLHIndexConfigParamMapKeys.NUM_STAGES.key]!!.toInt(),
                params[SuperBitSLHIndexConfigParamMapKeys.SEED.key]!!.toLong(),
                params[SuperBitSLHIndexConfigParamMapKeys.CONSIDER_IMAGINARY.key]!!.toInt() != 0,
                SuperBit.SamplingMethod.valueOf(params[SuperBitSLHIndexConfigParamMapKeys.SAMPLING_METHOD.key]!!))
    }

}

