package org.vitrivr.cottontail.database.index.pq

import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.model.values.types.VectorValue

data class PQIndexConfig (val numSubspaces: Int,
                          val numCentroids: Int,
                          val learningDataFraction: Double,
                          val lookuptablePrecision: LookupTablePrecision,
                          val kApproxScan: Int,
                          val seed: Long,
                          val type: ColumnType<out VectorValue<*>>,
                          val complexStrategy: ComplexStrategy
                          ) {
    companion object Serializer: org.mapdb.Serializer<PQIndexConfig> {
        /**
         * Serializes the content of the given value into the given
         * [DataOutput2].
         *
         * @param out DataOutput2 to save object into
         * @param value Object to serialize
         *
         * @throws IOException in case of an I/O error
         */
        override fun serialize(out: DataOutput2, value: PQIndexConfig) {
            out.packInt(value.numSubspaces)
            out.packInt(value.numCentroids)
            out.writeDouble(value.learningDataFraction)
            out.packInt(value.lookuptablePrecision.ordinal)
            out.packInt(value.kApproxScan)
            out.packLong(value.seed)
            out.writeUTF(value.type.name)
            out.packInt(value.complexStrategy.ordinal)
        }

        /**
         * Deserializes and returns the content of the given [DataInput2].
         *
         * @param input DataInput2 to de-serialize data from
         * @param available how many bytes that are available in the DataInput2 for
         * reading, may be -1 (in streams) or 0 (null).
         *
         * @return the de-serialized content of the given [DataInput2]
         * @throws IOException in case of an I/O error
         */
        @Suppress("UNCHECKED_CAST")
        override fun deserialize(input: DataInput2, available: Int)
            = PQIndexConfig(input.unpackInt(),
                input.unpackInt(),
                input.readDouble(),
                LookupTablePrecision.values()[input.unpackInt()],
                input.unpackInt(),
                input.unpackLong(),
                ColumnType.forName(input.readUTF()) as ColumnType<out VectorValue<*>>, // todo: change to precision because we want to easily find the corresponding real vector type to a given complex vector type...
                ComplexStrategy.values()[input.unpackInt()]
        )

        @Suppress("UNCHECKED_CAST")
        fun fromParamsMap(params: Map<String, String>) =
                PQIndexConfig(
                        numSubspaces = (params[PQIndexConfigParamMapKeys.NUM_SUBSPACES.key] ?: error("num_subspaces not found")).toInt(),
                        numCentroids = (params[PQIndexConfigParamMapKeys.NUM_CENTROIDS.key] ?: error("num_centroids not found")).toInt(),
                        learningDataFraction = (params[PQIndexConfigParamMapKeys.LEARNING_DATA_FRACTION.key] ?: error("learning_data_fraction not found")).toDouble(),
                        lookuptablePrecision = LookupTablePrecision.valueOf(params[PQIndexConfigParamMapKeys.LOOKUPTABLE_PRECISION.key] ?: error("lookuptable_precision not found")),
                        kApproxScan = (params[PQIndexConfigParamMapKeys.K_APPROX_SCAN.key] ?: error("k_approx_scan not found")).toInt(),
                        seed = (params[PQIndexConfigParamMapKeys.SEED.key] ?: error("seed not found")).toLong(),
                        type = ColumnType.forName(params[PQIndexConfigParamMapKeys.TYPE.key] ?: error("type not found")) as ColumnType<out VectorValue<*>>,
                        complexStrategy = ComplexStrategy.valueOf(params[PQIndexConfigParamMapKeys.COMPLEX_STRATEGY.key] ?: error("complex_strategy not found")),
                )
    }

    enum class ComplexStrategy {
        DIRECT, SPLIT
    }
}