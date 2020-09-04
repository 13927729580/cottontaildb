package org.vitrivr.cottontail.database.index.pq

import org.mapdb.DataInput2
import org.mapdb.DataOutput2

data class PQIndexConfig (val numSubspaces: Int, val numCentroids: Int, /*val learningDataFraction: Double,*/ val seed: Long) {
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
//            out.writeDouble(value.learningDataFraction)
            out.packLong(value.seed)
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
        override fun deserialize(input: DataInput2, available: Int)
            = PQIndexConfig(input.unpackInt(), input.unpackInt(), /*input.readDouble(),*/ input.unpackLong())

        fun fromParamsMap(params: Map<String, String>) =
                PQIndexConfig(
                        numSubspaces = (params[PQIndexConfigParamMapKeys.NUM_SUBSPACES.key] ?: error("num_subspaces not found")).toInt(),
                        numCentroids = (params[PQIndexConfigParamMapKeys.NUM_CENTROIDS.key] ?: error("num_centroids not found")).toInt(),
//                        learningDataFraction = (params[PQIndexConfigParamMapKeys.LEARNING_DATA_FRACTION.key] ?: error("learning_data_fraction not found")).toDouble(),
                        seed = (params[PQIndexConfigParamMapKeys.SEED.key] ?: error("seed not found")).toLong()
                )
    }
}