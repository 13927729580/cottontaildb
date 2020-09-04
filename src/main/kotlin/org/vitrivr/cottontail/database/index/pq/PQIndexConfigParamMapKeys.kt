package org.vitrivr.cottontail.database.index.pq

enum class PQIndexConfigParamMapKeys(val key: String) {
    NUM_SUBSPACES("num_subspaces"),
    NUM_CENTROIDS("num_centroids"),
    LEARNING_DATA_FRACTION("learning_data_fraction"),
    SEED("seed")
}