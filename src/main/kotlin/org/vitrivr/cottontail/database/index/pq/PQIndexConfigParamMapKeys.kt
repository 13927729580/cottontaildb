package org.vitrivr.cottontail.database.index.pq

enum class PQIndexConfigParamMapKeys(val key: String) {
    NUM_SUBSPACES("num_subspaces"),
    NUM_CENTROIDS("num_centroids"),
    LEARNING_DATA_FRACTION("learning_data_fraction"),
    LOOKUPTABLE_PRECISION("lookuptable_precision"), // todo: this should be a query-time configuration
    K_APPROX_SCAN("k_approx_scan"), // todo: this as well!
    SEED("seed"),
    TYPE("type"),
}