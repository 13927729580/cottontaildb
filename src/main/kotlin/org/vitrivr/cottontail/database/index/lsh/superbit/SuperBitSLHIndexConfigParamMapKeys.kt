package org.vitrivr.cottontail.database.index.lsh.superbit

/*
keys used by the SBLSH index config classes to access the parameter maps when creating them with maps,
which is the case via GRPC
 */
enum class SuperBitSLHIndexConfigParamMapKeys(val key: String) {
    SUPERBIT_DEPTH("superbitdepth"),
    SUPERBITS_PER_STAGE("superbitsperstage"),
    SEED("seed"),
    NUM_STAGES("stages"),
    NUM_BUCKETS("buckets"),
    CONSIDER_IMAGINARY("considerimaginary"),
    SAMPLING_METHOD("samplingmethod")
}