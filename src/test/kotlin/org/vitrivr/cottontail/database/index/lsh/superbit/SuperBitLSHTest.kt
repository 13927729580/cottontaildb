package org.vitrivr.cottontail.database.index.lsh.superbit

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.junit.jupiter.params.provider.ValueSource
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.CosineDistance
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import java.util.*
import kotlin.time.ExperimentalTime

internal class SuperBitLSHTest {
    companion object {
        private val stages = arrayOf(1, 2, 4, 8)
        private val buckets = arrayOf(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
        private val seeds = arrayOf(1234L, 4321L, 82134L, 1337L, 42L)

        @JvmStatic
        fun provideConfigurationsForSBLSH(): List<Arguments> {
            return stages.flatMap { s ->
                buckets.flatMap { b ->
                    seeds.slice(0 until 1).map { seed ->
                        Arguments.of(s, b, seed)
                    }
                }
            }
        }
    }

    @ExperimentalTime
    @ParameterizedTest
    @MethodSource("provideConfigurationsForSBLSH")
    fun testRandomComplex64Vectors(stages: Int, buckets: Int, seed: Long) {
        println("stages: $stages buckets: $buckets seed: $seed")
        val rng = SplittableRandom(seed)
        val numDim = 20
        val lsh = SuperBitLSH(stages, buckets, numDim, seed, Complex64VectorValue.zero(numDim))
        val vectors = Array(20) {
            val v = Complex64VectorValue.random(numDim, rng)
            v / v.norm2()
        }
        val bucketSignatures = vectors.map {
            lsh.hash(it)
        }

        // compare pair-wise bucket signature and cosine distance
        println("i,j,cosinedistance,numDiffBuckets")
        for (i in vectors.indices) {
            for (j in i until vectors.size) {
                val a = vectors[i]
                val b = vectors[j]
                val d = AbsoluteInnerProductDistance(a, b)
                val d2 = (bucketSignatures[i] zip bucketSignatures[j]).map { (a, b) ->
                    if (a == b) 0 else 1
                }.sum()
                println("$i,$j,${d.value},${d2}")
            }
        }
    }
}