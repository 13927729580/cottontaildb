package org.vitrivr.cottontail.database.index.lsh.superbit

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.vitrivr.cottontail.math.basics.isApproximatelyTheSame
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.RealInnerProductDistance
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.util.*

internal class SuperBitTest {

    companion object {
        val numDim = 20
        val numVecs = 100
        val Ns = 1 until 10
        val Ls = 1 until 10
        @JvmStatic
        fun provideConfigurationsForSB(): List<Arguments> {
            return Ns.flatMap { N ->
                Ls.map { L ->
                    Arguments.of(N, L)
                }
            }
        }
    }
    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun signatureComplex(N: Int, L: Int) {
        println("N: $N, L: $L")
        val rng = SplittableRandom(1234)
        val sb = SuperBit(numDim, N, L, 1234, Complex64VectorValue.zero(20))
        // check orthogonality
        for (i in sb.hyperplanes.indices) {
            for (j in i until sb.hyperplanes.size) {
                println("$i, $j: ${(sb.hyperplanes[i] dot sb.hyperplanes[j]).abs().value}")
            }
        }
        val vectors = Array(numVecs) {
            val v = Complex64VectorValue.random(numDim, rng)
            v / v.norm2()
        }
        val signatures = vectors.map {
            sb.signature(it)
        }

        // compare pair-wise bucket signature and cosine distance
        // we build a bucket-hamming distance on the basis of how many buckets are different... (probably shitty this way)
        println("i,j,absoluteIPDist,realIPDist,hammingDist")
        for (i in vectors.indices) {
            for (j in i until vectors.size) {
                val a = vectors[i]
                val b = vectors[j]
                val d = AbsoluteInnerProductDistance(a, b)
                val dreal = RealInnerProductDistance(a, b)
                val hd = (signatures[i] zip signatures[j]).map { (a, b) ->
                    if (a == b) 0 else 1
                }.sum()
                println("$i,$j,${d.value},${dreal.value},${hd}")
            }
        }
    }
    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun signatureReal(N: Int, L: Int) {
        println("N: $N, L: $L")
        val rng = SplittableRandom(1234)
        val sb = SuperBit(numDim, N, L, 1234, DoubleVectorValue.zero(20))
        val vectors = Array(numVecs) {
            val v = DoubleVectorValue.random(numDim, rng)
            v / v.norm2()
        }
        val signatures = vectors.map {
            sb.signature(it)
        }

        // compare pair-wise bucket signature and cosine distance
        // we build a bucket-hamming distance on the basis of how many buckets are different... (probably shitty this way)
        println("i,j,absoluteIPDist,realIPDist,hammingDist")
        for (i in vectors.indices) {
            for (j in i until vectors.size) {
                val a = vectors[i]
                val b = vectors[j]
                val d = AbsoluteInnerProductDistance(a, b)
                val dreal = RealInnerProductDistance(a, b)
                val hd = (signatures[i] zip signatures[j]).map { (a, b) ->
                    if (a == b) 0 else 1
                }.sum()
                println("$i,$j,${d.value},${dreal.value},${hd}")
            }
        }
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun testOrthogonalityReal(N: Int, L: Int) {
        testOrthogonality2(N, L, DoubleVectorValue.zero(numDim))
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun testOrthogonalityComplex(N: Int, L: Int) {
        testOrthogonality2(N, L, Complex64VectorValue.zero(numDim))
    }

    private fun testOrthogonality2(N: Int, L: Int, vec: Any) {
        val sb = SuperBit(numDim, N, L, 1234, vec as VectorValue<*>)
        for (l in 0 until L) {
            for (n in 1 until N) {
                isApproximatelyTheSame(0.0f,
                        (sb.hyperplanes[l * N] dot sb.hyperplanes[l * N + n]).abs().value)
            }
        }
    }
}