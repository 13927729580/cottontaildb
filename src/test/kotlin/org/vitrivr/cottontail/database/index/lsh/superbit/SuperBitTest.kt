package org.vitrivr.cottontail.database.index.lsh.superbit

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.vitrivr.cottontail.math.basics.isApproximatelyTheSame
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.RealInnerProductDistance
import org.vitrivr.cottontail.model.values.Complex64Value
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.io.File
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
        val outDir = File("testOut/complex64")
        val rng = Random(1234)
        val sb = SuperBit(numDim, N, L, 1234, Complex64VectorValue.zero(20))
        val vectors = Array(numVecs) {
            val v = Complex64VectorValue(DoubleArray(numDim * 2) { rng.nextGaussian() })
            v / v.norm2()
        }
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, sb, File(outDir, "complex64SBLSHSignaturesN${N}L${L}.csv"))
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun signatureComplexFromFile(N: Int, L: Int) {
        println("N: $N, L: $L")
        val outDir = File("testOut/fromCsv")
        val sb = SuperBit(numDim, N, L, 1234, Complex64VectorValue.zero(20))
        val vecs = csvReader().open("src/test/resources/sampledVectors.csv") {
            readAllAsSequence().drop(1).toList()
        }
        println("${vecs.size} vectors read.")
        val vectors = vecs.map { row ->
            val v = Complex64VectorValue((0 until numDim).map { Complex64Value(row[it + 1].toDouble(), row[it + 2].toDouble()) }.toTypedArray())
            v / v.norm2()
        }.toTypedArray()
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, sb, File(outDir, "complex64SBLSHSignaturesN${N}L${L}.csv"))
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun signatureReal(N: Int, L: Int) {
        val outDir = File("testOut/real")
        println("N: $N, L: $L")
        val rng = Random(1234)
        val sb = SuperBit(numDim, N, L, 1234, DoubleVectorValue.zero(20))
        val vectors = Array(numVecs) {
            val v = DoubleVectorValue(DoubleArray(numDim) { rng.nextGaussian() })
            v / v.norm2()
        }
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, sb, File(outDir, "doubleSBLSHSignaturesN${N}L${L}.csv"))
    }

    private fun compareNormalizedVectors(vectors: Array<VectorValue<*>>, sb: SuperBit, outCsvFile: File) {
        val signatures = vectors.map {
            sb.signature(it)
        }

        outCsvFile.parentFile.mkdirs()
        csvWriter().open(outCsvFile) {
            val header = listOf("i", "j", "absoluteIPDist", "realIPDist", "hammingDist")
            writeRow(header)
            for (i in vectors.indices) {
                for (j in i until vectors.size) {
                    val a = vectors[i]
                    val b = vectors[j]
                    val d = AbsoluteInnerProductDistance(a, b)
                    val dreal = RealInnerProductDistance(a, b)
                    val hd = (signatures[i] zip signatures[j]).map { (a, b) ->
                        if (a == b) 0 else 1
                    }.sum()
                    val data = listOf(i, j, d.value, dreal.value, hd)
                    writeRow(data)
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun testOrthogonalityReal(N: Int, L: Int) {
        testOrthogonality(N, L, DoubleVectorValue.zero(numDim))
    }

    @ParameterizedTest
    @MethodSource("provideConfigurationsForSB")
    fun testOrthogonalityComplex(N: Int, L: Int) {
        testOrthogonality(N, L, Complex64VectorValue.zero(numDim))
    }

    private fun testOrthogonality(N: Int, L: Int, vec: Any) {
        val sb = SuperBit(numDim, N, L, 1234, vec as VectorValue<*>)
        for (l in 0 until L) {
            for (n in 1 until N) {
                isApproximatelyTheSame(0.0f,
                        (sb.hyperplanes[l * N] dot sb.hyperplanes[l * N + n]).abs().value)
            }
        }
    }
    /* to plot the compareNormalizedVectors output:
    In [81]: fig, axes = plt.subplots(9, 9, sharex=True)

In [82]: for l in L:
    ...:     for n in N:
    ...:         pd.read_csv(f"complex64/complex64SBLSHSignaturesN{n}L{l}_includingImag.csv").plot("realIPDist", "hammingDist", ax=axes[n-1][l-1], kind="scatter", s=1)
    ...:         axes[n-1][l-1].set_title(f"N={n}, L={l}")
    ...:

In [83]: plt.suptitle("Complex including imaginary part in signature")
Out[83]: Text(0.5, 0.98, 'Complex including imaginary part in signature')

     */
}