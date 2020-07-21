package org.vitrivr.cottontail.database.index.lsh.superbit

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.metrics.RealInnerProductDistance
import org.vitrivr.cottontail.model.values.Complex64Value
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import java.io.File
import java.util.*
import kotlin.time.ExperimentalTime

internal class SuperBitLSHTest {
    companion object {
        private val stages = arrayOf(8, 20)
        private val buckets = arrayOf(4, 16, 32, 64, 128, 256, 512)
        private val seeds = arrayOf(1234L, 4321L, 82134L, 1337L, 42L)
        private val numVectors = 100

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
        val rng = Random(seed)
        val numDim = 20
        val lsh = SuperBitLSH(stages, buckets, numDim, seed, Complex64VectorValue.zero(numDim), false, SuperBit.SamplingMethod.UNIFORM)
        val vectors = Array(numVectors) {
            val v = Complex64VectorValue(DoubleArray(numDim * 2) { rng.nextGaussian() })
            v / v.norm2()
        }
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, lsh, File("testOut", "complex64/complexVectorsBucketDistances_stages${stages}buckets${buckets}seed${seed}.csv"))
    }

    private fun compareNormalizedVectors(vectors: Array<VectorValue<*>>, lsh: SuperBitLSH, outCSVFile: File) {
        outCSVFile.parentFile.mkdirs()
        val bucketSignatures = vectors.map {
            lsh.hash(it)
        }

        // compare pair-wise bucket signature and IP distances
        csvWriter().open(outCSVFile) {
        writeRow(listOf("i", "j", "absoulteIPDistance", "realIPDistance", "numDiffBuckets"))
            for (i in vectors.indices) {
                for (j in i until vectors.size) {
                    val a = vectors[i]
                    val b = vectors[j]
                    val d = AbsoluteInnerProductDistance(a, b)
                    val r = RealInnerProductDistance(a, b)
                    val d2 = (bucketSignatures[i] zip bucketSignatures[j]).map { (a, b) ->
                        if (a == b) 0 else 1
                    }.sum()
                    writeRow(listOf(i, j, d.value, r.value, d2))
                }
            }
        }
    }

    @ExperimentalTime
    @ParameterizedTest
    @MethodSource("provideConfigurationsForSBLSH")
    fun testRandomDoubleVectors(stages: Int, buckets: Int, seed: Long) {
        println("stages: $stages buckets: $buckets seed: $seed")
        val rng = Random(seed)
        val numDim = 20
        val lsh = SuperBitLSH(stages, buckets, numDim, seed, DoubleVectorValue.zero(numDim), false, SuperBit.SamplingMethod.UNIFORM)
        val vectors = Array(numVectors) {
            val v = DoubleVectorValue(DoubleArray(numDim) { rng.nextGaussian() })
            v / v.norm2()
        }
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, lsh, File("testOut", "real/realVectorsBucketDistances_stages${stages}buckets${buckets}seed${seed}.csv"))
    }

    @ExperimentalTime
    @ParameterizedTest
    @MethodSource("provideConfigurationsForSBLSH")
    fun testComplexVectorsFromFile(stages: Int, buckets: Int, seed: Long) {
        println("stages: $stages buckets: $buckets seed: $seed")
        val numDim = 20
        val lsh = SuperBitLSH(stages, buckets, numDim, seed, DoubleVectorValue.zero(numDim), false, SuperBit.SamplingMethod.UNIFORM)
        val file = File("src/test/resources/sampledVectors.csv")
        if (!file.exists()) {
            sampleVectorsFromCsv("src/test/resources/complexVectors.csv", "src/test/resources/sampledVectors.csv")
        }
        val vecs = csvReader().open(file) {
            readAllAsSequence().drop(1).toList()
        }
        println("${vecs.size} vectors read.")
        val vectors = vecs.map { row ->
            val v = Complex64VectorValue((0 until numDim).map { Complex64Value(row[it + 1].toDouble(), row[it + 2].toDouble()) }.toTypedArray())
            v / v.norm2()
        }.toTypedArray()
        compareNormalizedVectors(vectors as Array<VectorValue<*>>, lsh, File("testOut", "fromCsv/CsvVectorsBucketDistances_stages${stages}buckets${buckets}seed${seed}.csv"))
    }

    private fun sampleVectorsFromCsv(infile: String, outfile: String) {
        val rng = Random()
        csvReader().open(infile) {
            csvWriter().open(outfile) {
                writeRow(readNext()!!)
                writeRows(readAllAsSequence().filter { rng.nextDouble() < 1e-5 }.toList())
            }
        }
    }
}