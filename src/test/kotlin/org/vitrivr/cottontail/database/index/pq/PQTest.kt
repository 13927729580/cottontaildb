package org.vitrivr.cottontail.database.index.pq

import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import org.junit.jupiter.api.Test
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.model.values.DoubleVectorValue

import org.vitrivr.cottontail.testutils.getComplexVectorsFromFile
import org.vitrivr.cottontail.testutils.sampleVectorsFromCsv
import java.io.File
import java.util.*
import kotlin.math.pow
import kotlin.time.ExperimentalTime
import kotlin.time.measureTimedValue

internal class PQTest {

    @Test
    fun getCodebooks() {
    }

    @ExperimentalTime
    @Test
    fun testPQ() {
        val vectorsFile = File("src/test/resources/sampledVectors90000.csv")
        if (!vectorsFile.exists()) {
            sampleVectorsFromCsv("src/test/resources/complexVectors.csv", vectorsFile.toString(), Random(1234L), 1e-2)
        }
        val data = getComplexVectorsFromFile(vectorsFile.toString())
        val realData = Array(data.size) { i ->
            DoubleArray(data[i].logicalSize) { j ->
                data[i][j].real.value
            }
        }
        val imagData = Array(data.size) { i ->
            DoubleArray(data[i].logicalSize) { j ->
                data[i][j].imaginary.value
            }
        }
        val numSubspaces = 4
        val numCentroids = 128
        val seed = 1234L
        val rng = SplittableRandom(seed)
        val (permutation, reversePermutation) = PQIndex.generateRandomPermutation(realData[0].size, rng)
        val permutedRealData = Array(realData.size) { n ->
            DoubleArray(realData[n].size) {i ->
                realData[n][permutation[i]]
            }
        }
        val permutedImagData = Array(imagData.size) { n ->
            DoubleArray(imagData[n].size) {i ->
                imagData[n][permutation[i]]
            }
        }
        val pqRealTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedRealData) }
        val pqImagTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedImagData) }
        val pqReal = pqRealTimed.value
        val pqImag = pqImagTimed.value
        val outFileDir = File("testOut/pq/numSubspaces${numSubspaces}numCentroids${numCentroids}seed$seed/")
        outFileDir.mkdirs()
        File(outFileDir, "learning_time_real_ms.txt").writeText(pqRealTimed.duration.inMilliseconds.toString())
        File(outFileDir, "learning_time_imag_ms.txt").writeText(pqImagTimed.duration.inMilliseconds.toString())
        csvWriter().open(File(outFileDir, "permutation.csv")) {
            writeRow(reversePermutation.map { it.toString() })
        }
        for (k in 0 until numSubspaces) {
            csvWriter().open(File(outFileDir, "codebookReal$k.csv")) {
                writeRow((1..pqReal.first.dimensionsPerSubspace).map { "subspaceDim$it" })
                pqReal.first.codebooks[k].centroids.forEach { writeRow(it.map { d -> d.toString() }) }
            }
            csvWriter().open(File(outFileDir, "codebookImag$k.csv")) {
                writeRow((1..pqImag.first.dimensionsPerSubspace).map { "subspaceDim$it" })
                pqImag.first.codebooks[k].centroids.forEach { writeRow(it.map { d -> d.toString() }) }
            }
        }
        csvWriter().open(File(outFileDir, "dataSignaturesReal.csv")) {
            writeRow((1..numSubspaces).map{ "subspace${it}CentroidNumber"})
            pqReal.second.forEach {
                writeRow(it.map { i -> i.toString()})
            }
        }
        csvWriter().open(File(outFileDir, "dataSignaturesImag.csv")) {
            writeRow((1..numSubspaces).map{ "subspace${it}CentroidNumber"})
            pqReal.second.forEach {
                writeRow(it.map { i -> i.toString()})
            }
        }
        println("Calculating MSE of IP approximations")
        var MSERealReal = 0.0
        var MSEImagImag = 0.0
        var MSERealImag = 0.0
        var MSEImagReal = 0.0
        var MSEAbs = 0.0
        var n = 0
        val vRealData = realData.map {
            DoubleVectorValue(it)
        }
        val vImagData = imagData.map {
            DoubleVectorValue(it)
        }
        print("i = ")
        csvWriter().open(File(outFileDir, "IPs.csv")) {
            writeRow(listOf("i", "j", "exactAbsIP", "exactRealReal", "exactImagImag", "exactImagReal", "exactRealImag", "approxAbsIP", "approxRealReal", "approxImagImag", "approxImagReal", "approxRealImag"))
            for (i in data.indices) {
                if (i % 1000 == 0) {
                    if (i % 10000 == 0) {
                        println("$i")
                    } else {
                        print(".")
                    }
                }
                for (j in i until data.size) {
                    if (j % 10 == 0) {
                        val exactAbsIP = 1.0 - AbsoluteInnerProductDistance(data[i], data[j]).value
                        val exactRealReal = vRealData[i].dot(vRealData[j]).value
                        val exactImagImag = vImagData[i].dot(vImagData[j]).value
                        val exactImagReal = vImagData[i].dot(vRealData[j]).value
                        val exactRealImag = vRealData[i].dot(vImagData[j]).value
                        val approxRealReal = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedRealData[j])
                        val approxImagImag = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedImagData[j])
                        val approxImagReal = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedRealData[j])
                        val approxRealImag = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedImagData[j])
                        val approxAbsIP = ((approxRealReal + approxImagImag).pow(2) + (approxImagReal - approxRealImag).pow(2)).pow(0.5)
                        if (j % 10000 == 0) {
                            writeRow(listOf(i, j, exactAbsIP, exactRealReal, exactImagImag, exactImagReal, exactRealImag, approxAbsIP, approxRealReal, approxImagImag, approxImagReal, approxRealImag).map { it.toString() })
                        }
                        MSEAbs += (exactAbsIP - approxAbsIP).pow(2)
                        MSERealReal += (exactRealReal - approxRealReal).pow(2)
                        MSEImagImag += (exactImagImag - approxImagImag).pow(2)
                        MSEImagReal += (exactImagReal - approxImagReal).pow(2)
                        MSERealImag += (exactRealImag - approxRealImag ).pow(2)
                        n++
                    }
                }
            }
        }
        println("")
        MSEAbs /= n
        MSERealReal /= n
        MSEImagImag /= n
        MSEImagReal /= n
        MSERealImag /= n
        println("n = $n")
        println("MSEAbs = $MSEAbs")
        println("MSERealReal = $MSERealReal")
        println("MSEImagImag = $MSEImagImag")
        println("MSEImagReal = $MSEImagReal")
        println("MSERealImag = $MSERealImag")
        File(outFileDir, "MSE.txt").writeText("n: $n\nMSEAsymmetric: $MSEAbs\nMSERealReal: $MSERealReal\nMSEImagImag: $MSEImagImag\nMSEImagReal: $MSEImagReal\nMSERealImag: $MSERealImag")
        vectorsFile.copyTo(File(outFileDir, vectorsFile.name))
    }
}