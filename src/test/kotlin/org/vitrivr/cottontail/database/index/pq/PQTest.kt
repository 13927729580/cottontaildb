package org.vitrivr.cottontail.database.index.pq

import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import org.junit.jupiter.api.Test
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue

import org.vitrivr.cottontail.testutils.getComplexVectorsFromFile
import org.vitrivr.cottontail.testutils.sampleVectorsFromCsv
import java.io.File
import java.util.*
import kotlin.math.pow
import kotlin.time.ExperimentalTime
import kotlin.time.measureTimedValue

internal class PQTest {

    /*
    todo: test with noisy data
     */

    @ExperimentalTime
    @Test
    fun testPQ() {
        val queryVectorsFile = File("src/test/resources/queryVectors.csv")
        val vectorsFile = File("src/test/resources/sampledVectors90000.csv")
        if (!vectorsFile.exists()) {
            sampleVectorsFromCsv("src/test/resources/complexVectors.csv", vectorsFile.toString(), Random(1234L), 1e-2)
        }
        val dbData = getComplexVectorsFromFile(vectorsFile.toString(), 1, 20)
        val queryData = getComplexVectorsFromFile(queryVectorsFile.toString(), 2, 20)
        val realDbData = getRealPart(dbData)
        val realQData = getRealPart(queryData)
        val imagDbData = getImagPart(dbData)
        val imagQData = getImagPart(queryData)
        val numSubspaces = 4
        val numCentroids = 64
        val seed = 1234L
        val rng = SplittableRandom(seed)
        val (permutation, reversePermutation) = PQIndex.generateRandomPermutation(realDbData[0].size, rng)
        val permutedRealDbData = permuteData(realDbData, permutation)
        val permutedRealQData = permuteData(realQData, permutation)
        val permutedImagDbData = permuteData(imagDbData, permutation)
        val permutedImagQData = permuteData(imagQData, permutation)
//        val pqRealTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedRealDbData, permutedRealQData) }
        val pqRealTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedRealDbData, null) }
//        val pqImagTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedImagDbData, permutedImagQData) }
        val pqImagTimed = measureTimedValue { PQ.fromPermutedData(numSubspaces, numCentroids, permutedImagDbData, null) }
        val pqReal = pqRealTimed.value
        val pqImag = pqImagTimed.value
        val outFileDir = File("testOut/pq/nss${numSubspaces}nc${numCentroids}seed${seed}WithoutQData/")
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
            csvWriter().open(File(outFileDir, "codebookReal${k}inverseCovMatrix.csv")) {
                val cb = pqReal.first.codebooks[k]
                writeRow(listOf("row") + (1..pqReal.first.dimensionsPerSubspace).map { "col$it" })
                cb.inverseDataCovarianceMatrix.data.forEachIndexed { i, row ->
                    writeRow(listOf(i.toString()) + row.map { it.toString() })
                }
            }
            csvWriter().open(File(outFileDir, "codebookImag${k}inverseCovMatrix.csv")) {
                val cb = pqImag.first.codebooks[k]
                writeRow(listOf("row") + (1..pqImag.first.dimensionsPerSubspace).map { "col$it" })
                cb.inverseDataCovarianceMatrix.data.forEachIndexed { i, row ->
                    writeRow(listOf(i.toString()) + row.map { it.toString() })
                }
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
            pqImag.second.forEach {
                writeRow(it.map { i -> i.toString()})
            }
        }
        println("Calculating MSE of IP approximations")
        var MSERealReal = 0.0
        var MSERealRealQ = 0.0
        var MSEImagImag = 0.0
        var MSEImagImagQ = 0.0
        var MSERealImag = 0.0
        var MSERealImagQ = 0.0
        var MSEImagReal = 0.0
        var MSEImagRealQ = 0.0
        var MSEAbs = 0.0
        var MSEAbsQ = 0.0
        val vRealDbData = realDbData.map {
            DoubleVectorValue(it)
        }
        val vRealQData = realQData.map {
            DoubleVectorValue(it)
        }
        val vImagDbData = imagDbData.map {
            DoubleVectorValue(it)
        }
        val vImagQData = imagQData.map {
            DoubleVectorValue(it)
        }
        val numPairs = 100000
        print("i = ")
        csvWriter().open(File(outFileDir, "IPs.csv")) db@{
            csvWriter().open(File(outFileDir, "IPsQ.csv")) q@{
                this@db.writeRow(listOf("i", "j", "exactAbsIP", "exactRealReal", "exactImagImag", "exactImagReal", "exactRealImag", "approxAbsIP", "approxRealReal", "approxImagImag", "approxImagReal", "approxRealImag", "exactIPReal", "exactIPImag"))
                this@q.writeRow(listOf("i", "j", "exactAbsIP", "exactRealReal", "exactImagImag", "exactImagReal", "exactRealImag", "approxAbsIP", "approxRealReal", "approxImagImag", "approxImagReal", "approxRealImag", "exactIPReal", "exactIPImag"))
                for (n in 0 until numPairs) {
                    if (n % 1000 == 0) {
                        if (n % 10000 == 0) {
                            println("$n")
                        } else {
                            print(".")
                        }
                    }
                    val i = rng.nextInt(dbData.size)
                    val j = rng.nextInt(dbData.size)
                    var jQ = rng.nextInt(queryData.size)
                    while (queryData[jQ].norm2().value == 0.0) jQ = rng.nextInt(queryData.size)
                    val exactAbsIP = 1.0 - AbsoluteInnerProductDistance(dbData[i], dbData[j]).value
                    val exactAbsIPQ = 1.0 - AbsoluteInnerProductDistance(dbData[i], queryData[jQ]).value
                    val exactRealReal = vRealDbData[i].dot(vRealDbData[j]).value
                    val exactRealRealQ = vRealDbData[i].dot(vRealQData[jQ]).value
                    val exactImagImag = vImagDbData[i].dot(vImagDbData[j]).value
                    val exactImagImagQ = vImagDbData[i].dot(vImagQData[jQ]).value
                    val exactImagReal = vImagDbData[i].dot(vRealDbData[j]).value
                    val exactImagRealQ = vImagDbData[i].dot(vRealQData[jQ]).value
                    val exactRealImag = vRealDbData[i].dot(vImagDbData[j]).value
                    val exactRealImagQ = vRealDbData[i].dot(vImagQData[jQ]).value
                    val approxRealReal = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedRealDbData[j])
                    val approxRealRealQ = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedRealQData[jQ])
                    val approxImagImag = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedImagDbData[j])
                    val approxImagImagQ = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedImagQData[jQ])
                    val approxImagReal = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedRealDbData[j])
                    val approxImagRealQ = pqImag.first.approximateAsymmetricIP(pqImag.second[i], permutedRealQData[jQ])
                    val approxRealImag = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedImagDbData[j])
                    val approxRealImagQ = pqReal.first.approximateAsymmetricIP(pqReal.second[i], permutedImagQData[jQ])
                    val approxAbsIP = ((approxRealReal + approxImagImag).pow(2) + (approxImagReal - approxRealImag).pow(2)).pow(0.5)
                    val approxAbsIPQ = ((approxRealRealQ + approxImagImagQ).pow(2) + (approxImagRealQ - approxRealImagQ).pow(2)).pow(0.5)
                    val exactIP = dbData[i].dot(dbData[j])
                    val exactIPQ = dbData[i].dot(queryData[jQ])
                    this@db.writeRow(listOf(i, j, exactAbsIP, exactRealReal, exactImagImag, exactImagReal, exactRealImag, approxAbsIP, approxRealReal, approxImagImag, approxImagReal, approxRealImag, exactIP.real.value, exactIP.imaginary.value).map { it.toString() })
                    this@q.writeRow(listOf(i, jQ, exactAbsIPQ, exactRealRealQ, exactImagImagQ, exactImagRealQ, exactRealImagQ, approxAbsIPQ, approxRealRealQ, approxImagImagQ, approxImagRealQ, approxRealImagQ, exactIPQ.real.value, exactIPQ.imaginary.value).map { it.toString() })
                    MSEAbs += (exactAbsIP - approxAbsIP).pow(2)
                    MSEAbsQ += (exactAbsIPQ - approxAbsIPQ).pow(2)
                    MSERealReal += (exactRealReal - approxRealReal).pow(2)
                    MSERealRealQ += (exactRealRealQ - approxRealRealQ).pow(2)
                    MSEImagImag += (exactImagImag - approxImagImag).pow(2)
                    MSEImagImagQ += (exactImagImagQ - approxImagImagQ).pow(2)
                    MSEImagReal += (exactImagReal - approxImagReal).pow(2)
                    MSEImagRealQ += (exactImagRealQ - approxImagRealQ).pow(2)
                    MSERealImag += (exactRealImag - approxRealImag).pow(2)
                    MSERealImagQ += (exactRealImagQ - approxRealImagQ).pow(2)
                }
            }
        }
        println("")
        MSEAbs /= numPairs
        MSEAbsQ /= numPairs
        MSERealReal /= numPairs
        MSERealRealQ /= numPairs
        MSEImagImag /= numPairs
        MSEImagImagQ /= numPairs
        MSEImagReal /= numPairs
        MSEImagRealQ /= numPairs
        MSERealImag /= numPairs
        MSERealImagQ /= numPairs
        println("n = $numPairs")
        println("MSEAbs = $MSEAbs")
        println("MSEAbsQ = $MSEAbsQ")
        println("MSERealReal = $MSERealReal")
        println("MSERealRealQ = $MSERealRealQ")
        println("MSEImagImag = $MSEImagImag")
        println("MSEImagImagQ = $MSEImagImagQ")
        println("MSEImagReal = $MSEImagReal")
        println("MSEImagRealQ = $MSEImagRealQ")
        println("MSERealImag = $MSERealImag")
        println("MSERealImagQ = $MSERealImagQ")
        File(outFileDir, "MSE.txt").writeText("n: $numPairs\nMSEAsymmetric: $MSEAbs\nMSERealReal: $MSERealReal\nMSEImagImag: $MSEImagImag\nMSEImagReal: $MSEImagReal\nMSERealImag: $MSERealImag")
        File(outFileDir, "MSEQ.txt").writeText("n: $numPairs\nMSEAsymmetric: $MSEAbsQ\nMSERealReal: $MSERealRealQ\nMSEImagImag: $MSEImagImagQ\nMSEImagReal: $MSEImagRealQ\nMSERealImag: $MSERealImagQ")
        vectorsFile.copyTo(File(outFileDir, vectorsFile.name), overwrite = true)
    }

    private fun getImagPart(dbData: Array<Complex64VectorValue>): Array<DoubleArray> {
        val imagDbData = Array(dbData.size) { i ->
            DoubleArray(dbData[i].logicalSize) { j ->
                dbData[i][j].imaginary.value
            }
        }
        return imagDbData
    }

    private fun getRealPart(dbData: Array<Complex64VectorValue>): Array<DoubleArray> {
        val realDbData = Array(dbData.size) { i ->
            DoubleArray(dbData[i].logicalSize) { j ->
                dbData[i][j].real.value
            }
        }
        return realDbData
    }

    private fun permuteData(realDbData: Array<DoubleArray>, permutation: IntArray): Array<DoubleArray> {
        val permutedRealData = Array(realDbData.size) { n ->
            DoubleArray(realDbData[n].size) { i ->
                realDbData[n][permutation[i]]
            }
        }
        return permutedRealData
    }
}