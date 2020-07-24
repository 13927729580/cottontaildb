package org.vitrivr.cottontail.database.index.lsh.superbit

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import org.vitrivr.cottontail.model.values.Complex64Value
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleVectorValue
import java.io.File
import java.util.*

fun sampleVectorsFromCsv(infile: String, outfile: String, rng: Random, probability: Double = 1e-5) {
    println("Sampling vectors from $infile with probability of $probability. Writing to $outfile")
    csvReader().open(infile) {
        csvWriter().open(outfile) {
            writeRow(readNext()!!)
            writeRows(readAllAsSequence().filter { rng.nextDouble() < probability }.toList())
        }
    }
}

fun getComplexVectorsFromFile(file: String): Array<Complex64VectorValue> {
    val f = File(file)
    if (!f.exists()) {
        sampleVectorsFromCsv("src/test/resources/complexVectors.csv", file, Random(1234), 2e-5)
    }
    val vecs = csvReader().open(file) {
        readAllAsSequence().drop(1).toList()
    }
    println("${vecs.size} vectors read.")
    val vectors = vecs.map { row ->
        val v = Complex64VectorValue((0 until SuperBitTest.numDim).map { Complex64Value(row[it + 1].toDouble(), row[it + 2].toDouble()) }.toTypedArray())
        v / v.norm2()
    }.toTypedArray()
    return vectors
}

fun getRandomRealVectors(rng: Random, numVecs: Int, numDim: Int): Array<DoubleVectorValue> {
    val vectors = Array(numVecs) {
        val v = DoubleVectorValue(DoubleArray(numDim) { rng.nextGaussian() })
        v / v.norm2()
    }
    return vectors
}

fun getRandomComplexVectors(rng: Random, numVecs: Int, numDim: Int): Array<Complex64VectorValue> {
    val vectors = Array(numVecs) {
        val v = Complex64VectorValue(DoubleArray(numDim * 2) { rng.nextGaussian() })
        v / v.norm2()
    }
    return vectors
}
