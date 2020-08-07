package org.vitrivr.cottontail.database.index.vaplus

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import org.vitrivr.cottontail.model.values.DoubleVectorValue

internal class MarksGeneratorTest {

    val random = java.util.Random(1234)
    @Test
    fun getEquidistantMarks() {
        println("data")
        val realdata = Array(5) {
            val vec = DoubleArray(20) { random.nextGaussian() }
            println(vec.joinToString())
            vec
        }
        println("marks")
        val marks = MarksGenerator.getEquidistantMarks(realdata, IntArray(realdata.first().size) { 1 })
        for (m in marks) {
            println(m.joinToString())
        }
    }
}