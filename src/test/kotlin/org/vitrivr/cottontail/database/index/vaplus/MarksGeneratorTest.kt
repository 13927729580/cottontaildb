package org.vitrivr.cottontail.database.index.vaplus

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import org.vitrivr.cottontail.model.values.DoubleVectorValue

internal class MarksGeneratorTest {

    val random = java.util.Random(1234)
    val realdata = Array(30) {
        DoubleArray(20) { random.nextGaussian() }
    }
    val marksPerDim = 10

    @Test
    fun getEquidistantMarks() {
        println("data")
        realdata.forEach { println(it.joinToString()) }
        println("marks")
        val marks = MarksGenerator.getEquidistantMarks(realdata, IntArray(realdata.first().size) { marksPerDim })
        for (m in marks) {
            assertTrue(m.toList() == m.toList().sorted(), "Marks not sorted in ascending order!")
            println(m.joinToString())
        }
    }

    @Test
    fun getNonUniformMarks() {
        println("data")
        realdata.forEach { println(it.joinToString()) }
        println("marks (each dim a row)")
        val marks = MarksGenerator.getNonUniformMarks(realdata, IntArray(realdata.first().size) { marksPerDim })
        marks.forEach { m ->
            assertTrue(m.toList() == m.toList().sorted(), "Marks not sorted in ascending order!")
            println(m.joinToString()) }
    }

    @Test
    fun getEquallyPopulatedMarks() {
        println("data")
        realdata.forEach { println(it.joinToString()) }
        println("marks (each dim a row)")
        val marks = MarksGenerator.getEquallyPopulatedMarks(realdata, IntArray(realdata.first().size) { marksPerDim })
        val vap = VAPlus()
        println("cells for vecs")
        val cells = realdata.map {
            val c = vap.getCells(it, marks)
            println(c.joinToString())
            c
        }
        realdata.first().indices.map {dim ->
            println("dim $dim counts per cell in cell order")
            val map = (0 until marksPerDim - 1).map { cell ->
                cells.count { it[dim] == cell }
            }
            println(map.joinToString())
            assertTrue(map.max()!! - map.min()!! <= 1, "Dim $dim Difference between most and least per cell larger than 1! Not optimal!")
        }
        marks.forEachIndexed { i, m ->
            assertTrue(m.toList() == m.toList().sorted(), "Marks not sorted in ascending order!")
            println(m.joinToString())

        }
    }
}