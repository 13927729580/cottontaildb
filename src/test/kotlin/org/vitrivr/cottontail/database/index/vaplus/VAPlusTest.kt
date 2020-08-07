package org.vitrivr.cottontail.database.index.vaplus

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.vitrivr.cottontail.model.values.DoubleVectorValue

import java.util.*
import kotlin.math.max
import kotlin.math.pow

internal class VAPlusTest {
    val vap = VAPlus()
    val random = Random(1234)
    val data = Array(10) {
        DoubleArray(20) { random.nextGaussian() }
    }
    val marks = MarksGenerator.getEquidistantMarks(data, IntArray(data.first().size) { 100 })

    @Test
    fun computeBounds() {
        val vector = data.first()
        val bounds = vap.computeBounds(vector, marks)
        println("vector")
        println(vector.joinToString())
        println("bounds first (lbounds)")
        bounds.first.forEach { println(it.joinToString()) }
        println("bounds second (ubounds)")
        bounds.second.forEach { println(it.joinToString()) }
    }

    @Test
    fun getCells() {
        val cells = vap.getCells(data.first(), marks)
        println("marks")
        for (m in marks) {
            println(m.joinToString())
        }
        println("cells")
        println(cells.joinToString())
    }

    @Test
    fun boundL2() {
        // this is a test implementing what's on p8 top of blott&weber 1997
        for (vector in data) {
            /*
            todo: there will be an indexOutOfBounds error if the vector component is not contained in the marks!
                  this can happen if the marks are not generated based on the entire data in the DB, but just sampled
                  from there! We could artificially add the Double.Max_VALUE and MIN_VALUE to the marks to catch this
                  see MarksGenerator
            */
            val query = DoubleArray(vector.size) { random.nextGaussian() }
            val cellsVec = vap.getCells(vector, marks)
            val cellsQuery = vap.getCells(query, marks)
            val lbl2sq = vector.mapIndexed { i, v ->
                when {
                    cellsVec[i] < cellsQuery[i] -> {
                        query[i] - marks[i][cellsVec[i] + 1]
                    }
                    cellsVec[i] == cellsQuery[i] -> {
                        0.0
                    }
                    else -> {
                        marks[i][cellsVec[i]] - query[i]
                    }
                }
            }.map { it.pow(2) }.sum()
            val ubl2sq = cellsVec.mapIndexed { i, v ->
                val a = { query[i] - marks[i][v] }
                val b = { marks[i][v + 1] - query[i] }
                val cq = cellsQuery[i]
                when {
                    v < cq -> {
                        a()
                    }
                    v == cq -> {
                        max(a(), b())
                    }
                    else -> {
                        b()
                    }
                }
            }.map { it.pow(2) }.sum()
            val l2sq = DoubleVectorValue(vector).l2(DoubleVectorValue(query)).pow(2).value
            println("actual squared l2 norm $l2sq")
            println("ub of squared l2 norm $ubl2sq")
            println("lb of squared l2 norm $lbl2sq")
            assertTrue(l2sq <= ubl2sq, "actual l2 larger than upper bound!!")
            assertTrue(l2sq >= lbl2sq, "actual l2 smaller than lower bound!!")
        }
    }
}