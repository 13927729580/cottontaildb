package org.vitrivr.cottontail.database.index.va.marks

inline class Marks(val marks: Array<DoubleArray>) {

    /**
     * This methods calculates the signature of the vector.
     * This method checks for every mark if the corresponding vector is beyond or not.
     * If so, mark before is the corresponding mark.
     *
     * Note that this can return -1, which means that the component is smaller than the smallest mark!
     * This can arise e.g. if marks are not generated from entire dataset, but just a sampled subset thereof!
     *
     * @param vector The vector.
     * @return An [IntArray] containing the signature of the vector.
     */
    fun getCells(vector: DoubleArray): IntArray = IntArray(vector.size) {
        val index = marks[it].indexOfFirst { i -> i > vector[it] }
        if (index == -1) { // all marks are less or equal than the vector component! last mark is what we're looking for!
            marks[it].size - 1
        } else {
            index - 1
        }
    }
}