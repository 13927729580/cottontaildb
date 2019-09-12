package ch.unibas.dmi.dbis.cottontail.model.basics

import ch.unibas.dmi.dbis.cottontail.model.recordset.Recordset

/**
 * An objects that holds [Record] values and allows for scanning operation on those [Record] values.
 *
 * @see Record
 *
 * @author Ralph Gasser
 * @version 1.0
 */
interface Scanable {
    /**
     * Applies the provided action to each [Record] in this [Recordset].
     *
     * @param action The action that should be applied.
     */
    fun forEach(action: (Record) -> Unit)

    /**
     * Applies the provided action to each [Record] in this [Recordset].
     *
     * @param action The action that should be applied.
     * @param from
     * @param to
     */
    fun forEach(from: Long, to: Long, action: (Record) -> Unit)

    /**
     * Applies the provided mapping function to each [Record] in this [Filterable].
     *
     * @param action The mapping function that should be applied.
     * @return Collection of the results of the mapping function.
     */
    fun <R> map(action: (Record) -> R): Collection<R>
}