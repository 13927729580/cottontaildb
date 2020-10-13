package org.vitrivr.cottontail.storage.engine.hare.access.cursor

import org.vitrivr.cottontail.model.basics.TupleId
import org.vitrivr.cottontail.model.values.types.Value
import org.vitrivr.cottontail.storage.engine.hare.access.EntryDeletedException
import org.vitrivr.cottontail.storage.engine.hare.access.NullValueNotAllowedException

/**
 * A [WritableCursor] is a writeable proxy that allows for navigation in and editing of HARE data
 * structures such as columns. Access to entries is facilitated by [TupleId]s, that uniquely
 * identify each entry in the underlying data structure.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
interface WritableCursor<T : Value> : ReadableCursor<T> {
    /**
     * Updates the [Value] for the [TupleId] this [WritableCursor] is currently pointing to.
     *
     * @param value The value [T] the updated entry should contain after the update. Can be null, if the underlying data structure permits it.
     *
     * @throws EntryDeletedException If entry identified by [TupleId] has been deleted.
     * @throws NullValueNotAllowedException If [value] is null but the underlying data structure does not support null values.
     */
    fun update(value: T?)

    /**
     * Updates the [Value] for the [TupleId] this [WritableCursor] is currently pointing to if, and only if, it is equal to the expected value.
     *
     * @param expectedValue The value [T] the entry is expected to contain before the update. May be null.
     * @param newValue The  value [T] the updated entry should contain after the update. Can be null, if the underlying data structure permits it.
     * @return True if entry was updated, false otherwise.
     *
     * @throws EntryDeletedException If entry identified by [TupleId] has been deleted.
     * @throws NullValueNotAllowedException If [newValue] is null but the underlying data structure does not support null values.
     */
    fun compareAndUpdate(expectedValue: T?, newValue: T?): Boolean

    /**
     * Deletes the [Value] for the [TupleId] this [WritableCursor] is currently pointing to.
     *
     * @return The value of the entry before deletion.
     *
     * @throws EntryDeletedException If entry identified by [TupleId] has been deleted.
     */
    fun delete(): T?

    /**
     * Appends the provided [Value] to the underlying data structure, assigning it a new [TupleId].
     *
     * @param value The value to append. Can be null, if the underlying data structure permits it.
     * @return The [TupleId] of the new value.
     *
     * @throws NullValueNotAllowedException If [value] is null but the underlying data structure does not support null values.
     */
    fun append(value: T?): TupleId
}