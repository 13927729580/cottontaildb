package org.vitrivr.cottontail.model.exceptions

import org.mapdb.DBException
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.TransactionId
import org.vitrivr.cottontail.model.basics.TupleId
import java.util.*


/**
 * These exceptions are thrown whenever a [Transaction][org.vitrivr.cottontail.database.general.Tx]
 * or an action making up a [Transaction][org.vitrivr.cottontail.database.general.Tx] fails for some reason.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
open class TransactionException(message: String) : DatabaseException(message) {
    /** [Transaction][org.vitrivr.cottontail.database.general.Tx]
     * could not be created because enclosing DBO was closed.
     *
     * @param tid The ID of the [Tx] in which this error occurred.
     */
    class TransactionDBOClosedException(tid: TransactionId) : TransactionException("The enclosing DBO has been closed. Transaction $tid could not be created!")

    /**
     * [Transaction][org.vitrivr.cottontail.database.general.Tx]
     * cannot be used anymore, because it is in error.
     *
     * @param tid The ID of the [Tx] in which this error occurred.
     */
    class TransactionClosedException(tid: TransactionId) : TransactionException("Transaction $tid has been closed and cannot be used anymore.")

    /**
     * [Transaction][org.vitrivr.cottontail.database.general.Tx]
     * cannot be used anymore, because it was closed already.
     *
     * @param tid The ID of the [Tx] in which this error occurred.
     */
    class TransactionInErrorException(tid: TransactionId) : TransactionException("Transaction $tid is in error and cannot be used, until it is rolled back.")

    /**
     * Write could not be executed, because [Transaction][org.vitrivr.cottontail.database.general.Tx]
     * is read-only.
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     */
    class TransactionReadOnlyException(tid: TransactionId) : TransactionException("Transaction $tid is read-only and cannot be used to alter data.")

    /**
     * Write could not be executed, because [Transaction][org.vitrivr.cottontail.database.general.Tx]
     * was unable to acquire the necessary locks (usually on DBOs).
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     */
    class TransactionWriteLockException(tid: TransactionId) : TransactionException("Transaction $tid was unable to obtain the necessary locks.")

    /**
     * Write could not be executed because it failed a validation step. This is usually caused by a user error, providing wrong data.
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     * @param message Description of the validation error.
     */
    class TransactionValidationException(tid: TransactionId, message: String) : TransactionException("Transaction $tid reported validation error: $message")

    /**
     * Read/write could not be executed because it caused an error in the underlying data store. This is usually a critical condition and
     * can be caused by either system failure, external manipulation or serious bugs.
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     * @param message Description of the storage error.
     */
    class TransactionStorageException(tid: TransactionId, message: String) : TransactionException("Transaction $tid reported storage error: $message")

    /**
     * Read/write could not be executed because some of the tuple IDs was invalid
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     * @param tupleId The tupleId that was wrong.
     */
    class InvalidTupleId(tid: TransactionId, tupleId: TupleId) : DBException("Transaction $tid reported an error: The provided tuple ID $tupleId is out of bounds and therefore invalid.")

    /**
     * Read/write could not be executed because some of the colums don't exist.
     *
     * @param tid The ID of the [Transaction][org.vitrivr.cottontail.database.general.Tx] in which this error occurred.
     * @param column The definition of the [Column][org.vitrivr.cottontail.database.column.Column] that is missing.
     */
    class ColumnUnknownException(tid: TransactionId, column: ColumnDef<*>) : TransactionException("Transaction $tid could not be executed, because column '${column.name}' (type=${column.type.name}, size=${column.logicalSize}) does not either not exist or has a different type.")
}