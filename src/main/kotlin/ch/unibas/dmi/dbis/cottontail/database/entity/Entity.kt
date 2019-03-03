package ch.unibas.dmi.dbis.cottontail.database.entity

import ch.unibas.dmi.dbis.cottontail.database.column.Column
import ch.unibas.dmi.dbis.cottontail.database.general.DBO
import ch.unibas.dmi.dbis.cottontail.database.general.Transaction
import ch.unibas.dmi.dbis.cottontail.database.general.TransactionStatus
import ch.unibas.dmi.dbis.cottontail.database.column.ColumnTransaction
import ch.unibas.dmi.dbis.cottontail.database.column.mapdb.MapDBColumn
import ch.unibas.dmi.dbis.cottontail.database.index.Index
import ch.unibas.dmi.dbis.cottontail.database.index.IndexTransaction
import ch.unibas.dmi.dbis.cottontail.database.index.IndexType
import ch.unibas.dmi.dbis.cottontail.database.queries.AtomicBooleanPredicate
import ch.unibas.dmi.dbis.cottontail.database.queries.BooleanPredicate
import ch.unibas.dmi.dbis.cottontail.database.queries.Predicate
import ch.unibas.dmi.dbis.cottontail.database.schema.Schema
import ch.unibas.dmi.dbis.cottontail.model.basics.*
import ch.unibas.dmi.dbis.cottontail.model.recordset.Recordset
import ch.unibas.dmi.dbis.cottontail.model.recordset.StandaloneRecord

import ch.unibas.dmi.dbis.cottontail.model.exceptions.DatabaseException
import ch.unibas.dmi.dbis.cottontail.model.exceptions.QueryException
import ch.unibas.dmi.dbis.cottontail.model.exceptions.TransactionException
import ch.unibas.dmi.dbis.cottontail.model.values.Value

import org.mapdb.*
import org.mapdb.volume.MappedFileVol
import java.nio.file.Files
import java.nio.file.Path
import java.util.*

import java.util.concurrent.locks.ReentrantReadWriteLock
import java.util.logging.Filter
import java.util.stream.Collectors
import kotlin.concurrent.read
import kotlin.concurrent.write

/**
 * Represents a single entity in the Cottontail DB data model. An [Entity] has name that must remain unique within a [Schema].
 * The [Entity] contains one to many [Column]s holding the actual data. Hence, it can be seen as a table containing tuples.
 *
 * Calling the default constructor for [Entity] opens that [Entity]. It can only be opened once due to file locks and it
 * will remain open until the [Entity.close()] method is called.
 *
 * @see Schema
 * @see Column
 * @see Entity.Tx
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class Entity(override val name: String, schema: Schema) : DBO {
    /** The [Path] to the [Entity]'s main folder. */
    override val path: Path = schema.path.resolve("entity_$name")

    /** The parent [DBO], which is the [Schema] in case of an [Entity]. */
    override val parent: Schema? = schema

    /** Internal reference to the [StoreWAL] underpinning this [Entity]. */
    private val store: StoreWAL = try {
        StoreWAL.make(file = this.path.resolve(FILE_CATALOGUE).toString(), volumeFactory = MappedFileVol.FACTORY, fileLockWait = this.parent!!.parent.config.lockTimeout)
    } catch (e: DBException) {
        throw DatabaseException("Failed to open entity '$fqn': ${e.message}'.")
    }

    /** The header of this [Entity]. */
    private val header: EntityHeader
        get() = this.store.get(HEADER_RECORD_ID, EntityHeaderSerializer)
                ?: throw DatabaseException.DataCorruptionException("Failed to open header of entity '$fqn'!")

    /** A internal lock that is used to synchronize [Entity.Tx] affecting this [Entity]. */
    private val txLock = ReentrantReadWriteLock()

    /** A internal lock that is used to synchronize closing of an [Entity] with running [Entity.Tx]. */
    private val globalLock = ReentrantReadWriteLock()

    /** List of all the [Column]s associated with this [Entity]. */
    private val columns: Collection<Column<*>> = this.header.columns.map {
        MapDBColumn<Any>(this.store.get(it, Serializer.STRING)
                ?: throw DatabaseException.DataCorruptionException("Failed to open entity '$fqn': Could not read column definition at position $it!"), this)
    }

    /** List of all the [Index]es associated with this [Entity]. */
    private val indexes: Collection<Index> = this.header.indexes.map {
        val index = this.store.get(it, IndexEntrySerializer)
                ?: throw DatabaseException.DataCorruptionException("Failed to open entity '$fqn': Could not read index definition at position $it!")
        index.type.open(index.name, this)
    }

    /**
     * Status indicating whether this [Entity] is open or closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Creates and returns an [EntityStatistics] snapshot.
     *
     * @return [EntityStatistics] for this [Entity].
     */
    val statistics: EntityStatistics = this.header.let { EntityStatistics(it.columns.size, it.size) }

    /**
     * Returns all [ColumnDef] for the [Column]s contained in this [Entity].
     *
     * @return Collection of [ColumnDef].
     */
    fun allColumns(): Collection<ColumnDef<*>> = this.columns.map { it.columnDef }

    /**
     * Returns the [ColumnDef] for the specified name.
     *
     * @param name The name of the [Column].
     * @return [ColumnDef] of the [Column].
     */
    fun columnForName(name: String): ColumnDef<*>? = this.columns.find { it.name == name }?.columnDef


    /**
     * Creates the [Index] with the given settings
     *
     * @param name Name of the [Index] to create.
     * @param type Type of the [Index] to create.
     * @param columns The list of [columns] to [Index].
     */
    fun createIndex(name: String, type: IndexType, columns: Array<ColumnDef<*>>) = this.globalLock.write {
        val indexEntry = this.header.indexes.map {
            Pair(it, this.store.get(it, IndexEntrySerializer) ?: throw DatabaseException.DataCorruptionException("Failed to create index '$fqn.$name': Could not read index definition at position $it!"))
        }.find { it.second.name == name }

        if (indexEntry != null) throw DatabaseException.IndexAlreadyExistsException("$fqn.$name")


        /* Creates and opens the index. */
        val index = type.open(name, this)
        index.Tx(true).update(columns)

        /* Update catalogue + header. */
        try {
            /* Update catalogue. */
            val sid = this.store.put(IndexEntry(name, type, false), IndexEntrySerializer)

            /* Update header. */
            val new = this.header.let { EntityHeader(it.size, it.created, System.currentTimeMillis(), it.columns, it.indexes.copyOf(it.indexes.size + 1)) }
            new.indexes[new.indexes.size-1] = sid
            this.store.update(Entity.HEADER_RECORD_ID, new, EntityHeaderSerializer)
            this.store.commit()
        } catch (e: DBException) {
            this.store.rollback()
            val pathsToDelete = Files.walk(index.path).sorted(Comparator.reverseOrder()).collect(Collectors.toList())
            pathsToDelete.forEach { Files.delete(it) }
            throw DatabaseException("Failed to create index '$.fqn.$name' due to a storage exception: ${e.message}")
        }
    }

    /**
     * Drops the [Index] with the given name.
     *
     * @param name Name of the [Index] to drop.
     */
    fun dropIndex(name: String) = this.globalLock.write {
        val indexEntry = this.header.indexes.map {
            Pair(it, this.store.get(it, IndexEntrySerializer) ?: throw DatabaseException.DataCorruptionException("Failed to drop index '$fqn.$name': Could not read index definition at position $it!"))
        }.find { it.second.name == name }?.let { ie ->
            Triple(ie.first, ie.second, this.indexes.find { it.name == ie.second.name })
        } ?: throw DatabaseException.IndexDoesNotExistException("$fqn.$name")

        /* Close index. */
        indexEntry.third?.close()

        /* Update header. */
        try {
            val new = this.header.let { EntityHeader(it.size, it.created, System.currentTimeMillis(), it.columns, it.indexes.filter { it != indexEntry.first }.toLongArray()) }
            this.store.update(Entity.HEADER_RECORD_ID, new, EntityHeaderSerializer)
            this.store.commit()
        } catch (e: DBException) {
            this.store.rollback()
            throw DatabaseException("Failed to drop index '$fqn.$name' due to a storage exception: ${e.message}")
        }

        /* Delete files that belong to the index. */
        if (indexEntry.third != null) {
            val pathsToDelete = Files.walk(indexEntry.third!!.path).sorted(Comparator.reverseOrder()).collect(Collectors.toList())
            pathsToDelete.forEach { Files.delete(it) }
        }
    }

    /**
     * Updates the [Index] with the given name.
     *
     * @param name The name of the [Index]
     */
    fun updateIndex(name: String) {
        val index = this.indexes.find { it.name == name }
        index?.Tx(false)?.update()
    }

    /**
     * Updates all [Index]es for this [Entity]
     */
    fun updateAllIndexes() {
        this.indexes.forEach {
            it.Tx(false).update()
        }
    }

    /**
     * Closes the [Entity]. Closing an [Entity] is a delicate matter since ongoing [Entity.Tx] objects as well as all involved [Column]s are involved.
     * Therefore, access to the method is mediated by an global [Entity] wide lock.
     */
    override fun close() = this.globalLock.write {
        this.columns.forEach { it.close() }
        this.store.close()
        this.closed = true
    }

    /**
     * Handles finalization, in case the Garbage Collector reaps a cached [Entity] soft-reference.
     */
    @Synchronized
    protected fun finalize() {
        if (!this.closed) {
            this.close()
        }
    }

    /**
     * Companion object of the [Entity]
     */
    companion object {
        /** Filename for the [Entity] catalogue.  */
        internal const val FILE_CATALOGUE = "index.db"

        /** Filename for the [Entity] catalogue.  */
        internal const val HEADER_RECORD_ID = 1L
    }

    /**
     * A [Tx] that affects this [Entity].
     *
     * Opening such a [Tx] will spawn a associated [Column.Tx] for every [Column] associated with this [Entity].
     */
    inner class Tx(override val readonly: Boolean, override val tid: UUID = UUID.randomUUID(), columns: Array<ColumnDef<*>>? = null, ommitIndex: Boolean = false) : EntityTransaction {
        /** List of [ColumnTransaction]s associated with this [Entity.Tx]. */
        private val columns: Map<ColumnDef<*>, ColumnTransaction<*>> = if (columns != null && this.readonly) {
            this@Entity.columns.filter{columns.contains(it.columnDef)}.associateBy({ColumnDef(it.name, it.type, it.size)}, {it.newTransaction(readonly, tid)})
        } else {
            this@Entity.columns.associateBy({ColumnDef(it.name, it.type, it.size)}, {it.newTransaction(readonly, tid)})
        }

        /** List of [IndexTransaction] associated with this [Entity.Tx]. */
        private val indexes: Collection<IndexTransaction> = if (!ommitIndex) {
            this@Entity.indexes.map { it.Tx(true, tid) }
        } else {
            emptyList()
        }

        /** Flag indicating whether or not this [Entity.Tx] was closed */
        @Volatile
        override var status: TransactionStatus = TransactionStatus.CLEAN
            private set

        /** Tries to acquire a global read-lock on this [Entity]. */
        init {
            if (this@Entity.closed) {
                throw TransactionException.TransactionDBOClosedException(tid)
            }
            this@Entity.globalLock.readLock().lock()
        }

        /**
         * Commits all changes made through this [Entity.Tx] since the last commit or rollback.
         */
        @Synchronized
        override fun commit() {
            if (this.status == TransactionStatus.DIRTY) {
                this.columns.values.forEach { it.commit() }
                this@Entity.store.commit()
                this@Entity.txLock.writeLock().unlock()
                this.status = TransactionStatus.CLEAN
            }
        }

        /**
         * Rolls all changes made through this [Entity.Tx] back to the last commit.
         */
        @Synchronized
        override fun rollback() {
            if (this.status == TransactionStatus.DIRTY) {
                this.columns.values.forEach { it.rollback() }
                this@Entity.store.rollback()
                this@Entity.txLock.writeLock().unlock()
                this.status = TransactionStatus.CLEAN
            }
        }

        /**
         * Closes this [Entity.Tx] and thereby releases all the [Column.Tx] and the global lock. Closed [Entity.Tx] cannot be used anymore!
         */
        @Synchronized
        override fun close() {
            if (this.status != TransactionStatus.CLOSED) {
                if (this.status == TransactionStatus.DIRTY) {
                    this@Entity.store.rollback()
                    this@Entity.txLock.writeLock().unlock()
                }
                this.status = TransactionStatus.CLOSED
                this@Entity.globalLock.readLock().unlock()
            }
        }

        /**
         * Reads the values of one or many [Column]s and returns it as a [Tuple]
         *
         * @param tupleId The ID of the desired entry.
         * @return The desired [Tuple].
         *
         * @throws DatabaseException If tuple with the desired ID doesn't exist OR is invalid.
         */
        fun read(tupleId: Long): Record = this@Entity.txLock.read {
            checkValidOrThrow()
            checkValidTupleId(tupleId)


            /* Return value of all the desired columns. */
            val columns = this.columns.keys.toTypedArray()
            StandaloneRecord(tupleId, columns).assign(columns.map { this.columns.getValue(it).read(tupleId) }.toTypedArray() )
        }

        /**
         * Reads the specified values of one or many [Column]s and returns them as a [Recordset]
         *
         * @param tupleId The ID of the desired entry.
         * @return The resulting [Recordset].
         *
         * @throws DatabaseException If tuple with the desired ID doesn't exist OR is invalid.
         */
        fun readMany(tupleIds: Collection<Long>): Recordset = this@Entity.txLock.read {
            checkValidOrThrow()
            val columns = this.columns.keys.toTypedArray()
            val dataset = Recordset(columns)
            tupleIds.forEach { tid ->
                checkValidTupleId(tid)
                dataset.addRow(tid, columns.map { this.columns.getValue(it).read(tid) }.toTypedArray())
            }
            dataset
        }

        /**
         * Reads all values of one or many [Column]s and returns them as a [Recordset].
         *
         * @return The resulting [Recordset].
         */
        fun readAll(): Recordset = this@Entity.txLock.read {
            checkValidOrThrow()

            val columns = this.columns.keys.toTypedArray()
            val dataset = Recordset(columns)
            val data = Array<Value<*>?>(columns.size) { null }

            this.columns.getValue(columns[0]).forEach {
                data[0] = it.values[0]
                for (i in 1 until columns.size) {
                    data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                }
                dataset.addRow(it.tupleId, data)
            }
            return dataset
        }

        /**
         * Returns the number of entries in this [Entity].
         *
         * @return The number of entries in this [Entity].
         */
        override fun count(): Long = this@Entity.txLock.read {
            checkValidOrThrow()
            return this@Entity.header.size
        }

        /**
         * Applies the provided function to each entry found in this [Column]. The provided function cannot not change
         * the data stored in the [Column]!
         *
         * @param action The function to apply to each [Column] entry.
         */
        override fun forEach(action: (Record) -> Unit) = this@Entity.txLock.read {
            checkValidOrThrow()
            val columns = this.columns.keys.toTypedArray()
            val data = Array<Value<*>?>(columns.size) { null }

            this.columns.getValue(columns[0]).forEach {
                data[0] = it.values[0]
                for (i in 1 until columns.size) {
                    data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                }
                action(StandaloneRecord(it.tupleId, columns).assign(data))
            }
        }

        /**
         * Applies the provided mapping function on each [Tuple] found in this [Entity], returning a collection of the desired output values.
         *
         * @param action The mapping that should be applied to each [Tuple].
         *
         * @return A collection of Pairs mapping the tupleId to the generated value.
         */
        override fun <R> map(action: (Record) -> R) = this@Entity.txLock.read {
            checkValidOrThrow()

            val columns = this.columns.keys.toTypedArray()
            val data = Array<Value<*>?>(columns.size) { null }
            val list = mutableListOf<R>()

            this.columns.getValue(columns[0]).forEach {
                data[0] = it.values[0]
                for (i in 1 until columns.size) {
                    data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                }
                list.add(action(StandaloneRecord(it.tupleId, columns).assign(data)))
            }
            list
        }

        /**
         * Applies the provided action to each [Record] that matches the given [Predicate].
         *
         * @param predicate [Predicate] to filter [Record]s.
         * @param action The action that should be applied.
         */
        override fun forEach(predicate: Predicate, action: (Record) -> Unit)= this@Entity.txLock.read {
            if (predicate is BooleanPredicate) {
                checkValidOrThrow()
                checkColumnsExist(*predicate.columns.toTypedArray())

                /* Extract necessary data structures. */
                val columns = this.columns.keys.toTypedArray()
                val data = Array<Value<*>?>(columns.size) { null }
                val filterable = this.indexes.find { it.canProcess(predicate) }

                /* Handle forEach() for different cases. */
                when {
                    /* Case 1: Predicate satisfiable by index. */
                    filterable != null -> filterable.forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        action(StandaloneRecord(it.tupleId, columns).assign(data))
                    }
                    /* Case 2: Predicate affects single column (AtomicBooleanPredicate). */
                    predicate is AtomicBooleanPredicate<*> -> this.columns.getValue(predicate.columns.first()).forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        action(StandaloneRecord(it.tupleId, columns).assign(data))
                    }
                    /* Case 3 (general): Multi-column predicate. */
                    else -> this.columns.getValue(columns[0]).forEach {
                        data[0] = it.values[0]
                        for (i in 1 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        val record = StandaloneRecord(it.tupleId, columns).assign(data)
                        if (predicate.matches(record)) {
                            action(record)
                        }
                    }
                }
            }  else {
                throw QueryException.UnsupportedPredicateException("Only boolean predicates are supported for invocation of forEach() on an Entity.")
            }
        }

        /**
         * Applies the provided mapping function to each [Record] that matches the given [Predicate].
         *
         * @param predicate [Predicate] to filter [Record]s.
         * @param action The mapping function that should be applied.
         * @return Collection of the results of the mapping function.
         */
        override fun <R> map(predicate: Predicate, action: (Record) -> R): Collection<R> = this@Entity.txLock.read {
            if (predicate is BooleanPredicate) {
                checkValidOrThrow()
                checkColumnsExist(*predicate.columns.toTypedArray())

                val columns = this.columns.keys.toTypedArray()
                val data = Array<Value<*>?>(columns.size) { null }
                val list = mutableListOf<R>()
                val filterable = this.indexes.find { it.canProcess(predicate) }

                /* Handle map() for different cases. */
                when {
                    /* Case 1: Predicate satisfiable by index. */
                    filterable != null -> filterable.forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        list.add(action(StandaloneRecord(it.tupleId, columns).assign(data)))
                    }
                    /* Case 2: Predicate affects single column (AtomicBooleanPredicate). */
                    predicate is AtomicBooleanPredicate<*> -> this.columns.getValue(predicate.columns.first()).forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        list.add(action(StandaloneRecord(it.tupleId, columns).assign(data)))
                    }
                    /* Case 3 (general): Multi-column predicate. */
                    else -> this.columns.getValue(columns[0]).forEach {
                        data[0] = it.values[0]
                        for (i in 1 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        val record = StandaloneRecord(it.tupleId, columns).assign(data)
                        if (predicate.matches(record)) {
                            list.add(action(record))
                        }
                    }
                }
                list
            }  else {
                throw QueryException.UnsupportedPredicateException("Only boolean predicates are supported for invocation of map() on an Entity.")
            }
        }

        /**
         * Reads all values of one or many [Column]s and returns those that match the provided predicate as a [Recordset].
         *
         * @param predicate The [BooleanPredicate] to apply. Only columns contained in that [BooleanPredicate] will be read.
         * @return The resulting [Recordset].
         */
        override fun filter(predicate: Predicate): Recordset = this@Entity.txLock.read {
            if (predicate is BooleanPredicate) {
                checkValidOrThrow()
                checkColumnsExist(*predicate.columns.toTypedArray())

                val columns = this.columns.keys.toTypedArray()
                val dataset = Recordset(columns)
                val data = Array<Value<*>?>(columns.size) { null }
                val filterable = this.indexes.find { it.canProcess(predicate) }

                /* Handle filter() for different cases. */
                when {
                    /* Case 1: Predicate satisfiable by index. */
                    filterable != null -> filterable.forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        dataset.addRow(it.tupleId, data)
                    }
                    /* Case 2: Predicate affects single column (AtomicBooleanPredicate). */
                    predicate is AtomicBooleanPredicate<*> -> this.columns.getValue(predicate.columns.first()).forEach(predicate) {
                        for (i in 0 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        dataset.addRow(it.tupleId, data)
                    }
                    /* Case 3 (general): Multi-column predicate. */
                    else -> this.columns.getValue(columns[0]).forEach {
                        data[0] = it.values[0]
                        for (i in 1 until columns.size) {
                            data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                        }
                        dataset.addRowIf(it.tupleId , predicate, data)
                    }
                }
                dataset
            } else {
                throw QueryException.UnsupportedPredicateException("Only boolean predicates are supported for invocation of filter() on an Entity.")
            }
        }

        /**
         * Applies the provided function to each entry found in this [Column] and does so in a parallel fashion using co-routines, which
         * each operate on a disjoint partition of the data. The provided function cannot not change the data stored in the [Column]!
         *
         * It is up to the developer of the function to make sure, that the provided function operates in a thread-safe manner.
         *
         * @param action The function to apply to each [Column] entry.
         * @param parallelism The desired amount of parallelism (i.e. the number of co-routines to spawn).
         */
        override fun forEach(parallelism: Short, action: (Record) -> Unit) = this@Entity.txLock.read {
            checkValidOrThrow()
            val columns = this.columns.keys.toTypedArray()
            this.columns.getValue(columns[0]).forEach(parallelism) {
                val data = Array<Value<*>?>(columns.size) { null }
                data[0] = it.values[0]
                for (i in 1 until columns.size) {
                    data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                }
                action(StandaloneRecord(it.tupleId, columns).assign(data))
            }
        }

        /**
         * Applies the provided mapping function to each [Record] in this [Filterable].
         *
         * @param parallelism The amount of parallelism. Defaults to 2.
         * @param action The mapping function that should be applied.
         * @return Collection of the results of the mapping function.
         */
        override fun <R> map(parallelism: Short, action: (Record) -> R): Collection<R> = this@Entity.txLock.read {
            checkValidOrThrow()
            val columns = this.columns.keys.toTypedArray()
            val results = Collections.synchronizedList(mutableListOf<R>())
            this.columns.getValue(columns[0]).forEach(parallelism) {
                val data = Array<Value<*>?>(columns.size) { null }
                data[0] = it.values[0]
                for (i in 1 until columns.size) {
                    data[i] = this.columns.getValue(columns[i]).read(it.tupleId)
                }
                results.add(action(StandaloneRecord(it.tupleId, columns).assign(data)))
            }
            results
        }

        /**
         * Attempts to insert the provided [Tuple] into the [Entity]. Columns specified in the [Tuple] that are not part
         * of the [Entity] will cause an error!
         *
         * @param record The [Record] that should be inserted.
         * @return The ID of the record or null, if nothing was inserted.
         * @throws TransactionException If some of the sub-transactions on [Column] level caused an error.
         * @throws DatabaseException If a general database error occurs during the insert.
         */
        fun insert(record: Record): Long? {
            checkColumnsExist(*record.columns) /* Perform sanity check on columns before locking. */
            acquireWriteLock()
            try {
                var lastRecId: Long? = null
                for ((column, tx) in this.columns) {
                    val recId = (tx as ColumnTransaction<Any>).insert(record[column] as Value<Any>?)
                    if (lastRecId != recId && lastRecId != null) {
                        throw DatabaseException.DataCorruptionException("Entity ${this@Entity.fqn} is corrupt. Insert did not yield same record ID for all columns involved!")
                    }
                    lastRecId = recId
                }

                /* Update the header of this entity. */
                if (lastRecId != null) {
                    val header = this@Entity.header
                    header.size += 1
                    header.modified = System.currentTimeMillis()
                    this@Entity.store.update(HEADER_RECORD_ID, header, EntityHeaderSerializer)
                }

                return lastRecId
            } catch (e: DatabaseException) {
                this.status = TransactionStatus.ERROR
                throw e
            } catch (e: DBException) {
                this.status = TransactionStatus.ERROR
                throw DatabaseException("Inserting record failed due to an error in the underlying storage: ${e.message}.")
            }
        }

        /**
         * Attempts to insert the provided [Tuple]s into the [Entity]. Columns specified in the [Tuple] that are not part
         * of the [Entity] will cause an error!
         *
         * @param tuples The [Tuple] that should be inserted.
         * @return The ID of the record or null, if nothing was inserted.
         * @throws TransactionException If some of the sub-transactions on [Column] level caused an error.
         * @throws DatabaseException If a general database error occurs during the insert.
         */
        fun insertAll(tuples: Collection<Record>): Collection<Long?> {
            tuples.forEach { checkColumnsExist(*it.columns) }  /* Perform sanity check on columns before locking. */
            acquireWriteLock()
            try {
                /* Perform delete on each column. */
                val tuplesIds = tuples.map {
                    var lastRecId: Long? = null
                    for ((column, tx) in this.columns) {
                        val recId = (tx as ColumnTransaction<Any>).insert(it[column] as Value<Any>?)
                        if (lastRecId != recId && lastRecId != null) {
                            throw DatabaseException.DataCorruptionException("Entity ${this@Entity.fqn} is corrupt. Insert did not yield same record ID for all columns involved!")
                        }
                        lastRecId = recId
                    }
                    lastRecId
                }

                /* Update header. */
                val header = this@Entity.header
                header.size += tuples.size
                header.modified = System.currentTimeMillis()
                store.update(HEADER_RECORD_ID, header, EntityHeaderSerializer)

                return tuplesIds
            } catch (e: DatabaseException) {
                this.status = TransactionStatus.ERROR
                throw e
            } catch (e: DBException) {
                this.status = TransactionStatus.ERROR
                throw DatabaseException("Inserting records failed due to an error in the underlying storage: ${e.message}.")
            }
        }

        /**
         * Attempts to delete the provided [Tuple] from the [Entity]. This tasks will set this [Entity.Tx] to
         * [TransactionStatus.DIRTY] and acquire a [Entity]-wide write lock until the [Entity.Tx] either commit
         * or rollback is issued.
         *
         * @param tupleId The ID of the [Tuple] that should be deleted.
         *
         * @throws TransactionException If some of the sub-transactions on [Column] level caused an error.
         * @throws DatabaseException If a general database error occurs during the insert.
         */
        fun delete(tupleId: Long) {
            acquireWriteLock()
            try {
                /* Perform delete on each column. */
                this.columns.values.forEach { it.delete(tupleId) }

                /* Update header. */
                val header = this@Entity.header
                header.size -= 1
                header.modified = System.currentTimeMillis()
                store.update(HEADER_RECORD_ID, header, EntityHeaderSerializer)
            } catch (e: DBException) {
                this.status = TransactionStatus.ERROR
                throw DatabaseException("Deleting record $tid failed due to an error in the underlying storage: ${e.message}.")
            }
        }

        /**
         * Attempts to delete all the provided [Tuple] from the [Entity]. This tasks will set this [Entity.Tx] to
         * [TransactionStatus.DIRTY] and acquire a [Entity]-wide write lock until the [Entity.Tx] either commit
         * or rollback is issued.
         *
         * @param tupleIds The IDs of the [Tuple]s that should be deleted.
         *
         * @throws TransactionException If some of the sub-transactions on [Column] level caused an error.
         * @throws DatabaseException If a general database error occurs during the insert.
         */
        fun deleteAll(tupleIds: Collection<Long>) {
            acquireWriteLock()
            try {
                /* Perform delete on each column. */
                tupleIds.forEach { tupleId ->
                    this.columns.values.forEach { it.delete(tupleId) }
                }

                /* Update header. */
                val header = this@Entity.header
                header.size -= tupleIds.size
                header.modified = System.currentTimeMillis()
                store.update(HEADER_RECORD_ID, header, EntityHeaderSerializer)
            } catch (e: DBException) {
                this.status = TransactionStatus.ERROR
                throw DatabaseException("Deleting records failed due to an error in the underlying storage: ${e.message}.")
            }
        }

        /**
         * Check if all the provided [Column]s exist on this [Entity] and that they have the type that was expected!
         *
         * @params The list of [Column]s that should be checked.
         */
        private fun checkColumnsExist(vararg columns: ColumnDef<*>) = columns.forEach {
            if (!this.columns.contains(it)) {
                throw TransactionException.ColumnUnknownException(tid, it, this@Entity.name)
            }
        }

        /**
         * Checks if the provided tupleID is valid. Otherwise, an exception will be thrown.
         */
        private fun checkValidTupleId(tupleId: Long) {
            if ((tupleId < 0L) or (tupleId == HEADER_RECORD_ID)) {
                throw TransactionException.InvalidTupleId(tid, tupleId)
            }
        }

        /**
         * Checks if this [Column.Tx] is still open. Otherwise, an exception will be thrown.
         */
        @Synchronized
        private fun checkValidOrThrow() {
            if (this.status == TransactionStatus.CLOSED) throw TransactionException.TransactionClosedException(tid)
            if (this.status == TransactionStatus.ERROR) throw TransactionException.TransactionInErrorException(tid)
        }

        /**
         * Tries to acquire a write-lock. If method fails, an exception will be thrown
         */
        @Synchronized
        private fun acquireWriteLock() {
            if (this.readonly) throw TransactionException.TransactionReadOnlyException(tid)
            if (this.status == TransactionStatus.CLOSED) throw TransactionException.TransactionClosedException(tid)
            if (this.status == TransactionStatus.ERROR) throw TransactionException.TransactionInErrorException(tid)
            if (this.status != TransactionStatus.DIRTY) {
                if (this@Entity.txLock.writeLock().tryLock()) {
                    this.status = TransactionStatus.DIRTY
                } else {
                    throw TransactionException.TransactionWriteLockException(this.tid)
                }
            }
        }
    }
}
