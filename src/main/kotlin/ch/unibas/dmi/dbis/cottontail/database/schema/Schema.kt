package ch.unibas.dmi.dbis.cottontail.database.schema

import ch.unibas.dmi.dbis.cottontail.database.catalogue.Catalogue
import ch.unibas.dmi.dbis.cottontail.database.column.mapdb.MapDBColumn
import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.entity.EntityHeader
import ch.unibas.dmi.dbis.cottontail.database.entity.EntityHeaderSerializer
import ch.unibas.dmi.dbis.cottontail.database.general.DBO
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.exceptions.DatabaseException
import ch.unibas.dmi.dbis.cottontail.utilities.extensions.read
import ch.unibas.dmi.dbis.cottontail.utilities.name.Name
import ch.unibas.dmi.dbis.cottontail.utilities.name.NameType
import org.mapdb.DBException
import org.mapdb.Serializer
import org.mapdb.Store
import org.mapdb.StoreWAL
import java.io.IOException
import java.lang.ref.SoftReference
import java.nio.file.Files
import java.nio.file.Path
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock
import java.util.concurrent.locks.StampedLock
import java.util.stream.Collectors
import kotlin.concurrent.read
import kotlin.concurrent.withLock
import kotlin.concurrent.write

/**
 * Represents an schema in the Cottontail DB data model. A [Schema] is a collection of [Entity] objects that belong together
 * (e.g., because they belong to the same application). Every [Schema] can be seen as a dedicated database and different
 * [Schema]s in Cottontail can reside in different locations.
 *
 * Calling the default constructor for [Schema] opens that [Schema]. It can only be opened once due to file locks and it
 * will remain open until the [Schema.close()] method is called.
 *
 * @see Entity
 * @see MapDBColumn
 *
 * @author Ralph Gasser
 * @version 1.2
 */
class Schema(override val name: Name, override val path: Path, override val parent: Catalogue) : DBO {

    /**
     * Companion object with different constants.
     */
    companion object {
        /** ID of the schema header! */
        const val HEADER_RECORD_ID: Long = 1L

        /** Filename for the [Schema] catalogue.  */
        const val FILE_CATALOGUE = "index.db"
    }

    /** Constant FQN of the [Schema] object. */
    override val fqn: Name = this.parent.fqn.append(this.name)

    /** Internal reference to the [Store] underpinning this [MapDBColumn]. */
    private val store: StoreWAL = try {
        StoreWAL.make(
                file = this.path.resolve(FILE_CATALOGUE).toString(),
                volumeFactory = this.parent.config.memoryConfig.volumeFactory,
                allocateIncrement = 1L shl this.parent.config.memoryConfig.cataloguePageShift,
                fileLockWait = this.parent.config.lockTimeout
        )
    } catch (e: DBException) {
        throw DatabaseException("Failed to open schema $name at '$path': ${e.message}'")
    }

    /** Reference to the [SchemaHeader] of the [Schema]. */
    private val header
        get() = this.store.get(HEADER_RECORD_ID, SchemaHeaderSerializer)
                ?: throw DatabaseException.DataCorruptionException("Failed to open header of schema $fqn!")

    /** A lock used to mediate access the closed state of this [Schema]. */
    private val closeLock = StampedLock()

    /** A lock used to mediate access to changes to the entities contained in this [Schema]. */
    private val entityLock = ReentrantReadWriteLock()

    /** A lock used to mediate access to [loaded] cache. */
    private val cacheLock = ReentrantLock()

    /** A map of loaded [Entity] references. The [SoftReference] allow for coping with changing memory conditions. */
    private val loaded = ConcurrentHashMap<Name, SoftReference<Entity>>()

    /** Returns a list of [Entity] held by this [Schema]. */
    val entities: List<Name>
        get() = this.header.entities.map {
            Name(this.store.get(it, Serializer.STRING)
                    ?: throw DatabaseException.DataCorruptionException("Failed to read schema $fqn ($path): Could not find entity name of ID $it."))
        }

    /** Size of the [Schema] in terms of [Entity] objects it contains. */
    val size
        get() = this.entityLock.read { this.header.entities.size }

    /** Flag indicating whether or not this [Schema] has been closed. */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Creates a new [Entity] in this [Schema].
     *
     * @param name The name of the [Entity] that should be created.
     */
    fun createEntity(name: Name, vararg columns: ColumnDef<*>) = this.closeLock.read {
        /* Check closed status. */
        if (this.closed) {
            throw IllegalStateException("Schema ${this.name} has been closed and cannot be used anymore.")
        }

        /* Check the type of name. */
        if (name.type != NameType.SIMPLE) {
            throw IllegalArgumentException("The provided name '$name' is of type '${name.type} and cannot be used to access an entity through a schema.")
        }

        if (columns.map { it.name }.distinct().size != columns.size) throw DatabaseException.DuplicateColumnException(this.fqn.append(name), columns.map { it.name })
        if (this.entities.contains(name)) throw DatabaseException.EntityAlreadyExistsException(this.fqn.append(name))

        this.entityLock.write {
            /* Create empty folder for entity. */
            val data = path.resolve("entity_$name")

            try {
                if (!Files.exists(data)) {
                    Files.createDirectories(data)
                } else {
                    throw DatabaseException("Failed to create entity '${this.fqn.append(name)}'. Data directory '$data' seems to be occupied.")
                }

                /* Store entry for new entity. */
                val recId = this.store.put(name.name, Serializer.STRING)

                /* Generate the entity. */
                val store = StoreWAL.make(
                        file = data.resolve(Entity.FILE_CATALOGUE).toString(),
                        volumeFactory = this.parent.config.memoryConfig.volumeFactory,
                        allocateIncrement = 1L shl this.parent.config.memoryConfig.cataloguePageShift,
                        fileLockWait = this.parent.config.lockTimeout
                )
                store.preallocate() /* Pre-allocates the header. */

                /* Initialize the entities header. */
                val columnIds = columns.map {
                    MapDBColumn.initialize(it, data, this.parent.config.memoryConfig)
                    store.put(it.name.name, Serializer.STRING)
                }.toLongArray()
                store.update(Entity.HEADER_RECORD_ID, EntityHeader(columns = columnIds), EntityHeaderSerializer)
                store.commit()
                store.close()

                /* Update schema header. */
                val header = this.header
                header.modified = System.currentTimeMillis()
                header.entities = header.entities.copyOf(header.entities.size + 1)
                header.entities[header.entities.size - 1] = recId
                this.store.update(HEADER_RECORD_ID, header, SchemaHeaderSerializer)

                /* Commit changes to local schema. */
                this.store.commit()
            } catch (e: DBException) {
                this.store.rollback()
                val pathsToDelete = Files.walk(data).sorted(Comparator.reverseOrder()).collect(Collectors.toList())
                pathsToDelete.forEach { Files.delete(it) }
                throw DatabaseException("Failed to create entity '${this.fqn.append(name)}' due to error in the underlying data store: {${e.message}")
            } catch (e: IOException) {
                throw DatabaseException("Failed to create entity '${this.fqn.append(name)}' due to an IO exception: {${e.message}")
            }
        }
    }

    /**
     * Drops an [Entity] in this [Schema]. The act of dropping an [Entity] requires a lock on that [Entity].
     *
     * @param name The name of the [Entity] that should be dropped.
     */
    fun dropEntity(name: Name) = this.closeLock.read {
        /* Check closed status. */
        if (this.closed) {
            throw IllegalStateException("Schema ${this.name} has been closed and cannot be used anymore.")
        }

        /* Check the type of name. */
        if (name.type != NameType.SIMPLE) {
            throw IllegalArgumentException("The provided name '$name' is of type '${name.type} and cannot be used to access an entity through a schema.")
        }

        this.entityLock.write {
            val entityRecId = this.header.entities.find { this.store.get(it, Serializer.STRING) == name.name }
                    ?: throw DatabaseException.EntityDoesNotExistException(fqn.append(name))

            val entity = this.entityForName(name)
            entity.allIndexes().map { it.name }.forEach { entity.dropIndex(it) }

            /* Unload the entity and remove it. */
            this.loaded.remove(name)?.get()?.close()

            /* Remove entity. */
            try {
                /* Delete entity name from list. */
                this.store.delete(entityRecId, Serializer.STRING)

                /* Update header. */
                val header = this.header
                header.modified = System.currentTimeMillis()
                header.entities = header.entities.filter { it != entityRecId }.toLongArray()
                this.store.update(HEADER_RECORD_ID, header, SchemaHeaderSerializer)

                /* Commit. */
                this.store.commit()
            } catch (e: DBException) {
                this.store.rollback()
                throw DatabaseException("Entity '${this.fqn.append(name)}' could not be dropped, because of an error in the underlying data store: ${e.message}!")
            }

            /* Delete all files associated with the entity. */
            val pathsToDelete = Files.walk(this.path.resolve("entity_$name")).sorted(Comparator.reverseOrder()).collect(Collectors.toList())
            pathsToDelete.forEach { Files.deleteIfExists(it) }
        }
    }


    /**
     * Returns an instance of [Entity] if such an instance exists. If the [Entity] has been loaded before,
     * that [Entity] is re-used. Otherwise, the [Entity] will be loaded from disk.
     *
     * @param name Name of the [Entity] to access.
     */
    fun entityForName(name: Name): Entity = this.closeLock.read {
        /* Check closed status. */
        if (this.closed) {
            throw IllegalStateException("Schema ${this.name} has been closed and cannot be used anymore.")
        }

        /* Check the type of name. */
        if (name.type != NameType.SIMPLE) {
            throw IllegalArgumentException("The provided name '$name' is of type '${name.type}  and cannot be used to access an entity through a schema.")
        }

        this.entityLock.read {
            if (!this.entities.contains(name)) throw DatabaseException.EntityDoesNotExistException(this.fqn.append(name))
            this.cacheLock.withLock {
                var ret = this.loaded[name]?.get()
                return if (ret != null) {
                    ret
                } else {
                    ret = Entity(name, this)
                    this.loaded[name] = SoftReference(ret)
                    ret
                }
            }
        }
    }

    /**
     * Closes this [Schema] and all the [Entity] objects that are contained within.
     *
     * Since locks to [DBO] or [Transaction][ch.unibas.dmi.dbis.cottontail.database.general.Transaction]
     * objects may be held by other threads, it can take a
     * while for this method to complete.
     */
    override fun close() = this.closeLock.read {
        if (!this.closed) {
            this.cacheLock.withLock {
                this.loaded.entries.removeIf {
                    it.value.get()?.close()
                    true
                }
            }
            this.store.close()
            this.closed = true
        }
    }
}



