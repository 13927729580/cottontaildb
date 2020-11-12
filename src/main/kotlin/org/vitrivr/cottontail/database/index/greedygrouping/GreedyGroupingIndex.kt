package org.vitrivr.cottontail.database.index.greedygrouping

import org.mapdb.DBMaker
import org.mapdb.Serializer
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.pq.PQIndex
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.database.queries.predicates.KnnPredicateHint
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.Complex32VectorValue
import org.vitrivr.cottontail.model.values.Complex64VectorValue
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import java.util.*

class GreedyGroupingIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>,
                          ): Index() {
        companion object {
            const val CONFIG_NAME = "gg_config"
            const val GROUPS_NAME = "gg_means"
            val LOGGER = LoggerFactory.getLogger(org.vitrivr.cottontail.database.index.greedygrouping.GreedyGroupingIndex::class.java)!!

        }

    /** The [Path] to the [DBO]'s main file OR folder. */
    override val path: Path = this.parent.path.resolve("idx_gg_$name.db")

    /** The [PQIndex] implementation returns exactly the columns that is indexed. */
    override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("distance"), ColumnType.forName("DOUBLE")))

    /** The type of [Index]. */
    override val type = IndexType.GG

    private val numGroups = 50 // todo: config
    private val queryConsiderNumGroupsDefault = (numGroups + 9) / 10

    /** The internal [DB] reference. */
    private val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }

    val rng = SplittableRandom(1234L)

    val groupMeans = mutableListOf<Complex32VectorValue>()
    val tIdsOfGroups = mutableListOf<LongArray>()
    private val groupsStore = db.hashMap(GROUPS_NAME, Serializer.FLOAT_ARRAY, Serializer.LONG_ARRAY).counterEnable().createOrOpen()


    init {
        if (columns.size != 1) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing a single column")
        }
        if (!columns.all { it.type == ColumnType.forName("COMPLEX32_VEC") }) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing complex32 vector columns, not ${columns.first()::class.java}")
        }

        loadGroupsFromDisk()
    }

    private fun loadGroupsFromDisk() {
        LOGGER.debug("Index ${name.simple} loading groups from disk.")
        tIdsOfGroups.clear()
        groupMeans.clear()
        groupsStore.forEach { mean, tId ->
            tIdsOfGroups.add(tId)
            groupMeans.add(Complex32VectorValue(mean))
        }
        LOGGER.debug("Loaded ${groupMeans.size} groups from disk.")
    }

    /**
     * Checks if this [Index] can process the provided [Predicate] and returns true if so and false otherwise.
     *
     * @param predicate [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate) =
            predicate is KnnPredicate<*>
                    && predicate.query.all { it is ComplexVectorValue<*> }
                    && predicate.columns.first() == this.columns[0]
                    && predicate.distance is AbsoluteInnerProductDistance


    /**
     * Calculates the cost estimate if this [Index] processing the provided [Predicate].
     *
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate) = Cost.ZERO // todo...

    /**
     * Returns true, if the [Index] supports incremental updates, and false otherwise.
     *
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate() = false // todo...

    /**
     * (Re-)builds the [Index]. Invoking this method should rebuild the [Index] immediately, without the
     * need to commit (i.e. commit actions must take place inside).
     *
     * This is an internal method! External invocation is only possible through a [Index.Tx] object.
     *
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun rebuild(tx: Entity.Tx) {
        /*
        Parameters: numGroups
        1. Take one dictionary element (random is probably easisest to start with)
        2. Go through all yet ungrouped elements and find k = groupSize = numElementsTotal/numGroups most similar ones (AbsIP?)
        3. Build mean vector of those k in the group and store as group representation
        4. Don't do any PCA/SVD as we only have 18-25 ish dims...
        5. Repeat with a new randomly selected element from the remaining ones until no elements remain.
         */


        // get a set of all TIDs that exist.
        val remainingTIds = mutableSetOf<Long>()
        tx.forEach { remainingTIds.add(it.tupleId) }
        val groupSize = (remainingTIds.size + numGroups - 1) / numGroups  // ceildiv
        val finishedTIds = mutableSetOf<Long>()
        LOGGER.info("Index ${name.simple} rebuilding. Grouping ${remainingTIds.size} into $numGroups groups. GroupSize <= $groupSize.")
        groupsStore.clear()
        while (remainingTIds.isNotEmpty()) {
            val groupSeedTid = remainingTIds.elementAt(rng.nextInt(remainingTIds.size))
            LOGGER.debug("Processing group ${groupsStore.size}. Grouping around TID $groupSeedTid.")
            val groupSeedValue = tx.read(groupSeedTid)[columns[0]] as Complex32VectorValue
            val knn = MinHeapSelection<ComparablePair<Long, DoubleValue>>(groupSize)
            //sequential access probably fastest...
            //by scanning all, we will also include the seed as it will have maximum similarity
            LOGGER.trace("Scanning remaining elements.")
            tx.forEach {
                if (!finishedTIds.contains(it.tupleId)) {
                    knn.offer(ComparablePair(it.tupleId, AbsoluteInnerProductDistance.invoke(it[columns[0]] as Complex32VectorValue, groupSeedValue)))
                }
            }
            // get mean vector and TIDs
            // update remaining and finished TIDs
            LOGGER.trace("Calculating mean and storing.")
            var groupMean = Complex64VectorValue(DoubleArray(groupSeedValue.data.size))  // faster than Complex64VectorValue.ZERO... use double as this could get large
            val groupTids = mutableListOf<Long>()
            for (i in 0 until knn.size) {
                val elementTid = knn[i].first
                groupMean += tx.read(elementTid)[columns[0]] as Complex32VectorValue
                groupTids.add(elementTid)
                check(remainingTIds.remove(elementTid)) {"${name.simple} processed an element that should not have been remaining."}
                check(finishedTIds.add(elementTid)) {"${name.simple} processed an element that was already processed."}
            }
            groupMean /= DoubleValue(knn.size)
            groupsStore[FloatArray(groupMean.data.size) {groupMean.data[it].toFloat()}] = groupTids.toLongArray()
        }
        check(groupsStore.size == numGroups) {"${name.simple} did not group into the expected number of groups (expected: $numGroups, actual: ${groupsStore.size})."}
        LOGGER.debug("Commiting to disk.")
        db.commit()
        loadGroupsFromDisk()
        LOGGER.info("Index ${name.simple} done rebuilding.")
    }

    /**
     * Updates the [Index] with the provided [DataChangeEvent]s. The updates take effect immediately, without the need to
     * commit (i.e. commit actions must take place inside).
     *
     * Not all [Index] implementations support incremental updates. Should be indicated by [IndexTransaction#supportsIncrementalUpdate()]
     *
     * @param update [Record]s to update this [Index] with wrapped in the corresponding [DataChangeEvent].
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @throws [ValidationException.IndexUpdateException] If update of [Index] fails for some reason.
     */
    override fun update(update: Collection<DataChangeEvent>, tx: Entity.Tx) {
        TODO("Not yet implemented")
    }

    /**
     * Performs a lookup through this [Index] and returns [Recordset]. This is an internal method! External
     * invocation is only possible through a [Index.Tx] object.
     *
     * This is the minimal method any [Index] implementation must support.
     *
     * @param predicate The [Predicate] to perform the lookup.
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     * @return The resulting [Recordset].
     *
     * @throws QueryException.UnsupportedPredicateException If predicate is not supported by [Index].
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset {
        /*
        Parameters: Group Match Threshold (absolute or relative), or fixed number of groups
        1. Find Groups to consider by comparing query to representative group signals
           and taking into account the Threshold or numGroups above
        2. Consider the members of the resulting groups as candidates for exact comparison
         */

        require(canProcess(predicate)) {"The supplied predicate $predicate cannot be processed by index ${this.name}"}
        predicate as KnnPredicate<*>
        // check if name hint, then look for param in there, if not name hint or no param use default
        val considerNumGroups = if (predicate.hint is KnnPredicateHint.KnnIndexNamePredicateHint) {
            val paramName = "queryConsiderNumGroups"
            val v = predicate.hint.parameters[paramName]
            if (v != null) {
                LOGGER.debug("Found '$paramName' override parameter.")
                val vNum = v.toIntOrNull()
                if (vNum != null) {
                    LOGGER.info("Found '$paramName' override parameter with value '$vNum'.")
                    vNum
                } else {
                    LOGGER.info("Found '$paramName' override parameter '$v' but could not parse it as int.")
                    queryConsiderNumGroupsDefault
                }
            } else {
                queryConsiderNumGroupsDefault
            }
        }
        else {
            queryConsiderNumGroupsDefault
        }

        LOGGER.info("Index '${this.name}' Filtering ${predicate.query.size} queries. Considering $considerNumGroups groups")
        // todo: the following is not yet fully correct, but should serve as a documentation hint that the filter
        //  algorithm should consider the k depending on the group size (need to consider more groups if k large)!
        require(predicate.k < tx.maxTupleId() / numGroups * considerNumGroups) {"k too large for this index considering $considerNumGroups groups."}

        val groupKnns = predicate.query.map { _ ->
            MinHeapSelection<ComparablePair<Int, DoubleValue>>(considerNumGroups)
        }

        LOGGER.debug("Scanning group mean signals.")
        predicate.query.indices.toList().parallelStream().forEach { queryIndex ->
            LOGGER.trace("Query $queryIndex: Scanning groups.")
            val query = predicate.query[queryIndex] as Complex32VectorValue
            groupMeans.forEachIndexed { i, gm ->
                groupKnns[queryIndex].offer(ComparablePair(i, predicate.distance.invoke(gm, query)))
            }
        }
        // transform groupKnns to
        LOGGER.debug("Transforming query group results.")
        val queriesForGroup = Array(numGroups) { emptyList<Int>().toMutableList() }
        groupKnns.forEachIndexed { i, knn ->
            for (j in 0 until knn.size) {
                queriesForGroup[knn[j].first].add(i)
            }
        }
        // now scan the groups
        // parallelize group scanning is easiest on groups, but will likely have large load imbalance
        // this requires that the selection objects are thead safe which I think they are

        val knns = predicate.query.map { _ ->
            if (predicate.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection(predicate.k)
        }

        LOGGER.debug("Scanning group members.")
        queriesForGroup.indices.toList().parallelStream().forEach { i ->
            val queryIndexes = queriesForGroup[i]
            LOGGER.trace("Scanning group members of group $i for ${queryIndexes.size} interested queries.")
            val tIdsOfGroup = tIdsOfGroups[i]
            tIdsOfGroup.forEach {
                val value = tx.read(it)[columns[0]] as Complex32VectorValue
                queryIndexes.forEach { queryIndex ->
                    knns[queryIndex].offer(ComparablePair(it, predicate.distance.invoke(value, predicate.query[queryIndex])))
                }
            }
        }

        LOGGER.info("Done filtering.")
        return KnnUtilities.selectToRecordset(this.produces.first(), knns)
    }

    /**
     * Flag indicating if this [GreedyGroupingIndex] has been closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Closes this [GreedyGroupingIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!closed) {
            db.close()
            closed = true
        }
    }
}