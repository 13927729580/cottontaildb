package org.vitrivr.cottontail.database.index.va

import org.mapdb.Atomic
import org.mapdb.DBMaker
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.index.va.marks.Marks
import org.vitrivr.cottontail.database.index.va.marks.MarksGenerator
import org.vitrivr.cottontail.database.queries.components.KnnPredicate
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.execution.tasks.entity.knn.KnnUtilities.selectToRecordset
import org.vitrivr.cottontail.math.knn.metrics.AbsoluteInnerProductDistance
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.ComplexVectorValue
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class VAFIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>): Index() {

    companion object {
        val SIGNATURE_FIELD_NAME = "vaf_signatures"
        val REAL_MARKS_FIELD_NAME = "vaf_marks_real"
        val IMAG_MARKS_FIELD_NAME = "vaf_marks_imag"
        val MARKS_PER_DIM = 25  // doesn't have too much of an influence on execution time with current implementation -> something is quite inefficient... Influence on filter ratio is visible
        // considering the fact that the approximation is calculated for all query-db pairs, this suggests that the exact calculation is much more efficient
        val LOGGER: Logger = LoggerFactory.getLogger(VAFIndex::class.java)
    }
    /** The [Path] to the [DBO]'s main file OR folder. */
    override val path: Path = this.parent.path.resolve("idx_vaf_$name.db")

    /** The [VAFIndex] implementation returns exactly the columns that is indexed. */
    override val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef(this.parent.name.column("distance"), ColumnType.forName("DOUBLE")))

    /** The type of [Index]. */
    override val type = IndexType.VAF

    /** The internal [DB] reference. */
    private val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }
    /** Store for the marks. */
    private val marksRealStore: Atomic.Var<Marks> = db.atomicVar(REAL_MARKS_FIELD_NAME, Marks.MarksSerializer).createOrOpen()
    private val marksImagStore: Atomic.Var<Marks> = db.atomicVar(IMAG_MARKS_FIELD_NAME, Marks.MarksSerializer).createOrOpen()

    /** Store for the signatures. */
    private val signatures = this.db.indexTreeList(SIGNATURE_FIELD_NAME, VectorApproximationSignatureSerializer).createOrOpen()

    init {
        if (columns.size != 1) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing a single column")
        }
        if (!columns.all { it.type == ColumnType.forName("COMPLEX32_VEC") || it.type == ColumnType.forName("COMPLEX64_VEC") }) {
            throw DatabaseException.IndexNotSupportedException(name, "${this::class.java} currently only supports indexing complex vector columns, not ${columns.first()::class.java}")
        }
    }

    /**
     * Flag indicating if this [VAFIndex] has been closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Closes this [VAFIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!closed) {
            db.close()
            closed = true
        }
    }

    /**
     * Calculates the cost estimate if this [Index] processing the provided [Predicate].
     * todo: get real cost estimate
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate) = Cost.ZERO

    override fun canProcess(predicate: Predicate) =
            predicate is KnnPredicate<*>
                    && predicate.query.all { it is ComplexVectorValue<*> }
                    && predicate.columns.first() == this.columns[0]
                    && predicate.distance is AbsoluteInnerProductDistance

    /**
     * Returns true, if the [Index] supports incremental updates, and false otherwise.
     * false for now
     * @return True if incremental [Index] updates are supported.
     */
    override fun supportsIncrementalUpdate() = false

    /**
     * (Re-)builds the [Index]. Invoking this method should rebuild the [Index] immediately, without the
     * need to commit (i.e. commit actions must take place inside).
     *
     * This is an internal method! External invocation is only possible through a [Index.Tx] object.
     *
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun rebuild(tx: Entity.Tx) {
        LOGGER.info("Rebuilding $name")
        val specimen = tx.read(tx.maxTupleId())[columns[0]] as ComplexVectorValue<*>
        // find min max for each dim to get marks
        val minReal = DoubleArray(specimen.logicalSize)
        val minImag = DoubleArray(specimen.logicalSize)
        val maxReal = DoubleArray(specimen.logicalSize)
        val maxImag = DoubleArray(specimen.logicalSize)
        LOGGER.debug("Finding min and max for each dim")
        tx.forEach { r ->
            (r[columns.first()] as ComplexVectorValue<*>).forEachIndexed { i, v ->
                minReal[i] = min(minReal[i], v.real.value.toDouble())
                minImag[i] = min(minImag[i], v.imaginary.value.toDouble())
                maxReal[i] = max(maxReal[i], v.real.value.toDouble())
                maxImag[i] = max(maxImag[i], v.imaginary.value.toDouble())
            }
        }
        val marksReal = MarksGenerator.getEquidistantMarks(minReal, maxReal, IntArray(specimen.logicalSize) { MARKS_PER_DIM })
        marksRealStore.set(marksReal)
        val marksImag = MarksGenerator.getEquidistantMarks(minImag, maxImag, IntArray(specimen.logicalSize) { MARKS_PER_DIM })
        marksImagStore.set(marksImag)

        LOGGER.debug("Generating signatures for vectors")
        tx.forEach {
            val (valueReal, valueImag) = (it[columns.first()] as ComplexVectorValue<*>).map { it.real.value.toDouble() to it.imaginary.value.toDouble() }.unzip()

            signatures.add(VectorApproximationSignature(it.tupleId, marksReal.getCells(valueReal.toDoubleArray())))
            signatures.add(VectorApproximationSignature(it.tupleId, marksImag.getCells(valueImag.toDoubleArray())))
        }
        LOGGER.info("Done.")
        db.commit()
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
        require(canProcess(predicate)) { "The supplied predicate $predicate is not supported by the index" }
        LOGGER.info("filtering")
        predicate as KnnPredicate<*>
        val marksReal = marksRealStore.get()!!
        val marksImag = marksImagStore.get()!!
        val knns = predicate.query.map {
            if (predicate.k == 1) MinSingleSelection<ComparablePair<Long, DoubleValue>>() else MinHeapSelection<ComparablePair<Long, DoubleValue>>(predicate.k)
        }
//        val queriesSplit = predicate.query.map { (it as ComplexVectorValue<*>).map { qc -> qc.real.value.toDouble() to qc.imaginary.value.toDouble() }.unzip() }
        // need as array of primitives for performance
        val queriesSplit = Array(predicate.query.size) {i ->
            DoubleArray(predicate.query[i].logicalSize) {j ->
                (predicate.query[i] as ComplexVectorValue<*>).real(j).value.toDouble()
            } to DoubleArray(predicate.query[i].logicalSize) { j ->
                (predicate.query[i] as ComplexVectorValue<*>).imaginary(j).value.toDouble()
            }
        }
        var countCandidates = 0
        var countRejected = 0
        signatures.chunked(2).forEach {sigRealImag ->
            val sigReal = sigRealImag[0]!!
            val sigImag = sigRealImag[1]!!
            check(sigReal.tupleId == sigImag.tupleId)
            predicate.query.forEachIndexed { i, query ->
                val absIPDistLB = 1.0 - absoluteComplexInnerProductSqUpperBound(sigReal.signature, sigImag.signature, queriesSplit[i].first, queriesSplit[i].second, marksReal, marksImag).pow(0.5)
                if (knns[i].size < predicate.k || knns[i].peek()!!.second > absIPDistLB) {
                    countCandidates++
                    val tid = sigReal.tupleId
                    knns[i].offer(ComparablePair(tid, predicate.distance((tx.read(tid)[columns.first()] as ComplexVectorValue<*>), query)))
                } else {
                    countRejected++
                }
            }
        }
        LOGGER.info("Done. Considered candidates: $countCandidates, rejected candidates: $countRejected (${countRejected.toDouble() / (countRejected + countCandidates) * 100} %)")

        return selectToRecordset(this.produces.first(), knns.toList())
    }
}