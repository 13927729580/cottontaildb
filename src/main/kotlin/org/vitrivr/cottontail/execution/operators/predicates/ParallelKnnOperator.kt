package org.vitrivr.cottontail.execution.operators.predicates

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.*
import org.vitrivr.cottontail.database.column.ColumnDef
import org.vitrivr.cottontail.database.queries.predicates.knn.KnnPredicate
import org.vitrivr.cottontail.execution.TransactionContext
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.math.knn.selection.ComparablePair
import org.vitrivr.cottontail.math.knn.selection.MinHeapSelection
import org.vitrivr.cottontail.math.knn.selection.MinSingleSelection
import org.vitrivr.cottontail.math.knn.selection.Selection
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.recordset.StandaloneRecord
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.Value
import org.vitrivr.cottontail.model.values.types.VectorValue
import org.vitrivr.cottontail.utilities.math.KnnUtilities

/**
 * An [Operator.PipelineOperator] used during query execution. Performs a kNN lookup on the input
 * generated by the parent [Operator]s using the given [KnnPredicate]. The incoming branches are
 * executed in parallel (if possible) and merged afterwards. This merging step involves materialization.
 *
 * Produces querySize * k [Record]s. Acts as pipeline breaker.
 *
 * @author Ralph Gasser
 * @version 1.1.1
 */
class ParallelKnnOperator(parents: List<Operator>, val knn: KnnPredicate) : Operator.MergingPipelineOperator(parents) {

    /** The columns produced by this [ParallelKnnOperator]. */
    override val columns: Array<ColumnDef<*>> = arrayOf(
            *this.parents.first().columns,
            KnnUtilities.distanceColumnDef(this.knn.column.name.entity())
    )

    /** [ParallelKnnOperator] does act as a pipeline breaker. */
    override val breaker: Boolean = true

    /**
     * Converts this [ParallelKnnOperator] to a [Flow] and returns it.
     *
     * @param context The [TransactionContext] used for execution
     * @return [Flow] representing this [ParallelKnnOperator]
     */
    override fun toFlow(context: TransactionContext): Flow<Record> {
        /* Prepare data structures and logic for kNN. */
        val knnSet: List<Selection<ComparablePair<Record, DoubleValue>>> = if (this.knn.k == 1) {
            knn.query.map { MinSingleSelection() }
        } else {
            knn.query.map { MinHeapSelection(this.knn.k) }
        }
        val action: (Record) -> Unit = if (this.knn.weights != null) {
            {
                val value = it[this.knn.column]
                if (value is VectorValue<*>) {
                    this.knn.query.forEachIndexed { i, query ->
                        knnSet[i].offer(ComparablePair(it, this.knn.distance(query, value, this.knn.weights[i])))
                    }
                }
            }
        } else {
            {
                val value = it[this.knn.column]
                if (value is VectorValue<*>) {
                    this.knn.query.forEachIndexed { i, query ->
                        knnSet[i].offer(ComparablePair(it, this.knn.distance(query, value)))
                    }
                }
            }
        }

        /* Obtain parent flows amd compose new flow. */
        val parentFlows = this.parents.map { it.toFlow(context) }
        return flow {
            /* Execute incoming flows and wait for completion. */
            parentFlows.map { flow ->
                flow.onEach { record ->
                    action(record)
                }.launchIn(CoroutineScope(context.dispatcher))
            }.forEach {
                it.join()
            }

            /* Emit kNN values. */
            val values = ArrayList<Value?>(this@ParallelKnnOperator.columns.size + 1)
            for (knn in knnSet) {
                for (i in 0 until knn.size) {
                    values.clear()
                    knn[i].first.forEach { _, value -> values.add(value) }
                    values.add(DoubleValue(knn[i].second))
                    emit(StandaloneRecord(knn[i].first.tupleId, this@ParallelKnnOperator.columns, values.toTypedArray()))
                }
            }
        }
    }
}