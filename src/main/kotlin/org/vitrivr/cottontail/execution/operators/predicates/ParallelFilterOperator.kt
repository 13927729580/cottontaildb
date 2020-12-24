package org.vitrivr.cottontail.execution.operators.predicates

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import org.vitrivr.cottontail.database.queries.components.BooleanPredicate
import org.vitrivr.cottontail.execution.TransactionContext
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Record

/**
 * An [Operator.MergingPipelineOperator] used during query execution. Filters the input generated
 * by the parent [Operator]s using the given [BooleanPredicate].
 *
 * This [Operator.MergingPipelineOperator] merges input generated by multiple [Operator]s.
 * Their output order may be arbitrary.
 *
 * @author Ralph Gasser
 * @version 1.0.0
 */
class ParallelFilterOperator(parents: List<Operator>, private val predicate: BooleanPredicate) : Operator.MergingPipelineOperator(parents) {

    /** Columns returned by [ParallelFilterOperator] depend on the parent [Operator]. */
    override val columns: Array<ColumnDef<*>> = this.parents.first().columns

    /** [ParallelFilterOperator] does not act as a pipeline breaker. */
    override val breaker: Boolean = false

    /**
     * Converts this [ParallelFilterOperator] to a [Flow] and returns it.
     *
     * @param context The [TransactionContext] used for execution
     * @return [Flow] representing this [FilterOperator]
     */
    @ExperimentalCoroutinesApi
    override fun toFlow(context: TransactionContext): Flow<Record> {

        /* Obtain parent flows amd compose new flow. */
        val list = mutableListOf<Record>()
        val parentFlows = this.parents.map { it.toFlow(context) }
        return flow {
            /* Execute incoming flows and wait for completion. */
            parentFlows.map { flow ->
                flow.onEach { record ->
                    if (this@ParallelFilterOperator.predicate.matches(record)) {
                        list.add(record)
                    }
                }.launchIn(CoroutineScope(context.dispatcher))
            }.forEach {
                it.join()
            }

            list.forEach {
                emit(it)
            }
        }
    }
}