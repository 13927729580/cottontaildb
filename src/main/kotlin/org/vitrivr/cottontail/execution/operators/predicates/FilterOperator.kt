package org.vitrivr.cottontail.execution.operators.predicates

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.filter
import org.vitrivr.cottontail.database.queries.components.BooleanPredicate
import org.vitrivr.cottontail.execution.TransactionContext
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Record

/**
 * An [Operator.PipelineOperator] used during query execution. Filters the input generated by the
 * parent [Operator] using the given [BooleanPredicate].
 *
 * @author Ralph Gasser
 * @version 1.1.1
 */
class FilterOperator(parent: Operator, private val predicate: BooleanPredicate) : Operator.PipelineOperator(parent) {

    /** Columns returned by [FilterOperator] depend on the parent [Operator]. */
    override val columns: Array<ColumnDef<*>> = this.parent.columns

    /** [FilterOperator] does not act as a pipeline breaker. */
    override val breaker: Boolean = false

    /**
     * Converts this [FilterOperator] to a [Flow] and returns it.
     *
     * @param context The [TransactionContext] used for execution
     * @return [Flow] representing this [FilterOperator]
     */
    override fun toFlow(context: TransactionContext): Flow<Record> =
        this.parent.toFlow(context).filter { this.predicate.matches(it) }
}