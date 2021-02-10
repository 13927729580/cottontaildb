package org.vitrivr.cottontail.database.queries.planning.nodes.physical.predicates

import org.vitrivr.cottontail.database.queries.QueryContext
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.database.queries.planning.nodes.physical.UnaryPhysicalNodeExpression
import org.vitrivr.cottontail.database.queries.predicates.bool.BooleanPredicate
import org.vitrivr.cottontail.execution.TransactionContext
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.execution.operators.predicates.FilterOperator
import org.vitrivr.cottontail.execution.operators.predicates.ParallelFilterOperator
import org.vitrivr.cottontail.model.basics.ColumnDef

/**
 * A [UnaryPhysicalNodeExpression] that represents application of a [BooleanPredicate] on some intermediate result.
 *
 * @author Ralph Gasser
 * @version 1.1.0
 */
class FilterPhysicalNodeExpression(val predicate: BooleanPredicate) :
    UnaryPhysicalNodeExpression() {

    companion object {
        const val MAX_PARALLELISM = 4
    }

    private val selectivity: Float = Cost.COST_DEFAULT_SELECTIVITY

    /** The [FilterPhysicalNodeExpression] returns the [ColumnDef] of its input. */
    override val columns: Array<ColumnDef<*>>
        get() = this.input.columns

    override val outputSize: Long
        get() = (this.input.outputSize * this.selectivity).toLong()

    override val cost: Cost
        get() = Cost(cpu = this.input.outputSize * this.predicate.cost * Cost.COST_MEMORY_ACCESS)

    override fun copy() = FilterPhysicalNodeExpression(this.predicate)

    override fun toOperator(tx: TransactionContext, ctx: QueryContext): Operator {
        val parallelisation = Integer.min(this.cost.parallelisation(), MAX_PARALLELISM)
        return if (this.canBePartitioned && parallelisation > 1) {
            val operators = this.input.partition(parallelisation).map { it.toOperator(tx, ctx) }
            ParallelFilterOperator(operators, this.predicate.bind(ctx))
        } else {
            FilterOperator(this.input.toOperator(tx, ctx), this.predicate.bind(ctx))
        }
    }

    /**
     * Calculates and returns the digest for this [FilterPhysicalNodeExpression].
     *
     * @return Digest for this [FilterPhysicalNodeExpression]e
     */
    override fun digest(): Long = 31L * super.digest() + this.predicate.digest()
}