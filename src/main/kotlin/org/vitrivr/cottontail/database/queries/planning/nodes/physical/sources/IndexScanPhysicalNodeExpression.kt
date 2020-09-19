package org.vitrivr.cottontail.database.queries.planning.nodes.physical.sources

import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.queries.components.BooleanPredicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.database.queries.planning.cost.Costs
import org.vitrivr.cottontail.database.queries.planning.nodes.physical.NullaryPhysicalNodeExpression
import org.vitrivr.cottontail.execution.ExecutionEngine
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.execution.operators.sources.EntityIndexScanOperator

/**
 * A [AbstractEntityPhysicalNodeExpression] that represents a predicated lookup using an [Index].
 *
 * @author Ralph Gasser
 * @version 1.1
 */
class IndexScanPhysicalNodeExpression(val entity: Entity, val index: Index, val predicate: BooleanPredicate, val selectivity: Float = Costs.DEFAULT_SELECTIVITY) : NullaryPhysicalNodeExpression() {
    override val canBePartitioned: Boolean = false
    override val outputSize: Long = (this.entity.statistics.rows * this.selectivity).toLong()
    override val cost: Cost = this.index.cost(this.predicate)
    override fun copy() = IndexScanPhysicalNodeExpression(this.entity, this.index, this.predicate, this.selectivity)
    override fun toOperator(context: ExecutionEngine.ExecutionContext): Operator = EntityIndexScanOperator(context, this.entity, this.entity.allColumns().toTypedArray(), this.predicate, this.index.type)

    override fun partition(p: Int): List<NullaryPhysicalNodeExpression> {
        /* TODO: May actually be possible for certain index structures. */
        throw IllegalStateException("IndexKnnPhysicalNodeExpression cannot be partitioned.")
    }
}