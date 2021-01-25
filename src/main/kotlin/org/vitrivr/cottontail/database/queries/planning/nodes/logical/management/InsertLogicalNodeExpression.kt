package org.vitrivr.cottontail.database.queries.planning.nodes.logical.management

import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.queries.planning.nodes.logical.UnaryLogicalNodeExpression
import org.vitrivr.cottontail.model.basics.Record

/**
 * A [InsertLogicalNodeExpression] that formalizes a INSERT operation on an [Entity].
 *
 * @author Ralph Gasser
 * @version 1.0.0
 */
class InsertLogicalNodeExpression(val entity: Entity, val record: Record) :
    UnaryLogicalNodeExpression() {

    /**
     * Returns a copy of this [DeleteLogicalNodeExpression]
     *
     * @return Copy of this [DeleteLogicalNodeExpression]
     */
    override fun copy(): InsertLogicalNodeExpression =
        InsertLogicalNodeExpression(this.entity, this.record)
}