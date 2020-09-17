package org.vitrivr.cottontail.database.queries.planning.rules.physical.implementation

import org.vitrivr.cottontail.database.queries.planning.nodes.interfaces.NodeExpression
import org.vitrivr.cottontail.database.queries.planning.nodes.interfaces.RewriteRule
import org.vitrivr.cottontail.database.queries.planning.nodes.logical.predicates.FilterLogicalNodeExpression
import org.vitrivr.cottontail.database.queries.planning.nodes.physical.predicates.FilterPhysicalNodeExpression

/**
 * A [RewriteRule] that implements a [FilterLogicalNodeExpression] by a [FilterPhysicalNodeExpression].
 *
 * This is a simple 1:1 replacement (implementation).
 *
 * @author Ralph Gasser
 * @version 1.0
 */
object FilterImplementationRule : RewriteRule {
    override fun canBeApplied(node: NodeExpression): Boolean = node is FilterLogicalNodeExpression
    override fun apply(node: NodeExpression): NodeExpression? {
        if (node is FilterLogicalNodeExpression) {
            val parent = (node.copyWithInputs() as FilterLogicalNodeExpression).input
            val p = FilterPhysicalNodeExpression(node.predicate)
            p.addInput(parent)
            node.copyOutput()?.addInput(p)
            return p
        }
        return null
    }
}