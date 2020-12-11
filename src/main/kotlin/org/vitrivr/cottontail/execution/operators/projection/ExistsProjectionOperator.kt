package org.vitrivr.cottontail.execution.operators.projection

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import org.vitrivr.cottontail.execution.ExecutionEngine
import org.vitrivr.cottontail.execution.operators.basics.AbortFlowException
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.recordset.StandaloneRecord
import org.vitrivr.cottontail.model.values.BooleanValue

/**
 * An [Operator.PipelineOperator] used during query execution. It returns a single [Record] containing
 * either true or false depending on whether there are incoming [Record]s.
 *
 * Only produces a single [Record]. Acts as pipeline breaker.
 *
 * @author Ralph Gasser
 * @version 1.1.1
 */
class ExistsProjectionOperator(parent: Operator) : Operator.PipelineOperator(parent) {

    /** Column returned by [ExistsProjectionOperator]. */
    override val columns: Array<ColumnDef<*>> = arrayOf(ColumnDef.withAttributes(parent.columns.first().name.entity()?.column("exists()")
            ?: Name.ColumnName("exists()"), "BOOLEAN"))


    /** [ExistsProjectionOperator] does act as a pipeline breaker. */
    override val breaker: Boolean = true

    /**
     * Converts this [ExistsProjectionOperator] to a [Flow] and returns it.
     *
     * @param context The [ExecutionEngine.ExecutionContext] used for execution
     * @return [Flow] representing this [ExistsProjectionOperator]
     */
    override fun toFlow(context: ExecutionEngine.ExecutionContext): Flow<Record> {
        val parentFlow = this.parent.toFlow(context)
        return flow {
            var exists = false
            try {
                parentFlow.collect {
                    exists = true
                    throw AbortFlowException(this)
                }
            } catch (e: AbortFlowException) {
                e.checkOwnership(this)
            }
            emit(StandaloneRecord(0L, this@ExistsProjectionOperator.columns, arrayOf(BooleanValue(exists))))
        }
    }
}