package org.vitrivr.cottontail.execution.operators.management

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.execution.ExecutionEngine
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.execution.operators.basics.OperatorStatus
import org.vitrivr.cottontail.execution.operators.basics.SinkOperator
import org.vitrivr.cottontail.execution.operators.predicates.FilterOperator
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.recordset.StandaloneRecord
import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.LongValue
import org.vitrivr.cottontail.model.values.types.Value
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

class UpdateOperator(parent: Operator, context: ExecutionEngine.ExecutionContext, val entity: Entity, val values: Map<ColumnDef<*>,Value?>): SinkOperator(parent, context) {
    /** Columns returned by [UpdateOperator]. */
    override val columns: Array<ColumnDef<*>> = arrayOf(
        ColumnDef.withAttributes(Name.ColumnName("updated"), "LONG", -1, false),
        ColumnDef.withAttributes(Name.ColumnName("duration_ms"), "DOUBLE", -1, false),
    )

    private var transaction: Entity.Tx? = null

    override fun prepareOpen() {
        this.transaction = this.context.requestTransaction(this.entity, this.entity.allColumns().toTypedArray(), false)
    }

    override fun prepareClose() {
        this.transaction?.close()
        this.transaction = null
    }

    /**
     * Converts this [FilterOperator] to a [Flow] and returns it.
     *
     * @param scope The [CoroutineScope] used for execution
     * @return [Flow] representing this [FilterOperator]
     *
     * @throws IllegalStateException If this [Operator.status] is not [OperatorStatus.OPEN]
     */
    @ExperimentalTime
    override fun toFlow(scope: CoroutineScope): Flow<Record> {
        var updated = 0L
        val parent = this.parent.toFlow(scope)
        return flow {
            val time = measureTime {
                parent.collect {
                    this@UpdateOperator.transaction?.
                    updated += 1
                }
            }
            this@UpdateOperator.transaction?.commit()
            emit(StandaloneRecord(0L, this@UpdateOperator.columns, arrayOf(LongValue(updated), DoubleValue(time.inMilliseconds))))
        }
    }
}