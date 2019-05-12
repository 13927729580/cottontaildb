package ch.unibas.dmi.dbis.cottontail.execution.tasks.recordset.projection

import ch.unibas.dmi.dbis.cottontail.database.queries.Projection
import ch.unibas.dmi.dbis.cottontail.execution.cost.Costs
import ch.unibas.dmi.dbis.cottontail.execution.tasks.basics.ExecutionTask
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.recordset.Recordset
import ch.unibas.dmi.dbis.cottontail.model.values.DoubleValue
import com.github.dexecutor.core.task.Task
import com.github.dexecutor.core.task.TaskExecutionException

/**
 * A [Task] used during query execution. It takes a single [Recordset] and determines the mean of a specific [ColumnDef]. It thereby creates a 1x1 [Recordset].
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class RecordsetMeanProjectionTask(val projection: Projection, estimatedSize: Int = 1000): ExecutionTask("RecordsetMeanProjectionTask") {

    /** The cost of this [RecordsetMaxProjectionTask] is constant */
    override val cost = estimatedSize * Costs.MEMORY_ACCESS_READ

    /**
     * Executes this [RecordsetCountProjectionTask]
     */
    override fun execute(): Recordset {
        assertUnaryInput()

        /* Get records from parent task. */
        val parent = this.first() ?: throw TaskExecutionException("Projection could not be executed because parent task has failed.")

        /* Calculate mean(). */
        val column = projection.columns.first()
        val resultsColumn = ColumnDef.withAttributes(this.projection.fields[column.name] ?: "mean(${column.name})", "DOUBLE")
        val results = Recordset(arrayOf(resultsColumn))
        if (parent.rowCount > 0) {
            var sum = 0.0
            parent.forEach {
                when (val value = it[column]?.value) {
                    is Byte -> sum += value
                    is Short -> sum += value
                    is Int -> sum += value
                    is Long -> sum += value
                    is Float -> sum += value
                    is Double -> sum += value
                    else -> {}
                }
            }
            results.addRowUnsafe(arrayOf(DoubleValue(sum/parent.rowCount)))
            return results
        } else {
            results.addRowUnsafe(arrayOf(DoubleValue(0.0)))
            return results
        }
    }
}