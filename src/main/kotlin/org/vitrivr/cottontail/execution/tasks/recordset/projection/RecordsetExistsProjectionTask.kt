package org.vitrivr.cottontail.execution.tasks.recordset.projection

import com.github.dexecutor.core.task.Task
import com.github.dexecutor.core.task.TaskExecutionException
import org.vitrivr.cottontail.execution.cost.Costs
import org.vitrivr.cottontail.execution.tasks.basics.ExecutionTask
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.BooleanValue
import org.vitrivr.cottontail.utilities.name.Name

/**
 * A [Task] used during query execution. It takes a single [Recordset] as input, counts the number of of rows and returns it as [Recordset].
 *
 * @author Ralph Gasser
 * @version 1.0.1
 */
class RecordsetExistsProjectionTask : ExecutionTask("RecordsetExistsProjectionTask") {

    /** The cost of this [RecordsetExistsProjectionTask] is constant. */
    override val cost = Costs.MEMORY_ACCESS_READ

    /**
     * Executes this [RecordsetExistsProjectionTask]
     */
    override fun execute(): Recordset {
        assertUnaryInput()

        /* Get records from parent task. */
        val parent = this.first()
                ?: throw TaskExecutionException("EXISTS projection could not be executed because parent task has failed.")

        /* Create new Recordset with new columns. */
        val recordset = Recordset(arrayOf(ColumnDef.withAttributes(Name("exists(*)"), "BOOLEAN")))
        recordset.addRowUnsafe(arrayOf(BooleanValue(parent.rowCount > 0)))
        return recordset
    }
}