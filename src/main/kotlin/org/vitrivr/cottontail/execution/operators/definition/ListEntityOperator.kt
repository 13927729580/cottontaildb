package org.vitrivr.cottontail.execution.operators.definition

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.vitrivr.cottontail.client.language.basics.Constants
import org.vitrivr.cottontail.database.catalogue.CatalogueTx
import org.vitrivr.cottontail.database.catalogue.DefaultCatalogue
import org.vitrivr.cottontail.database.column.ColumnDef
import org.vitrivr.cottontail.database.schema.SchemaTx
import org.vitrivr.cottontail.execution.TransactionContext
import org.vitrivr.cottontail.execution.operators.basics.Operator
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.basics.Type
import org.vitrivr.cottontail.model.recordset.StandaloneRecord
import org.vitrivr.cottontail.model.values.StringValue
import kotlin.time.ExperimentalTime

/**
 * An [Operator.SourceOperator] used during query execution. Lists all available [Entity]s.
 *
 * @author Ralph Gasser
 * @version 1.0.1
 */
class ListEntityOperator(val catalogue: DefaultCatalogue, val schema: Name.SchemaName? = null) : Operator.SourceOperator() {

    companion object {
        val COLUMNS: Array<ColumnDef<*>> = arrayOf(
            ColumnDef(Name.ColumnName(Constants.COLUMN_NAME_DBO), Type.String, false),
            ColumnDef(Name.ColumnName(Constants.COLUMN_NAME_CLASS), Type.String, false)
        )
    }

    override val columns: Array<ColumnDef<*>> = COLUMNS

    @ExperimentalTime
    override fun toFlow(context: TransactionContext): Flow<Record> {
        val txn = context.getTx(this.catalogue) as CatalogueTx
        val schemas = if (this.schema != null) {
            listOf(txn.schemaForName(this.schema))
        } else {
            txn.listSchemas()
        }
        return flow {
            for (schema in schemas) {
                val schemaTxn = context.getTx(schema) as SchemaTx
                for (entity in schemaTxn.listEntities()) {
                    emit(
                        StandaloneRecord(
                            0L,
                            this@ListEntityOperator.columns,
                            arrayOf(StringValue(entity.name.toString()), StringValue("ENTITY"))
                        )
                    )
                }
            }
        }
    }
}