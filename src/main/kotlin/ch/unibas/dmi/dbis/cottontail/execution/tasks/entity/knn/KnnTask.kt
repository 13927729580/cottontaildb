package ch.unibas.dmi.dbis.cottontail.execution.tasks.entity.knn

import ch.unibas.dmi.dbis.cottontail.database.column.DoubleVectorColumnType
import ch.unibas.dmi.dbis.cottontail.database.column.FloatVectorColumnType
import ch.unibas.dmi.dbis.cottontail.database.column.IntVectorColumnType
import ch.unibas.dmi.dbis.cottontail.database.column.LongVectorColumnType
import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.queries.BooleanPredicate
import ch.unibas.dmi.dbis.cottontail.database.queries.KnnPredicate
import ch.unibas.dmi.dbis.cottontail.execution.tasks.basics.ExecutionTask
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.exceptions.QueryException

internal object KnnTask {
    /** Threshold under which parallelism starts to kick in. TODO: Find optimal value experimentally. */
    private const val KNN_OP_PARALLELISM_THRESHOLD = 81920000L

    /**
     * Constructs a [ExecutionTask] for kNN lookup given the [KnnPredicate] and the optional [BooleanPredicate]. This method
     * optimizes for parallelism given the required operations and the size of the [Entity] involved.
     *
     * @param entity [Entity] on which to perform the kNN lookup.
     * @param knnClause The [KnnPredicate] that specifies the kNN lookup
     * @param whereClause The optional [BooleanPredicate] to apply before doing the kNN lookup
     * @return  The resulting [ExecutionTask]
     *
     * @throws QueryException.QueryBindException If the provided [ColumnDef] does not support kNN lookups.
     */
    @Suppress("UNCHECKED_CAST")
    fun entityScanTaskForPredicate(entity: Entity, knnClause: KnnPredicate<*>, whereClause: BooleanPredicate?) : ExecutionTask {
        val operations = knnClause.query.first().size * entity.statistics.rows * (knnClause.operations + (whereClause?.operations ?: 0))
        val parallelism = Math.min(Math.floorDiv(operations, KNN_OP_PARALLELISM_THRESHOLD).toInt(), Runtime.getRuntime().availableProcessors()).toShort()
        return when {
            parallelism > 1 && knnClause.column.type is DoubleVectorColumnType -> ParallelEntityScanDoubleKnnTask(entity, knnClause as KnnPredicate<DoubleArray>, whereClause, parallelism)
            parallelism > 1 && knnClause.column.type is FloatVectorColumnType -> ParallelEntityScanFloatKnnTask(entity, knnClause as KnnPredicate<FloatArray>, whereClause, parallelism)
            parallelism > 1 && knnClause.column.type is LongVectorColumnType -> ParallelEntityScanLongKnnTask(entity, knnClause as KnnPredicate<LongArray>, whereClause, parallelism)
            parallelism > 1 && knnClause.column.type is IntVectorColumnType -> ParallelEntityScanIntKnnTask(entity, knnClause as KnnPredicate<IntArray>, whereClause, parallelism)
            parallelism <= 1 && knnClause.column.type is DoubleVectorColumnType -> LinearEntityScanDoubleKnnTask(entity, knnClause as KnnPredicate<DoubleArray>, whereClause)
            parallelism <= 1 && knnClause.column.type is FloatVectorColumnType -> LinearEntityScanFloatKnnTask(entity, knnClause as KnnPredicate<FloatArray>, whereClause)
            parallelism <= 1 && knnClause.column.type is LongVectorColumnType -> LinearEntityScanLongKnnTask(entity, knnClause as KnnPredicate<LongArray>, whereClause)
            parallelism <= 1 && knnClause.column.type is IntVectorColumnType -> LinearEntityScanIntKnnTask(entity, knnClause as KnnPredicate<IntArray>, whereClause)
            else -> throw QueryException.QueryBindException("A column of type '${knnClause.column.type} is not supported for kNN queries.")
        }
    }
}