package ch.unibas.dmi.dbis.cottontail.server.grpc.helper

import ch.unibas.dmi.dbis.cottontail.database.catalogue.Catalogue
import ch.unibas.dmi.dbis.cottontail.database.column.*
import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.index.IndexType
import ch.unibas.dmi.dbis.cottontail.database.queries.*

import ch.unibas.dmi.dbis.cottontail.execution.ExecutionEngine
import ch.unibas.dmi.dbis.cottontail.execution.ExecutionPlan
import ch.unibas.dmi.dbis.cottontail.execution.ExecutionPlanFactory
import ch.unibas.dmi.dbis.cottontail.execution.tasks.basics.ExecutionTask

import ch.unibas.dmi.dbis.cottontail.grpc.CottontailGrpc
import ch.unibas.dmi.dbis.cottontail.math.knn.metrics.*

import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.exceptions.DatabaseException
import ch.unibas.dmi.dbis.cottontail.model.exceptions.QueryException
import ch.unibas.dmi.dbis.cottontail.model.values.*
import ch.unibas.dmi.dbis.cottontail.model.values.complex.Complex
import ch.unibas.dmi.dbis.cottontail.utilities.name.doesNameMatch
import ch.unibas.dmi.dbis.cottontail.utilities.name.normalizeColumnName
import java.util.*

/**
 * This helper class parses and binds queries issued through the GRPC endpoint. The process encompasses three steps:
 *
 * 1) The [CottontailGrpc.Query] is decomposed into its components.
 * 2) The GRPC query components are bound to Cottontail DB [DBO] objects and internal query objects are constructed. This step includes some basic validation.
 * 3) A [ExecutionPlan] is constructed from the internal query objects.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class GrpcQueryBinder(val catalogue: Catalogue, engine: ExecutionEngine) {

    /** [ExecutionPlanFactor] used to generate [ExecutionPlan]s from query definitions. */
    private val factory = ExecutionPlanFactory(engine)

    /**
     * Binds the given [CottontailGrpc.Query] to the database objects and thereby creates an [ExecutionPlan].
     *
     * @param query The [CottontailGrpc.Query] that should be bound.
     * @return [ExecutionPlan]
     *
     * @throws QueryException.QuerySyntaxException If [CottontailGrpc.Query] is structurally incorrect.
     */
    fun parseAndBind(query: CottontailGrpc.Query): ExecutionPlan {
        if (!query.hasFrom()) throw QueryException.QuerySyntaxException("The query lacks a valid FROM-clause.")
        return when {
            query.from.hasEntity() -> parseAndBindSimpleQuery(query)
            else -> throw QueryException.QuerySyntaxException("The query lacks a valid FROM-clause.")
        }
    }

    /**
     * Parses and binds a simple [CottontailGrpc.Query] without any joins and thereby generates an [ExecutionPlan].
     *
     * @param query The simple [CottontailGrpc.Query] object.
     */
    private fun parseAndBindSimpleQuery(query: CottontailGrpc.Query): ExecutionPlan {
        val entity = try {
            this.catalogue.schemaForName(query.from.entity.schema.name).entityForName(query.from.entity.name)
        } catch (e: DatabaseException.SchemaDoesNotExistException) {
            throw QueryException.QueryBindException("Failed to bind '${query.from.entity.fqn()}'. Schema does not exist!")
        } catch (e: DatabaseException.EntityDoesNotExistException) {
            throw QueryException.QueryBindException("Failed to bind '${query.from.entity.fqn()}'. Entity does not exist!")
        } catch (e: DatabaseException) {
            throw QueryException.QueryBindException("Failed to bind ${query.from.entity.fqn()}. Database error!")
        }

        /* Create projection clause. */
        val projectionClause = if (query.hasProjection()) {
            parseAndBindProjection(entity, query.projection)
        } else {
            parseAndBindProjection(entity, CottontailGrpc.Projection.newBuilder().setOp(CottontailGrpc.Projection.Operation.SELECT).putAttributes("*", "").build())
        }
        val knnClause = if (query.hasKnn()) parseAndBindKnnPredicate(entity, query.knn) else null
        val whereClause = if (query.hasWhere()) parseAndBindBooleanPredicate(entity, query.where) else null

        /* Transform to ExecutionPlan. */
        return this.factory.simpleExecutionPlan(entity, projectionClause, knnClause = knnClause, whereClause = whereClause, limit = query.limit, skip = query.skip)
    }

    /**
     * Parses and binds a [CottontailGrpc.Where] clause.
     *
     * @param entity The [Entity] from which fetch columns.
     * @param where The [CottontailGrpc.Where] object.
     *
     * @return The resulting [AtomicBooleanPredicate].
     */
    private fun parseAndBindBooleanPredicate(entity: Entity, where: CottontailGrpc.Where): BooleanPredicate = when (where.predicateCase) {
        CottontailGrpc.Where.PredicateCase.ATOMIC -> parseAndBindAtomicBooleanPredicate(entity, where.atomic)
        CottontailGrpc.Where.PredicateCase.COMPOUND -> parseAndBindCompoundBooleanPredicate(entity, where.compound)
        CottontailGrpc.Where.PredicateCase.PREDICATE_NOT_SET -> throw QueryException.QuerySyntaxException("WHERE clause without a predicate is invalid!")
        null -> throw QueryException.QuerySyntaxException("WHERE clause without a predicate is invalid!")
    }

    /**
     * Parses and binds an atomic boolean predicate
     *
     * @param entity The [Entity] from which fetch columns.
     * @param projection The [CottontailGrpc.Knn] object.
     *
     * @return The resulting [AtomicBooleanPredicate].
     */
    private fun parseAndBindCompoundBooleanPredicate(entity: Entity, compound: CottontailGrpc.CompoundBooleanPredicate): CompoundBooleanPredicate {
        val left = when (compound.leftCase) {
            CottontailGrpc.CompoundBooleanPredicate.LeftCase.ALEFT -> parseAndBindAtomicBooleanPredicate(entity, compound.aleft)
            CottontailGrpc.CompoundBooleanPredicate.LeftCase.CLEFT -> parseAndBindCompoundBooleanPredicate(entity, compound.cleft)
            CottontailGrpc.CompoundBooleanPredicate.LeftCase.LEFT_NOT_SET -> throw QueryException.QuerySyntaxException("Unbalanced predicate! A compound boolean predicate must have a left and a right side.")
            null -> throw QueryException.QuerySyntaxException("Unbalanced predicate! A compound boolean predicate must have a left and a right side.")
        }

        val right = when (compound.rightCase) {
            CottontailGrpc.CompoundBooleanPredicate.RightCase.ARIGHT -> parseAndBindAtomicBooleanPredicate(entity, compound.aright)
            CottontailGrpc.CompoundBooleanPredicate.RightCase.CRIGHT -> parseAndBindCompoundBooleanPredicate(entity, compound.cright)
            CottontailGrpc.CompoundBooleanPredicate.RightCase.RIGHT_NOT_SET -> throw QueryException.QuerySyntaxException("Unbalanced predicate! A compound boolean predicate must have a left and a right side.")
            null -> throw QueryException.QuerySyntaxException("Unbalanced predicate! A compound boolean predicate must have a left and a right side.")
        }

        return try {
            CompoundBooleanPredicate(ConnectionOperator.valueOf(compound.op.name), left, right)
        } catch (e: IllegalArgumentException) {
            throw QueryException.QuerySyntaxException("'${compound.op.name}' is not a valid connection operator for a boolean predicate!")
        }
    }

    /**
     * Parses and binds an atomic boolean predicate
     *
     * @param entity The [Entity] from which fetch columns.
     * @param projection The [CottontailGrpc.Knn] object.
     *
     * @return The resulting [AtomicBooleanPredicate].
     */
    @Suppress("UNCHECKED_CAST")
    private fun parseAndBindAtomicBooleanPredicate(entity: Entity, atomic: CottontailGrpc.AtomicLiteralBooleanPredicate): AtomicBooleanPredicate<*> {
        val column = entity.columnForName(atomic.attribute)
                ?: throw QueryException.QueryBindException("Failed to bind column '${atomic.attribute}'. Column does not exist on entity '${entity.fqn}'.")
        val operator = try {
            ComparisonOperator.valueOf(atomic.op.name)
        } catch (e: IllegalArgumentException) {
            throw QueryException.QuerySyntaxException("'${atomic.op.name}' is not a valid comparison operator for a boolean predicate!")
        }

        /* Perform some sanity checks. */
        when {
            operator == ComparisonOperator.LIKE && !entity.hasIndexForColumn(column, IndexType.LUCENE) -> throw QueryException.QueryBindException("Failed to bind query '${atomic.attribute} LIKE :1' on entity '${entity.fqn}'. The entity does not have a text-index on the specified column '${column.name}', which is required for LIKE comparisons.")
        }

        /* Return the resulting AtomicBooleanPredicate. */
        return when (column.type) {
            is DoubleColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Double>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toDoubleValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is FloatColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Float>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toFloatValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is LongColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Long>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toLongValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is IntColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Int>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toIntValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is ShortColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Short>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toShortValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is ByteColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Byte>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toByteValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is BooleanColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Boolean>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toBooleanValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is StringColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<String>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toStringValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is ComplexColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Complex>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toComplexValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is FloatVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<FloatArray>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toFloatVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is DoubleVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<DoubleArray>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toDoubleVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is LongVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<LongArray>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toLongVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is IntVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<IntArray>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toIntVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is BooleanVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<BooleanArray>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toBooleanVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
            is ComplexVectorColumnType -> AtomicBooleanPredicate(column = column as ColumnDef<Value<Array<Complex>>>, operator = operator, not = atomic.not, values = atomic.dataList.map {
                it.toComplexVectorValue()
                        ?: throw QueryException.QuerySyntaxException("Cannot compare ${column.name} to NULL value with operator $operator.")
            })
        }
    }

    /**
     * Parses and binds the kNN-lookup part of a GRPC [CottontailGrpc.Query]
     *
     * @param entity The [Entity] from which fetch columns.
     * @param projection The [CottontailGrpc.Knn] object.
     *
     * @return The resulting [ExecutionTask].
     */
    @Suppress("UNCHECKED_CAST")
    private fun parseAndBindKnnPredicate(entity: Entity, knn: CottontailGrpc.Knn): KnnPredicate<*> {
        val column = entity.columnForName(knn.attribute)
                ?: throw QueryException.QueryBindException("Failed to bind column '${knn.attribute}'. Column does not exist on entity '${entity.fqn}'!")
        val weights = if (knn.weightsCount > 0) {
            knn.weightsList.map { w -> w.toFloatVectorValue() }
        } else {
            null
        }

        return when (column.type) {
            is DoubleVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toDoubleVectorValue() }
                val distance = DoubleVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<DoubleArray>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            is FloatVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toFloatVectorValue() }
                val distance = FloatVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<FloatArray>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            is LongVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toLongVectorValue() }
                val distance = LongVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<LongArray>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            is IntVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toIntVectorValue() }
                val distance = IntVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<IntArray>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            is BooleanVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toBooleanVectorValue() }
                val distance = BoolVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<BitSet>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            is ComplexVectorColumnType -> {
                val query = knn.queryList.map { q -> q.toComplexVectorValue() }
                val distance = ComplexVectorDistance.valueOf(knn.distance.name)
                KnnPredicate(column = column as ColumnDef<Array<Complex>>, k = knn.k, query = query, weights = weights, distance = distance)
            }
            else -> throw QueryException.QuerySyntaxException("A kNN predicate does not contain a valid query vector!")
        }
    }

    /**
     * Parses and binds the projection part of a GRPC [CottontailGrpc.Query]
     *
     * @param involvedEntities The list of [Entity] objects involved in the projection.
     * @param projection The [CottontailGrpc.Projection] object.
     *
     * @return The resulting [Projection].
     */
    private fun parseAndBindProjection(entity: Entity, projection: CottontailGrpc.Projection): Projection = try {
        val availableColumns = entity.allColumns()
        val requestedColumns = mutableListOf<ColumnDef<*>>()

        val fields = projection.attributesMap.map { (expr, alias) ->
            /* Fetch columns that match field and add them to list of requested columns */
            val field = expr.normalizeColumnName(entity)
            availableColumns.filter { field.doesNameMatch(it.name) }.let { requestedColumns.addAll(it) }

            /* Return field to alias mapping. */
            field to if (alias.isEmpty()) {
                null
            } else {
                alias
            }
        }.toMap()

        Projection(type = ProjectionType.valueOf(projection.op.name), columns = requestedColumns.distinct().toTypedArray(), fields = fields)
    } catch (e: java.lang.IllegalArgumentException) {
        throw QueryException.QuerySyntaxException("The query lacks a valid SELECT-clause (projection): ${projection.op} is not supported.")
    }
}