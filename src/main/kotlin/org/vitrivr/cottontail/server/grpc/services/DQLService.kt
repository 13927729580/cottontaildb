package org.vitrivr.cottontail.server.grpc.services

import com.google.protobuf.Empty
import io.grpc.Status
import io.grpc.stub.StreamObserver
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.catalogue.Catalogue
import org.vitrivr.cottontail.database.locking.DeadlockException
import org.vitrivr.cottontail.database.queries.planning.CottontailQueryPlanner
import org.vitrivr.cottontail.database.queries.planning.rules.logical.DeferredFetchAfterFilterRewriteRule
import org.vitrivr.cottontail.database.queries.planning.rules.logical.DeferredFetchAfterKnnRewriteRule
import org.vitrivr.cottontail.database.queries.planning.rules.logical.LeftConjunctionRewriteRule
import org.vitrivr.cottontail.database.queries.planning.rules.logical.RightConjunctionRewriteRule
import org.vitrivr.cottontail.database.queries.planning.rules.physical.implementation.*
import org.vitrivr.cottontail.database.queries.planning.rules.physical.index.KnnIndexScanRule
import org.vitrivr.cottontail.database.queries.planning.rules.physical.pushdown.CountPushdownRule
import org.vitrivr.cottontail.execution.TransactionManager
import org.vitrivr.cottontail.execution.exceptions.ExecutionException
import org.vitrivr.cottontail.execution.operators.system.ExplainQueryOperator
import org.vitrivr.cottontail.grpc.CottontailGrpc
import org.vitrivr.cottontail.grpc.DQLGrpc
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.exceptions.TransactionException
import org.vitrivr.cottontail.server.grpc.helper.GrpcQueryBinder
import org.vitrivr.cottontail.server.grpc.operators.SpoolerSinkOperator
import java.util.*
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

/**
 * Implementation of [DQLGrpc.DQLImplBase], the gRPC endpoint for querying data in Cottontail DB.
 *
 * @author Ralph Gasser
 * @version 1.4.1
 */
@ExperimentalTime
class DQLService(val catalogue: Catalogue, override val manager: TransactionManager) : DQLGrpc.DQLImplBase(), TransactionService {

    /** Logger used for logging the output. */
    companion object {
        private val LOGGER = LoggerFactory.getLogger(DQLService::class.java)
    }

    /** [GrpcQueryBinder] used to generate [org.vitrivr.cottontail.database.queries.planning.nodes.logical.LogicalNodeExpression] tree from a gRPC query. */
    private val binder = GrpcQueryBinder(catalogue = this@DQLService.catalogue)

    /** [CottontailQueryPlanner] used to generate execution plans from query definitions. */
    private val planner = CottontailQueryPlanner(
        logicalRewriteRules = listOf(
            LeftConjunctionRewriteRule,
            RightConjunctionRewriteRule,
            DeferredFetchAfterFilterRewriteRule,
            DeferredFetchAfterKnnRewriteRule
        ),
        physicalRewriteRules = listOf(
            KnnIndexScanRule,
            CountPushdownRule,
            EntityScanImplementationRule,
            FilterImplementationRule,
            KnnImplementationRule,
            LimitImplementationRule,
            ProjectionImplementationRule,
            FetchImplementationRule
        )
    )

    /**
     * gRPC endpoint for executing queries.
     */
    override fun query(request: CottontailGrpc.QueryMessage, responseObserver: StreamObserver<CottontailGrpc.QueryResponseMessage>) = try {
        this.withTransactionContext(request.txId) { tx, q ->
            try {
                /* Start query execution. */
                val totalDuration = measureTime {
                    /* Bind query and create logical plan. */
                    val bindTimedValue = measureTimedValue {
                        this.binder.parseAndBindQuery(request.query, tx)
                    }

                    LOGGER.trace(formatMessage(tx, q, "Parsing & binding query took ${bindTimedValue.duration}."))

                    /* Plan query and create execution plan. */
                    val planTimedValue = measureTimedValue {
                        val candidates = this.planner.plan(bindTimedValue.value)
                        val operator = candidates.minByOrNull { it.totalCost }!!.toOperator(this.manager)
                        SpoolerSinkOperator(operator, q, 0, responseObserver)
                    }

                    LOGGER.trace(formatMessage(tx, q, "Planning query took ${planTimedValue.duration}."))

                    /* Execute query in transaction context. */
                    tx.execute(planTimedValue.value)
                }

                /* Finalize transaction. */
                LOGGER.trace(formatMessage(tx, q, "Executing query took ${totalDuration}."))
                responseObserver.onCompleted()
            } catch (e: QueryException.QuerySyntaxException) {
                val message = formatMessage(tx, q, "Could not execute query due to syntax error: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(message).asException())
            } catch (e: QueryException.QueryBindException) {
                val message = formatMessage(tx, q, "Could not execute query because DBO could not be found: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(message).asException())
            } catch (e: QueryException.QueryPlannerException) {
                val message = formatMessage(tx, q, "Could not execute query because of an error during query planning: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INTERNAL.withDescription(message).asException())
            } catch (e: DeadlockException) {
                val message = formatMessage(tx, q, "Could not execute query due to deadlock with other transaction: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.ABORTED.withDescription(message).asException())
            } catch (e: ExecutionException) {
                val message = formatMessage(tx, q, "Could not execute query due to an unhandled execution error: ${e.message}")
                LOGGER.error(message, e)
                responseObserver.onError(Status.INTERNAL.withDescription(message).asException())
            } catch (e: Throwable) {
                val message = formatMessage(tx, q, "Could not execute query due to an unhandled error: ${e.message}")
                LOGGER.error(message, e)
                responseObserver.onError(Status.UNKNOWN.withDescription(message).asException())
            }
        }
    } catch (e: TransactionException.TransactionNotFoundException) {
        val message = "Execution failed because transaction ${request.txId.value} could not be resumed."
        LOGGER.info(message)
        responseObserver.onError(Status.FAILED_PRECONDITION.withDescription(message).asException())
    }

    /**
     * gRPC endpoint for explaining queries.
     */
    override fun explain(request: CottontailGrpc.QueryMessage, responseObserver: StreamObserver<CottontailGrpc.QueryResponseMessage>) = try {
        this.withTransactionContext(request.txId) { tx, q ->
            try {
                /* Start query execution. */
                val totalDuration = measureTime {
                    /* Bind query and create logical plan. */
                    val bindTimedValue = measureTimedValue {
                        this.binder.parseAndBindQuery(request.query, tx)
                    }

                    LOGGER.trace(formatMessage(tx, q, "Parsing & binding query took ${bindTimedValue.duration}."))

                    /* Plan query and create execution plan. */
                    val planTimedValue = measureTimedValue {
                        val candidates = this.planner.plan(bindTimedValue.value)
                        SpoolerSinkOperator(ExplainQueryOperator(candidates), q, 0, responseObserver)
                    }
                    LOGGER.trace(formatMessage(tx, q, "Planning query took ${planTimedValue.duration}."))

                    /* Execute query in transaction context. */
                    tx.execute(planTimedValue.value)
                }

                LOGGER.trace(formatMessage(tx, q, "Explaining query took ${totalDuration}."))
                responseObserver.onCompleted()
            } catch (e: QueryException.QuerySyntaxException) {
                val message = formatMessage(tx, q, "Could not explain query due to syntax error: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(message).asException())
            } catch (e: QueryException.QueryBindException) {
                val message = formatMessage(tx, q, "Could not explain query because DBO could not be found: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(message).asException())
            } catch (e: QueryException.QueryPlannerException) {
                val message = formatMessage(tx, q, "Could not execute query because of an error during query planning: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.INTERNAL.withDescription(message).asException())
            } catch (e: DeadlockException) {
                val message = formatMessage(tx, q, "Could not explain query due to deadlock with other transaction: ${e.message}")
                LOGGER.info(message)
                responseObserver.onError(Status.ABORTED.withDescription(message).asException())
            } catch (e: ExecutionException) {
                val message = formatMessage(tx, q, "Could not explain query due to an unhandled execution error: ${e.message}")
                LOGGER.error(message, e)
                responseObserver.onError(Status.INTERNAL.withDescription(message).asException())
            } catch (e: Throwable) {
                val message = formatMessage(tx, q, "Could not explain query due to an unhandled error: ${e.message}")
                LOGGER.error(message, e)
                responseObserver.onError(Status.UNKNOWN.withDescription(message).asException())
            }
        }
    } catch (e: TransactionException.TransactionNotFoundException) {
        val message = "Execution failed because transaction ${request.txId.value} could not be resumed."
        LOGGER.info(message)
        responseObserver.onError(Status.FAILED_PRECONDITION.withDescription(message).asException())
    }

    /**
     * gRPC endpoint for handling PING requests.
     */
    override fun ping(request: Empty, responseObserver: StreamObserver<Empty>) {
        responseObserver.onCompleted()
    }
}
