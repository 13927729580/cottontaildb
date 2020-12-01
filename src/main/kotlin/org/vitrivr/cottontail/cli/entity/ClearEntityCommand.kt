package org.vitrivr.cottontail.cli.entity

import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.prompt
import org.vitrivr.cottontail.grpc.CottonDMLGrpc
import org.vitrivr.cottontail.grpc.CottontailGrpc
import org.vitrivr.cottontail.server.grpc.helper.protoFrom
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

/**
 * Command to clear [org.vitrivr.cottontail.database.entity.Entity], i.e., delete all data from it
 *
 * @author Ralph Gasser
 * @version 1.0
 */
@ExperimentalTime
class ClearEntityCommand(private val dmlStub: CottonDMLGrpc.CottonDMLBlockingStub) : AbstractEntityCommand(name = "clear", help = "Clears the given entity, i.e., deletes all data it contains. Usage: entity clear <schema>.<entity>") {

    private val confirm by option().prompt(text = "Do you really want to delete all data from entity ${this.entityName} (y/N)?")

    override fun exec() {
        if (this.confirm.toLowerCase() == "y") {
            val time = measureTime {
                this.dmlStub.delete(CottontailGrpc.DeleteMessage.newBuilder().setFrom(this.entityName.protoFrom()).build())
            }
            println("Successfully cleared entity ${this.entityName} (took $time).")
        } else {
            println("Clear entity aborted.")
        }
    }
}