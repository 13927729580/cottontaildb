package ch.unibas.dmi.dbis.cottontail.calcite.adapter

import ch.unibas.dmi.dbis.cottontail.Cottontail
import ch.unibas.dmi.dbis.cottontail.database.catalogue.Catalogue

import org.apache.calcite.schema.Schema
import org.apache.calcite.schema.SchemaFactory
import org.apache.calcite.schema.SchemaPlus

/**
 * Part of Cottontail DB's Apache Calcite adapter. Produces instances of [CottontailCatalogue] objects.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal object CottontailCatalogueFactory : SchemaFactory {
    /**
     * Returns singleton instance of [CottontailCatalogue].
     *
     * @param parentSchema Always null.
     * @param name Always null.
     * @param operand Always null.
     */
    override fun create(parentSchema: SchemaPlus?, name: String?, operand: MutableMap<String, Any>?): Schema = CottontailCatalogue(Cottontail.CATALOGUE!!)
}