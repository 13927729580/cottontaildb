package ch.unibas.dmi.dbis.cottontail.database.index

import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.index.hash.NonUniqueHashIndex
import ch.unibas.dmi.dbis.cottontail.database.index.hash.UniqueHashIndex
import ch.unibas.dmi.dbis.cottontail.database.index.lsh.LSHIndex
import ch.unibas.dmi.dbis.cottontail.database.index.lucene.LuceneIndex
import ch.unibas.dmi.dbis.cottontail.database.index.vaplus.VAPlusIndex
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.utilities.name.Name

enum class IndexType(val inexact: Boolean) {
    HASH_UQ(false), /* A hash based index with unique values. */
    HASH(false), /* A hash based index. */
    BTREE(false), /* A BTree based index. */
    LUCENE(false), /* A Lucene based index (fulltext search). */
    VAF(false), /* A VA+ file based index (for exact kNN lookup). */
    PQ(true), /* A product quantization based index (for approximate kNN lookup). */
    SH(true), /* A spectral hashing based index (for approximate kNN lookup). */
    LSH(true); /* A locality sensitive hashing based index (for approximate kNN lookup). */

    /**
     * Opens an index of this [IndexType] using the given name and [Entity].
     *
     * @param name Name of the [Index]
     * @param entity The [Entity] the desired [Index] belongs to.
     */
    fun open(name: Name, entity: Entity, columns: Array<ColumnDef<*>>): Index = when (this) {
        HASH_UQ -> UniqueHashIndex(name, entity, columns)
        HASH -> NonUniqueHashIndex(name, entity, columns)
        LUCENE -> LuceneIndex(name, entity, columns)
        VAF -> VAPlusIndex(name, entity, columns)
        LSH -> LSHIndex<Any>(name, entity, columns, null)
        else -> TODO()
    }

    /**
     * Creates an index of this [IndexType] using the given name and [Entity].
     *
     * @param name Name of the [Index]
     * @param entity The [Entity] the desired [Index] belongs to.
     * @param columns The [ColumnDef] for which to create the [Index]
     * @param params Additions configuration params.
     */
    fun create(name: Name, entity: Entity, columns: Array<ColumnDef<*>>, params: Map<String, String> = emptyMap()) = when (this) {
        HASH_UQ -> UniqueHashIndex(name, entity, columns)
        HASH -> NonUniqueHashIndex(name, entity, columns)
        LUCENE -> LuceneIndex(name, entity, columns)
        VAF -> VAPlusIndex(name, entity, columns)
        LSH -> LSHIndex<Any>(name, entity, columns, params)
        else -> TODO()
    }

}