package ch.unibas.dmi.dbis.cottontail.database.index.lucene

import ch.unibas.dmi.dbis.cottontail.database.queries.*
import ch.unibas.dmi.dbis.cottontail.model.exceptions.QueryException
import ch.unibas.dmi.dbis.cottontail.model.values.StringValue

import org.apache.lucene.index.Term
import org.apache.lucene.search.*

/**
 * Converts an [AtomicBooleanPredicate] to a [Query] supported by Apache Lucene.
 */
internal fun AtomicBooleanPredicate<*>.toLuceneQuery(): Query = if (this.operator == ComparisonOperator.LIKE && this.values.first() is StringValue) {
    val column = this.columns.first()
    val value = (this.values.first().value as StringValue).value.replace("%","*").replace("_","?")
    if (value.contains("*") || value.contains("_")) {
        WildcardQuery(Term(column.name, value))
    } else if (value.contains(" ")) {
        val builder = PhraseQuery.Builder()
        value.split(" ").forEach { builder.add(Term(column.name, it)) }
        builder.build()
    } else if (value.matches(Regex("~([0-9](.[0-9]*)?)$"))) {
        FuzzyQuery(Term(column.name, value))
    } else {
        TermQuery(Term(column.name, value))
    }
} else {
    throw QueryException("Only LIKE queries with String values can be mapped to Apache Lucene!")
}

/**
 * Converts a [CompoundBooleanPredicate] to a [Query] supported by Apache Lucene.
 */
internal fun CompoundBooleanPredicate.toLuceneQuery(): Query {
    val clause = when (this.connector) {
        ConnectionOperator.AND -> BooleanClause.Occur.MUST
        ConnectionOperator.OR -> BooleanClause.Occur.SHOULD
    }
    val left = when(this.p1) {
        is AtomicBooleanPredicate<*> -> this.p1.toLuceneQuery()
        is CompoundBooleanPredicate -> this.p1.toLuceneQuery()
    }
    val right = when(this.p2) {
        is AtomicBooleanPredicate<*> -> this.p2.toLuceneQuery()
        is CompoundBooleanPredicate -> this.p2.toLuceneQuery()
    }

    val builder = BooleanQuery.Builder()
    builder.add(left, clause)
    builder.add(right, clause)
    return builder.build()
}

