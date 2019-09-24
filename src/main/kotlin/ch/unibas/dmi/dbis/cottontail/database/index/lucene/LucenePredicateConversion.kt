package ch.unibas.dmi.dbis.cottontail.database.index.lucene

import ch.unibas.dmi.dbis.cottontail.database.queries.*
import ch.unibas.dmi.dbis.cottontail.model.exceptions.QueryException
import ch.unibas.dmi.dbis.cottontail.model.values.StringValue
import org.apache.lucene.analysis.standard.StandardAnalyzer

import org.apache.lucene.index.Term
import org.apache.lucene.queryparser.flexible.standard.QueryParserUtil
import org.apache.lucene.search.*

/**
 * Converts an [AtomicBooleanPredicate] to a [Query] supported by Apache Lucene.
 */
fun AtomicBooleanPredicate<*>.toLuceneQuery(): Query = if (this.values.first() is StringValue) {
    val column = this.columns.first()
    val value = (this.values.first() as StringValue).value
    when (this.operator){
        ComparisonOperator.LIKE -> QueryParserUtil.parse(arrayOf(value), arrayOf("${column.name}_txt"), StandardAnalyzer())
        ComparisonOperator.EQUAL -> TermQuery(Term("${column.name}_str", value))
        else -> throw QueryException("Only EQUALS and LIKE queries can be mapped to Apache Lucene!")
    }
} else {
    throw QueryException("Only String values can be handled by Apache Lucene!")
}

/**
 * Converts a [CompoundBooleanPredicate] to a [Query] supported by Apache Lucene.
 */
fun CompoundBooleanPredicate.toLuceneQuery(): Query {
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

