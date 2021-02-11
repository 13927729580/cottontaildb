package org.vitrivr.cottontail.database.entity


/**
 * A class containing statistics regarding an [DefaultEntity]. This class acts as snapshot of the [DefaultEntity]. It may become invalid
 * the moment is was created, as other processes make changes to the [DefaultEntity].
 *
 * @author Ralph Gasser
 * @version 1.0
 */
data class EntityStatistics(val columns: Int, val rows: Long, val maxTupleId: Long)