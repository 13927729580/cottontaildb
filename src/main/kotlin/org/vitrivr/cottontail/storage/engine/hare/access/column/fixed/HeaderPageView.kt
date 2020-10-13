package org.vitrivr.cottontail.storage.engine.hare.access.column.fixed

import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.TupleId

import org.vitrivr.cottontail.storage.engine.hare.DataCorruptionException
import org.vitrivr.cottontail.storage.engine.hare.basics.Page
import org.vitrivr.cottontail.storage.engine.hare.disk.DataPage
import org.vitrivr.cottontail.storage.engine.hare.views.AbstractPageView
import org.vitrivr.cottontail.storage.engine.hare.views.ViewConstants

/**
 * The [HeaderPageView] of this [FixedHareColumnFile]. The [HeaderPageView] is located on the first
 * [DataPage] in the [FixedHareColumnFile] file.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class HeaderPageView : AbstractPageView() {

    companion object {
        /** The offset into the [HeaderPageView]'s header to obtain the type of column. */
        const val HEADER_OFFSET_TYPE = 4

        /** The offset into the [HeaderPageView]'s header to obtain the logical size of column. */
        const val HEADER_OFFSET_LSIZE = 8

        /** The offset into the [HeaderPageView]'s header to obtain the physical size of column. */
        const val HEADER_OFFSET_PSIZE = 12

        /** The offset into the [HeaderPageView]'s header to obtain the file flags. */
        const val HEADER_OFFSET_FLAGS = 16

        /** The offset into the [HeaderPageView]'s header to obtain the number of entries. */
        const val HEADER_OFFSET_COUNT = 24

        /** The offset into the [HeaderPageView]'s header to obtain the number of deleted entries. */
        const val HEADER_OFFSET_DELETED = 32

        /** The offset into the [HeaderPageView]'s header to obtain the maximum [TupleId]. */
        const val HEADER_OFFSET_MAXTID = 40

        /** Bit Masks (for flags) */

        /** Mask for 'NULLABLE' bit in this [HeaderPageView]. */
        const val HEADER_MASK_NULLABLE = 1L shl 0
    }

    /** The [pageTypeIdentifier] for the [HeaderPageView]. */
    override val pageTypeIdentifier: Int
        get() = ViewConstants.HEADER_FIXED_COLUMN

    /** The [ColumnType] held by this [FixedHareColumnFile]. */
    val type: ColumnType<*>
        get() = ColumnType.forOrdinal(this.page?.getInt(HEADER_OFFSET_TYPE)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction."))

    /** The logical size of the [ColumnDef] held by this [FixedHareColumnFile]. */
    val size: Int
        get() = this.page?.getInt(HEADER_OFFSET_LSIZE)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")

    /** The physical size of an entry in bytes. */
    val entrySize: Int
        get() = this.page?.getInt(HEADER_OFFSET_PSIZE)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")

    /** Special flags set for this [FixedHareColumnFile], such as, nullability. */
    val flags: Long
        get() = this.page?.getLong(HEADER_OFFSET_FLAGS)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")

    /** True if this [FixedHareColumnFile] supports null values. */
    val nullable: Boolean
        get() = ((this.flags and HEADER_MASK_NULLABLE) > 0L)

    /** The total number of entries in this [FixedHareColumnFile]. */
    var count: Long
        get() = this.page?.getLong(HEADER_OFFSET_COUNT)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")
        set(v) {
            check(this.page != null) { "This HeaderPageView is not wrapping any page and can therefore not be used for interaction." }
            this.page!!.putLong(HEADER_OFFSET_COUNT, v)
        }

    /** The number of deleted entries in this [FixedHareColumnFile]. */
    var deleted: Long
        get() = this.page?.getLong(HEADER_OFFSET_DELETED)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")
        set(v) {
            check(this.page != null) { "This HeaderPageView is not wrapping any page and can therefore not be used for interaction." }
            this.page!!.putLong(HEADER_OFFSET_DELETED, v)
        }

    /** The number of deleted entries in this [FixedHareColumnFile]. */
    var maxTupleId: TupleId
        get() = this.page?.getLong(HEADER_OFFSET_MAXTID)
                ?: throw IllegalStateException("This HeaderPageView is not wrapping any page and can therefore not be used for interaction.")
        set(v) {
            check(this.page != null) { "This HeaderPageView is not wrapping any page and can therefore not be used for interaction." }
            this.page!!.putLong(HEADER_OFFSET_MAXTID, v)
        }

    /**
     * Wraps a [Page] for usage as a [HeaderPageView].
     *
     * @param page [Page] that should be wrapped.
     */
    override fun wrap(page: Page): HeaderPageView {
        super.wrap(page)
        require(this.count >= 0) { DataCorruptionException("Negative number of entries in HARE variable length column file.") }
        require(this.deleted >= 0) { DataCorruptionException("Negative number of deleted entries in HARE variable length column file.") }
        return this
    }

    /**
     * Using this method is prohibited; throws a [UnsupportedOperationException]
     *
     * @param page [Page] that should be wrapped.
     */
    @Deprecated("Usage of initializeAndWrap() without specifying a ColumnDef is prohibited for HeaderPageView.")
    override fun initializeAndWrap(page: Page): AbstractPageView {
        throw UnsupportedOperationException("Usage of initializeAndWrap() without specifying a ColumnDef is prohibited for HeaderPageView.")
    }

    /**
     * Initializes and wraps a [Page] for usage as a [HeaderPageView].
     *
     * @param page [Page] that should be wrapped.
     * @param columnDef The [ColumnDef] of the [FixedHareColumnFile] this [HeaderPageView] belongs to.
     */
    fun initializeAndWrap(page: Page, columnDef: ColumnDef<*>): HeaderPageView {
        super.initializeAndWrap(page)
        page.putInt(HEADER_OFFSET_TYPE, columnDef.type.ordinal)                                            /* 4: Type of column. See ColumnDef.forOrdinal() */
        page.putInt(HEADER_OFFSET_LSIZE, columnDef.logicalSize)                                            /* 8: Logical size of column (for structured data types). */
        page.putInt(HEADER_OFFSET_PSIZE, columnDef.serializer.physicalSize
                + FixedHareColumnFile.ENTRY_HEADER_SIZE)                                                   /* 12: Physical size of a column entry in bytes. */
        page.putLong(18, if (columnDef.nullable) {                                                   /* 16: Column flags; 64 bits, one bit reserved. */
            (0L or HEADER_MASK_NULLABLE)
        } else {
            0L
        })
        page.putLong(HEADER_OFFSET_COUNT, 0L)                                                        /* 24: Number of entries (count) in column. */
        page.putLong(HEADER_OFFSET_DELETED, 0L)                                                      /* 32: Number of deleted entries (count) in column. */
        page.putLong(HEADER_OFFSET_MAXTID, 0L)                                                       /* 40: Number of deleted entries (count) in column. */
        return this
    }
}