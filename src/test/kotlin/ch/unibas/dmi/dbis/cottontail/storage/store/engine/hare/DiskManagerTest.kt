package ch.unibas.dmi.dbis.cottontail.storage.store.engine.hare

import ch.unibas.dmi.dbis.cottontail.storage.basics.Units
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk.*

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach

import java.nio.file.Paths
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.RepeatedTest
import java.nio.ByteBuffer
import java.nio.file.Files
import java.util.*
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

class DiskManagerTest {
    val path = Paths.get("./test-db.hare")

    var manager: DiskManager? = null

    val random = SplittableRandom(System.currentTimeMillis())

    @BeforeEach
    fun beforeEach() {
        DiskManager.init(this.path)
        this.manager = DiskManager(this.path)
    }

    @AfterEach
    fun afterEach() {
        this.manager!!.close()
        Files.delete(this.path)
    }

    @Test
    fun testCreationAndLoading() {
        assertEquals(this.manager!!.pages, 0)
        assertEquals(this.manager!!.size, DiskManager.FILE_HEADER_SIZE_BYTES)
    }

    /**
     * Appends [Page]s of random bytes and checks, if those [Page]s' content remains the same after reading.
     */
    @RepeatedTest(5)
    fun testOutOfBounds() {
        val page = Page(ByteBuffer.allocateDirect(BufferPool.PAGE_MEMORY_SIZE))
        val data = this.initWithData(random.nextInt(65536))
        assertThrows(PageIdOutOfBoundException::class.java){ this.manager!!.read((data.size + 1 + random.nextInt()).toLong(), page) }
    }

    /**
     * Appends [Page]s of random bytes and checks, if those [Page]s' content remains the same after reading.
     */
    @ExperimentalTime
    @RepeatedTest(5)
    fun testAppendPage() {
        val data = this.initWithData(random.nextInt(65536))

        /* Check if data remains the same. */
        this.compareData(data)
    }

    /**
     * Appends [Page]s of random bytes and checks, if those [Page]s' content remains the same after reading.
     */
    @ExperimentalTime
    @RepeatedTest(5)
    fun testPersistence() {
        val data = this.initWithData(random.nextInt(65536))

        /** Close and re-open this DiskManager. */
        this.manager!!.close()
        this.manager = DiskManager(this.path)

        /* Check if data remains the same. */
        this.compareData(data)
    }

    /**
     * Updates [Page]s with random bytes and checks, if those [Page]s' content remains the same after reading.
     */
    @ExperimentalTime
    @RepeatedTest(5)
    fun testUpdatePage() {
        val page = Page(ByteBuffer.allocateDirect(BufferPool.PAGE_MEMORY_SIZE))
        val data = this.initWithData(random.nextInt(65536))

        val newData = Array(data.size) {
            val bytes = ByteArray(Page.Constants.PAGE_DATA_SIZE_BYTES)
            random.nextBytes(bytes)
            bytes
        }

        /* Update data with new data. */
        for (i in newData.indices) {
            this.manager!!.read(i.toLong(), page)

            assertFalse(page.dirty)

            page.putBytes(0, newData[i])

            assertTrue(page.dirty)

            this.manager!!.update(page)
            assertArrayEquals(newData[i], page.getBytes(0))
            assertEquals(i.toLong(), page.id)
            assertFalse(page.dirty)
        }

        /* Check if data remains the same. */
        this.compareData(newData)
    }

    /**
     * Compares the data stored in this [DiskManager] with the data provided as array of [ByteArray]s
     */
    @ExperimentalTime
    private fun compareData(ref: Array<ByteArray>) {
        val page = Page(ByteBuffer.allocateDirect(BufferPool.PAGE_MEMORY_SIZE))
        var readTime = Duration.ZERO
        for (i in ref.indices) {
            readTime += measureTime {
                this.manager!!.read(i.toLong(), page)
            }
            assertArrayEquals(ref[i], page.getBytes(0))
            assertEquals(i.toLong(), page.id)
            assertFalse(page.dirty)
        }
        println("Reading ${this.manager!!.size `in` Units.MEGABYTE} took $readTime (${(this.manager!!.size `in` Units.MEGABYTE).value / readTime.inSeconds} MB/s).")
    }

    /**
     * Initializes this [DiskManager] with random data.
     *
     * @param size The number of [Page]s to write.
     */
    private fun initWithData(size: Int) : Array<ByteArray> {
        val page = Page(ByteBuffer.allocateDirect(BufferPool.PAGE_MEMORY_SIZE))
        val data = Array(size) {
            val bytes = ByteArray(Page.Constants.PAGE_DATA_SIZE_BYTES)
            random.nextBytes(bytes)
            bytes
        }

        for (i in data.indices) {
            page.putBytes(0, data[i])

            assertTrue(page.dirty)

            this.manager!!.append(page)
            assertEquals(this.manager!!.pages, i+1L)
            assertEquals(this.manager!!.size.value.toLong(), DiskManager.FILE_HEADER_SIZE_BYTES + (i+1)*Page.Constants.PAGE_DATA_SIZE_BYTES)
            assertEquals(i.toLong(), page.id)
            assertFalse(page.dirty)
        }

        return data
    }
}