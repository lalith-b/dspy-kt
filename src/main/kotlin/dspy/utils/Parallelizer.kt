package dspy.utils

import org.slf4j.LoggerFactory
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

private val logger = LoggerFactory.getLogger(ParallelExecutor::class.java)

/**
 * Parallel executor for DSPy evaluation/metric computation.
 *
 * Offers isolation between the tasks (dspy.settings) irrespective of whether numThreads == 1 or > 1.
 * Handles straggler timeouts.
 *
 * Port of `dspy/utils/parallelizer.py`
 */
class ParallelExecutor(
    numThreads: Int? = null,
    maxErrors: Int? = null,
    private val disableProgressBar: Boolean = false,
    private val provideTraceback: Boolean? = null,
    private val compareResults: Boolean = false,
    private val timeout: Int = 120,
    private val stragglerLimit: Int = 3,
) {
    private val numThreads: Int
    private val maxErrors: Int
    private val _provideTraceback: Boolean
    private val errorCount: AtomicInteger = AtomicInteger(0)
    private val cancelled: AtomicBoolean = AtomicBoolean(false)
    private val failedIndices: MutableList<Int> = CopyOnWriteArrayList()
    private val exceptionsMap: MutableMap<Int, Throwable> = CopyOnWriteArrayList<MutableMap<Int, Throwable>>().firstOrNull() ?: CopyOnWriteArrayList<Throwable>().let { mutableMapOf<Int, Throwable>() }

    init {
        this.numThreads = numThreads ?: Settings.numThreads
        this.maxErrors = maxErrors ?: Settings.maxErrors ?: Int.MAX_VALUE
        this._provideTraceback = provideTraceback ?: (Settings.get("provide_traceback") as? Boolean) ?: false
    }

    /**
     * Execute a function over a data list, either sequentially or in parallel.
     */
    fun <T : Any> execute(function: (T) -> Any?, data: List<T>): List<Any?> {
        val wrapped: (T) -> Any? = { item ->
            if (cancelled.get()) null
            else runCatching { function(item) }.getOrElse { e ->
                val prev = errorCount.incrementAndGet()
                if (prev >= maxErrors) {
                    cancelled.set(true)
                }
                if (_provideTraceback) {
                    logger.error("Error for $item: ${e.message}", e)
                } else {
                    logger.error("Error for $item: ${e.message}. Set provideTraceback=true for traceback.")
                }
                failedIndices.add(data.indexOf(item))
                null
            }
        }

        return if (numThreads == 1) {
            executeSequential(wrapped, data)
        } else {
            executeParallel(wrapped, data)
        }
    }

    private fun <T : Any> executeSequential(function: (T) -> Any?, data: List<T>): List<Any?> {
        val results = MutableList<Any?>(data.size) { null }
        var completed = 0

        for (idx in data.indices) {
            if (cancelled.get()) break

            val outcome = function(data[idx])
            processOutcome(results, idx, outcome)

            if (outcome != null) completed++
            if (!disableProgressBar) {
                reportProgress(completed, data.size)
            }
        }

        if (cancelled.get()) {
            logger.warn("Execution cancelled due to errors or interruption.")
            throw RuntimeException("Execution cancelled due to errors or interruption.")
        }

        return results
    }

    private fun <T : Any> executeParallel(function: (T) -> Any?, data: List<T>): List<Any?> {
        val results = MutableList<Any?>(data.size) { null }
        val latch = CountDownLatch(data.size)
        val completionLock = Object()

        for (idx in data.indices) {
            thread(start = true) {
                val item = data[idx]
                val outcome = try {
                    if (cancelled.get()) null else function(item)
                } catch (e: Exception) {
                    null
                }

                synchronized(completionLock) {
                    if (results[idx] == null) {
                        processOutcome(results, idx, outcome)
                    }
                    completionLock.notifyAll()
                }
                latch.countDown()
            }
        }

        var completed = 0
        var lastCompleted = 0

        while (completed < data.size && !cancelled.get()) {
            synchronized(completionLock) {
                completed = results.count { it != null }

                if (!disableProgressBar && completed > lastCompleted) {
                    reportProgress(completed, data.size)
                    lastCompleted = completed
                }

                if (completed < data.size) {
                    completionLock.wait(1000)
                }
            }
        }

        if (cancelled.get()) {
            logger.warn("Execution cancelled due to errors or interruption.")
            throw RuntimeException("Execution cancelled due to errors or interruption.")
        }

        return results
    }

    private fun processOutcome(results: MutableList<Any?>, idx: Int, outcome: Any?) {
        if (outcome is Throwable) {
            failedIndices.add(idx)
            exceptionsMap[idx] = outcome
        } else {
            results[idx] = outcome
        }
    }

    private fun reportProgress(completed: Int, total: Int) {
        if (compareResults && total > 0) {
            val pct = String.format("%.1f", 100.0 * completed / total)
            logger.debug("Processed $completed / $total examples (${pct}%)")
        } else {
            logger.debug("Processed $completed / $total examples")
        }
    }

    /**
     * Get the indices that failed during execution.
     */
    fun failedIndices(): List<Int> = failedIndices.toList()

    /**
     * Get the exceptions map from execution.
     */
    fun exceptionsMap(): Map<Int, Throwable> = exceptionsMap.toMap()

    /**
     * Get the current error count.
     */
    fun errorCount(): Int = errorCount.get()
}
