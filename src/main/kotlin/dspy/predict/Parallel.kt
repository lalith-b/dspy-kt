package dspy.predict

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.utils.Settings
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * A utility class for parallel, multi-threaded execution of (module, example) pairs.
 *
 * Supports various example formats (Example, dict, list), robust error handling,
 * optional progress tracking, and can optionally return failed examples and exceptions.
 *
 * Port of `dspy/predict/parallel.py`
 *
 * Uses Kotlin coroutines instead of Python threading.
 */
class Parallel(
    numThreads: Int? = null,
    maxErrors: Int? = null,
    val accessExamples: Boolean = true,
    val returnFailedExamples: Boolean = false,
    val provideTraceback: Boolean? = null,
    val disableProgressBar: Boolean = false,
    val timeout: Int = 120,
    val stragglerLimit: Int = 3,
) {
    val numThreads: Int = numThreads ?: Settings.numThreads
    val maxErrors: Int = maxErrors ?: (Settings.maxErrors ?: Int.MAX_VALUE)

    private val errorLock = Mutex()
    private var errorCount = 0
    val failedExamples: MutableList<Any> = mutableListOf()
    val exceptions: MutableList<Throwable> = mutableListOf()

    /**
     * Forward pass: execute the processing function over the execution pairs.
     */
    suspend fun forward(
        execPairs: List<Pair<Any, Any>>,
        numThreads: Int? = null,
    ): Any {
        val actualNumThreads = numThreads ?: this.numThreads
        val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        val semaphore = Semaphore(actualNumThreads)

        data class AsyncResult(
            val success: Boolean = false,
            val value: Any? = null,
            val exception: Throwable? = null,
        )

        val results = MutableList<AsyncResult>(execPairs.size) { AsyncResult() }

        val jobs = execPairs.mapIndexed { index, pair ->
            scope.async {
                semaphore.withLock {
                    try {
                        val result = processPair(pair)
                        results[index] = AsyncResult(success = true, value = result)
                    } catch (e: Exception) {
                        errorLock.withLock {
                            errorCount++
                        }
                        results[index] = AsyncResult(success = false, exception = e)
                    }
                }
            }
        }

        // Wait for all jobs with timeout
        try {
            withTimeout(timeout * 1000L) {
                jobs.forEach { it.await() }
            }
        } catch (e: TimeoutCancellationException) {
            jobs.forEach { it.cancel() }
        }

        // Collect results
        val collectedResults = mutableListOf<Any>()
        val failedIndices = mutableSetOf<Int>()
        val exceptionsMap = mutableMapOf<Int, Throwable>()

        results.forEachIndexed { index, result ->
            if (result.success) {
                collectedResults.add(result.value!!)
            } else {
                failedIndices.add(index)
                result.exception?.let { exceptionsMap[index] = it }
            }
        }

        // Populate failed examples and exceptions
        if (returnFailedExamples) {
            failedIndices.sorted().forEach { idx ->
                if (idx < execPairs.size) {
                    failedExamples.add(execPairs[idx].second)
                    exceptionsMap[idx]?.let { exceptions.add(it) }
                }
            }
            return Triple(collectedResults, failedExamples.toList(), exceptions.toList())
        }

        return collectedResults
    }

    private suspend fun processPair(pair: Pair<Any, Any>): Any {
        val module = pair.first
        val example = pair.second

        return when (example) {
            is Example -> {
                val mod = module as Module
                if (accessExamples) {
                    mod.invoke(example.inputs().toMap())
                } else {
                    mod.invoke(mapOf("example" to example))
                }
            }
            is Map<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                (module as Module).invoke(example as Map<String, Any?>)
            }
            is List<*> -> {
                if (module is Parallel) {
                    @Suppress("UNCHECKED_CAST")
                    module.forward(example as List<Pair<Any, Any>>)
                } else {
                    throw NotImplementedError("List examples not supported for non-Parallel modules")
                }
            }
            else -> {
                throw IllegalArgumentException(
                    "Invalid example type: ${example::class.simpleName}, " +
                        "only supported types are Example, Map, and List"
                )
            }
        }
    }

    /**
     * Call the forward method.
     */
    suspend operator fun invoke(
        execPairs: List<Pair<Any, Any>>,
        numThreads: Int? = null,
    ): Any {
        return forward(execPairs, numThreads)
    }
}

/**
 * Simple semaphore implementation using Mutex.
 */
class Semaphore(private val permits: Int) {
    private val mutex = Mutex()
    private var available = permits

    suspend fun withLock(block: suspend () -> Unit) {
        mutex.lock()
        while (available <= 0) {
            mutex.unlock()
            delay(10)
            mutex.lock()
        }
        available--
        mutex.unlock()
        try {
            block()
        } finally {
            mutex.lock()
            available++
            mutex.unlock()
        }
    }
}
