package dspy.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.slf4j.LoggerFactory

private val logger = LoggerFactory.getLogger(Unbatchify::class.java)

/**
 * Converts a batch-processing function into a single-item interface by batching
 * requests internally and processing them together.
 *
 * Uses Kotlin coroutines (Channel + CompletableDeferred) instead of Java threading APIs.
 *
 * Port of `dspy/utils/unbatchify.py`
 *
 * Example:
 * ```kotlin
 * val unbatchify = Unbatchify(
 *     batchFn = { items -> items.map { it * 2 } },
 *     maxBatchSize = 32,
 *     maxWaitTimeMs = 100L
 * )
 * val result = unbatchify.process(5)
 * unbatchify.close()
 * ```
 */
class Unbatchify<T : Any, R : Any>(
    private val batchFn: suspend (List<T>) -> List<R>,
    private val maxBatchSize: Int = 32,
    private val maxWaitTimeMs: Long = 100L,
) : AutoCloseable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val channel = Channel<Pair<T, CompletableDeferred<R>>>(Channel.UNLIMITED)
    @Volatile
    private var isClosed = false

    init {
        scope.launch { workerLoop() }
    }

    /**
     * Submit a single input and get the corresponding output.
     */
    suspend fun process(inputItem: T): R {
        require(!isClosed) { "Unbatchify is closed" }
        val deferred = CompletableDeferred<R>()
        channel.send(inputItem to deferred)
        return try {
            deferred.await()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
    }

    suspend fun aprocess(inputItem: T): R = process(inputItem)

    private suspend fun workerLoop() {
        while (scope.isActive && !isClosed) {
            val batch = mutableListOf<T>()
            val deferreds = mutableListOf<CompletableDeferred<R>>()
            val startTime = System.currentTimeMillis()

            while (batch.size < maxBatchSize && (System.currentTimeMillis() - startTime) < maxWaitTimeMs) {
                val remaining = maxWaitTimeMs - (System.currentTimeMillis() - startTime)
                if (remaining <= 0L) break

                val received = channel.tryReceive()
                if (received.isSuccess) {
                    val (input, deferred) = received.getOrNull()!!
                    batch.add(input)
                    deferreds.add(deferred)
                } else {
                    delay(remaining.coerceAtMost(10L))
                }
            }

            if (batch.isNotEmpty()) {
                runCatching { batchFn(batch) }.fold(
                    onSuccess = { outputs ->
                        for (i in batch.indices) {
                            if (i < outputs.size) {
                                deferreds[i].complete(outputs[i])
                            } else {
                                deferreds[i].completeExceptionally(
                                    RuntimeException("Batch returned fewer outputs than inputs")
                                )
                            }
                        }
                    },
                    onFailure = { e ->
                        for (d in deferreds) {
                            d.completeExceptionally(e)
                        }
                    }
                )
            } else {
                delay(10)
            }
        }
        drainRemaining()
        logger.info("Unbatchify worker coroutine has been terminated.")
    }

    private fun drainRemaining() {
        while (true) {
            val result = channel.tryReceive()
            if (result.isSuccess) {
                result.getOrNull()?.second?.completeExceptionally(
                    RuntimeException("Unbatchify is closed")
                )
            } else {
                break
            }
        }
    }

    override fun close() {
        if (!isClosed) {
            isClosed = true
            scope.cancel()
            channel.close()
        }
    }
}
