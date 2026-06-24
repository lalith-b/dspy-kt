package dspy.streaming

/**
 * Streaming listener for LM responses.
 */
interface StreamingListener {
    /**
     * Called when a new token is received.
     */
    fun onToken(token: String)

    /**
     * Called when streaming is complete.
     */
    fun onComplete(fullResponse: String)

    /**
     * Called when an error occurs.
     */
    fun onError(error: String)
}

/**
 * Base class for streaming responses.
 */
abstract class StreamResponse {
    var tokens: MutableList<String> = mutableListOf()
    var isComplete: Boolean = false
    var error: String? = null

    abstract fun nextToken(): String?
}

/**
 * Token stream implementation.
 */
class TokenStream(
    private val initialTokens: List<String>,
    private val listeners: List<StreamingListener> = emptyList()
) : StreamResponse() {
    private var currentIndex = 0

    override fun nextToken(): String? {
        if (currentIndex >= initialTokens.size) {
            isComplete = true
            listeners.forEach { it.onComplete(initialTokens.joinToString("")) }
            return null
        }
        val token = initialTokens[currentIndex++]
        tokens.add(token)
        listeners.forEach { it.onToken(token) }
        return token
    }
}

/**
 * Async token stream.
 */
class AsyncTokenStream : StreamResponse() {
    private val queue = java.util.concurrent.LinkedBlockingQueue<String>()
    private var closed = false

    fun send(token: String) {
        if (!closed) queue.offer(token)
    }

    fun close() {
        closed = true
    }

    override fun nextToken(): String? {
        val token = queue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS)
        if (token != null) {
            tokens.add(token)
            return token
        }
        if (closed) {
            isComplete = true
        }
        return null
    }
}
