package dspy.clients

/**
 * LiteLLM integration utilities for DSPy.
 *
 * This module provides cached access to LiteLLM and applies DSPy-specific defaults.
 */
object LiteLLMClient {
    @Volatile
    private var litellmModule: Any? = null
    private var configured = false

    /**
     * Get the LiteLLM module, applying DSPy defaults once on first access.
     */
    @Synchronized
    fun getLiteLLM(feature: String): Any {
        if (litellmModule != null) return litellmModule!!
        // In Kotlin, LiteLLM would be used via JPython or an HTTP API
        throw NotImplementedError("LiteLLM integration requires JPython bridge or HTTP API. feature=$feature")
    }

    /**
     * Check if an exception is LiteLLM's context-window error.
     */
    fun isLiteLLMContextWindowError(error: Exception): Boolean {
        return error.message?.contains("ContextWindowExceededError") == true ||
               error.message?.contains("context length") == true
    }
}
