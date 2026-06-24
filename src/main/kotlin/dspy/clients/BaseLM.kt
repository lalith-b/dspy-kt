package dspy.clients

import dspy.core.types.*
import dspy.utils.CallbackHandler
import dspy.utils.Settings
import kotlinx.serialization.json.Json
import kotlin.random.Random

/**
 * Base class for DSPy language models.
 *
 * Most users should use [LM], which is a [BaseLM] subclass.
 * For advanced use cases, such as custom language model backends, users can
 * subclass [BaseLM] and implement [forward].
 */
abstract class BaseLM(
    model: String,
    modelType: String = "chat",
    temperature: Double? = null,
    maxTokens: Int? = null,
    cache: Boolean = true,
    callbacks: List<CallbackHandler>? = null,
    numRetries: Int = 3,
    vararg extraKwargs: Pair<String, Any?>,
) {
    val model: String = model
    val modelType: String = modelType
    val cache: Boolean = cache
    val callbacks: MutableList<CallbackHandler> = (callbacks ?: emptyList()).toMutableList()
    val numRetries: Int = numRetries
    val kwargs: MutableMap<String, Any?> = mutableMapOf<String, Any?>().apply {
        put("temperature", temperature)
        put("maxTokens", maxTokens)
        putAll(extraKwargs)
    }
    val history: MutableList<Any> = mutableListOf()
    private var _warnedZeroTempRollout: Boolean = false

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        isLenient = true
    }

    // ===================
    // Support flags
    // ===================

    open val supportsFunctionCalling: Boolean = false
    open val supportsReasoning: Boolean = false
    open val supportsResponseSchema: Boolean = false
    open val supportedParams: Set<String> = emptySet()

    // ===================
    // Forward (abstract)
    // ===================

    /**
     * Forward pass for the language model.
     * Subclasses must implement this method.
     */
    abstract suspend fun forward(
        prompt: String? = null,
        messages: List<Map<String, Any>>? = null,
        extraKwargs: Map<String, Any?> = emptyMap(),
    ): Any

    /**
     * Async forward pass for the language model.
     * Default implementation delegates to [forward].
     */
    open suspend fun aforward(
        prompt: String? = null,
        messages: List<Map<String, Any>>? = null,
        extraKwargs: Map<String, Any?> = emptyMap(),
    ): Any = forward(prompt, messages, extraKwargs)

    // ===================
    // Invoke operator
    // ===================

    /**
     * Call the language model.
     *
     * @param prompt Optional prompt string.
     * @param messages Optional chat messages.
     * @param request Optional explicit normalized request.
     * @param kwargs Per-call generation parameters.
     * @return Either [LMResponse] or legacy list of outputs.
     */
    suspend operator fun invoke(
        prompt: String? = null,
        messages: List<Map<String, Any>>? = null,
        request: LMRequest? = null,
        kwargs: Map<String, Any?> = emptyMap(),
    ): Any {
        // If an explicit LMRequest is provided, use typed path
        if (request != null) {
            val effectiveRequest = if (kwargs.isNotEmpty()) {
                val nonNull = kwargs.filterValues { it != null }.mapValues { it.value as Any }
                request.withConfigOverrides(nonNull)
            } else {
                request
            }
            return _legacyForwardAsLMResponse(effectiveRequest)
        }

        // Legacy direct call path
        return _legacyCallDirect(prompt = prompt, messages = messages, kwargs = kwargs)
    }

    // ===================
    // Legacy call path
    // ===================

    private suspend fun _legacyCallDirect(
        prompt: String?,
        messages: List<Map<String, Any>>?,
        kwargs: Map<String, Any?>,
    ): List<Any> {
        val response = forward(prompt = prompt, messages = messages, extraKwargs = kwargs)
        return _processLMResponse(response, prompt, messages, kwargs)
    }

    private suspend fun _legacyAcallDirect(
        prompt: String?,
        messages: List<Map<String, Any>>?,
        kwargs: Map<String, Any?>,
    ): List<Any> {
        val response = aforward(prompt = prompt, messages = messages, extraKwargs = kwargs)
        return _processLMResponse(response, prompt, messages, kwargs)
    }

    private suspend fun _legacyForwardAsLMResponse(request: LMRequest): LMResponse {
        val data = _legacyForwardKwargs(request)
        val messages = data.remove("messages") as? List<Map<String, Any>>
        val response = forward(prompt = _promptFromLMRequest(request), messages = messages, extraKwargs = data)
        val outputs = _processLMResponse(response, _promptFromLMRequest(request), messages, data)
        return _legacyOutputsToLMResponse(outputs, request = request, providerResponse = response)
    }

    private fun _legacyForwardKwargs(request: LMRequest): MutableMap<String, Any?> {
        val data: MutableMap<String, Any?> = _toOpenAIChatRequest(request).toMutableMap()
        data.remove("model")
        request.config.cache?.let { cacheConfig ->
            if (cacheConfig.enabled != null) data["cache"] = cacheConfig.enabled
            if (cacheConfig.rolloutId != null) data["rollout_id"] = cacheConfig.rolloutId
        }
        return data
    }

    private fun _promptFromLMRequest(request: LMRequest): String? {
        if (request.messages.size != 1) return null
        val message = request.messages[0]
        if (message.role != "user" || message.parts.size != 1) return null
        val part = message.parts[0]
        return if (part is LMTextPart) part.text else null
    }

    private fun _toOpenAIChatRequest(request: LMRequest): Map<String, Any?> {
        val data = mutableMapOf<String, Any?>()
        data["model"] = request.model
        data["messages"] = request.messages.map { msg ->
            mutableMapOf<String, Any?>(
                "role" to msg.role,
                "content" to (msg.parts.filterIsInstance<LMTextPart>().joinToString("") { it.text } ?: ""),
            ).apply {
                msg.name?.let { this["name"] = it }
            }
        }
        request.tools.isNotEmpty().let {
            if (it) data["tools"] = request.tools.map { tool ->
                mapOf("type" to "function", "function" to mapOf(
                    "name" to tool.name,
                    "description" to (tool.description ?: ""),
                    "parameters" to (tool.parameters ?: emptyMap()),
                ))
            }
        }
        request.config.temperature?.let { data["temperature"] = it }
        request.config.maxTokens?.let { data["max_tokens"] = it }
        request.config.topP?.let { data["top_p"] = it }
        request.config.stop?.let { data["stop"] = it }
        request.config.n?.let { data["n"] = it }
        return data
    }

    // ===================
    // Response processing
    // ===================

    /**
     * Process the response of OpenAI chat completion API and extract outputs.
     */
    protected fun _processCompletion(response: Any, mergedKwargs: Map<String, Any?>): List<Map<String, Any?>> {
        val outputs = mutableListOf<Map<String, Any?>>()
        val choices = when (response) {
            is Map<*, *> -> (response["choices"] as? List<Map<String, Any?>>) ?: emptyList()
            else -> return emptyList()
        }

        for (c in choices) {
            val output = mutableMapOf<String, Any?>()
            val message = when (c) {
                is Map<*, *> -> c["message"] as? Map<*, *>
                else -> c
            }
            output["text"] = when (message) {
                is Map<*, *> -> message["content"]
                else -> c["text"]
            }

            if (message is Map<*, *>) {
                message["reasoning_content"]?.let { output["reasoning_content"] = it }
                message["tool_calls"]?.let { output["tool_calls"] = it }
            }

            mergedKwargs["logprobs"]?.let {
                (c as? Map<*, *>)?.get("logprobs")?.let { logprobs -> output["logprobs"] = logprobs }
            }

            outputs.add(output)
        }

        // If every output only has "text" key, return a list of strings
        return if (outputs.all { it.size == 1 }) {
            outputs.map { mutableMapOf<String, Any?>("text" to (it["text"] ?: "")) }
        } else {
            outputs
        }
    }

    protected fun _processResponse(response: Any): List<Map<String, Any?>> {
        val textOutputs = mutableListOf<String>()
        val toolCalls = mutableListOf<Map<String, Any>>()
        val reasoningContents = mutableListOf<String>()

        val output = when (response) {
            is Map<*, *> -> (response["output"] as? List<Any>) ?: emptyList()
            else -> emptyList()
        }

        for (item in output) {
            val itemMap = item as? Map<*, *> ?: continue
            val itemType = itemMap["type"] as? String ?: continue
            when (itemType) {
                "message" -> {
                    val content = itemMap["content"] as? List<Any> ?: emptyList()
                    for (ci in content) {
                        val ciMap = ci as? Map<*, *> ?: continue
                        (ciMap["text"] as? String)?.let { textOutputs.add(it) }
                    }
                }
                "function_call" -> toolCalls.add(itemMap.mapKeys { it.key.toString() }.mapValues { it.value as Any })
                "reasoning" -> {
                    val content = itemMap["content"] as? List<Any> ?: emptyList()
                    if (content.isNotEmpty()) {
                        for (ci in content) {
                            val ciMap = ci as? Map<*, *> ?: continue
                            (ciMap["text"] as? String)?.let { reasoningContents.add(it) }
                        }
                    } else {
                        val summary = itemMap["summary"] as? List<Any> ?: emptyList()
                        for (si in summary) {
                            val siMap = si as? Map<*, *> ?: continue
                            (siMap["text"] as? String)?.let { reasoningContents.add(it) }
                        }
                    }
                }
            }
        }

        val result = mutableMapOf<String, Any?>()
        if (textOutputs.isNotEmpty()) result["text"] = textOutputs.joinToString("")
        if (toolCalls.isNotEmpty()) result["tool_calls"] = toolCalls
        if (reasoningContents.isNotEmpty()) result["reasoning_content"] = reasoningContents.joinToString("")
        return listOf(result)
    }

    private fun _processLMResponse(
        response: Any,
        prompt: String?,
        messages: List<Map<String, Any>>?,
        extraKwargs: Map<String, Any?>,
    ): List<Map<String, Any?>> {
        val mergedKwargs = kwargs + extraKwargs

        val outputs = if (modelType == "responses") {
            _processResponse(response)
        } else {
            _processCompletion(response, mergedKwargs)
        }

        // Logging (simplified)
        if (Settings.maxHistorySize > 0) {
            val entry = mutableMapOf<String, Any?>(
                "prompt" to prompt,
                "messages" to messages,
                "kwargs" to extraKwargs.filterKeys { !it.startsWith("api_") },
                "outputs" to outputs,
                "timestamp" to java.time.Instant.now().toString(),
                "uuid" to java.util.UUID.randomUUID().toString(),
                "model" to model,
                "modelType" to modelType,
            )
            if (history.size >= Settings.maxHistorySize) history.removeAt(0)
            history.add(entry)
        }

        return outputs
    }

    private fun _legacyOutputsToLMResponse(
        outputs: List<Any>,
        request: LMRequest,
        providerResponse: Any,
    ): LMResponse {
        val lmOutputs = outputs.map { output ->
            val parts = mutableListOf<LMPart>()
            when (output) {
                is String -> parts.add(LMTextPart(text = output))
                is Map<*, *> -> {
                    (output["text"] as? String)?.let { parts.add(LMTextPart(text = it)) }
                    (output["reasoning_content"] as? String)?.let { parts.add(LMThinkingPart(text = it)) }
                    (output["tool_calls"] as? List<*>)?.forEach { tc ->
                        val tcMap = tc as? Map<*, *> ?: return@forEach
                        parts.add(LMToolCallPart(
                            id = tcMap["id"] as? String,
                            name = tcMap["function"]?.let {
                                (it as? Map<*, *>)?.get("name") as? String
                            } ?: (tcMap["name"] as? String) ?: "",
                            args = (tcMap["function"] as? Map<*, *>)?.get("arguments") as? Map<String, Any?>
                                ?: emptyMap(),
                        ))
                    }
                }
            }
            LMOutput(parts = parts)
        }

        return LMResponse(
            model = model,
            outputs = lmOutputs,
        )
    }

    // ===================
    // State management
    // ===================

    open fun dumpState(): Map<String, Any?> {
        val filteredKwargs = kwargs.filterKeys { it !in listOf("api_key", "apiKey") }
        return mapOf(
            "_dspy_lm_class" to "${this::class.qualifiedName}",
            "model" to model,
            "modelType" to modelType,
            "cache" to cache,
            "numRetries" to numRetries,
        ) + filteredKwargs
    }

    companion object {
        @JvmStatic
        fun loadState(state: Map<String, Any?>, allowCustomLMClass: Boolean = false): BaseLM {
            // For the Kotlin port, we just reconstruct with the base params
            // Custom class loading would require reflection/serialization
            return state["model"]?.toString()?.let { m ->
                LM(model = m)
            } ?: LM(model = "unknown")
        }
    }

    // ===================
    // Copy
    // ===================

    fun copy(vararg kwargs: Pair<String, Any?>): BaseLM {
        val copy = LM(model = model)
        copy.kwargs.putAll(this.kwargs)
        copy.kwargs.putAll(kwargs.toMap())
        return copy
    }

    // ===================
    // History
    // ===================

    fun inspectHistory(n: Int = 1) {
        println("=== DSPy LM History (last $n entries) ===")
        history.takeLast(n).forEachIndexed { i, entry ->
            println("\n--- Entry ${i + 1} ---")
            println(entry)
        }
    }

    fun updateHistory(entry: Any) {
        if (history.size >= Settings.maxHistorySize) history.removeAt(0)
        history.add(entry)
    }
}
