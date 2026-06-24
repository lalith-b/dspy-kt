package dspy.core.types

/**
 * Normalized request, response, and stream types for DSPy language models.
 */

// ============================================================
// Part types
// ============================================================

/**
 * Sealed class for all LM content parts.
 */
sealed class LMPart {
    abstract val type: String
    open val metadata: Map<String, Any?> = emptyMap()
}

data class LMTextPart(
    override val type: String = "text",
    val text: String,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

open class LMSourcePart(
    override val type: String,
    open val mediaType: String,
    open val data: String? = null,
    open val url: String? = null,
    open val fileId: String? = null,
    open val path: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart() {
    init {
        require(listOfNotNull(data, url, fileId, path).size <= 1) {
            "Only one source may be set"
        }
    }
}

data class LMImagePart(
    override val mediaType: String = "image/png",
    override val data: String? = null,
    override val url: String? = null,
    override val fileId: String? = null,
    override val path: String? = null,
    val detail: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMSourcePart("image", mediaType, data, url, fileId, path, metadata)

data class LMAudioPart(
    override val mediaType: String = "audio/wav",
    override val data: String? = null,
    override val url: String? = null,
    override val fileId: String? = null,
    override val path: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMSourcePart("audio", mediaType, data, url, fileId, path, metadata)

data class LMVideoPart(
    override val mediaType: String = "video/mp4",
    override val data: String? = null,
    override val url: String? = null,
    override val fileId: String? = null,
    override val path: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMSourcePart("video", mediaType, data, url, fileId, path, metadata)

data class LMDocumentPart(
    override val type: String = "document",
    val mediaType: String = "application/pdf",
    val data: String? = null,
    val url: String? = null,
    val fileId: String? = null,
    val path: String? = null,
    val source: Map<String, Any?>? = null,
    val citations: Map<String, Any?> = emptyMap(),
    val title: String? = null,
    val context: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

data class LMBinaryPart(
    override val mediaType: String = "application/octet-stream",
    override val data: String? = null,
    override val url: String? = null,
    override val fileId: String? = null,
    override val path: String? = null,
    val filename: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMSourcePart("binary", mediaType, data, url, fileId, path, metadata)

data class LMToolCallPart(
    override val type: String = "tool_call",
    val id: String? = null,
    val name: String,
    val args: Map<String, Any?> = emptyMap(),
    val providerData: Map<String, Any?> = emptyMap(),
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart() {
    companion object {
        fun toolCall(id: String? = null, name: String, args: Map<String, Any?> = emptyMap()): LMToolCallPart {
            return LMToolCallPart(id = id, name = name, args = args)
        }
    }
}

data class LMToolResultPart(
    override val type: String = "tool_result",
    val callId: String? = null,
    val name: String? = null,
    val content: List<LMPart> = emptyList(),
    val isError: Boolean = false,
    val providerData: Map<String, Any?> = emptyMap(),
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

data class LMThinkingPart(
    override val type: String = "thinking",
    val text: String,
    val redacted: Boolean = false,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

data class LMCitationPart(
    override val type: String = "citation",
    val text: String? = null,
    val title: String? = null,
    val url: String? = null,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

data class LMRefusalPart(
    override val type: String = "refusal",
    val text: String,
    override val metadata: Map<String, Any?> = emptyMap()
) : LMPart()

// ============================================================
// Message
// ============================================================

data class LMMessage(
    val role: String,
    val parts: List<LMPart> = emptyList(),
    val name: String? = null,
    val metadata: Map<String, Any?> = emptyMap()
) {
    val text: String?
        get() = parts.filterIsInstance<LMTextPart>().joinToString("") { it.text }.takeIf { it.isNotEmpty() }

    companion object {
        fun system(content: String) = LMMessage("system", listOf(LMTextPart(text = content)))
        fun user(content: String) = LMMessage("user", listOf(LMTextPart(text = content)))
        fun assistant(content: String) = LMMessage("assistant", listOf(LMTextPart(text = content)))
    }
}

// Role shorthands
fun system(content: String) = LMMessage.system(content)
fun user(content: String) = LMMessage.user(content)
fun assistant(content: String) = LMMessage.assistant(content)
fun developer(content: String) = LMMessage("developer", listOf(LMTextPart(text = content)))

// ============================================================
// Tool & Config
// ============================================================

data class LMToolSpec(
    val type: String = "function",
    val name: String,
    val description: String? = null,
    val parameters: Map<String, Any?> = emptyMap(),
    val providerData: Map<String, Any?> = emptyMap()
)

data class LMReasoningConfig(
    val effort: String? = null,
    val maxTokens: Int? = null,
    val summary: String? = null
) {
    companion object {
        fun fromValue(value: Any? = null, overrides: Map<String, Any?> = emptyMap()): LMReasoningConfig {
            val effort = overrides["effort"] as? String ?: value?.toString()
            return LMReasoningConfig(effort = effort)
        }
    }
}

data class LMToolChoice(
    val mode: String = "auto",
    val allowed: List<String>? = null,
    val parallel: Boolean? = null
) {
    companion object {
        fun fromValue(value: Any? = null, overrides: Map<String, Any?> = emptyMap()): LMToolChoice {
            val mode = value as? String ?: "auto"
            val parallel = overrides["parallel"] as? Boolean
            return LMToolChoice(mode = mode, parallel = parallel)
        }
    }
}

data class LMCacheConfig(
    val enabled: Boolean? = null,
    val rolloutId: Any? = null
) {
    companion object {
        fun fromValue(value: Any? = null, overrides: Map<String, Any?> = emptyMap()): LMCacheConfig {
            val enabled = when (value) {
                is Boolean -> value
                is Map<*, *> -> value["enabled"] as? Boolean
                null -> null
                else -> null
            }
            val rolloutId = overrides["rolloutId"]
            return LMCacheConfig(enabled = enabled, rolloutId = rolloutId)
        }
    }
}

data class LMPromptCacheConfig(
    val enabled: Boolean? = null,
    val key: String? = null
) {
    companion object {
        fun fromValue(value: Any? = null, overrides: Map<String, Any?> = emptyMap()): LMPromptCacheConfig {
            val enabled = when (value) {
                is Boolean -> value
                is Map<*, *> -> value["enabled"] as? Boolean
                null -> null
                else -> null
            }
            val key = (value as? Map<*, *>)?.get("key") as? String
            return LMPromptCacheConfig(enabled = enabled, key = key)
        }
    }
}

// ============================================================
// Config
// ============================================================

data class LMConfig(
    val temperature: Float? = null,
    val maxTokens: Int? = null,
    val topP: Float? = null,
    val stop: List<String>? = null,
    val n: Int? = null,
    val logprobs: Any? = null,
    val responseFormat: Any? = null,
    val reasoning: LMReasoningConfig? = null,
    val toolChoice: LMToolChoice? = null,
    val cache: LMCacheConfig? = null,
    val promptCache: LMPromptCacheConfig? = null,
    val extensions: Map<String, Any?> = emptyMap()
) {
    companion object {
        fun fromKwargs(kwargs: Map<String, Any?>): LMConfig {
            return LMConfig(
                temperature = kwargs["temperature"] as? Float,
                maxTokens = kwargs["max_tokens"] as? Int,
                topP = kwargs["top_p"] as? Float,
                stop = kwargs["stop"] as? List<String>,
                n = kwargs["n"] as? Int,
                logprobs = kwargs["logprobs"],
                responseFormat = kwargs["response_format"],
                extensions = kwargs.filterKeys { it !in setOf("temperature", "max_tokens", "top_p", "stop", "n", "logprobs", "response_format") }
            )
        }
    }
}

// ============================================================
// Request & Response
// ============================================================

data class LMRequest(
    val model: String,
    val messages: List<LMMessage> = emptyList(),
    val tools: List<LMToolSpec> = emptyList(),
    val config: LMConfig = LMConfig(),
    val metadata: Map<String, Any?> = emptyMap()
) {
    fun withConfigOverrides(overrides: Map<String, Any?>): LMRequest {
        val newConfig = config.copy(
            temperature = overrides["temperature"] as? Float ?: config.temperature,
            maxTokens = overrides["maxTokens"] as? Int ?: overrides["max_tokens"] as? Int ?: config.maxTokens,
            topP = overrides["topP"] as? Float ?: overrides["top_p"] as? Float ?: config.topP,
            stop = overrides["stop"] as? List<String> ?: config.stop,
            n = overrides["n"] as? Int ?: config.n,
            logprobs = overrides["logprobs"] ?: config.logprobs,
            responseFormat = overrides["responseFormat"] ?: overrides["response_format"] ?: config.responseFormat,
            extensions = config.extensions + (overrides.filterKeys { it !in setOf("temperature", "maxTokens", "max_tokens", "topP", "top_p", "stop", "n", "logprobs", "responseFormat", "response_format") })
        )
        return copy(config = newConfig)
    }

    companion object {
        fun fromCall(
            model: String,
            messages: List<Any> = emptyList(),
            tools: List<Any>? = null,
            vararg kwargs: Pair<String, Any?>
        ): LMRequest {
            val config = LMConfig.fromKwargs(kwargs.associate { it.first to it.second })
            val msgList = messages.map {
                when (it) {
                    is LMMessage -> it
                    is Map<*, *> -> LMMessage(it["role"] as String, emptyList())
                    is String -> LMMessage.user(it)
                    else -> LMMessage("user", emptyList())
                }
            }
            return LMRequest(model = model, messages = msgList, config = config)
        }
    }
}

data class LMOutput(
    val parts: List<LMPart> = emptyList(),
    val finishReason: String? = null,
    val truncated: Boolean = false,
    val logprobs: Any? = null,
    val providerData: Map<String, Any?> = emptyMap(),
    val metadata: Map<String, Any?> = emptyMap()
) {
    val text: String?
        get() = parts.filterIsInstance<LMTextPart>().joinToString("") { it.text }.takeIf { it.isNotEmpty() }
    val reasoningContent: String?
        get() = parts.filterIsInstance<LMThinkingPart>().joinToString("") { it.text }.takeIf { it.isNotEmpty() }
    val toolCalls: List<LMToolCallPart>
        get() = parts.filterIsInstance<LMToolCallPart>()
    val citations: List<LMCitationPart>
        get() = parts.filterIsInstance<LMCitationPart>()
}

data class LMUsage(
    val inputTokens: Int? = null,
    val outputTokens: Int? = null,
    val totalTokens: Int? = null,
    val promptTokens: Int? = null,
    val completionTokens: Int? = null,
    val reasoningTokens: Int? = null,
    val cacheReadTokens: Int? = null,
    val cacheWriteTokens: Int? = null,
    val inputAudioTokens: Int? = null,
    val outputAudioTokens: Int? = null,
    val details: Map<String, Any?> = emptyMap()
) {
    // Note: Kotlin data classes can't alias properties; accessors handle cross-references
}

data class LMResponse(
    val model: String? = null,
    val outputs: List<LMOutput> = listOf(LMOutput()),
    val usage: Any? = null,
    val cost: Float? = null,
    val cacheHit: Boolean = false,
    val responseId: String? = null,
    val providerData: Map<String, Any?> = emptyMap(),
    val metadata: Map<String, Any?> = emptyMap()
) {
    val output: LMOutput get() = outputs.first()
    val text: String? get() = output.text
    val reasoningContent: String? get() = output.reasoningContent
    val toolCalls: List<LMToolCallPart> get() = output.toolCalls
    val citations: List<LMCitationPart> get() = output.citations

    companion object {
        fun fromText(text: String, model: String? = null, usage: Any? = null, cost: Float? = null): LMResponse {
            return LMResponse(
                model = model,
                outputs = listOf(LMOutput(parts = listOf(LMTextPart(text = text)))),
                usage = usage,
                cost = cost
            )
        }
    }
}

// ============================================================
// History Entry
// ============================================================

data class LMHistoryEntry(
    val request: LMRequest,
    val response: LMResponse,
    val timestamp: String = java.time.Instant.now().toString(),
    val uuid: String = java.util.UUID.randomUUID().toString()
) {
    val model: String get() = request.model
    val prompt: String get() = request.messages.joinToString("\n") { it.text ?: "" }
    val outputs: List<String> get() = response.outputs.map { it.text ?: "" }
    val responseModel: String? get() = response.model
    val usage: Map<String, Any?> get() = response.usage as? Map<String, Any?> ?: emptyMap()
    val messages: List<Map<String, Any?>>? get() = null

    operator fun get(key: String): Any? {
        return when (key) {
            "model" -> model
            "prompt" -> prompt
            "outputs" -> outputs
            "responseModel" -> responseModel
            "usage" -> usage
            "messages" -> messages
            "timestamp" -> timestamp
            "uuid" -> uuid
            else -> null
        }
    }
}

// ============================================================
// Request Patch
// ============================================================

data class LMRequestPatch(
    val messages: List<LMMessage> = emptyList(),
    val systemParts: List<LMPart> = emptyList(),
    val userParts: List<LMPart> = emptyList(),
    val assistantParts: List<LMPart> = emptyList(),
    val tools: List<LMToolSpec> = emptyList(),
    val config: LMConfig? = null,
    val deleteInputFields: List<String> = emptyList(),
    val deleteOutputFields: List<String> = emptyList(),
    val metadata: Map<String, Any?> = emptyMap()
) {
    fun merge(other: LMRequestPatch): LMRequestPatch {
        return LMRequestPatch(
            messages = messages + other.messages,
            systemParts = systemParts + other.systemParts,
            userParts = userParts + other.userParts,
            assistantParts = assistantParts + other.assistantParts,
            tools = tools + other.tools,
            config = config?.let { mergeLmConfig(it, other.config) } ?: other.config,
            deleteInputFields = deleteInputFields + other.deleteInputFields,
            deleteOutputFields = deleteOutputFields + other.deleteOutputFields,
            metadata = metadata + other.metadata
        )
    }

    fun toKwargs(): Map<String, Any?> {
        val kwargs = mutableMapOf<String, Any?>()
        config?.let { cfg ->
            cfg.temperature?.let { kwargs["temperature"] = it }
            cfg.maxTokens?.let { kwargs["maxTokens"] = it }
            cfg.topP?.let { kwargs["topP"] = it }
            cfg.stop?.let { kwargs["stop"] = it }
            cfg.n?.let { kwargs["n"] = it }
            cfg.logprobs?.let { kwargs["logprobs"] = it }
            cfg.responseFormat?.let { kwargs["responseFormat"] = it }
        }
        if (tools.isNotEmpty()) kwargs["tools"] = tools
        if (messages.isNotEmpty()) kwargs["messages"] = messages
        return kwargs
    }
}

fun mergeLmConfig(left: LMConfig?, right: LMConfig?): LMConfig? {
    if (left == null) return right
    if (right == null) return left
    return LMConfig(
        temperature = right.temperature ?: left.temperature,
        maxTokens = right.maxTokens ?: left.maxTokens,
        topP = right.topP ?: left.topP,
        stop = right.stop ?: left.stop,
        n = right.n ?: left.n,
        logprobs = right.logprobs ?: left.logprobs,
        responseFormat = right.responseFormat ?: left.responseFormat,
        reasoning = right.reasoning ?: left.reasoning,
        toolChoice = right.toolChoice ?: left.toolChoice,
        cache = right.cache ?: left.cache,
        promptCache = right.promptCache ?: left.promptCache,
        extensions = left.extensions + right.extensions
    )
}
