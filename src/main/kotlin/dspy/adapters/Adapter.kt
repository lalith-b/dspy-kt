package dspy.adapters

import dspy.adapters.types.Citations
import dspy.adapters.types.History
import dspy.adapters.types.Reasoning
import dspy.adapters.types.Tool
import dspy.adapters.types.ToolCalls
import dspy.clients.BaseLM
import dspy.core.types.LMConfig
import dspy.core.types.LMMessage
import dspy.core.types.LMOutput
import dspy.core.types.LMPart
import dspy.core.types.LMRequest
import dspy.core.types.LMResponse
import dspy.core.types.LMTextPart
import dspy.signatures.FieldInfo
import dspy.signatures.Signature
import dspy.utils.BaseCallback
import dspy.utils.CallbackHandler
import dspy.core.AdapterParseError
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlin.reflect.KClass

private val defaultNativeResponseTypes = listOf(Citations::class, Reasoning::class)

/**
 * Base Adapter class for the DSPy framework.
 *
 * The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs).
 * It handles the complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.
 */
abstract class Adapter(
    val callbacks: List<CallbackHandler> = emptyList(),
    var useNativeFunctionCalling: Boolean = false,
    var nativeResponseTypes: List<KClass<*>> = defaultNativeResponseTypes,
    var parallelToolCalls: Boolean? = null
) {
    protected fun callPreprocess(
        lm: BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        inputs: Map<String, Any?>
    ): Signature {
        if (!useNativeFunctionCalling) {
            for (key in listOf("tools", "tool_choice", "parallel_tool_calls")) {
                lmKwargs.remove(key)
            }
        } else {
            val toolCallOutputFieldName = getToolCallOutputFieldName(signature)
            val toolCallInputFieldName = getToolCallInputFieldName(signature)

            if (toolCallOutputFieldName != null && toolCallInputFieldName == null) {
                throw IllegalArgumentException(
                    "You provided an output field $toolCallOutputFieldName to receive the tool calls information, " +
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an " +
                    "input field with type `list[dspy.Tool]`."
                )
            }

            if (toolCallOutputFieldName != null && lm.supportsFunctionCalling) {
                val tools = inputs[toolCallInputFieldName] as? List<Tool> ?: listOf(inputs[toolCallInputFieldName] as Tool)
                val lmTools = tools.map { it.formatAsLiteLLMFunctionCall() }
                lmKwargs["tools"] = lmTools
                if (parallelToolCalls != null && lmKwargs["parallel_tool_calls"] == null) {
                    lmKwargs["parallel_tool_calls"] = parallelToolCalls
                }
            }
        }

        return signature
    }

    protected fun callPostprocess(
        processedSignature: Signature,
        originalSignature: Signature,
        outputs: List<Any>,
        lm: BaseLM,
        lmKwargs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val values = mutableListOf<Map<String, Any?>>()
        val toolCallOutputFieldName = getToolCallOutputFieldName(originalSignature)

        for (output in outputs) {
            var text: Any = output
            var outputLogprobs: Any? = null
            var toolCalls: List<Map<String, Any?>>? = null

            if (output is Map<*, *>) {
                text = output["text"] ?: output
                outputLogprobs = output["logprobs"]
                toolCalls = output["tool_calls"] as? List<Map<String, Any?>>
            }

            val value = mutableMapOf<String, Any?>()

            if (text != null && !(toolCalls != null && toolCallOutputFieldName != null)) {
                val parsed = parse(processedSignature, text.toString())
                value.putAll(parsed)
            } else if (toolCalls != null && toolCallOutputFieldName != null) {
                try {
                    if (text != null && processedSignature.outputFields.isNotEmpty()) {
                        val parsed = parse(processedSignature, text.toString())
                        value.putAll(parsed)
                    }
                } catch (_: AdapterParseError) {
                    // Ignore parse errors
                }
            } else {
                throw AdapterParseError(
                    adapterName = this::class.simpleName ?: "Adapter",
                    signature = originalSignature,
                    lmResponse = output.toString(),
                    message = "The LM returned an empty or null response."
                )
            }

            // Ensure all output fields are present
            for (field in originalSignature.outputFields) {
                value.putIfAbsent(field.name, null)
            }

            if (toolCalls != null && toolCallOutputFieldName != null) {
                val processedToolCalls = toolCalls.map { providerToolCallToToolCallDict(it) }
                value[toolCallOutputFieldName] = ToolCalls.fromDictList(processedToolCalls)
            }

            if (outputLogprobs != null) {
                value["logprobs"] = outputLogprobs
            }

            values.add(value)
        }

        return values
    }

    protected fun renderRequest(
        lm: BaseLM,
        lmKwargs: Map<String, Any?>,
        messages: List<Any>
    ): LMRequest {
        val msgList = messages.map { msg ->
            if (msg is LMMessage) msg
            else coerceChatDictToLmMessage(msg as Map<String, Any?>)
        }
        return LMRequest(
            model = lm.model,
            messages = msgList
        )
    }

    private fun coerceChatDictToLmMessage(message: Map<String, Any?>): LMMessage {
        return try {
            LMMessage(role = message["role"] as String, parts = listOf(LMTextPart(text = (message["content"] as? String) ?: "")))
        } catch (_: Exception) {
            LMMessage(role = message["role"] as String, parts = listOf(LMTextPart(text = message["content"]?.toString() ?: "")))
        }
    }

    protected fun callLm(lm: BaseLM, request: LMRequest): LMResponse {
        val data = toOpenAIChatRequest(request).toMutableMap()
        data.remove("model")
        @Suppress("UNCHECKED_CAST")
        val messages = data.remove("messages") as? List<Map<String, Any>> ?: emptyList()
        val result = runBlocking { lm(messages = messages, kwargs = data) }
        return when (result) {
            is LMResponse -> result
            is List<*> -> lmResponseFromLegacyOutputs(result as List<Any>, request)
            else -> lmResponseFromLegacyOutputs(listOf(result), request)
        }
    }

    suspend fun acallLm(lm: BaseLM, request: LMRequest): LMResponse {
        val data = toOpenAIChatRequest(request).toMutableMap()
        data.remove("model")
        @Suppress("UNCHECKED_CAST")
        val messages = data.remove("messages") as? List<Map<String, Any>> ?: emptyList()
        val result = lm(messages = messages, kwargs = data)
        return when (result) {
            is LMResponse -> result
            is List<*> -> lmResponseFromLegacyOutputs(result as List<Any>, request)
            else -> lmResponseFromLegacyOutputs(listOf(result), request)
        }
    }

    /**
     * Execute the adapter pipeline: format inputs, call LM, and parse outputs.
     */
    open operator fun invoke(
        lm: BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val processedSignature = callPreprocess(lm, lmKwargs, signature, inputs)
        val messages = format(processedSignature, demos, inputs)
        val request = renderRequest(lm, lmKwargs, messages)
        val response = callLm(lm, request)
        val outputs = legacyOutputsFromLmResponse(response)
        return callPostprocess(processedSignature, signature, outputs, lm, lmKwargs)
    }

    suspend fun acall(
        lm: BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val processedSignature = callPreprocess(lm, lmKwargs, signature, inputs)
        val messages = format(processedSignature, demos, inputs)
        val request = renderRequest(lm, lmKwargs, messages)
        val response = acallLm(lm, request)
        val outputs = legacyOutputsFromLmResponse(response)
        return callPostprocess(processedSignature, signature, outputs, lm, lmKwargs)
    }

    /**
     * Format the input messages for the LM call.
     */
    open fun format(
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val messages = mutableListOf<Map<String, Any?>>()

        val systemMessage = formatSystemMessage(signature)
        messages.add(mapOf("role" to "system", "content" to systemMessage))
        messages.addAll(formatDemos(signature, demos))

        val content = formatUserMessageContent(signature, inputs, mainRequest = true)
        if (content.isNotEmpty()) {
            messages.add(mapOf("role" to "user", "content" to content))
        }

        return messages
    }

    /**
     * Format the system message for the LM call.
     */
    open fun formatSystemMessage(signature: Signature): String {
        return "${formatFieldDescription(signature)}\n${formatFieldStructure(signature)}\n${formatTaskDescription(signature)}"
    }

    abstract fun formatFieldDescription(signature: Signature): String
    abstract fun formatFieldStructure(signature: Signature): String
    open fun formatTaskDescription(signature: Signature): String {
        return signature.instructions ?: ""
    }

    abstract fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String = "",
        suffix: String = "",
        mainRequest: Boolean = false
    ): String

    abstract fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String? = null
    ): String

    open fun formatDemos(
        signature: Signature,
        demos: List<Map<String, Any?>>
    ): List<Map<String, Any?>> {
        val allFieldNames = (signature.inputFields + signature.outputFields).map { it.name }.toSet()
        val completeDemos = demos.filter { demo ->
            allFieldNames.all { k -> k in demo && demo[k] != null }
        }
        val incompleteDemos = demos.filter { demo ->
            !completeDemos.contains(demo) &&
            signature.inputFields.any { it.name in demo } &&
            signature.outputFields.any { it.name in demo }
        }

        val messages = mutableListOf<Map<String, Any?>>()
        val incompleteDemoPrefix = "This is an example of the task, though some input or output fields are not supplied."

        for (demo in incompleteDemos) {
            messages.add(mapOf("role" to "user", "content" to formatUserMessageContent(signature, demo, prefix = incompleteDemoPrefix)))
            messages.add(mapOf("role" to "assistant", "content" to formatAssistantMessageContent(signature, demo, missingFieldMessage = "Not supplied for this particular example.")))
        }

        for (demo in completeDemos) {
            messages.add(mapOf("role" to "user", "content" to formatUserMessageContent(signature, demo)))
            messages.add(mapOf("role" to "assistant", "content" to formatAssistantMessageContent(signature, demo, missingFieldMessage = "Not supplied for this conversation history message.")))
        }

        return messages
    }

    abstract fun parse(signature: Signature, completion: String): Map<String, Any?>

    private fun getHistoryFieldName(signature: Signature): String? {
        return signature.inputFields.find { it.annotation == History::class }?.name
    }

    private fun getToolCallInputFieldName(signature: Signature): String? {
        return signature.inputFields.find { it.annotation == Tool::class }?.name
    }

    private fun getToolCallOutputFieldName(signature: Signature): String? {
        return signature.outputFields.find { it.annotation == ToolCalls::class }?.name
    }

    private fun providerToolCallToToolCallDict(toolCall: Map<String, Any?>): Map<String, Any?> {
        val function = (toolCall["function"] as? Map<String, Any?>) ?: emptyMap()
        var arguments: Map<String, Any?> = emptyMap()
        val rawArgs = function["arguments"]
        if (rawArgs is String) {
            arguments = try {
                Json { ignoreUnknownKeys = true }.decodeFromString<Map<String, Any?>>(rawArgs)
            } catch (_: Exception) {
                emptyMap()
            }
        } else if (rawArgs is Map<*, *>) {
            arguments = rawArgs as Map<String, Any?>
        }
        return mapOf(
            "id" to (toolCall["id"] ?: toolCall["call_id"]),
            "name" to (function["name"] ?: toolCall["name"]),
            "args" to arguments
        )
    }
}

// ============================================================
// Legacy compatibility functions
// ============================================================

fun legacyOutputsFromLmResponse(response: LMResponse): List<Map<String, Any?>> {
    return response.outputs.map { output ->
        val text = output.parts.filterIsInstance<LMTextPart>().joinToString("") { it.text }
        mutableMapOf<String, Any?>("text" to text)
    }
}

fun lmResponseFromLegacyOutputs(outputs: List<Any>, request: LMRequest): LMResponse {
    val lmOutputs = outputs.map { output ->
        val text = if (output is Map<*, *>) output["text"]?.toString() ?: "" else output.toString()
        LMOutput(parts = listOf(LMTextPart(text = text)))
    }
    return LMResponse(model = request.model, outputs = lmOutputs)
}

fun toOpenAIChatRequest(request: LMRequest): Map<String, Any?> {
    val data = mutableMapOf<String, Any?>()
    data["messages"] = request.messages.map { msg ->
        mapOf(
            "role" to msg.role,
            "content" to msg.parts.filterIsInstance<LMTextPart>().joinToString("") { it.text }
        )
    }
    lmConfigToKwargs(request.config)?.let { data.putAll(it) }
    return data
}

fun lmConfigToKwargs(config: LMConfig): Map<String, Any?> {
    val kwargs = mutableMapOf<String, Any?>()
    config.temperature?.let { kwargs["temperature"] = it }
    config.maxTokens?.let { kwargs["max_tokens"] = it }
    config.n?.let { kwargs["n"] = it }
    config.logprobs?.let { kwargs["logprobs"] = it }
    config.topP?.let { kwargs["top_p"] = it }
    config.extensions.forEach { (k, v) -> kwargs[k] = v }
    return kwargs
}
