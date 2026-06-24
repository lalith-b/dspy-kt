package dspy.clients

import dspy.core.types.LMMessage
import dspy.core.types.LMPart
import dspy.core.types.LMTextPart
import dspy.core.types.LMToolCallPart
import dspy.core.types.LMToolResultPart
import dspy.core.types.LMThinkingPart
import dspy.core.types.LMImagePart
import dspy.core.types.LMAudioPart
import dspy.core.types.LMVideoPart
import dspy.core.types.LMCitationPart
import dspy.core.types.LMDocumentPart
import dspy.core.types.LMBinaryPart
import dspy.core.types.LMRefusalPart
import dspy.core.types.LMToolSpec
import kotlinx.serialization.json.Json

/**
 * OpenAI format conversion utilities.
 *
 * Converts between DSPy internal types and OpenAI API format.
 */
object OpenAIFormat {
    /**
     * Convert LMMessage to OpenAI message dict.
     */
    fun messageToOpenAI(message: LMMessage): Map<String, Any?> {
        val content = message.parts.mapNotNull { part -> part.toOpenAIContent() }
        return mapOf(
            "role" to message.role,
            "content" to (if (content.size == 1) content[0] else content),
            "name" to message.name
        ).filterValues { it != null }
    }

    /**
     * Convert LMPart to OpenAI content.
     */
    fun LMPart.toOpenAIContent(): Any? = when (this) {
        is LMTextPart -> text
        is LMToolCallPart -> mapOf(
            "type" to "function",
            "function" to mapOf(
                "name" to name,
                "arguments" to serializeMapToJson(args)
            )
        )
        is LMImagePart -> mapOf(
            "type" to "image_url",
            "image_url" to mapOf(
                "url" to (url ?: "data:$mediaType;base64,$data")
            )
        )
        is LMToolResultPart -> mapOf(
            "type" to "function",
            "function" to mapOf("result" to content.joinToString { (it as? LMTextPart)?.text ?: "" })
        )
        is LMThinkingPart -> null
        else -> null
    }

private fun serializeMapToJson(map: Map<String, Any?>): String {
    val obj = kotlinx.serialization.json.buildJsonObject {
        for ((k, v) in map) put(k, when (v) {
            null -> kotlinx.serialization.json.JsonNull
            is String -> kotlinx.serialization.json.JsonPrimitive(v)
            is Number -> kotlinx.serialization.json.JsonPrimitive(v)
            is Boolean -> kotlinx.serialization.json.JsonPrimitive(v)
            else -> kotlinx.serialization.json.JsonPrimitive(v.toString())
        })
    }
    return obj.toString()
}

    /**
     * Convert LMToolSpec to OpenAI tool dict.
     */
    fun toolSpecToOpenAI(tool: LMToolSpec): Map<String, Any?> {
        return mapOf(
            "type" to tool.type,
            "function" to mapOf(
                "name" to tool.name,
                "description" to tool.description,
                "parameters" to tool.parameters
            )
        ).filterValues { it != null }
    }

    /**
     * Convert OpenAI response to LMResponse.
     */
    fun openAIResponseToLMResponse(response: Map<String, Any?>): List<Any> {
        val choices = response["choices"] as? List<Map<String, Any?>> ?: emptyList()
        return choices.map { choice ->
            val message = (choice as? Map<String, Any?>)?.get("message") as? Map<String, Any?> ?: emptyMap()
            mapOf(
                "text" to (message["content"]?.toString() ?: ""),
                "tool_calls" to (message["tool_calls"] as? List<*>)
            )
        }
    }

    /**
     * Format tool calls for OpenAI.
     */
    fun formatToolCalls(toolCalls: List<Map<String, Any?>>): List<Map<String, Any?>> {
        return toolCalls.map { tc ->
            mapOf(
                "id" to (tc["id"] ?: ""),
                "type" to "function",
                "function" to mapOf(
                    "name" to ((tc["function"] as? Map<String, Any?>)?.get("name") ?: tc["name"] ?: ""),
                    "arguments" to kotlinx.serialization.json.Json.encodeToString(
                        kotlinx.serialization.serializer<Map<String, Any?>>(),
                        ((tc["function"] as? Map<String, Any?>)?.get("arguments") ?: tc["args"]) as? Map<String, Any?> ?: emptyMap()
                    )
                )
            )
        }
    }

    /**
     * Format messages for OpenAI.
     */
    fun formatMessages(messages: List<Map<String, Any?>>): List<Map<String, Any?>> {
        return messages.map { msg ->
            mapOf(
                "role" to (msg["role"] ?: "user"),
                "content" to (msg["content"] ?: "")
            )
        }
    }
}
