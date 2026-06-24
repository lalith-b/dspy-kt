package dspy.adapters.types

import dspy.adapters.CUSTOM_TYPE_START_IDENTIFIER
import dspy.adapters.CUSTOM_TYPE_END_IDENTIFIER
import kotlin.reflect.KClass
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

abstract class Type {
    /**
     * Format the type for LM consumption.
     * Returns a list of dictionaries (same as Array of content parts in the OpenAI API user message's content field)
     * or a string.
     */
    abstract fun format(): Any

    /**
     * Description of the custom type for use in prompts.
     */
    open fun description(): String = ""

    /**
     * Serialize the model using custom type identifiers.
     */
    open fun serializeModel(): String {
        val formatted = format()
        return if (formatted is List<*>) {
            val jsonStrings = formatted.map { item ->
                if (item is Map<*, *>) {
                    val pairs = item.entries.joinToString(", ") { (k, v) ->
                        val key = k.toString().replace("\"", "\\\"")
                        val value = when (v) {
                            null -> "null"
                            is String -> "\"${v.replace("\"", "\\\"")}\""
                            is Number -> v.toString()
                            is Boolean -> v.toString()
                            else -> "\"${v.toString().replace("\"", "\\\"")}\""
                        }
                        "\"$key\": $value"
                    }
                    "{$pairs}"
                } else {
                    "\"${item.toString().replace("\"", "\\\"")}\""
                }
            }
            "$CUSTOM_TYPE_START_IDENTIFIER[${jsonStrings.joinToString(", ")}]$CUSTOM_TYPE_END_IDENTIFIER"
        } else {
            formatted.toString()
        }
    }

    /**
     * Adapt the custom type to the native LM feature if possible.
     */
    open fun adaptToNativeLmFeature(signature: Any, fieldName: String, lm: Any, lmKwargs: MutableMap<String, Any?>): Any {
        return signature
    }

    /**
     * Whether the custom type is streamable.
     */
    open fun isStreamable(): Boolean = false

    /**
     * Parse a stream chunk into the custom type.
     */
    open fun parseStreamChunk(chunk: Any): Type? = null

    /**
     * Parse a LM response into the custom type.
     */
    open fun parseLmResponse(response: Any): Type? = null

    /**
     * Extract all custom types from the annotation.
     */
    companion object {
        fun extractCustomTypeFromAnnotation(annotation: KClass<*>): List<KClass<out Type>> {
            if (Type::class.java.isAssignableFrom(annotation.java)) {
                return listOf(annotation as KClass<out Type>)
            }
            return emptyList()
        }
    }
}

/**
 * Split user message content into a list of content blocks around custom types content.
 */
fun splitMessageContentForCustomTypes(messages: List<Map<String, Any?>>): List<Map<String, Any?>> {
    val result = messages.toMutableList()
    val pattern = Regex("$CUSTOM_TYPE_START_IDENTIFIER(.*?)$CUSTOM_TYPE_END_IDENTIFIER", RegexOption.DOT_MATCHES_ALL)
    
    for (i in result.indices) {
        val message = result[i]
        if (message["role"] != "user") continue
        
        val content = message["content"] as? String ?: continue
        val blocks = mutableListOf<Map<String, Any?>>()
        var lastEnd = 0
        
        for (match in pattern.findAll(content)) {
            val start = match.range.first
            val end = match.range.last
            if (start > lastEnd) {
                blocks.add(mapOf("type" to "text", "text" to content.substring(lastEnd, start)))
            }
            val customTypeContent = match.groupValues[1].trim()
            try {
                val parsed = Json { ignoreUnknownKeys = true }.parseToJsonElement(customTypeContent)
                if (parsed is JsonArray) {
                    for (element in parsed) {
                        if (element is JsonObject) {
                            blocks.add(element.toMap())
                        }
                    }
                } else {
                    blocks.add(mapOf("type" to "text", "text" to customTypeContent))
                }
            } catch (_: Exception) {
                blocks.add(mapOf("type" to "text", "text" to customTypeContent))
            }
            lastEnd = end
        }
        
        if (lastEnd == 0) continue
        if (lastEnd < content.length) {
            blocks.add(mapOf("type" to "text", "text" to content.substring(lastEnd)))
        }
        
        val mutableMessage = message.toMutableMap()
        mutableMessage["content"] = blocks
        result[i] = mutableMessage
    }
    return result
}

private fun JsonObject.toMap(): Map<String, Any?> {
    return this.mapValues { (_, value) ->
        when (value) {
            is kotlinx.serialization.json.JsonPrimitive -> value.content
            is JsonObject -> value.toMap()
            is JsonArray -> value.map {
                when (it) {
                    is kotlinx.serialization.json.JsonPrimitive -> it.content
                    is JsonObject -> it.toMap()
                    is JsonArray -> it.map { e -> (e as? kotlinx.serialization.json.JsonPrimitive)?.content ?: e }
                    else -> null
                }
            }
            else -> null
        }
    }
}
