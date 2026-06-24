package dspy.adapters

import dspy.adapters.types.ToolCalls
import dspy.adapters.utils.formatFieldValue
import dspy.adapters.utils.getAnnotationName
import dspy.adapters.utils.parseValue
import dspy.adapters.utils.serializeForJson
import dspy.adapters.utils.deserializeFromJson
import dspy.adapters.utils.translateFieldType
import dspy.clients.BaseLM
import dspy.core.AdapterParseError
import dspy.core.LMError
import dspy.signatures.FieldInfo
import dspy.signatures.Signature
import dspy.utils.CallbackHandler
import kotlinx.serialization.json.Json
import kotlin.reflect.KClass

/**
 * JSONAdapter - uses JSON mode or structured output mode for LM responses.
 */
class JSONAdapter(
    callbacks: List<CallbackHandler> = emptyList(),
    useNativeFunctionCalling: Boolean = true,
    parallelToolCalls: Boolean? = null
) : ChatAdapter(callbacks = callbacks ?: emptyList(), useNativeFunctionCalling = useNativeFunctionCalling, parallelToolCalls = parallelToolCalls) {
    
    private fun jsonAdapterCallCommon(
        lm: BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>,
        callFn: () -> List<Map<String, Any?>>
    ): List<Map<String, Any?>>? {
        if ("response_format" !in lm.supportedParams) return null

        val hasToolCalls = signature.outputFields.any { it.annotation == ToolCalls::class }
        val hasOpenEndedMapping = signature.outputFields.any { it.annotation == Map::class }

        if (hasOpenEndedMapping || (!useNativeFunctionCalling && hasToolCalls) || !lm.supportsResponseSchema) {
            lmKwargs["response_format"] = mapOf("type" to "json_object")
            return callFn()
        }
        return null
    }

    override fun invoke(
        lm: BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val result = jsonAdapterCallCommon(lm, lmKwargs, signature, demos, inputs) {
            super.invoke(lm, lmKwargs, signature, demos, inputs)
        }
        if (result != null) return result

        try {
            val structuredOutputModel = getStructuredOutputsResponseFormat(signature, useNativeFunctionCalling)
            lmKwargs["response_format"] = structuredOutputModel
            return super.invoke(lm, lmKwargs, signature, demos, inputs)
        } catch (e: LMError) {
            throw e
        } catch (e: Exception) {
            println("Warning: Failed to use structured output format, falling back to JSON mode.")
            lmKwargs["response_format"] = mapOf("type" to "json_object")
            return super.invoke(lm, lmKwargs, signature, demos, inputs)
        }
    }

    override fun formatFieldStructure(signature: Signature): String {
        val parts = mutableListOf<String>()
        parts.add("All interactions will be structured in the following way, with the appropriate values filled in.")
        parts.add("Inputs will have the following structure:")
        parts.add(formatFieldsForStructure(signature.inputFields))
        parts.add("Outputs will be a JSON object with the following fields.")
        parts.add(formatFieldsForStructure(signature.outputFields))
        return parts.joinToString("\n\n")
    }

    private fun formatFieldsForStructure(fields: List<FieldInfo>): String {
        return fields.map { field ->
            "${field.name}: ${translateFieldType(field)}"
        }.joinToString("\n")
    }

    private val json = Json { prettyPrint = true }

    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?
    ): String {
        val obj = kotlinx.serialization.json.buildJsonObject {
            for (field in signature.outputFields) {
                val value = outputs[field.name] ?: missingFieldMessage
                put(field.name, serializeForJson(value))
            }
        }
        return obj.toString()
    }

    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        var fields: Map<String, Any?>
        try {
            val json = Json { ignoreUnknownKeys = true }
            val obj = json.decodeFromString<kotlinx.serialization.json.JsonObject>(completion)
            fields = obj.mapValues { (_, v) -> deserializeFromJson(v) }
        } catch (_: Exception) {
            // Try to extract JSON object from text
            val jsonMatch = Regex("""\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}""").find(completion)
            if (jsonMatch != null) {
                val json = Json { ignoreUnknownKeys = true }
                val obj = json.decodeFromString<kotlinx.serialization.json.JsonObject>(jsonMatch.value)
                fields = obj.mapValues { (_, v) -> deserializeFromJson(v) }
            } else {
                throw AdapterParseError(
                    message = "LM response cannot be serialized to a JSON object.",
                    adapterName = "JSONAdapter",
                    signature = signature,
                    lmResponse = completion
                )
            }
        }

        fields = fields.filterKeys { it in signature.outputFields.map { f -> f.name } }

        val parsed = fields.toMutableMap()
        for ((k, v) in parsed) {
            if (k in signature.outputFields.map { it.name }) {
                parsed[k] = parseValue(v.toString(), signature.outputFields.find { it.name == k }?.annotation ?: String::class)
            }
        }

        if (parsed.keys.toSet() != signature.outputFields.map { it.name }.toSet()) {
            throw AdapterParseError(
                message = "Missing output fields: ${signature.outputFields.map { it.name }.toSet() - parsed.keys}",
                adapterName = "JSONAdapter",
                signature = signature,
                lmResponse = completion,
                parsedResult = parsed
            )
        }

        return parsed
    }

    fun formatFieldWithValue(fieldsWithValues: Map<String, Any?>, role: String = "user"): String {
        if (role == "user") {
            return fieldsWithValues.entries.joinToString("\n\n") { (name, value) ->
                "[[ ## $name ## ]]\n$value"
            }
        } else {
            val obj = kotlinx.serialization.json.buildJsonObject {
                for ((k, v) in fieldsWithValues) put(k, serializeForJson(v))
            }
            return obj.toString()
        }
    }
}

fun getStructuredOutputsResponseFormat(
    signature: Signature,
    useNativeFunctionCalling: Boolean = true
): Map<String, KClass<*>> {
    for (field in signature.outputFields) {
        if (field.annotation == Map::class) {
            throw IllegalArgumentException("Field '${field.name}' has an open-ended mapping type which is not supported by Structured Outputs.")
        }
    }

    return signature.outputFields
        .filter { useNativeFunctionCalling || it.annotation != ToolCalls::class }
        .associate { it.name to it.annotation }
}
