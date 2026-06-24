package dspy.adapters

import dspy.adapters.types.ToolCalls
import dspy.adapters.utils.formatFieldValue
import dspy.adapters.utils.getAnnotationName
import dspy.adapters.utils.getFieldDescriptionString
import dspy.adapters.utils.parseValue
import dspy.adapters.utils.translateFieldType
import dspy.clients.BaseLM
import dspy.core.AdapterParseError
import dspy.core.ContextWindowExceededError
import dspy.core.LMError
import dspy.signatures.FieldInfo
import dspy.signatures.Signature
import dspy.utils.CallbackHandler
import kotlin.reflect.KClass

/**
 * Default Adapter for most language models.
 *
 * The ChatAdapter formats DSPy signatures into a format compatible with most language models.
 * It uses delimiter patterns like `[[ ## field_name ## ]]` to clearly separate input and output fields in
 * the message content.
 *
 * Key features:
 * - Structures inputs and outputs using field header markers for clear field delineation.
 * - Provides automatic fallback to JSONAdapter if the chat format fails.
 */
open class ChatAdapter(
    callbacks: List<CallbackHandler>? = null,
    useNativeFunctionCalling: Boolean = false,
    nativeResponseTypes: List<KClass<*>>? = null,
    val useJsonAdapterFallback: Boolean = true,
    parallelToolCalls: Boolean? = null,
) : Adapter(
    callbacks = callbacks ?: emptyList(),
    useNativeFunctionCalling = useNativeFunctionCalling,
    nativeResponseTypes = nativeResponseTypes ?: emptyList(),
    parallelToolCalls = parallelToolCalls,
) {
    /**
     * Format the field description for the system message.
     */
    override fun formatFieldDescription(signature: Signature): String {
        return buildString {
            append("Your input fields are:\n${getFieldDescriptionString(signature.inputFields)}\n")
            append("Your output fields are:\n${getFieldDescriptionString(signature.outputFields)}")
        }
    }

    /**
     * Format the field structure for the system message.
     *
     * ChatAdapter requires input and output fields to be in their own sections, with section header using markers
     * `[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
     * output fields section to indicate the end of the output fields.
     */
    override fun formatFieldStructure(signature: Signature): String {
        val parts = mutableListOf<String>()
        parts.add("All interactions will be structured in the following way, with the appropriate values filled in.")
        parts.add(formatFieldsForStructure(signature.inputFields))
        parts.add(formatFieldsForStructure(signature.outputFields))
        parts.add("[[ ## completed ## ]]\n")
        return parts.joinToString("\n\n").trim()
    }

    private fun formatFieldsForStructure(fields: List<FieldInfo>): String {
        val output = fields.map { field ->
            val formattedValue = translateFieldType(field)
            "[[ ## ${field.name} ## ]]\n$formattedValue"
        }
        return output.joinToString("\n\n").trim()
    }

    /**
     * Format the task description for the system message.
     */
    override fun formatTaskDescription(signature: Signature): String {
        val instructions = signature.instruction.trimIndent()
        val objective = instructions.lines().joinToString("\n        ") { "        $it" }
        return "In adhering to this structure, your objective is: $objective"
    }

    /**
     * Format the user message content.
     */
    override fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String,
        suffix: String,
        mainRequest: Boolean,
    ): String {
        val messages = mutableListOf<String>()
        if (prefix.isNotEmpty()) messages.add(prefix)

        for (field in signature.inputFields) {
            val value = inputs[field.name]
            if (value != null) {
                val formattedFieldValue = formatFieldValue(field, value)
                messages.add("[[ ## ${field.name} ## ]]\n$formattedFieldValue")
            }
        }

        if (mainRequest) {
            val outputRequirements = userMessageOutputRequirements(signature)
            if (outputRequirements != null) {
                messages.add(outputRequirements)
            }
        }

        if (suffix.isNotEmpty()) messages.add(suffix)
        return messages.joinToString("\n\n").trim()
    }

    /**
     * Returns a simplified format reminder for the language model.
     */
    fun userMessageOutputRequirements(signature: Signature): String? {
        if (signature.outputFields.isEmpty()) return null

        val typeInfos = signature.outputFields.map { field ->
            val typeInfo = when {
                field.annotation == ToolCalls::class ->
                    " (must be a JSON object like {\"tool_calls\": [{\"name\": \"...\", \"args\": {...}}]})"
                field.annotation != String::class ->
                    " (must be formatted as a valid Kotlin ${getAnnotationName(field.annotation)})"
                else -> ""
            }
            "`[[ ## ${field.name} ## ]]`$typeInfo"
        }

        return buildString {
            append("Respond with the corresponding output fields, starting with the field ")
            append(typeInfos.joinToString(", then "))
            append(", and then ending with the marker for `[[ ## completed ## ]]`.")
        }
    }

    /**
     * Format the assistant message content.
     */
    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?,
    ): String {
        val content = signature.outputFields.map { field ->
            val value = outputs[field.name] ?: missingFieldMessage
            val formattedFieldValue = formatFieldValue(field, value)
            "[[ ## ${field.name} ## ]]\n$formattedFieldValue"
        }.joinToString("\n\n").trim()

        return "$content\n\n[[ ## completed ## ]]\n"
    }

    /**
     * Parse the LM output into a dictionary of the output fields.
     *
     * Uses the `[[ ## field_name ## ]]` delimiter pattern to split the response into sections.
     */
    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        // Split by field header pattern [[ ## field_name ## ]]
        val fieldHeaderPattern = Regex("\\[\\[ ## (\\w+) ## \\]\\]")
        val lines = completion.split("\n")

        val parsedSections = mutableListOf<Pair<String?, MutableList<String>>>()
        parsedSections.add(Pair(null, mutableListOf()))

        for (line in lines) {
            val trimmed = line.trim()
            val match = fieldHeaderPattern.find(trimmed)
            if (match != null) {
                val header = match.groupValues[1]
                // Get content after the match
                val remainingContent = trimmed.substring(match.range.last + 1).trim()
                parsedSections.add(Pair(header, if (remainingContent.isEmpty()) mutableListOf() else mutableListOf(remainingContent)))
            } else {
                parsedSections.last().second.add(line)
            }
        }

        // Join section contents
        val sectionContents = parsedSections.map { (k, v) ->
            k to v.joinToString("\n").trim()
        }

        // Parse fields
        val fields = mutableMapOf<String, Any?>()
        for ((key, value) in sectionContents) {
            if (key != null && key in signature.outputFields.map { it.name }.toSet() && key !in fields) {
                val field = signature.outputFields.find { it.name == key }
                try {
                    fields[key] = parseValue(value, field?.annotation ?: String::class)
                } catch (e: Exception) {
                    throw AdapterParseError(
                        adapterName = "ChatAdapter",
                        signature = signature,
                        lmResponse = completion,
                        message = "Failed to parse field $key with value $value from the LM response. Error message: ${e.message}",
                    )
                }
            }
        }

        // Check all output fields were parsed
        val outputFieldNames = signature.outputFields.map { it.name }.toSet()
        if (fields.keys != outputFieldNames) {
            throw AdapterParseError(
                adapterName = "ChatAdapter",
                signature = signature,
                lmResponse = completion,
                parsedResult = fields,
                message = "Missing output fields: ${outputFieldNames - fields.keys}",
            )
        }

        return fields
    }

    /**
     * Format the call data into finetuning data according to the OpenAI API specifications.
     */
    fun formatFinetuneData(
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>,
        outputs: Map<String, Any?>,
    ): Map<String, Any> {
        val systemUserMessages = format(signature, demos, inputs)
        val assistantMessageContent = formatAssistantMessageContent(signature, outputs, missingFieldMessage = null)
        val assistantMessage = mapOf("role" to "assistant", "content" to assistantMessageContent)
        return mapOf("messages" to (systemUserMessages + listOf(assistantMessage)))
    }

    /**
     * Format field with value (used by formatFieldStructure for custom field-value mappings).
     */
    fun formatFieldWithValue(
        fieldsWithValues: Map<Pair<String, FieldInfo>, Any?>,
    ): String {
        return fieldsWithValues.map { (fieldKey, value) ->
            val formattedFieldValue = formatFieldValue(fieldKey.second, value)
            "[[ ## ${fieldKey.first} ## ]]\n$formattedFieldValue"
        }.joinToString("\n\n").trim()
    }

    /**
     * Call the adapter with JSON adapter fallback support.
     * Wraps the parent's [invoke] to add fallback on non-LM errors.
     */
    suspend fun callWithFallback(
        lm: BaseLM,
        lmKwargs: Map<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>,
    ): List<Map<String, Any?>> {
        return try {
            invoke(lm, lmKwargs.toMutableMap(), signature, demos, inputs)
        } catch (e: Exception) {
            if (e is LMError && e !is ContextWindowExceededError) throw e
            if (!useJsonAdapterFallback) throw e
            // In Python, this creates a JSONAdapter and retries
            // For now, re-raise since JSONAdapter would need its own implementation
            throw e
        }
    }

    /**
     * Async variant of [callWithFallback].
     */
    suspend fun acallWithFallback(
        lm: BaseLM,
        lmKwargs: Map<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>,
    ): List<Map<String, Any?>> {
        return try {
            acall(lm, lmKwargs.toMutableMap(), signature, demos, inputs)
        } catch (e: Exception) {
            if (e is LMError && e !is ContextWindowExceededError) throw e
            if (!useJsonAdapterFallback) throw e
            throw e
        }
    }
}
