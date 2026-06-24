package dspy.adapters

import dspy.adapters.utils.parseValue
import dspy.adapters.utils.translateFieldType
import dspy.core.AdapterParseError
import dspy.signatures.FieldInfo
import dspy.signatures.Signature
import dspy.utils.CallbackHandler
import kotlin.reflect.KClass

/**
 * XMLAdapter - wraps input and output fields in XML tags like `<field_name>`.
 */
class XMLAdapter(
    callbacks: List<CallbackHandler> = emptyList(),
    useNativeFunctionCalling: Boolean = false,
    parallelToolCalls: Boolean? = null
) : ChatAdapter(callbacks ?: emptyList(), useNativeFunctionCalling, nativeResponseTypes = null, parallelToolCalls = parallelToolCalls) {
    companion object {
        val fieldPattern = Regex("<(\\w+)>(.*?)</\\1>", RegexOption.DOT_MATCHES_ALL)
    }

    override fun formatFieldDescription(signature: Signature): String {
        return "Your input fields are:\n${signature.inputFields.joinToString("\n") { "<${it.name}>${it.desc}</${it.name}>" }}\nYour output fields are:\n${signature.outputFields.joinToString("\n") { "<${it.name}>${it.desc}</${it.name}>" }}"
    }

    override fun formatFieldStructure(signature: Signature): String {
        val parts = mutableListOf<String>()
        parts.add("All interactions will be structured in the following way, with the appropriate values filled in.")
        parts.add(signature.inputFields.joinToString("\n\n") { field ->
            "<${field.name}>\n${translateFieldType(field)}\n</${field.name}>"
        })
        parts.add(signature.outputFields.joinToString("\n\n") { field ->
            "<${field.name}>\n${translateFieldType(field)}\n</${field.name}>"
        })
        return parts.joinToString("\n\n")
    }

    override fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String,
        suffix: String,
        mainRequest: Boolean
    ): String {
        val messages = mutableListOf<String>()
        if (prefix.isNotEmpty()) messages.add(prefix)
        for (field in signature.inputFields) {
            if (field.name in inputs) {
                messages.add("<${field.name}>\n${inputs[field.name]}\n</${field.name}>")
            }
        }
        if (mainRequest) {
            val outputRequirements = signature.outputFields.joinToString(", ") { "`${it.name}`" }
            messages.add("Respond with the corresponding output fields wrapped in XML tags $outputRequirements.")
        }
        if (suffix.isNotEmpty()) messages.add(suffix)
        return messages.joinToString("\n\n")
    }

    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?
    ): String {
        return signature.outputFields.map { field ->
            "<${field.name}>\n${outputs[field.name] ?: missingFieldMessage}\n</${field.name}>"
        }.joinToString("\n\n")
    }

    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        val fields = mutableMapOf<String, Any?>()
        for (match in fieldPattern.findAll(completion)) {
            val name = match.groupValues[1]
            val content = match.groupValues[2].trim()
            if (name in signature.outputFields.map { it.name } && name !in fields) {
                val field = signature.outputFields.find { it.name == name }
                fields[name] = try {
                    parseValue(content, field?.annotation ?: String::class)
                } catch (e: Exception) {
                    throw AdapterParseError(
                        message = "Failed to parse field $name with value $content: ${e.message}",
                        adapterName = "XMLAdapter",
                        signature = signature,
                        lmResponse = completion
                    )
                }
            }
        }

        if (fields.keys.toSet() != signature.outputFields.map { it.name }.toSet()) {
            throw AdapterParseError(
                message = "Missing output fields: ${signature.outputFields.map { it.name }.toSet() - fields.keys}",
                adapterName = "XMLAdapter",
                signature = signature,
                lmResponse = completion,
                parsedResult = fields
            )
        }
        return fields
    }
}
