package dspy.adapters

import dspy.signatures.Signature
import kotlin.reflect.KClass

/**
 * BAML (Bare Metal Language) adapter for DSPy.
 *
 * Provides integration with BAML-generated code for type-safe LLM calls.
 */
class BamlAdapter(
    private val bamlPackage: Any? = null,
    private val bamlArgs: Map<String, Any?> = emptyMap()
) : Adapter() {
    override fun formatFieldDescription(signature: Signature): String {
        return "// BAML field description\n${signature.inputFields.joinToString("\n") { "${it.name}: ${it.annotation.simpleName ?: "Any"}" }}\n${signature.outputFields.joinToString("\n") { "${it.name}: ${it.annotation.simpleName ?: "Any"}" }}"
    }

    override fun formatFieldStructure(signature: Signature): String {
        return "// BAML field structure\n${signature.inputFields.joinToString("\n") { "- ${it.name}: ${it.annotation.simpleName ?: "Any"}" }}\n${signature.outputFields.joinToString("\n") { "- ${it.name}: ${it.annotation.simpleName ?: "Any"}" }}"
    }

    override fun formatTaskDescription(signature: Signature): String {
        return signature.instructions
    }

    override fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String,
        suffix: String,
        mainRequest: Boolean
    ): String {
        val sb = StringBuilder()
        if (prefix.isNotEmpty()) sb.append(prefix).append("\n")
        for (field in signature.inputFields) {
            if (field.name in inputs && inputs[field.name] != null) {
                sb.append("${field.name}: ${inputs[field.name]}\n")
            }
        }
        if (suffix.isNotEmpty()) sb.append(suffix)
        return sb.toString()
    }

    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?
    ): String {
        return signature.outputFields.map { field -> "${field.name}: ${outputs[field.name] ?: missingFieldMessage}" }.joinToString("\n")
    }

    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        return signature.outputFields.associate { it.name to null }
    }
}
