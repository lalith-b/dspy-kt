package dspy.signatures

import kotlin.reflect.KClass

/**
 * Utility functions for Signature manipulation.
 */
object SignatureUtils {
    /**
     * Validate signature field types.
     */
    fun validateSignature(signature: Signature): List<String> {
        val errors = mutableListOf<String>()
        if (signature.inputFields.isEmpty()) {
            errors.add("Signature must have at least one input field")
        }
        if (signature.outputFields.isEmpty()) {
            errors.add("Signature must have at least one output field")
        }
        val allKeys = (signature.inputFields + signature.outputFields).map { it.name }.toSet()
        val inputKeys = signature.inputFields.map { it.name }.toSet()
        val outputKeys = signature.outputFields.map { it.name }.toSet()
        if (inputKeys.intersect(outputKeys).isNotEmpty()) {
            errors.add("Input and output fields must not overlap")
        }
        return errors
    }

    /**
     * Validate field annotations.
     */
    fun validateFieldAnnotations(fields: List<FieldInfo>): List<String> {
        return fields.filter { field ->
            field.annotation == Any::class || field.annotation == KClass::class
        }.map { field ->
            "Field '${field.name}' has unsupported type annotation: ${field.annotation.simpleName}"
        }
    }

    /**
     * Get field type string.
     */
    fun fieldTypeString(field: FieldInfo): String {
        return field.annotation.simpleName ?: "Any"
    }

    /**
     * Convert signature to string representation.
     */
    fun signatureToString(signature: Signature): String {
        val sb = StringBuilder()
        if (signature.instructions.isNotBlank()) {
            sb.append("Instructions: ${signature.instructions}\n\n")
        }
        sb.append("Input fields:\n")
        for (field in signature.inputFields) {
            sb.append("  ${field.name}: ${fieldTypeString(field)}\n")
        }
        sb.append("Output fields:\n")
        for (field in signature.outputFields) {
            sb.append("  ${field.name}: ${fieldTypeString(field)}\n")
        }
        return sb.toString()
    }

    /**
     * Merge two signatures.
     */
    fun mergeSignatures(sig1: Signature, sig2: Signature): Signature {
        return Signature(
            instruction = sig1.instructions + if (sig2.instructions.isNotEmpty()) "\n" + sig2.instructions else "",
            inputFields = sig1.inputFields + sig2.inputFields,
            outputFields = sig1.outputFields + sig2.outputFields,
        )
    }

    /**
     * Ensure the given value is a Signature. If it's a string, parse it.
     */
    fun ensureSignature(signature: Any): Signature {
        return when (signature) {
            is Signature -> signature
            is String -> Signature.fromString(signature)
            else -> throw IllegalArgumentException("signature must be a Signature or a string")
        }
    }
}
