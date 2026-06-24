package dspy.adapters

import dspy.adapters.types.ToolCalls
import dspy.clients.BaseLM
import dspy.core.AdapterParseError
import dspy.core.LMError
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.utils.CallbackHandler
import kotlin.reflect.KClass

/**
 * A two-stage adapter that:
 *   1. Uses a simpler, more natural prompt for the main LM
 *   2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
 *
 * This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
 * are known to struggle with structured outputs.
 */
class TwoStepAdapter(
    val extractionModel: BaseLM,
    callbacks: List<CallbackHandler> = emptyList(),
    useNativeFunctionCalling: Boolean = false,
    parallelToolCalls: Boolean? = null
) : Adapter(callbacks ?: emptyList(), useNativeFunctionCalling, emptyList(), parallelToolCalls) {
    
    override fun format(
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>
    ): List<Map<String, Any?>> {
        val messages = mutableListOf<Map<String, Any?>>()
        messages.add(mapOf("role" to "system", "content" to formatTaskDescription(signature)))
        messages.addAll(formatDemos(signature, demos))
        messages.add(mapOf("role" to "user", "content" to formatUserMessageContent(signature, inputs)))
        return messages
    }

    override fun formatFieldDescription(signature: Signature): String = ""

    override fun formatFieldStructure(signature: Signature): String = ""

    override fun formatTaskDescription(signature: Signature): String {
        val parts = mutableListOf<String>()
        parts.add("You are a helpful assistant that can solve tasks based on user input.")
        parts.add("As input, you will be provided with:\n${signature.inputFields.joinToString("\n") { "${it.name}: ${it.desc ?: it.name}" }}")
        parts.add("Your outputs must contain:\n${signature.outputFields.joinToString("\n") { "${it.name}: ${it.desc ?: it.name}" }}")
        parts.add("You should lay out your outputs in detail so that your answer can be understood by another agent")
        if (signature.instructions.isNotBlank()) {
            parts.add("Specific instructions: ${signature.instructions}")
        }
        return parts.joinToString("\n")
    }

    override fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String,
        suffix: String,
        mainRequest: Boolean
    ): String {
        val parts = mutableListOf<String>()
        if (prefix.isNotEmpty()) parts.add(prefix)
        for (field in signature.inputFields) {
            if (field.name in inputs) {
                parts.add("${field.name}: ${inputs[field.name] ?: ""}")
            }
        }
        if (suffix.isNotEmpty()) parts.add(suffix)
        return parts.joinToString("\n\n")
    }

    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?
    ): String {
        return signature.outputFields
            .map { field -> "${field.name}: ${outputs[field.name] ?: missingFieldMessage}" }
            .joinToString("\n\n")
    }

    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        val extractorSignature = createExtractorSignature(signature)
        return try {
            val chatAdapter = ChatAdapter()
            val parsed = chatAdapter.invoke(
                lm = extractionModel,
                lmKwargs = mutableMapOf(),
                signature = extractorSignature,
                demos = emptyList(),
                inputs = mapOf("text" to completion)
            )
            parsed.firstOrNull() ?: emptyMap()
        } catch (e: LMError) {
            throw e
        } catch (e: Exception) {
            throw AdapterParseError(
                message = "Failed to parse response from the original completion: ${e.message}",
                adapterName = "TwoStepAdapter",
                signature = signature,
                lmResponse = completion
            )
        }
    }

    private fun createExtractorSignature(originalSignature: Signature): Signature {
        // Create a new signature with "text" input field and all output fields
        val inputFields = listOf(
            InputField(name = "text", desc = "The text containing the information to extract", annotation = String::class)
        )
        val outputFields = originalSignature.outputFields.map { field ->
            OutputField(name = field.name, desc = field.desc, annotation = field.annotation)
        }
        val outputsStr = originalSignature.outputFields.joinToString(", ") { "`${it.name}`" }
        val instructions = "The input is a text that should contain all the necessary information to produce the fields $outputsStr. Your job is to extract the fields from the text verbatim."
        return Signature(instruction = instructions, inputFields = inputFields, outputFields = outputFields)
    }
}
