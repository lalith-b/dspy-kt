package dspy.adapters.types

import dspy.clients.BaseLM
import kotlinx.serialization.Serializable

/**
 * Reasoning type in DSPy.
 *
 * This type is useful when you want the DSPy output to include the reasoning of the LM.
 * This is a str-like type that delegates string operations to the underlying content.
 */
@Serializable
data class Reasoning(
    val content: String
) : Type() {
    override fun format(): String {
        return content
    }

    companion object {
        fun adaptToNativeLmFeature(
            signature: Any,
            fieldName: String,
            lm: BaseLM,
            lmKwargs: MutableMap<String, Any?>
        ): Any {
            var reasoningEffort: Any? = lmKwargs["reasoning_effort"]
                ?: lm.kwargs["reasoning_effort"]
                ?: "low"

            if (reasoningEffort == null || !lm.supportsReasoning) {
                return signature
            }

            if (lm.model.contains("gpt-5") && lm.modelType == "chat") {
                return signature // GPT-5 chat workaround
            }

            lmKwargs["reasoning_effort"] = reasoningEffort
            return signature // In practice, delete the field from the signature
        }

        fun parseLmResponse(response: Any): Reasoning? {
            if (response is Map<*, *>) {
                val reasoningContent = response["reasoning_content"] as? String
                return reasoningContent?.let { Reasoning(content = it) }
            }
            return null
        }

        fun isStreamable(): Boolean = true

        fun parseStreamChunk(chunk: Any): String? {
            return try {
                val delta = (chunk as? Map<*, *>)?.get("choices") as? List<*>
                val firstChoice = delta?.get(0) as? Map<*, *>
                firstChoice?.get("delta") as? Map<*, *>
            } catch (_: Exception) {
                null
            }?.get("reasoning_content") as? String
        }
    }

    // String-like operations
    override fun toString(): String = content
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other is Reasoning) return content == other.content
        if (other is String) return content == other
        return false
    }
    override fun hashCode(): Int = content.hashCode()
    operator fun plus(other: Reasoning): Reasoning = Reasoning(content + other.content)
    operator fun plus(other: String): String = content + other
    operator fun plusAssign(other: Reasoning) { /* immutable, no-op */ }
    operator fun get(index: Int): Char = content[index]
    operator fun contains(item: Char): Boolean = item in content
    operator fun iterator(): Iterator<Char> = content.iterator()
    fun length(): Int = content.length
}
