package dspy.clients

import dspy.clients.utils_finetune.TrainDataFormat
import dspy.clients.utils_finetune.TrainingStatus
import dspy.clients.utils_finetune.saveData

/**
 * Embedder class for computing embeddings for text inputs.
 *
 * Provides a unified interface for both hosted embedding models (via litellm)
 * and custom embedding functions.
 */
class Embedder(
    val model: Any,
    val batchSize: Int = 200,
    val caching: Boolean = true,
    val kwargs: Map<String, Any?> = emptyMap()
) {
    /**
     * Compute embeddings for a list of strings.
     */
    fun invoke(inputs: List<String>): List<List<Float>> {
        return if (model is String) {
            embedWithLiteLLM(inputs)
        } else if (model is kotlin.jvm.functions.Function1<*, *>) {
            @Suppress("UNCHECKED_CAST")
            val func = model as kotlin.jvm.functions.Function1<Any, Any>
            val result = func.invoke(inputs)
            if (result is Array<*>) {
                (result as Array<DoubleArray>).map { it.map { v -> v.toFloat() } }
            } else if (result is List<*>) {
                result.map { row ->
                    if (row is List<*>) row.map { it as? Float ?: (it as? Double)?.toFloat() ?: 0f }
                    else listOf(0f)
                }
            } else {
                throw IllegalArgumentException("Embedding function must return 2D array")
            }
        } else {
            throw IllegalArgumentException("model must be a string or callable")
        }
    }

    private fun embedWithLiteLLM(inputs: List<String>): List<List<Float>> {
        // LiteLLM embedding call - placeholder for actual implementation
        throw NotImplementedError("LiteLLM embedding integration not yet implemented in Kotlin")
    }
}
