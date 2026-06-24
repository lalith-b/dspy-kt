package dspy.teleprompt.gepa

import dspy.primitives.Example
import dspy.primitives.Prediction

/**
 * GEPA (Generative Pseudo-Alignment) utility types and functions.
 */

/**
 * Reflective example containing inputs, generated outputs, and feedback.
 */
typealias ReflectiveExample = Map<String, Any?>

/**
 * Trace data for GEPA optimization.
 */
data class TraceData(
    val trace: List<String>,
    val score: Double,
    val feedback: String
)

/**
 * Failed prediction with error information.
 */
data class FailedPrediction(
    val inputs: Map<String, Any?>,
    val outputs: Map<String, Any?>,
    val error: String
)

/**
 * Score with feedback for evaluation.
 */
data class ScoreWithFeedback(
    val score: Float,
    val feedback: String
) : Prediction(mapOf("score" to score, "feedback" to feedback))

/**
 * GEPA adapter wrapper.
 */
class GEPAAdapter(
    private val logger: java.util.logging.Logger = java.util.logging.Logger.getLogger("gepa")
) {
    fun log(message: String) {
        logger.info(message)
    }
}

/**
 * Evaluation batch for GEPA.
 */
data class EvaluationBatch(
    val examples: List<Map<String, Any?>>,
    val predictions: List<Map<String, Any?>>,
    val scores: List<Double>
)

/**
 * DSPy trace type.
 */
typealias DSPyTrace = List<Triple<Any?, Map<String, Any?>, Prediction>>

/**
 * Create a reflective example.
 */
fun createReflectiveExample(
    inputs: Map<String, Any?>,
    outputs: Map<String, Any?>,
    feedback: String
): ReflectiveExample {
    return mapOf(
        "Inputs" to inputs,
        "Generated Outputs" to outputs,
        "Feedback" to feedback
    )
}

/**
 * Format trace for display.
 */
fun formatTrace(trace: DSPyTrace): String {
    return trace.joinToString("\n---\n") { (input, kwargs, prediction) ->
        "Input: $input\nKwargs: $kwargs\nPrediction: ${prediction.toDict()}"
    }
}
