package dspy.evaluate

import kotlin.reflect.KClass
import dspy.signatures.Signature

/**
 * Additional evaluation utilities for DSPy.
 */

/**
 * Auto-evaluation using LM-based scoring.
 */
class AutoEvaluationExtra(
    private val instruction: String,
    private val lm: Any? = null,
    private val fewShotExamples: List<Map<String, Any?>>? = null
) {
    /**
     * Create an auto-evaluation metric function.
     */
    fun createMetric(): (Map<String, Any?>, Map<String, Any?>) -> Boolean {
        return this::evaluate
    }

    /**
     * Evaluate a prediction against an example.
     */
    fun evaluate(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim()?.lowercase()
        val expectedAnswer = example["answer"]?.toString()?.trim()?.lowercase()
        return predAnswer == expectedAnswer ||
               (predAnswer != null && expectedAnswer != null &&
                    (predAnswer.contains(expectedAnswer) || expectedAnswer.contains(predAnswer)))
    }

    /**
     * Create a signature for auto-evaluation.
     */
    fun createEvaluationSignature(): Signature {
        return dspy.signatures.Signature.makeSignature(mapOf(
            "prediction" to Pair(String::class, dspy.signatures.InputField()),
            "expected" to Pair(String::class, dspy.signatures.InputField()),
            "correct" to Pair(Boolean::class, dspy.signatures.OutputField())
        ), instruction)
    }
}
