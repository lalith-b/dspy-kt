package dspy.evaluate

import dspy.primitives.Example
import dspy.primitives.Prediction
import dspy.primitives.Module

/**
 * A class that represents the result of an evaluation.
 */
class EvaluationResult(
    val score: Double,
    val results: List<Triple<Example, Prediction, Any?>>,
) : Prediction(mapOf("score" to score, "results" to results)) {
    override fun toString(): String {
        return "EvaluationResult(score=$score, results=<list of ${results.size} results>)"
    }
}

/**
 * DSPy Evaluate class.
 */
class Evaluate(
    devset: List<Example>,
    metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    numThreads: Int? = null,
    displayProgress: Boolean = false,
    displayTable: Boolean = false,
    maxErrors: Int? = null,
    provideTraceback: Boolean? = null,
    failureScore: Double = 0.0,
) : Module() {
    val devset: List<Example> = devset
    val metric: ((Example, Prediction, List<Any>?) -> Any?)? = metric
    val numThreads: Int? = numThreads
    val displayProgress: Boolean = displayProgress
    val displayTable: Boolean = displayTable
    val maxErrors: Int? = maxErrors
    val provideTraceback: Boolean? = provideTraceback
    val failureScore: Double = failureScore

    @Suppress("UNCHECKED_CAST")
    suspend fun __call__(
        program: Module,
        metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
        devset: List<Example>? = null,
        numThreads: Int? = null,
        displayProgress: Boolean? = null,
        displayTable: Boolean? = null,
    ): EvaluationResult {
        val m = metric ?: this.metric
        val d = devset ?: this.devset

        var totalScore = 0.0
        val results = mutableListOf<Triple<Example, Prediction, Any?>>()

        for (example in d) {
            try {
                val prediction = program(example.toMap())
                val score = if (m != null) m(example, prediction, null) else null
                totalScore += (score as? Number)?.toDouble() ?: 0.0
                results.add(Triple(example, prediction, score))
            } catch (e: Exception) {
                results.add(Triple(example, Prediction(emptyMap()), failureScore))
            }
        }

        val averageScore = if (d.isNotEmpty()) totalScore / d.size else 0.0
        return EvaluationResult(averageScore, results)
    }
}

/**
 * Auto-evaluation utilities for DSPy.
 */
object AutoEvaluation {
    fun createAutoMetric(
        instruction: String,
        lm: Any? = null,
        fewShotExamples: List<Map<String, Any?>>? = null,
    ): (Example, Prediction) -> Boolean {
        return { pred, example ->
            val predAnswer = pred["answer"]?.toString()?.trim()?.lowercase()
            val expectedAnswer = example["answer"]?.toString()?.trim()?.lowercase()
            predAnswer == expectedAnswer ||
                (predAnswer != null && expectedAnswer != null &&
                    (predAnswer.contains(expectedAnswer) || expectedAnswer.contains(predAnswer)))
        }
    }
}

/**
 * Metric functions for evaluation.
 */
object Metrics {
    fun exactMatch(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        return (pred["answer"]?.toString()?.trim()?.lowercase()) ==
               (example["answer"]?.toString()?.trim()?.lowercase())
    }

    fun contains(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim()?.lowercase() ?: return false
        val expectedAnswer = example["answer"]?.toString()?.trim()?.lowercase() ?: return false
        return predAnswer.contains(expectedAnswer)
    }

    fun multiChoice(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim()?.lowercase() ?: return false
        val expectedAnswer = example["answer"]?.toString()?.trim()?.lowercase() ?: return false
        return predAnswer == expectedAnswer
    }

    fun numeric(pred: Map<String, Any?>, example: Map<String, Any?>, tolerance: Double = 0.01): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim()?.toDoubleOrNull() ?: return false
        val expectedAnswer = example["answer"]?.toString()?.trim()?.toDoubleOrNull() ?: return false
        return kotlin.math.abs(predAnswer - expectedAnswer) < tolerance
    }
}
