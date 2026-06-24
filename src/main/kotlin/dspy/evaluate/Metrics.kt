package dspy.evaluate

import dspy.primitives.Example
import dspy.primitives.Prediction
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import org.slf4j.LoggerFactory
import java.text.Normalizer

private val logger = LoggerFactory.getLogger(Metrics::class.java)

/**
 * Compute the Exact Match (EM) metric between a prediction and reference answers.
 *
 * Returns true if any reference exactly matches the prediction after normalization;
 * otherwise false. Normalization applies Unicode NFD, lowercasing, punctuation
 * removal, English article removal ("a", "an", "the"), and whitespace collapse.
 *
 * Port of `dspy/evaluate/metrics.py::EM`
 */
fun EM(prediction: String, answersList: List<String>): Boolean {
    require(answersList is List<*>) { "`answersList` must be a list, got ${answersList::class}" }
    return answersList.maxOfOrNull { emScore(prediction, it) } ?: false
}

/**
 * Compute the maximum token-level F1 score against reference answers.
 *
 * Port of `dspy/evaluate/metrics.py::F1`
 */
fun F1(prediction: String, answersList: List<String>): Double {
    require(answersList is List<*>) { "`answersList` must be a list, got ${answersList::class}" }
    return answersList.map { f1Score(prediction, it) }.maxOrNull() ?: 0.0
}

/**
 * Compute the maximum HotPotQA-style F1 score against reference answers.
 *
 * Port of `dspy/evaluate/metrics.py::HotPotF1`
 */
fun HotPotF1(prediction: String, answersList: List<String>): Double {
    require(answersList is List<*>) { "`answersList` must be a list, got ${answersList::class}" }
    return answersList.map { hotpotF1Score(prediction, it) }.maxOrNull() ?: 0.0
}

/**
 * Normalize text for string and token comparisons.
 *
 * Steps:
 * 1) Unicode NFD normalization
 * 2) Lowercasing
 * 3) Punctuation removal
 * 4) English article removal ("a", "an", "the")
 * 5) Whitespace collapse
 *
 * Port of `dspy/evaluate/metrics.py::normalize_text`
 */
fun normalizeText(s: String): String {
    val step1 = Normalizer.normalize(s, Normalizer.Form.NFD)
    val step2 = step1.lowercase()
    val step3 = step2.replace(Regex("[^\\p{L}\\p{N}\\s]"), "")
    val step4 = step3.replace(Regex("\\b(a|an|the)\\b", RegexOption.IGNORE_CASE), " ")
    return step4.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
}

/**
 * Compute boolean exact match after normalization.
 *
 * Port of `dspy/evaluate/metrics.py::em_score`
 */
fun emScore(prediction: String, groundTruth: String): Boolean {
    return normalizeText(prediction) == normalizeText(groundTruth)
}

/**
 * Compute token-level F1 between prediction and reference (after normalization).
 *
 * Port of `dspy/evaluate/metrics.py::f1_score`
 */
fun f1Score(prediction: String, groundTruth: String): Double {
    val predictionTokens = normalizeText(prediction).split(Regex("\\s+")).filter { it.isNotEmpty() }
    val groundTruthTokens = normalizeText(groundTruth).split(Regex("\\s+")).filter { it.isNotEmpty() }

    if (predictionTokens.isEmpty() && groundTruthTokens.isEmpty()) {
        logger.warn("#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.")
    }

    val predCounter = predictionTokens.groupingBy { it }.eachCount()
    val gtCounter = groundTruthTokens.groupingBy { it }.eachCount()

    val common = predCounter.filterKeys { it in gtCounter }
        .mapValues { (k, v) -> minOf(v, gtCounter[k] ?: 0) }

    val numSame = common.values.sum()

    if (numSame == 0) return 0.0

    val precision = 1.0 * numSame / predictionTokens.size
    val recall = 1.0 * numSame / groundTruthTokens.size
    return (2.0 * precision * recall) / (precision + recall)
}

/**
 * Compute HotPotQA-style token F1 with special labels.
 *
 * Port of `dspy/evaluate/metrics.py::hotpot_f1_score`
 */
fun hotpotF1Score(prediction: String, groundTruth: String): Double {
    val normalizedPrediction = normalizeText(prediction)
    val normalizedGroundTruth = normalizeText(groundTruth)

    if (normalizedPrediction in listOf("yes", "no", "noanswer") && normalizedPrediction != normalizedGroundTruth) {
        return 0.0
    }
    if (normalizedGroundTruth in listOf("yes", "no", "noanswer") && normalizedPrediction != normalizedGroundTruth) {
        return 0.0
    }

    val predictionTokens = normalizedPrediction.split(Regex("\\s+")).filter { it.isNotEmpty() }
    val groundTruthTokens = normalizedGroundTruth.split(Regex("\\s+")).filter { it.isNotEmpty() }

    val predCounter = predictionTokens.groupingBy { it }.eachCount()
    val gtCounter = groundTruthTokens.groupingBy { it }.eachCount()

    val common = predCounter.filterKeys { it in gtCounter }
        .mapValues { (k, v) -> minOf(v, gtCounter[k] ?: 0) }

    val numSame = common.values.sum()

    if (numSame == 0) return 0.0

    val precision = 1.0 * numSame / predictionTokens.size
    val recall = 1.0 * numSame / groundTruthTokens.size
    return (2.0 * precision * recall) / (precision + recall)
}

/**
 * Compute token-level precision of prediction against reference (after normalization).
 *
 * Port of `dspy/evaluate/metrics.py::precision_score`
 */
fun precisionScore(prediction: String, groundTruth: String): Double {
    val predictionTokens = normalizeText(prediction).split(Regex("\\s+")).filter { it.isNotEmpty() }
    val groundTruthTokens = normalizeText(groundTruth).split(Regex("\\s+")).filter { it.isNotEmpty() }

    if (predictionTokens.isEmpty() && groundTruthTokens.isEmpty()) {
        logger.warn("#> Precision Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.")
    }

    val predCounter = predictionTokens.groupingBy { it }.eachCount()
    val gtCounter = groundTruthTokens.groupingBy { it }.eachCount()

    val common = predCounter.filterKeys { it in gtCounter }
        .mapValues { (k, v) -> minOf(v, gtCounter[k] ?: 0) }

    val numSame = common.values.sum()

    if (numSame == 0) return 0.0

    return 1.0 * numSame / predictionTokens.size
}

/**
 * Return true if any passage contains any answer (normalized & DPR-normalized).
 *
 * Port of `dspy/evaluate/metrics.py::_passage_match`
 */
private fun passageMatch(passages: List<String>, answers: List<String>): Boolean {
    return passages.any { passage ->
        answers.any { answer ->
            val normAnswer = normalizeText(answer)
            val normPassage = normalizeText(passage)
            normPassage.contains(normAnswer, ignoreCase = true)
        }
    }
}

/**
 * Return true if prediction matches any answer.
 *
 * When `frac >= 1.0`, require exact match (EM). Otherwise, return whether the
 * maximum token-level F1 across answers is at least `frac`.
 *
 * Port of `dspy/evaluate/metrics.py::_answer_match`
 */
private fun answerMatch(prediction: String, answers: List<String>, frac: Double = 1.0): Boolean {
    return if (frac >= 1.0) {
        EM(prediction, answers)
    } else {
        F1(prediction, answers) >= frac
    }
}

/**
 * Evaluate exact match or F1-thresholded match for an example/prediction pair.
 *
 * Port of `dspy/evaluate/metrics.py::answer_exact_match`
 */
fun answerExactMatch(example: Example, pred: Prediction, trace: Any? = null, frac: Double = 1.0): Boolean {
    val exampleAnswer = example["answer"]
    val predAnswer = pred["answer"]?.toString() ?: return false

    return when (exampleAnswer) {
        is String -> answerMatch(predAnswer, listOf(exampleAnswer), frac)
        is List<*> -> answerMatch(predAnswer, exampleAnswer.filterIsInstance<String>(), frac)
        else -> throw IllegalArgumentException("Invalid answer type: ${exampleAnswer?.let { it::class }}")
    }
}

/**
 * Return true if any passage in `pred.context` contains the answer(s).
 *
 * Port of `dspy/evaluate/metrics.py::answer_passage_match`
 */
fun answerPassageMatch(example: Example, pred: Prediction, trace: Any? = null): Boolean {
    val exampleAnswer = example["answer"]
    val context = (pred["context"] as? List<*>)?.filterIsInstance<String>() ?: emptyList()

    return when (exampleAnswer) {
        is String -> passageMatch(context, listOf(exampleAnswer))
        is List<*> -> passageMatch(context, exampleAnswer.filterIsInstance<String>())
        else -> throw IllegalArgumentException("Invalid answer type: ${exampleAnswer?.let { it::class }}")
    }
}

/**
 * DPR-style normalization for passage matching.
 * Simplified version of the Python DPR_normalize.
 */
fun DPRNormalize(s: String): String {
    return s.lowercase()
        .replace(Regex("[^a-z0-9\\s]"), "")
        .replace(Regex("\\s+"), " ")
        .trim()
}

/**
 * Check if text contains any of the tokenized answers.
 */
fun hasAnswer(tokenizedAnswers: List<String>, text: String): Boolean {
    return tokenizedAnswers.any { answer ->
        text.contains(answer, ignoreCase = true)
    }
}
