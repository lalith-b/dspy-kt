package dspy.predict

import dspy.primitives.Completions
import dspy.primitives.Prediction
import dspy.signatures.Signature

/**
 * Local normalization function for text comparison.
 */
fun normalizeText(s: String?): String? {
    if (s == null) return null
    return s.lowercase().trim().replace(Regex("\\s+"), " ")
}

/**
 * Default normalization function for majority voting.
 */
fun defaultNormalize(s: String?): String? {
    return normalizeText(s) ?: null
}

/**
 * Returns the most common completion for the target field (or the last field) in the signature.
 * When normalize returns None, that completion is ignored.
 * In case of a tie, earlier completions are prioritized.
 *
 * Port of `dspy/predict/aggregation.py::majority`
 */
fun majority(
    predictionOrCompletions: Any,
    normalize: ((String?) -> String?) = ::defaultNormalize,
    field: String? = null,
): Prediction {
    require(predictionOrCompletions is Prediction ||
            predictionOrCompletions is Completions ||
            predictionOrCompletions is List<*>) {
        "Input must be a Prediction, Completions, or list"
    }

    // Get the completions
    val completions: Completions = when (predictionOrCompletions) {
        is Prediction -> predictionOrCompletions.completions
            ?: throw IllegalStateException("Prediction has no completions")
        is Completions -> predictionOrCompletions
        is List<*> -> {
            val items = mutableMapOf<String, MutableList<Any>>()
            for (item in predictionOrCompletions) {
                if (item is Map<*, *>) {
                    for ((k, v) in item) {
                        items.getOrPut(k.toString()) { mutableListOf() }.add(v as Any)
                    }
                }
            }
            Completions(items, signature = null)
        }
        else -> throw IllegalArgumentException("Unsupported type")
    }

    val signature = try {
        completions.signature
    } catch (e: Exception) {
        null
    }

    // Determine the target field
    val targetField = field ?: run {
        if (signature != null) {
            signature.outputFields.lastOrNull()?.name
        } else {
            val firstCompletion = completions[0]
            firstCompletion.toMap().keys.lastOrNull()
        }
    }

    if (targetField == null) {
        throw IllegalArgumentException("Cannot determine target field")
    }

    // Normalize values
    val normalizedValues = (0 until completions.size).map { i ->
        normalize(completions[i].toMap()[targetField] as? String)
    }
    val normalizedValuesNonNone = normalizedValues.filterNotNull()

    // Count occurrences
    val valueCounts = mutableMapOf<String, Int>()
    for (value in (normalizedValuesNonNone.ifEmpty { normalizedValues.filterNotNull() })) {
        valueCounts[value] = valueCounts.getOrDefault(value, 0) + 1
    }

    // Find the majority value
    val majorityValue = valueCounts.maxByOrNull { it.value }?.key
        ?: throw IllegalStateException("No values to count")

    // Return the first completion with the majority value in the field
    var selectedCompletion: Map<String, Any?>? = null
    for (i in 0 until completions.size) {
        val completion = completions[i]
        val value = completion.toMap()[targetField] as? String
        if (normalize(value) == majorityValue) {
            selectedCompletion = completion.toMap()
            break
        }
    }

    return Prediction.fromCompletions(
        listOf(selectedCompletion ?: emptyMap()),
        signature = signature,
    )
}
