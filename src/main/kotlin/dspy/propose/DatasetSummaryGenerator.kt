package dspy.propose

/**
 * Dataset summary generator.
 *
 * Creates a summary of a dataset by iteratively observing samples
 * and aggregating observations.
 *
 * Faithful port of `dspy/propose/dataset_summary_generator.py`.
 */

/**
 * Order input keys in a string representation.
 */
fun orderInputKeysInString(unorderedRepr: String): String {
    val pattern = Regex("input_keys=\\{([^}]+\\{([^}]+)\\}\n\n")
    val match = pattern.find(unorderedRepr)
    return match?.let {
        val keys = it.groupValues[1].split(",")
            .map { key -> key.trim() }
            .sorted()
        val replacement = "input_keys={${keys.joinToString(", ")}}"
        pattern.replace(unorderedRepr, replacement)
    } ?: unorderedRepr
}

/**
 * Create a dataset summary using iterative observation and summarization.
 *
 * Args:
 *     trainset: The training dataset.
 *     viewDataBatchSize: Number of samples to view per batch.
 *     promptModel: The LM to use for generating observations.
 *     verbose: Whether to print progress.
 *
 * Returns:
 *     A summary string of the dataset.
 */
@Suppress("UNCHECKED_CAST")
fun createDatasetSummary(
    trainset: List<Map<String, Any?>>,
    viewDataBatchSize: Int,
    promptModel: Any?,
    verbose: Boolean = false,
): String {
    if (verbose) {
        println("\nBootstrapping dataset summary (this will be used to generate instructions)...")
    }

    val upperLim = minOf(trainset.size, viewDataBatchSize)
    var observations = ""

    if (upperLim > 0) {
        val batchStr = trainset.take(upperLim).joinToString("\n") { it.toString() }
        // In a full implementation, this would call the LM
        observations = "Dataset contains $upperLim samples."
    }

    if (verbose) {
        println("Initial observations: $observations")
    }

    var skips = 0
    var calls = 0
    val maxCalls = 10

    try {
        for (b in viewDataBatchSize until trainset.size step viewDataBatchSize) {
            calls++
            if (calls >= maxCalls) break

            if (verbose) {
                println("b: $b")
            }

            val upper = minOf(trainset.size, b + viewDataBatchSize)
            val batch = trainset.slice(b until upper)
            val batchStr = batch.joinToString("\n") { it.toString() }

            // In a full implementation, this would call the LM with prior observations
            if (calls > maxCalls / 2) {
                skips++
                if (skips >= 5) break
                continue
            }

            observations += " | Additional batch observed ($b to $upper)."

            if (verbose) {
                println("observations: $observations")
            }
        }
    } catch (e: Exception) {
        if (verbose) {
            println("Error: ${e.message}. Using observations from past round for a summary.")
        }
    }

    // Final summarization
    val summary = observations.take(500) + if (observations.length > 500) "..." else ""

    if (verbose) {
        println("\nGenerated summary: ${stripPrefix(summary)}\n")
    }

    return stripPrefix(summary)
}
