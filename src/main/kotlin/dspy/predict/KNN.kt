package dspy.predict

import dspy.clients.Embedder
import dspy.primitives.Example

/**
 * K-nearest neighbors retriever.
 *
 * Finds similar examples from a training set using dot product similarity
 * of embedding vectors.
 *
 * Port of `dspy/predict/knn.py`
 *
 * Note: NumPy operations are implemented natively in Kotlin.
 */
class KNN(
    val k: Int,
    val trainset: List<Example>,
    val embedding: Embedder,
) {
    // Pre-computed embeddings for the training set
    val trainsetVectors: List<List<Float>>

    init {
        // Cast training examples to strings for vectorization
        val trainsetCasted = trainset.map { example ->
            val inputKeys = example.inputKeys()
            example.toMap().filterKeys { it in inputKeys }
                .entries.map { entry -> "${entry.key}: ${entry.value}" }
                .joinToString(" | ")
        }

        trainsetVectors = embedding.invoke(trainsetCasted)
    }

    /**
     * Find the k nearest neighbors for the given input.
     */
    operator fun invoke(kwargs: Map<String, Any?>): List<Example> {
        // Create input vector
        val inputStr = kwargs.entries.map { entry -> "${entry.key}: ${entry.value}" }.joinToString(" | ")
        val inputVectors = embedding.invoke(listOf(inputStr))
        val inputVector = inputVectors.first()

        // Compute dot product scores
        val scores = trainsetVectors.map { trainVec ->
            val minLen = kotlin.math.min(trainVec.size, inputVector.size)
            var score = 0.0
            for (i in 0 until minLen) {
                score += trainVec[i] * inputVector[i]
            }
            score
        }

        // Find indices of top-k scores (descending)
        val nearestIndices = scores.indices.toList()
            .sortedByDescending { scores[it] }
            .takeLast(k)
            .reversed()

        return nearestIndices.map { trainset[it] }
    }
}
