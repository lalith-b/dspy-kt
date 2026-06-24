package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import kotlinx.coroutines.runBlocking

/**
 * KNN few-shot teleprompter.
 *
 * Uses an in-memory KNN retriever to find the k nearest neighbors in a trainset
 * at test time. For each input example in a forward call, it identifies the k most
 * similar examples from the trainset and attaches them as demonstrations.
 *
 * Faithful port of `dspy/teleprompt/knn_fewshot.py`.
 */
class KNNFewShot(
    private val k: Int,
    private val trainset: List<Example>,
    private val vectorizer: Embedder,
    private val fewShotBootstrapArgs: Map<String, Any?> = emptyMap(),
) : Teleprompter() {
    /**
     * Compile the student module with KNN-based few-shot learning.
     *
     * Args:
     *     student: The student module to compile.
     *     trainset: Ignored (uses the trainset from constructor).
     *     teacher: Optional teacher module for bootstrapping.
     *
     * Returns:
     *     The compiled student module.
     */
    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        val studentCopy = student.resetCopy()

        // Create KNN retriever
        val knnRetriever = KNNRetriever(k, this.trainset, vectorizer)

        // Wrap forward with KNN-based few-shot compilation
        return createKNNModule(studentCopy, knnRetriever, teacher)
    }

    private fun createKNNModule(
        base: Module,
        knn: KNNRetriever,
        teacher: Module?,
    ): Module {
        return KNNModule(base, knn, teacher, fewShotBootstrapArgs)
    }
}

/**
 * Embedder interface for vectorization.
 *
 * Faithful port of `dspy.clients.Embedder`.
 */
interface Embedder {
    /**
     * Generate embeddings for a list of texts.
     *
     * Returns a 2D array where each row is an embedding vector.
     */
    suspend operator fun invoke(texts: List<String>): Array<DoubleArray>
}

/**
 * KNN retriever for finding nearest neighbors.
 *
 * Faithful port of `dspy.predict.knn.KNN`.
 */
class KNNRetriever(
    private val k: Int,
    private val trainset: List<Example>,
    private val vectorizer: Embedder,
) {
    private val trainsetVectors: Array<DoubleArray>

    init {
        val textsToEmbed = trainset.map { example ->
            example.inputKeys().map { key ->
                val value = example[key]
                "$key: $value"
            }.joinToString(" | ")
        }
        trainsetVectors = kotlinx.coroutines.runBlocking {
            vectorizer(textsToEmbed)
        }
    }

    /**
     * Find the k nearest neighbors for the given input.
     */
    suspend operator fun invoke(input: Map<String, Any?>): List<Example> {
        val inputText = input.map { (key, value) -> "$key: $value" }.joinToString(" | ")
        val inputVector = vectorizer(listOf(inputText))

        if (inputVector.isEmpty()) return emptyList()

        val scores = trainsetVectors.mapIndexed { idx, vec ->
            val dot = dotProduct(vec, inputVector[0])
            Triple(idx, dot, trainset[idx])
        }

        val nearest = scores.sortedByDescending { it.second }.take(k)
        return nearest.map { it.third }
    }

    private fun dotProduct(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in 0 until minOf(a.size, b.size)) {
            sum += a[i] * b[i]
        }
        return sum
    }
}

/**
 * Module that wraps KNN-based few-shot compilation.
 *
 * At test time, finds k nearest neighbors and compiles with bootstrap.
 */
private class KNNModule(
    private val base: Module,
    private val knn: KNNRetriever,
    private val teacher: Module?,
    private val bootstrapArgs: Map<String, Any?>,
) : Module() {
    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction {
        // Find k nearest neighbors
        val knnTrainset = knn(kwargs)

        // Bootstrap with the nearest neighbors
        val optimizer = BootstrapFewShot(
            teacherSettings = bootstrapArgs["teacherSettings"] as? Map<String, Any?> ?: emptyMap(),
            maxBootstrappedDemos = bootstrapArgs["maxBootstrappedDemos"] as? Int ?: 4,
            maxLabeledDemos = bootstrapArgs["maxLabeledDemos"] as? Int ?: 16,
            maxRounds = bootstrapArgs["maxRounds"] as? Int ?: 1,
        )
        val compiled = optimizer.compile(base, trainset = knnTrainset, teacher = teacher, valset = null)

        return compiled(kwargs)
    }
}
