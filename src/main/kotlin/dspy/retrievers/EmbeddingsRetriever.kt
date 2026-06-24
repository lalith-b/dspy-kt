package dspy.retrievers

import dspy.primitives.Prediction

/**
 * DSPy Embeddings retriever.
 *
 * Port of `dspy/retrievers/embeddings.py`.
 *
 * Note: numpy and faiss are Python-specific. This Kotlin port provides the core logic
 * with pure Kotlin math operations.
 */
open class Embeddings(
    corpus: List<String>,
    embedder: (List<String>) -> List<DoubleArray>,
    k: Int = 5,
    normalize: Boolean = true,
    bruteForceThreshold: Int = 20_000,
) {
    private val embedder: (List<String>) -> List<DoubleArray> = embedder
    protected val k: Int = k
    val corpus: List<String> = corpus
    protected val normalize: Boolean = normalize
    protected var corpusEmbeddings: DoubleArray
    private var index: FaissIndex? = null

    init {
        val rawEmbeddings = embedder(corpus)
        corpusEmbeddings = if (normalize) {
            rawEmbeddings.flatMap { it.toList() }.toDoubleArray().let { arr ->
                val norm = Math.sqrt(arr.sumOf { it * it })
                if (norm > 1e-10) arr.map { it / norm }.toDoubleArray() else arr
            }
        } else {
            rawEmbeddings.flatMap { it.toList() }.toDoubleArray()
        }

        if (corpus.size >= bruteForceThreshold) {
            index = null // Would build FAISS index if Java bindings were available
        }
    }

    operator fun invoke(query: String): Prediction {
        return forward(query)
    }

    /**
     * Search for the top-k passages most similar to the query.
     */
    open fun forward(query: String): Prediction {
        val result = batchForward(listOf(query)).first()
        val (passages, indices, scores) = result
        return Prediction(base = mapOf("passages" to passages, "indices" to indices))
    }

    protected fun batchForward(queries: List<String>): List<Triple<List<String>, List<Int>, List<Double>>> {
        val qEmbeds = if (normalize) {
            embedder(queries).map { normalizeEmbedding(it) }
        } else {
            embedder(queries)
        }

        val candidateIndices: List<List<Int>> = if (index != null) {
            index!!.search(qEmbeds, k * 10)
        } else {
            queries.map { (0 until corpus.size).toList() }
        }

        return rerankAndPredict(qEmbeds, candidateIndices)
    }

    private fun normalizeEmbedding(embedding: DoubleArray): DoubleArray {
        val norm = Math.sqrt(embedding.sumOf { it * it })
        return if (norm > 1e-10) {
            embedding.map { it / norm }.toDoubleArray()
        } else {
            embedding
        }
    }

    private fun rerankAndPredict(
        qEmbeds: List<DoubleArray>,
        candidateIndices: List<List<Int>>
    ): List<Triple<List<String>, List<Int>, List<Double>>> {
        return qEmbeds.mapIndexed { queryIdx, qEmbed ->
            val candidates = candidateIndices[queryIdx]
            val scored = candidates.map { idx ->
                val passageEmbed = getPassageEmbedding(idx)
                val score = dotProduct(qEmbed, passageEmbed)
                idx to score
            }
            val topK = scored.sortedByDescending { it.second }.take(k)
            val passages = topK.map { corpus[it.first] }
            val indices = topK.map { it.first }
            val scores = topK.map { it.second }
            Triple(passages, indices, scores)
        }
    }

    private fun getPassageEmbedding(passageIdx: Int): DoubleArray {
        val dim = corpusEmbeddings.size / corpus.size
        val offset = passageIdx * dim
        return corpusEmbeddings.sliceArray(offset until (offset + dim))
    }

    private fun dotProduct(a: DoubleArray, b: DoubleArray): Double {
        return a.zip(b).sumOf { (x, y) -> x * y }
    }

    /**
     * Dummy FAISS index for Kotlin port.
     */
    private class FaissIndex(
        private val embeddings: DoubleArray,
        private val dim: Int
    ) {
        fun search(queries: List<DoubleArray>, k: Int): List<List<Int>> {
            return queries.map { query ->
                val scored = (0 until (embeddings.size / dim)).map { idx ->
                    val passageEmbed = embeddings.sliceArray(idx * dim until (idx + 1) * dim)
                    val score = dotProduct(query, passageEmbed)
                    idx to score
                }
                scored.sortedByDescending { it.second }.take(k).map { it.first }
            }
        }

        private fun dotProduct(a: DoubleArray, b: DoubleArray): Double {
            return a.zip(b).sumOf { (x, y) -> x * y }
        }
    }
}
