package dspy.dsp.utils

/**
 * DPR (Dense Passage Retrieval) utilities.
 */
object DPR {
    /**
     * Encode queries using DPR encoder.
     */
    fun encodeQueries(queries: List<String>): List<List<Float>> {
        throw NotImplementedError("DPR encoding requires HuggingFace integration")
    }

    /**
     * Encode passages using DPR encoder.
     */
    fun encodePassages(passages: List<String>): List<List<Float>> {
        throw NotImplementedError("DPR encoding requires HuggingFace integration")
    }

    /**
     * Search using DPR embeddings.
     */
    fun search(
        queryEmbeddings: List<List<Float>>,
        passageEmbeddings: List<List<Float>>,
        passages: List<String>,
        k: Int = 3
    ): List<Pair<String, Double>> {
        return queryEmbeddings.mapNotNull { queryEmb ->
            val scored = passageEmbeddings.mapIndexed { idx, passageEmb ->
                idx to cosineSimilarity(queryEmb, passageEmb)
            }
            scored.maxByOrNull { it.second }?.let { (idx, score) ->
                passages[idx] to score
            }
        }
    }

    private fun cosineSimilarity(a: List<Float>, b: List<Float>): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val dotProduct = a.zip(b).sumOf { (x, y) -> x.toDouble() * y.toDouble() }
        val normA = Math.sqrt(a.sumOf { it.toDouble() * it.toDouble() })
        val normB = Math.sqrt(b.sumOf { it.toDouble() * it.toDouble() })
        return if (normA == 0.0 || normB == 0.0) 0.0 else dotProduct / (normA * normB)
    }
}
