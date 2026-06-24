package dspy.datasets

/**
 * HotpotQA dataset loader for multi-hop question answering.
 */
object HotpotQADataset {
    /**
     * Load HotpotQA dataset.
     *
     * In production, this would load from HuggingFace datasets.
     */
    fun load(): Pair<List<Map<String, Any?>>, List<Map<String, Any?>>> {
        val trainData = listOf(
            mapOf<String, Any?>(
                "question" to "What castle did David Gregory inherit?",
                "answer" to "Gordon Castle"
            ),
            mapOf<String, Any?>(
                "question" to "Which film was released first, All Is Lost or The Great Gatsby?",
                "answer" to "All Is Lost"
            ),
        )
        val devData = listOf(
            mapOf<String, Any?>(
                "question" to "Who was the director of the film that featured the song \"Let It Be\"?",
                "answer" to "Michael Lindsay-Hogg"
            ),
        )
        return trainData to devData
    }

    fun evaluatePrediction(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim()?.lowercase() ?: return false
        val expectedAnswer = example["answer"]?.toString()?.trim()?.lowercase() ?: return false
        return predAnswer == expectedAnswer || predAnswer.contains(expectedAnswer) || expectedAnswer.contains(predAnswer)
    }
}
