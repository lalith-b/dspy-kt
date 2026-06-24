package dspy.datasets

/**
 * GSM8K dataset loader for grade school math problems.
 */
object GSM8KDataset {
    /**
     * Load GSM8K dataset.
     *
     * In production, this would load from HuggingFace datasets.
     */
    fun load(): Pair<List<Map<String, String>>, List<Map<String, String>>> {
        // Sample data - production would load from HF datasets
        val trainData = listOf(
            mapOf("question" to "There are 15 apples. 7 are eaten. How many remain?", "answer" to "15 - 7 = 8"),
            mapOf("question" to "If you have 3 boxes with 4 items each, how many total?", "answer" to "3 * 4 = 12"),
            mapOf("question" to "A car travels 60 miles per hour for 2 hours. How far?", "answer" to "60 * 2 = 120"),
        )
        val devData = listOf(
            mapOf("question" to "There are 20 oranges. 8 are given away. How many left?", "answer" to "20 - 8 = 12"),
            mapOf("question" to "If 5 students each get 3 pencils, how many total?", "answer" to "5 * 3 = 15"),
        )
        return trainData to devData
    }

    fun evaluatePrediction(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        val predAnswer = pred["answer"]?.toString()?.trim() ?: return false
        val expectedAnswer = example["answer"]?.toString()?.trim() ?: return false
        // Extract the final number from both answers
        val predNum = predAnswer.split(" ").lastOrNull()?.toIntOrNull()
        val expectedNum = expectedAnswer.split(" ").lastOrNull()?.toIntOrNull()
        return predNum == expectedNum
    }
}
