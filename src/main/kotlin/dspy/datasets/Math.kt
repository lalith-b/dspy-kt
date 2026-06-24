package dspy.datasets

/**
 * Math dataset loader for mathematical reasoning tasks.
 */
object MathDataset {
    fun load(): Pair<List<Map<String, String>>, List<Map<String, String>>> {
        val trainData = listOf(
            mapOf("problem" to "Solve: x^2 - 4 = 0", "answer" to "x = 2 or x = -2"),
            mapOf("problem" to "What is the derivative of x^3?", "answer" to "3x^2"),
            mapOf("problem" to "Integrate 2x from 0 to 3", "answer" to "9"),
        )
        val devData = listOf(
            mapOf("problem" to "Solve: 2x + 6 = 0", "answer" to "x = -3"),
            mapOf("problem" to "What is the integral of x?", "answer" to "x^2/2"),
        )
        return trainData to devData
    }
}
