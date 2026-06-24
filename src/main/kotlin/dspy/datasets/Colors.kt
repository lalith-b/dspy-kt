package dspy.datasets

/**
 * Colors dataset for testing DSPy programs.
 */
object ColorsDataset {
    private val colors = listOf("red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray")
    private val shapes = listOf("circle", "square", "triangle", "rectangle", "pentagon", "hexagon", "star", "diamond")

    fun trainset(numSamples: Int = 15): List<Map<String, String>> {
        val results = mutableListOf<Map<String, String>>()
        val random = kotlin.random.Random
        repeat(numSamples) {
            val color = colors.random(random)
            val shape = shapes.random(random)
            results.add(mapOf(
                "color" to color,
                "shape" to shape,
                "answer" to "The $color $shape"
            ))
        }
        return results
    }

    fun devset(numSamples: Int = 5): List<Map<String, String>> {
        return trainset(numSamples)
    }

    fun evaluatePrediction(pred: Map<String, Any?>, example: Map<String, Any?>): Boolean {
        return (pred["answer"] as? String) == (example["answer"] as? String)
    }
}
