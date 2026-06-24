package dspy.primitives

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class PredictionTest : FunSpec({
    test("prediction creation") {
        val prediction = Prediction(mapOf("answer" to "4", "score" to 0.9))
        prediction["answer"] shouldBe "4"
        prediction.toFloat() shouldBe 0.9
    }

    test("prediction with completions") {
        val completions = listOf(
            mapOf("answer" to "4", "score" to 0.9),
            mapOf("answer" to "5", "score" to 0.8),
        )
        val prediction = Prediction.fromCompletions(completions)
        prediction["answer"] shouldBe "4"
        prediction.completions?.size shouldBe 2
    }

    test("completions creation") {
        val completions = Completions(listOf(
            mapOf("answer" to "4", "score" to 0.9),
            mapOf("answer" to "5", "score" to 0.8),
        ))
        completions.size shouldBe 2
        completions["answer"] shouldBe listOf("4", "5")
        completions[0]["answer"] shouldBe "4"
        completions[1]["answer"] shouldBe "5"
    }

    test("completions from dict") {
        val completions = Completions(mapOf(
            "answer" to listOf("4", "5"),
            "score" to listOf(0.9, 0.8),
        ))
        completions.size shouldBe 2
        completions["answer"] shouldBe listOf("4", "5")
    }

    test("prediction dumpState and loadState") {
        val prediction = Prediction(mapOf("answer" to "4"))
        val state = prediction.dumpState()
        state["answer"] shouldBe "4"
        val newPrediction = Prediction()
        newPrediction.loadState(state)
        newPrediction["answer"] shouldBe "4"
    }
})
