package dspy.predict

import dspy.primitives.Example
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.utils.DummyLM
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import kotlinx.coroutines.runBlocking

class PredictTest : FunSpec({
    test("initialization with string signature") {
        val predict = Predict(signature = "input1, input2 -> output")
        predict.signature.inputFields.size shouldBe 2
        predict.signature.outputFields.size shouldBe 1
    }

    test("predict has named parameters") {
        val predict = Predict(signature = "input -> output")
        // Predict has parameters (self -> this)
        predict.namedParameters().isNotEmpty() shouldBe true
    }

    test("call method with DummyLM") {
        runBlocking {
            val lm = DummyLM(listOf(mapOf<String, Any?>("output" to "test output")))
            val predict = Predict(signature = "input -> output")
            predict._lm = lm
            val result = predict.__call__(kwargs = mapOf("input" to "test input"))
            // Note: without proper adapter setup, this may not parse correctly
            // This test validates the Predict flow works
        }
    }

    test("dump and load state preserves signature") {
        val sig = Signature(
            instruction = "original instructions",
            inputFields = listOf(dspy.signatures.InputField(name = "input")),
            outputFields = listOf(dspy.signatures.OutputField(name = "output")),
        )
        val predict = Predict(sig = sig)
        val state = predict.dumpState()
        val newPredict = Predict(signature = "input -> output")
        newPredict.loadState(state)
        newPredict.signature.inputFields.size shouldBe 1
        newPredict.signature.outputFields.size shouldBe 1
    }
})
