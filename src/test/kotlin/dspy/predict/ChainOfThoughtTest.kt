package dspy.predict

import dspy.primitives.Example
import dspy.signatures.Signature
import dspy.signatures.OutputField
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class ChainOfThoughtTest : FunSpec({
    test("creates ChainOfThought from Signature") {
        val sig = Signature.fromString("question -> answer")
        val cot = ChainOfThought(sig)
        cot.predict.signature.inputFields.size shouldBe 1
        cot.predict.signature.outputFields.size shouldBe 2 // reasoning + answer
    }

    test("creates ChainOfThought with custom rationale") {
        val sig = Signature.fromString("question -> answer")
        val rationale = OutputField(name = "thinking", desc = "Think step by step")
        val cot = ChainOfThought(sig, rationale)
        cot.predict.signature.outputFields.size shouldBe 2
        cot.predict.signature.outputFields[0].name shouldBe "thinking"
    }
})
