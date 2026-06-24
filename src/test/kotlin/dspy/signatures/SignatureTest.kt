package dspy.signatures

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class SignatureTest : FunSpec({
    test("creates Signature from string") {
        val sig = Signature.fromString("question -> answer")
        sig.inputFields.size shouldBe 1
        sig.inputFields[0].name shouldBe "question"
        sig.outputFields.size shouldBe 1
        sig.outputFields[0].name shouldBe "answer"
    }

    test("creates Signature with multiple fields") {
        val sig = Signature.fromString("input1, input2 -> output1, output2")
        sig.inputFields.size shouldBe 2
        sig.outputFields.size shouldBe 2
    }

    test("creates Signature with custom instruction") {
        val sig = Signature(
            instruction = "Custom instruction",
            inputFields = listOf(InputField(name = "question")),
            outputFields = listOf(OutputField(name = "answer")),
        )
        sig.instruction shouldBe "Custom instruction"
    }

    test("Signature toString contains fields") {
        val sig = Signature.fromString("question -> answer")
        sig.toString().contains("question") shouldBe true
        sig.toString().contains("answer") shouldBe true
    }

    test("ChatAdapter field header regex") {
        val pattern = Regex("\\[\\[ ## (\\w+) ## \\]\\]")
        val text = "[[ ## output ## ]]\ntest output"
        val lines = text.split("\n")
        lines[0] shouldBe "[[ ## output ## ]]"
        val match = pattern.find(lines[0])
        (match != null) shouldBe true
        match?.groupValues?.get(1) shouldBe "output"
    }
})
