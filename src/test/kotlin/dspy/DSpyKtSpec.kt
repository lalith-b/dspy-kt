package dspy

import dspy.core.types.LMConfig
import dspy.core.types.LMMessage
import dspy.core.types.LMTextPart
import dspy.primitives.Example
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.utils.Settings
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe

class CoreTypesTest : FunSpec({
    test("LMConfig creation") {
        val config = LMConfig(
            temperature = 0.7f,
            maxTokens = 1000,
            topP = 0.9f,
            stop = listOf("END"),
            n = 3,
            logprobs = true
        )
        config.temperature shouldBe 0.7f
        config.maxTokens shouldBe 1000
        config.topP shouldBe 0.9f
        config.stop shouldBe listOf("END")
        config.n shouldBe 3
        config.logprobs shouldBe true
    }

    test("LMConfig fromKwargs") {
        val config = LMConfig.fromKwargs(mapOf(
            "temperature" to 0.8f,
            "max_tokens" to 2000
        ))
        config.temperature shouldBe 0.8f
        config.maxTokens shouldBe 2000
    }

    test("LMConfig default values") {
        val config = LMConfig()
        config.temperature shouldBe null
        config.maxTokens shouldBe null
        config.topP shouldBe null
        config.stop shouldBe null
        config.n shouldBe null
        config.logprobs shouldBe null
    }
})

class MessageFactoryTest : FunSpec({
    test("creates user message") {
        val msg = LMMessage(role = "user", parts = listOf(LMTextPart(text = "Hello")))
        msg.role shouldBe "user"
        msg.text shouldBe "Hello"
    }

    test("creates assistant message") {
        val msg = LMMessage(role = "assistant", parts = listOf(LMTextPart(text = "Answer")))
        msg.role shouldBe "assistant"
        msg.text shouldBe "Answer"
    }

    test("creates system message") {
        val msg = LMMessage(role = "system", parts = listOf(LMTextPart(text = "You are helpful")))
        msg.role shouldBe "system"
        msg.text shouldBe "You are helpful"
    }
})

class SignatureBuilderTest : FunSpec({
    test("Signature from string") {
        val sig = Signature.fromString("question -> answer")
        sig.inputFields.size shouldBe 1
        sig.inputFields[0].name shouldBe "question"
        sig.outputFields.size shouldBe 1
        sig.outputFields[0].name shouldBe "answer"
    }

    test("Signature with builder") {
        val sig = Signature(
            instruction = "Translate the text",
            inputFields = listOf(dspy.signatures.InputField(name = "text", desc = "Text to translate")),
            outputFields = listOf(dspy.signatures.OutputField(name = "translation", desc = "Translated text"))
        )
        sig.inputFields.size shouldBe 1
        sig.outputFields.size shouldBe 1
        sig.inputFields[0].name shouldBe "text"
        sig.outputFields[0].name shouldBe "translation"
    }

    test("Signature with instruction") {
        val sig = Signature(
            instruction = "Answer the question",
            inputFields = listOf(dspy.signatures.InputField(name = "question")),
            outputFields = listOf(dspy.signatures.OutputField(name = "answer"))
        )
        sig.instruction shouldBe "Answer the question"
    }
})

class ExampleTest : FunSpec({
    test("Example creation and access") {
        val example = Example(mapOf(
            "question" to "What is 2+2?",
            "answer" to "4"
        ))
        example["question"] shouldBe "What is 2+2?"
        example["answer"] shouldBe "4"
    }
})

class PredictionTest : FunSpec({
    test("Prediction creation") {
        val prediction = Prediction(mapOf(
            "answer" to "4",
            "score" to 0.9
        ))
        prediction["answer"] shouldBe "4"
        prediction.toFloat() shouldBe 0.9
    }
})

class SettingsTest : FunSpec({
    test("configure and get LM") {
        Settings.configure("gpt-4")
        val lm = Settings.lm()
        (lm != null) shouldBe true
        lm?.model shouldBe "gpt-4"
    }
})

class AdapterTest : FunSpec({
    test("ChatAdapter format produces messages") {
        val adapter = dspy.adapters.ChatAdapter()
        val sig = Signature(
            instruction = "Answer",
            inputFields = listOf(dspy.signatures.InputField(name = "question")),
            outputFields = listOf(dspy.signatures.OutputField(name = "answer"))
        )
        val demos = emptyList<Map<String, Any?>>()
        val inputs = mapOf<String, Any?>("question" to "What is 2+2?")
        val messages = adapter.format(sig, demos, inputs)
        (messages.size >= 2) shouldBe true
        (messages[0]["role"] as String) shouldBe "system"
    }
})
