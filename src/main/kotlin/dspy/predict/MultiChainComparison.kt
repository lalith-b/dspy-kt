package dspy.predict

import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.signatures.SignatureUtils
import kotlinx.coroutines.runBlocking

/**
 * MultiChainComparison module.
 *
 * Takes multiple reasoning attempts and compares them to produce a final rationale and answer.
 *
 * Port of `dspy/predict/multi_chain_comparison.py`
 */
class MultiChainComparison(
    signature: Any,
    val M: Int = 3,
    temperature: Double = 0.7,
    config: Map<String, Any?> = emptyMap(),
) : Module() {
    val lastKey: String
    val predict: Predict

    init {
        val sig = SignatureUtils.ensureSignature(signature)

        // Get the last output field key
        val outputFieldNames = sig.outputFields.map { it.name }
        require(outputFieldNames.isNotEmpty()) { "Signature must have at least one output field" }
        lastKey = outputFieldNames.last()

        // Build the modified signature by appending reasoning attempts and prepending rationale
        var modifiedSig = sig
        for (idx in 0 until M) {
            modifiedSig = modifiedSig.append(
                "reasoning_attempt_${idx + 1}",
                InputField(
                    prefix = "Student Attempt #${idx + 1}:",
                    desc = "\${reasoning attempt}",
                ),
            )
        }

        modifiedSig = modifiedSig.prepend(
            "rationale",
            OutputField(
                prefix = "Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc = "\${corrected reasoning}",
            ),
        )

        predict = Predict(sig = modifiedSig).apply {
            this.config["temperature"] = temperature
            this.config.putAll(config)
        }
    }

    /**
     * Forward pass: process completions into attempts and call predict.
     */
    fun forward(completions: List<Map<String, Any?>>, kwargs: Map<String, Any?> = emptyMap()): Prediction {
        val attempts = mutableListOf<String>()

        for (c in completions) {
            val rationale = (c["rationale"] ?: c["reasoning"] ?: "").toString().trim().split("\n")[0].trim()
            val answer = c[lastKey]?.toString()?.trim()?.split("\n")?.get(0)?.trim() ?: ""
            attempts.add("\u00ABI'm trying to $rationale I'm not sure but my prediction is $answer\u00BB")
        }

        require(attempts.size == M) {
            "The number of attempts (${attempts.size}) doesn't match the expected number M ($M). " +
                "Please set the correct value for M when initializing MultiChainComparison."
        }

        val mergedKwargs = mutableMapOf<String, Any?>()
        mergedKwargs.putAll(kwargs)
        for ((idx, attempt) in attempts.withIndex()) {
            mergedKwargs["reasoning_attempt_${idx + 1}"] = attempt
        }

        return runBlocking { predict.invoke(kwargs = mergedKwargs) }
    }

    override suspend operator fun invoke(kwargs: Map<String, Any?>): Prediction {
        val completions = (kwargs["completions"] as? List<Map<String, Any?>>)
            ?: throw IllegalArgumentException("MultiChainComparison requires 'completions' in kwargs")
        val filteredKwargs = kwargs.toMutableMap().apply { remove("completions") }
        return forward(completions, filteredKwargs)
    }

    override fun namedParameters(): List<Pair<String, Parameter>> {
        return listOf("predict" to predict)
    }

    override fun deepcopy(): Module {
        return MultiChainComparison(
            signature = predict.sig.toString(),
            M = M,
            temperature = predict.config["temperature"] as? Double ?: 0.7,
            config = predict.config.toMap(),
        )
    }
}
