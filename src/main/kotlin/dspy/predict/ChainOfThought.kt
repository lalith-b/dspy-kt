package dspy.predict

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Parameter
import dspy.primitives.Prediction
import dspy.signatures.OutputField
import dspy.signatures.Signature

class ChainOfThought(
    signature: Signature,
    rationaleField: OutputField? = null,
    private val lm: dspy.clients.BaseLM? = null,
) : Module(), Parameter {
    val predict: Predict

    init {
        val rationale = rationaleField ?: OutputField(name = "reasoning", desc = "\${reasoning}")
        val extended = Signature(
            instruction = signature.instruction,
            inputFields = signature.inputFields,
            outputFields = listOf(rationale) + signature.outputFields,
        )
        predict = Predict(sig = extended)
        predict._lm = lm
    }

    suspend fun forward(example: Example): Prediction = predict.__call__(kwargs = example.toMap())

    override fun namedParameters(): List<Pair<String, Parameter>> = predict.namedParameters()

    override fun dumpState(): Map<String, Any?> = predict.dumpState()

    override fun loadState(state: Any?) = predict.loadState(state)
}
