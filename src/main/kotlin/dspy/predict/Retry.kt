package dspy.predict

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Parameter
import dspy.primitives.Prediction
import dspy.signatures.Signature

class Retry(
    val module: Predict,
    val originalSignature: Signature,
    val maxRetries: Int = 2,
) : Module(), Parameter {
    var error: Throwable? = null

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction {
        for (attempt in 0 until maxRetries + 1) {
            try {
                val prediction = module(kwargs.toMutableMap())
                if (error != null) {
                    val newKwargs = kwargs.toMutableMap()
                    newKwargs["error"] = error
                    error = null
                    return module(newKwargs)
                }
                return prediction
            } catch (e: Exception) {
                error = e
            }
        }
        throw IllegalStateException("Max retries exceeded")
    }

    override fun namedParameters(): List<Pair<String, Parameter>> {
        return listOf(Pair("self", this@Retry)) +
            (module as? Predict)?.namedParameters().orEmpty()
    }

    override fun dumpState(): Map<String, Any?> {
        return mapOf<String, Any?>(
            "module" to ((module as? Predict)?.dumpState() ?: mapOf()),
            "originalSignature" to originalSignature.dumpState(),
        )
    }

    override fun loadState(state: Any?) {
        (state as? Map<String, Any?>)?.let {
            (it["module"] as? Map<String, Any?>)?.let {
                module.loadState(it)
            }
        }
    }
}
