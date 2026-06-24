package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module

/**
 * DEPRECATED - replaced with COPRO.
 * Port of `dspy/teleprompt/signature_opt.py`.
 */
class SignatureOptimizer(
    promptModel: Any? = null,
    metric: Any? = null,
    breadth: Int = 10,
    depth: Int = 3,
    initTemperature: Double = 1.4,
    verbose: Boolean = false,
    trackStats: Boolean = false,
) : Teleprompter() {

    init {
        System.err.println(
            "[WARNING] SignatureOptimizer has been deprecated and replaced with COPRO. " +
                "SignatureOptimizer will be removed in a future release."
        )
    }

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        throw NotImplementedError(
            "SignatureOptimizer is deprecated and no longer functional. " +
                "Please use COPRO instead."
        )
    }
}
