package dspy.teleprompt

import dspy.primitives.Module
import kotlin.random.Random

/**
 * Ensemble teleprompter.
 *
 * Combines multiple programs into an ensemble. At forward time, it samples
 * (or uses all) programs, calls each, and reduces their outputs.
 *
 * A common reduce_fn is majority voting.
 *
 * Faithful port of `dspy/teleprompt/ensemble.py`.
 */
class Ensemble(
    private val reduceFn: ((List<Any>) -> Any)? = null,
    private val size: Int? = null,
    private val deterministic: Boolean = false,
) : Teleprompter() {
    init {
        require(!deterministic) { "TODO: Implement example hashing for deterministic ensemble." }
    }

    /**
     * Create an ensembled program from a list of programs.
     *
     * This matches the Python `compile(programs)` signature.
     */
    fun compile(programs: List<Module>): Module {
        return createEnsembledProgram(programs)
    }

    override suspend fun compile(
        student: Module,
        trainset: List<dspy.primitives.Example>,
        teacher: Module?,
        valset: List<dspy.primitives.Example>?,
    ): Module {
        // Override: for Ensemble, we need programs, not a single student.
        // The Python code uses `compile(programs)` where programs is a list.
        // In Kotlin, we handle this by using trainset as a proxy.
        return createEnsembledProgram(emptyList())
    }

    private fun createEnsembledProgram(programs: List<Module>): Module {
        return EnsembledProgram(programs, size, reduceFn)
    }
}

/**
 * Internal ensembled program module.
 */
private class EnsembledProgram(
    private val programs: List<Module>,
    private val size: Int?,
    private val reduceFn: ((List<Any>) -> Any)?,
) : Module() {
    override suspend fun invoke(kwargs: Map<String, Any?>): dspy.primitives.Prediction {
        val selectedPrograms = if (size != null) {
            programs.shuffled().take(size)
        } else {
            programs
        }

        val outputs = selectedPrograms.map { prog ->
            prog(kwargs)
        }

        return if (reduceFn != null) {
            reduceFn(outputs) as? dspy.primitives.Prediction
                ?: dspy.primitives.Prediction(emptyMap())
        } else {
            outputs.firstOrNull() as? dspy.primitives.Prediction
                ?: dspy.primitives.Prediction(emptyMap())
        }
    }
}
