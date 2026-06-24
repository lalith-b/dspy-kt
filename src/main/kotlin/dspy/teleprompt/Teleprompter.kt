package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings
import kotlin.reflect.full.memberProperties

/**
 * Base class for DSPy teleprompters (optimizers).
 *
 * A teleprompter compiles an uncompiled student module into an optimized one
 * using a training set, an optional teacher module, and optionally a validation set.
 *
 * Faithful port of `dspy/teleprompt/teleprompt.py`.
 */
abstract class Teleprompter {
    /**
     * Optimize the student program.
     *
     * Args:
     *     student: The student program to optimize.
     *     trainset: The training set to use for optimization.
     *     teacher: The teacher program to use for optimization.
     *     valset: The validation set to use for optimization.
     *
     * Returns:
     *     The optimized student program.
     */
    open suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module? = null,
        valset: List<Example>? = null,
    ): Module {
        throw NotImplementedError("compile not implemented for ${this::class.simpleName}")
    }

    /**
     * Get the parameters of the teleprompter.
     *
     * Returns:
     *     The parameters of the teleprompter.
     */
    fun getParams(): Map<String, Any?> {
        val params = mutableMapOf<String, Any?>()
        for (prop in this::class.memberProperties) {
            if (!prop.name.startsWith("_") && !prop.name.startsWith("get")) {
                val value = prop.getter.call(this)
                if (value != null) {
                    params[prop.name] = value
                }
            }
        }
        return params
    }
}
