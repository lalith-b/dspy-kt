package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import kotlin.random.Random

/**
 * Labeled few-shot teleprompter.
 *
 * Samples k random examples from the trainset and assigns them as demos
 * to each predictor in the student module.
 *
 * Faithful port of `dspy/teleprompt/vanilla.py`.
 */
class LabeledFewShot(private val k: Int = 16) : Teleprompter() {
    private var trainset: List<Example> = emptyList()

    /**
     * Compile the student module by assigning few-shot demos to its predictors.
     *
     * Args:
     *     student: The student module to optimize.
     *     trainset: The training set to sample demos from.
     *     sample: If true, randomly sample k examples. If false, use the first k.
     *
     * Returns:
     *     The compiled student module.
     */
    fun compile(
        student: Module,
        trainset: List<Example>,
        sample: Boolean = true,
    ): Module {
        val compiledStudent = student.resetCopy() as Module
        this.trainset = trainset

        if (trainset.isEmpty()) {
            return compiledStudent
        }

        val rng = Random(0)

        for ((_, predictor) in compiledStudent.namedPredictors()) {
            if (sample) {
                predictor.demos = trainset.shuffled(rng).take(minOf(k, trainset.size)).map { it.toDict() }.toMutableList()
            } else {
                predictor.demos = trainset.take(minOf(k, trainset.size)).map { it.toDict() }.toMutableList()
            }
        }

        return compiledStudent
    }
}
