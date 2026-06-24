package dspy.propose

import dspy.primitives.Module
import dspy.signatures.Signature
import dspy.teleprompt.getPromptModel
import dspy.teleprompt.getSignature
import kotlin.random.Random

/**
 * Proposer base class.
 *
 * Faithful port of `dspy/propose/propose_base.py`.
 */
abstract class Proposer {
    /**
     * Propose instructions for the program.
     *
     * Args:
     *     trainset: Training dataset.
     *     program: The DSPy program to optimize.
     *     demoCandidates: Demo candidates for each predictor.
     *     trialLogs: Logs of past trials.
     *     n: Number of instructions to propose per predictor.
     *
     * Returns:
     *     Map from predictor index to list of proposed instructions.
     */
    abstract fun proposeInstructionsForProgram(
        trainset: List<Map<String, Any?>>,
        program: Module,
        demoCandidates: Map<Int, List<List<Map<String, Any?>>>>,
        trialLogs: Map<Int, Map<String, Any?>>,
        n: Int,
    ): Map<Int, List<String>>

    /**
     * Propose a single instruction for a predictor.
     */
    open fun proposeInstructionForPredictor(
        program: Module,
        predictor: Any,
        predI: Int,
        demoCandidates: Map<Int, List<List<Map<String, Any?>>>>,
        demoSetI: Int,
        trialLogs: Map<Int, Map<String, Any?>>,
        tip: String?,
    ): String {
        throw NotImplementedError("Not implemented for ${this::class.simpleName}")
    }
}
