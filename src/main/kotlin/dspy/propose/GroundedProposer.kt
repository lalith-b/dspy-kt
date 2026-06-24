package dspy.propose

import dspy.primitives.Module

/**
 * Tips for instruction generation.
 */
val TIPS = mapOf(
    "none" to "",
    "creative" to "Don't be afraid to be creative when creating the new instruction!",
    "simple" to "Keep the instruction clear and concise.",
    "description" to "Make sure your instruction is very informative and descriptive.",
    "high_stakes" to "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona" to "Include a persona that is relevant to the task in the instruction"
)

/**
 * Grounded proposer for generating instructions based on program structure and task demonstrations.
 *
 * Simplified port of `dspy/propose/grounded_proposer.py`
 */
class GroundedProposer(
    val promptModel: dspy.clients.BaseLM,
    val program: Module,
    val trainset: List<dspy.primitives.Example>,
    val verbose: Boolean = false,
    val initTemperature: Float = 1.0f
) : Proposer() {
    
    override fun proposeInstructionsForProgram(
        trainset: List<Map<String, Any?>>,
        program: Module,
        demoCandidates: Map<Int, List<List<Map<String, Any?>>>>,
        trialLogs: Map<Int, Map<String, Any?>>,
        n: Int
    ): Map<Int, List<String>> {
        val proposedInstructions = mutableMapOf<Int, MutableList<String>>()
        
        // Simplified implementation - would use LLM to generate instructions
        for (predI in 0 until n) {
            proposedInstructions[predI] = mutableListOf("Proposed instruction for predictor $predI")
        }
        
        return proposedInstructions
    }
}

/**
 * Exception for value errors.
 */
class ValueError(message: String) : IllegalArgumentException(message)
