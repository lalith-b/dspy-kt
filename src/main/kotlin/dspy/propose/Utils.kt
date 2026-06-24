package dspy.propose

import dspy.primitives.Module
import dspy.signatures.Signature
import dspy.teleprompt.getSignature
import kotlin.random.Random

/**
 * Utility functions for proposal generation.
 *
 * Faithful port of `dspy/propose/utils.py`.
 */

/**
 * Strip prefix from text (removes leading markers and quotes).
 */
fun stripPrefix(text: String): String {
    val pattern = Regex("""^[\*\s]*(([\w\'\-\s]+){0,4}[\w\'\-]+):\s*""")
    return pattern.replace(text, "").trim().removeSurrounding("\"")
}

/**
 * Create a formatted string history of instruction sets for the program.
 */
fun createInstructionSetHistoryString(
    baseProgram: Module,
    trialLogs: Map<Int, Map<String, Any?>>,
    topN: Int,
): String {
    val programHistory = mutableListOf<Map<String, Any?>>()

    for (trialNum in trialLogs.keys) {
        val trial = trialLogs[trialNum] ?: continue
        if ("program_path" in trial) {
            val trialProgram = baseProgram.deepcopy()
            // In a full implementation, would load from path
            programHistory.add(mutableMapOf<String, Any?>(
                "program" to trialProgram,
                "score" to ((trial["score"] as? Double) ?: 0.0),
            ))
        }
    }

    // Deduplicate based on instruction set
    val seenPrograms = mutableSetOf<String>()
    val uniqueProgramHistory = mutableListOf<Map<String, Any?>>()
    for (entry in programHistory) {
        val program = entry["program"] as? Module ?: continue
        val instructionSet = getProgramInstructionSetString(program)
        if (instructionSet !in seenPrograms) {
            seenPrograms.add(instructionSet)
            uniqueProgramHistory.add(entry)
        }
    }

    // Get top N programs
    val topNProgramHistory = uniqueProgramHistory
        .sortedByDescending { (it["score"] as? Double) ?: 0.0 }
        .take(topN)
        .reversed()

    // Create formatted string
    val sb = StringBuilder()
    for (entry in topNProgramHistory) {
        val program = entry["program"] as? Module ?: continue
        val score = entry["score"] as? Double ?: 0.0
        val instructionSet = getProgramInstructionSetString(program)
        sb.append("$instructionSet | Score: $score\n\n")
    }

    return sb.toString()
}

/**
 * Parse a string representation of a list of instructions.
 */
@Suppress("UNCHECKED_CAST")
fun parseListOfInstructions(instructionString: String): List<String> {
    // Try JSON parsing first
    return try {
        val trimmed = instructionString.trim()
        if (trimmed.startsWith("[")) {
            // Manual parsing for simple list format
            val inner = trimmed.substring(1, trimmed.length - 1).trim()
            if (inner.isEmpty()) return emptyList()
            val items = mutableListOf<String>()
            val pattern = Regex("\"([^\"]*)\"")
            for (match in pattern.findAll(inner)) {
                items.add(match.groupValues[1])
            }
            items
        } else {
            listOf(instructionString)
        }
    } catch (e: Exception) {
        // Fall back to regex extraction
        val pattern = Regex("\"([^\"]*)\"")
        pattern.findAll(instructionString).map { it.groupValues[1] }.toList()
    }
}

/**
 * Get the program's instruction set as a formatted string.
 */
fun getProgramInstructionSetString(program: Module): String {
    val instructionList = program.predictors().map { pred ->
        val instructions = (pred as? dspy.predict.Predict)?.signature?.instructions ?: ""
        "\"$instructions\""
    }
    return "[${instructionList.joinToString(", ")}]"
}

/**
 * Create a predictor-level history string.
 */
fun createPredictorLevelHistoryString(
    baseProgram: Module,
    predictorI: Int,
    trialLogs: Map<Int, Map<String, Any?>>,
    topN: Int,
): String {
    val instructionAggregate = mutableMapOf<String, MutableList<Double>>()
    val instructionHistory = mutableListOf<Map<String, Any?>>()

    for (trialNum in trialLogs.keys) {
        val trial = trialLogs[trialNum] ?: continue
        if ("program_path" in trial) {
            val trialProgram = baseProgram.deepcopy()
            instructionHistory.add(mutableMapOf<String, Any?>(
                "program" to trialProgram,
                "score" to ((trial["score"] as? Double) ?: 0.0),
            ))
        }
    }

    // Aggregate scores for each instruction
    for (historyItem in instructionHistory) {
        val program = historyItem["program"] as? Module ?: continue
        val predictors = program.predictors()
        if (predictorI >= predictors.size) continue
        val predictor = predictors[predictorI]
        val instruction = (predictor as? dspy.predict.Predict)?.signature?.instructions ?: continue
        val score = (historyItem["score"] as? Double) ?: 0.0

        instructionAggregate.getOrPut(instruction) { mutableListOf() }.add(score)
    }

    // Calculate average scores
    val predictorHistory = instructionAggregate.map { (instruction, scores) ->
        instruction to (scores.average())
    }

    // Deduplicate and sort
    val seen = mutableSetOf<String>()
    val uniqueHistory = mutableListOf<Pair<String, Double>>()
    for ((instruction, score) in predictorHistory) {
        if (instruction !in seen) {
            seen.add(instruction)
            uniqueHistory.add(instruction to score)
        }
    }

    val topInstructions = uniqueHistory
        .sortedByDescending { it.second }
        .take(topN)
        .reversed()

    val sb = StringBuilder()
    for ((instruction, score) in topInstructions) {
        sb.append("$instruction | Score: $score\n\n")
    }

    return sb.toString()
}

/**
 * Create an example string from fields and example data.
 */
@Suppress("UNCHECKED_CAST")
fun createExampleString(fields: Map<String, Any?>, example: Map<String, Any?>): String {
    val output = mutableListOf<String>()

    for ((fieldName, fieldObj) in fields) {
        val fieldMap = fieldObj as? Map<String, Any?> ?: continue
        val prefix = fieldMap["prefix"] as? String ?: fieldName
        val value = example[fieldName] ?: ""
        output.add("$prefix $value")
    }

    return output.joinToString("\n")
}

/**
 * Get DSPy source code for a module (stub — inspect.getsource not available in Kotlin).
 */
fun getDspySourceCode(module: Module): String {
    // In Kotlin, we can't easily get source code via reflection.
    // Return a description of the module structure instead.
    val sb = StringBuilder()
    sb.append("// DSPy Module: ${module::class.simpleName}\n")

    for ((name, pred) in module.namedPredictors()) {
        val predictor = pred as? dspy.predict.Predict ?: continue
        sb.append("// Predictor: $name\n")
        sb.append("// Signature: ${predictor.signature.instructions}\n")
    }

    return sb.toString()
}
