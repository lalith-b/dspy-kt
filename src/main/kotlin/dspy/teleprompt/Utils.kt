package dspy.teleprompt

import dspy.evaluate.Evaluate
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.utils.Settings
import kotlin.random.Random

/**
 * Helper functions for DSPy optimizers.
 *
 * Faithful port of `dspy/teleprompt/utils.py`.
 */

// ============================================================================
// Optimizer Training Utils
// ============================================================================

/**
 * Create a minibatch from the trainset.
 */
fun createMinibatch(
    trainset: List<Example>,
    batchSize: Int = 50,
    rng: Random? = null,
): List<Example> {
    val effectiveBatchSize = minOf(batchSize, trainset.size)
    val effectiveRng = rng ?: Random.Default

    val sampledIndices = trainset.indices.shuffled(effectiveRng).take(effectiveBatchSize)
    return sampledIndices.map { trainset[it] }
}

/**
 * Evaluate a candidate program on the trainset, using the specified batch size.
 */
suspend fun evalCandidateProgram(
    batchSize: Int,
    trainset: List<Example>,
    candidateProgram: Module,
    evaluate: Evaluate,
    rng: Random? = null,
): Any {
    return try {
        if (batchSize >= trainset.size) {
            evaluate.__call__(candidateProgram)
        } else {
            val eval = Evaluate(
                devset = createMinibatch(trainset, batchSize, rng),
                metric = evaluate.metric,
                numThreads = evaluate.numThreads,
                maxErrors = evaluate.maxErrors,
                displayTable = false,
                displayProgress = true,
            )
            eval.__call__(candidateProgram)
        }
    } catch (e: Exception) {
        // TODO: Handle this better, as -ve scores are possible
        Prediction(mapOf("score" to 0.0, "results" to emptyList<Any>()))
    }
}

/**
 * Calculate the average and best quality of the last n programs proposed.
 * This is useful for seeing if our proposals are actually 'improving' over time or not.
 */
suspend fun calculateLastNProposedQuality(
    baseProgram: Module,
    trialLogs: Map<Int, Map<String, Any>>,
    evaluate: Evaluate,
    trainset: List<Example>,
    devset: List<Example>,
    n: Int,
): Quadruple<Double, Double, Double, Double> {
    val lastNTrialNums = trialLogs.keys.toList().takeLast(n)

    var totalTrainScore = 0.0
    var bestTrainScore = 0.0
    var totalDevScore = 0.0
    var bestDevScore = 0.0

    for (trialNum in lastNTrialNums) {
        val trialLog = trialLogs[trialNum] ?: continue
        val fullEval = trialLog["full_eval"] as? Boolean ?: false
        if (!fullEval) {
            throw NotImplementedError(
                "Still need to implement non full eval handling in calculateLastNProposedQuality"
            )
        }
        val trainScore = trialLog["score"] as? Double ?: 0.0
        val program = baseProgram.deepcopy()

        val devScore = evaluate.__call__(program).score

        totalTrainScore += trainScore
        totalDevScore += devScore
        if (trainScore > bestTrainScore) {
            bestTrainScore = trainScore
            bestDevScore = devScore
        }
    }

    return Quadruple(
        bestTrainScore,
        totalTrainScore / n,
        bestDevScore,
        totalDevScore / n,
    )
}

/**
 * Get the prompt model.
 */
fun getPromptModel(promptModel: dspy.clients.BaseLM?): dspy.clients.BaseLM? {
    return promptModel ?: Settings.lm()
}

/**
 * Get the signature from a predictor.
 */
fun getSignature(predictor: Any): Signature {
    require(predictor is dspy.predict.Predict) { "predictor must be a Predict instance" }
    return predictor.signature
}

/**
 * Set the signature on a predictor.
 */
fun setSignature(predictor: Any, updatedSignature: Signature) {
    require(predictor is dspy.predict.Predict) { "predictor must be a Predict instance" }
    predictor.signature = updatedSignature
}

/**
 * Create n few-shot demo sets using the same approach as random search.
 */
suspend fun createNFewShotDemoSets(
    student: Module,
    numCandidateSets: Int,
    trainset: List<Example>,
    maxLabeledDemos: Int,
    maxBootstrappedDemos: Int,
    metric: (Example, Prediction, List<Any>?) -> Any?,
    teacherSettings: Map<String, Any?>,
    maxErrors: Int? = null,
    maxRounds: Int = 1,
    labeledSample: Boolean = true,
    minNumSamples: Int = 1,
    metricThreshold: Double? = null,
    teacher: Module? = null,
    includeNonBootstrapped: Boolean = true,
    seed: Int = 0,
    rng: Random? = null,
): Map<Int, List<List<Map<String, Any?>>>> {
    val effectiveMaxErrors = maxErrors ?: Settings.maxErrors
    val effectiveRng = rng ?: Random(seed)

    // Account for 3 extra candidate sets (zero-shot, labels-only, unshuffled)
    val adjustedNumSets = numCandidateSets - 3

    val demoCandidates = student.predictors().indices.associateWith { mutableListOf<List<Map<String, Any?>>>() }

    for (seed in -3 until adjustedNumSets) {
        println("Bootstrapping set ${seed + 4}/${adjustedNumSets + 3}")

        val trainsetCopy = trainset.toMutableList()
        var program2: Module = when (seed) {
            -3 -> {
                if (includeNonBootstrapped) student.resetCopy() else continue
            }
            -2 -> {
                if (includeNonBootstrapped && maxLabeledDemos > 0) {
                    val teleprompter = LabeledFewShot(k = maxLabeledDemos)
                    teleprompter.compile(student, trainset = trainsetCopy)
                } else {
                    continue
                }
            }
            -1 -> {
                val program = BootstrapFewShot(
                    metric = metric,
                    maxErrors = effectiveMaxErrors,
                    metricThreshold = metricThreshold,
                    maxBootstrappedDemos = maxBootstrappedDemos,
                    maxLabeledDemos = maxLabeledDemos,
                    teacherSettings = teacherSettings,
                    maxRounds = maxRounds,
                )
                program.compile(student, trainset = trainsetCopy, teacher = teacher, valset = null)
            }
            else -> {
                // Python's shuffle mutates in-place; Kotlin's shuffled returns new list
                val shuffledTrainset = trainsetCopy.shuffled(effectiveRng)
                // Python's randint(a, b) is inclusive on both ends
                // Kotlin's nextInt(a, b) is exclusive on upper end, so use maxBootstrappedDemos + 1
                val size = effectiveRng.nextInt(minNumSamples, maxBootstrappedDemos + 1)

                val teleprompter = BootstrapFewShot(
                    metric = metric,
                    maxErrors = effectiveMaxErrors,
                    metricThreshold = metricThreshold,
                    maxBootstrappedDemos = size,
                    maxLabeledDemos = maxLabeledDemos,
                    teacherSettings = teacherSettings,
                    maxRounds = maxRounds,
                )
                teleprompter.compile(student, trainset = shuffledTrainset, teacher = teacher, valset = null)
            }
        }

        for (i in student.predictors().indices) {
            val demos = student.predictors()[i].demos
            demoCandidates[i]?.add(demos.toList())
        }
    }

    return demoCandidates
}

/**
 * Get a full trace of the task model's history for a given candidate program.
 */
suspend fun getTaskModelHistoryForFullExample(
    candidateProgram: Module,
    taskModel: dspy.clients.BaseLM,
    devset: List<Example>,
    evaluate: Evaluate,
) {
    evaluate.__call__(candidateProgram)
    taskModel.inspectHistory(n = candidateProgram.predictors().size)
}

/**
 * Print out the program's instructions & prefixes for each module.
 */
fun printFullProgram(program: Module) {
    for ((i, predictor) in program.predictors().withIndex()) {
        println("Predictor $i")
        println("i: ${predictor.signature.instruction}")
        val allFields = predictor.signature.inputFields + predictor.signature.outputFields
        if (allFields.isNotEmpty()) {
            val lastField = allFields.last()
            println("p: ${lastField.prefix ?: "${lastField.name.replace('_', ' ')}:"}")
        }
        println()
    }
}

/**
 * Save the candidate program to the log directory.
 */
fun saveCandidateProgram(
    program: Module,
    logDir: String?,
    trialNum: Int,
    note: String? = null,
): String? {
    if (logDir == null) return null

    val evalProgramsDir = "$logDir/evaluated_programs"
    val savePath = if (note != null) {
        "$evalProgramsDir/program_${trialNum}_${note}.json"
    } else {
        "$evalProgramsDir/program_${trialNum}.json"
    }

    // In Kotlin, we'd serialize the program state to JSON
    // For now, just return the path
    return savePath
}

/**
 * Extract total input tokens and output tokens from a model's interaction history.
 * Returns (totalInputTokens, totalOutputTokens).
 */
fun getTokenUsage(model: dspy.clients.BaseLM): Pair<Int, Int> {
    val history = model.history
    var totalInputTokens = 0
    var totalOutputTokens = 0

    for (interaction in history) {
        if (interaction is Map<*, *>) {
            val usage = interaction["usage"] as? Map<*, *> ?: continue
            totalInputTokens += (usage["prompt_tokens"] as? Int ?: 0)
            totalOutputTokens += (usage["completion_tokens"] as? Int ?: 0)
        }
    }

    return totalInputTokens to totalOutputTokens
}

/**
 * Extract total input and output tokens used by each model and log to trial_logs.
 */
fun logTokenUsage(
    trialLogs: MutableMap<Int, MutableMap<String, Any>>,
    trialNum: Int,
    modelDict: Map<String, dspy.clients.BaseLM>,
) {
    val tokenUsageDict = mutableMapOf<String, Map<String, Int>>()

    for ((modelName, model) in modelDict) {
        val (inTokens, outTokens) = getTokenUsage(model)
        tokenUsageDict[modelName] = mapOf(
            "total_input_tokens" to inTokens,
            "total_output_tokens" to outTokens,
        )
    }

    trialLogs.getOrPut(trialNum) { mutableMapOf() }["token_usage"] = tokenUsageDict
}

/**
 * Setup logging to a file in the log directory.
 */
fun setupLogging(logDir: String?) {
    if (logDir == null) return
    // In Kotlin, we'd use a proper logging framework like slf4j
    // For now, this is a placeholder
}

/**
 * Save a file to the log directory.
 */
fun saveFileToLogDir(sourceFilePath: String, logDir: String?) {
    if (logDir == null) return
    val dest = "$logDir/${sourceFilePath.substringAfterLast('/')}"
    // Copy file - would use java.nio.file.Files.copy in production
}

/**
 * Quadruple type for calculateLastNProposedQuality result.
 */
data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
)
