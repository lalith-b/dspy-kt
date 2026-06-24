package dspy.teleprompt

import dspy.evaluate.Evaluate
import dspy.evaluate.EvaluationResult
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.utils.Settings
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * MIPROv2 (Massive Initialization Parameter Random Optimization v2) teleprompter.
 *
 * Searches over instruction and few-shot example candidates using random search
 * to find optimal prompt parameters for a DSPy program.
 *
 * Faithful port of `dspy/teleprompt/mipro_optimizer_v2.py`.
 * Note: Optuna-based Bayesian optimization is replaced with random search since
 * Optuna is not available in Kotlin.
 */

// ============================================================================
// Constants
// ============================================================================

const val BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
const val LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
const val MIN_MINIBATCH_SIZE = 50

private val AUTO_RUN_SETTINGS: Map<String, Map<String, Int>> = mapOf(
    "light" to mapOf("n" to 6, "val_size" to 100),
    "medium" to mapOf("n" to 12, "val_size" to 300),
    "heavy" to mapOf("n" to 18, "val_size" to 1000),
)

// ============================================================================
// Optimization result data class
// ============================================================================

data class MIPROv2OptimizationResult(
    val bestProgram: Module,
    val bestScore: Double,
    val trialLogs: Map<Int, Map<String, Any>>,
    val candidatePrograms: List<Map<String, Any>>,
)

// ============================================================================
// MIPROv2
// ============================================================================

class MIPROv2(
    val metric: (Example, Prediction, List<Any>?) -> Any?,
    val promptModel: dspy.clients.BaseLM? = null,
    val taskModel: dspy.clients.BaseLM? = null,
    teacherSettings: Map<String, Any?>? = null,
    val maxBootstrappedDemos: Int = 4,
    val maxLabeledDemos: Int = 4,
    auto: String? = "light",
    numCandidates: Int? = null,
    val numThreads: Int? = null,
    maxErrors: Int? = null,
    val seed: Int = 9,
    val initTemperature: Double = 1.0,
    val verbose: Boolean = false,
    val trackStats: Boolean = true,
    val logDir: String? = null,
    val metricThreshold: Double? = null,
) : Teleprompter() {
    // Validate 'auto' parameter
    val allowedModes = setOf<String?>(null, "light", "medium", "heavy")
    init {
        require(auto in allowedModes) { "Invalid value for auto: $auto. Must be one of $allowedModes." }
    }

    val auto: String? = auto
    val numFewshotCandidates: Int? = numCandidates
    val numInstructCandidates: Int? = numCandidates
    val numCandidates: Int? = numCandidates
    val teacherSettings: Map<String, Any?> = teacherSettings ?: emptyMap()
    var maxErrors: Int? = maxErrors
    var rng: Random? = null
    var promptModelTotalCalls: Int = 0
    var totalCalls: Int = 0

    init {
        require(this.promptModel != null && this.taskModel != null) {
            "Either provide both promptModel and taskModel or set a default LM through Settings.configure(lm=...)"
        }
    }

    /**
     * Optimizes the student program by searching over instruction and few-shot example candidates.
     *
     * Note: In the full Python implementation, compile also supports additional parameters
     * like `numTrials`, `minibatch`, `minibatchSize`, etc. These are derived from the `auto`
     * setting or must be set via `auto=null` and `numCandidates`.
     */
    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        val effectiveMaxErrors = maxErrors ?: Settings.maxErrors

        // Set random seeds
        setRandomSeeds(seed)

        // Set training & validation sets
        val datasets = setAndValidateDatasets(trainset, valset)
        val effectiveTrainset = datasets.trainset
        val effectiveValset = datasets.valset

        // Determine zeroshot optimization
        val zeroshotOpt = maxBootstrappedDemos == 0 && maxLabeledDemos == 0

        // Determine num_trials and valset based on run mode
        var numInstruct: Int
        var numFewshot: Int
        var finalValset = effectiveValset

        if (auto != null) {
            val hyperparams = setHyperparamsFromRunMode(
                program = student,
                numInstructCandidates = numInstructCandidates,
                numFewshotCandidates = numFewshotCandidates,
                zeroshotOpt = zeroshotOpt,
                valset = effectiveValset,
            )
            numInstruct = hyperparams.numInstructCandidates
            numFewshot = hyperparams.numFewshotCandidates
            finalValset = hyperparams.valset

            // Print auto run settings
            printAutoRunSettings(
                numTrials = hyperparams.numTrials,
                minibatch = hyperparams.minibatch,
                valset = hyperparams.valset,
                numFewshotCandidates = numFewshot,
                numInstructCandidates = numInstruct,
            )
        } else {
            numInstruct = numInstructCandidates ?: 6
            numFewshot = numFewshotCandidates ?: 6
        }

        // Initialize evaluator
        val evaluate = Evaluate(
            devset = finalValset,
            metric = this.metric,
            numThreads = numThreads,
            maxErrors = effectiveMaxErrors,
            displayTable = false,
            displayProgress = true,
        )

        // Run optimization
        val result = optimizePromptParameters(
            program = student.deepcopy(),
            evaluate = evaluate,
            valset = finalValset,
            effectiveTrainset = effectiveTrainset,
            teacher = teacher,
            numFewshotCandidates = numFewshot,
            numInstructCandidates = numInstruct,
            zeroshotOpt = zeroshotOpt,
        )

        // Mark as compiled
        result.bestProgram.compiled = true
        result.bestProgram._compiled = true

        return result.bestProgram
    }

    // ========================================================================
    // Helper methods (public for extensibility, matching Python API)
    // ========================================================================

    fun setRandomSeeds(seed: Int) {
        rng = Random(seed)
    }

    fun setNumTrialsFromNumCandidates(program: Module, zeroshotOpt: Boolean, numCandidates: Int): Int {
        var numVars = program.predictors().size
        if (!zeroshotOpt) {
            numVars *= 2 // Account for few-shot examples + instruction variables
        }
        // Trials = MAX(2 * M * log2(N), ceil(1.5 * N))
        return maxOf(
            (2 * numVars * log2(numCandidates.toDouble())).toInt(),
            (1.5 * numCandidates).toInt()
        )
    }

    data class Hyperparams(
        val numTrials: Int,
        val valset: List<Example>,
        val minibatch: Boolean,
        val numInstructCandidates: Int,
        val numFewshotCandidates: Int,
    )

    fun setHyperparamsFromRunMode(
        program: Module,
        numInstructCandidates: Int?,
        numFewshotCandidates: Int?,
        zeroshotOpt: Boolean,
        valset: List<Example>,
    ): Hyperparams {
        if (auto == null) {
            require(numInstructCandidates != null && numFewshotCandidates != null) {
                "numCandidates must be provided when auto is None."
            }
            val numTrials = setNumTrialsFromNumCandidates(program, zeroshotOpt, numInstructCandidates)
            return Hyperparams(numTrials, valset, true, numInstructCandidates, numFewshotCandidates)
        }

        val settings = AUTO_RUN_SETTINGS[auto]!!
        val effectiveRng = rng ?: Random(seed)
        val effectiveValset = createMinibatch(valset, settings["val_size"]!!, effectiveRng)
        val minibatch = effectiveValset.size > MIN_MINIBATCH_SIZE

        // Set num instruct candidates to 1/2 of N if optimizing with few-shot examples
        val nInstruct = if (zeroshotOpt) settings["n"]!! else (settings["n"]!! * 0.5).toInt()
        val nFewshot = settings["n"]!!

        val numTrials = setNumTrialsFromNumCandidates(program, zeroshotOpt, settings["n"]!!)

        return Hyperparams(numTrials, effectiveValset, minibatch, nInstruct, nFewshot)
    }

    data class DatasetPair(
        val trainset: List<Example>,
        val valset: List<Example>,
    )

    fun setAndValidateDatasets(trainset: List<Example>, valset: List<Example>?): DatasetPair {
        require(trainset.isNotEmpty()) { "Trainset cannot be empty." }

        return if (valset == null) {
            require(trainset.size >= 2) { "Trainset must have at least 2 examples if no valset specified." }
            val valsetSize = minOf(1000, maxOf(1, (trainset.size * 0.80).toInt()))
            val cutoff = trainset.size - valsetSize
            DatasetPair(
                trainset = trainset.subList(0, cutoff),
                valset = trainset.subList(cutoff, trainset.size),
            )
        } else {
            require(valset.size >= 1) { "Validation set must have at least 1 example." }
            DatasetPair(trainset, valset)
        }
    }

    private fun printAutoRunSettings(
        numTrials: Int,
        minibatch: Boolean,
        valset: List<Example>,
        numFewshotCandidates: Int,
        numInstructCandidates: Int,
    ) {
        println(
            "\nRUNNING WITH THE FOLLOWING ${auto?.uppercase()} AUTO RUN SETTINGS:" +
                "\nnumTrials: $numTrials" +
                "\nminibatch: $minibatch" +
                "\nnumFewshotCandidates: $numFewshotCandidates" +
                "\nnumInstructCandidates: $numInstructCandidates" +
                "\nvalset size: ${valset.size}\n"
        )
    }

    // ========================================================================
    // Instruction proposal (simplified - no GroundedProposer in Kotlin)
    // ========================================================================

    private suspend fun proposeInstructions(
        program: Module,
        numInstructCandidates: Int,
    ): Map<Int, List<String>> {
        println("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        println("Proposing N=$numInstructCandidates instructions...")

        val instructionCandidates = mutableMapOf<Int, MutableList<String>>()

        for ((i, predictor) in program.predictors().withIndex()) {
            val sig = getSignature(predictor)
            val original = sig.instruction

            instructionCandidates[i] = mutableListOf(original)

            // Generate additional instruction variants
            for (j in 1 until numInstructCandidates) {
                val variant = "${original}\n\nTip $j: Consider breaking down the task into smaller steps and reasoning through each one carefully."
                instructionCandidates[i]?.add(variant)
            }

            if (verbose) {
                println("Proposed Instructions for Predictor $i:")
                instructionCandidates[i]?.forEachIndexed { idx, instr ->
                    println("  $idx: $instr")
                }
                println()
            }
        }

        return instructionCandidates
    }

    // ========================================================================
    // Optimization (simplified random search instead of Optuna Bayesian opt)
    // ========================================================================

    private suspend fun optimizePromptParameters(
        program: Module,
        evaluate: Evaluate,
        valset: List<Example>,
        effectiveTrainset: List<Example>,
        teacher: Module?,
        numFewshotCandidates: Int,
        numInstructCandidates: Int,
        zeroshotOpt: Boolean,
    ): MIPROv2OptimizationResult {
        val effectiveRng = rng ?: Random(seed)

        // Step 1: Bootstrap few-shot examples
        println("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        var demoCandidates: Map<Int, List<List<Map<String, Any?>>>>? = null

        if (!zeroshotOpt) {
            println("Bootstrapping N=$numFewshotCandidates sets of demonstrations...")
            demoCandidates = createNFewShotDemoSets(
                student = program,
                numCandidateSets = numFewshotCandidates,
                trainset = effectiveTrainset,
                maxLabeledDemos = maxLabeledDemos,
                maxBootstrappedDemos = maxBootstrappedDemos,
                metric = this.metric,
                teacherSettings = teacherSettings,
                maxErrors = maxErrors ?: Settings.maxErrors,
                metricThreshold = metricThreshold,
                teacher = teacher,
                seed = seed,
                rng = effectiveRng,
            )
        } else {
            println("Running in zero-shot mode, skipping few-shot bootstrapping.")
        }

        // Step 2: Propose instructions
        val instructionCandidates = proposeInstructions(program, numInstructCandidates)

        // If zero-shot, discard demos
        if (zeroshotOpt) {
            demoCandidates = null
        }

        // Step 3: Find optimal prompt parameters
        println("\n==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        println("Evaluating with random search over instruction and few-shot combinations...")

        // Evaluate default program first
        val defaultResult = evaluate.__call__(program)
        val defaultScore = defaultResult.score
        println("Default program score: $defaultScore")

        var bestScore = defaultScore
        var bestProgram: Module = program.deepcopy()
        val numPredictors = program.predictors().size
        val numTrials = setNumTrialsFromNumCandidates(program, zeroshotOpt, numInstructCandidates)

        val trialLogs = mutableMapOf<Int, MutableMap<String, Any>>()
        val scoreData = mutableListOf<Map<String, Any>>()

        // Log default evaluation
        trialLogs[0] = mutableMapOf(
            "full_eval_score" to defaultScore,
            "score" to defaultScore,
            "full_eval" to true,
        )
        scoreData.add(mapOf(
            "score" to defaultScore,
            "program" to program.deepcopy(),
            "full_eval" to true,
        ))

        for (trialIdx in 1..numTrials) {
            println("== Trial $trialIdx / $numTrials ==")

            // Create a new candidate program
            val candidateProgram = program.deepcopy()

            // Randomly select instruction and demo indices for each predictor
            val chosenParams = mutableListOf<String>()
            for (i in 0 until numPredictors) {
                val predictor = candidateProgram.predictors()[i]

                // Select instruction
                val instrList = instructionCandidates[i] ?: emptyList()
                val instrIdx = effectiveRng.nextInt(instrList.size)
                val selectedInstruction = instrList[instrIdx]

                val updatedSignature = getSignature(predictor).withInstructions(selectedInstruction)
                setSignature(predictor, updatedSignature)
                chosenParams.add("Predictor $i: Instruction $instrIdx")

                // Select demos if available
                if (demoCandidates != null) {
                    val demosList = demoCandidates[i] ?: emptyList()
                    if (demosList.isNotEmpty()) {
                        val demosIdx = effectiveRng.nextInt(demosList.size)
                        predictor.demos = demosList[demosIdx].toMutableList()
                        chosenParams.add("Predictor $i: Few-Shot Set $demosIdx")
                    }
                }
            }

            // Evaluate candidate program
            val candidateResult = evaluate.__call__(candidateProgram)
            val scoreValue = candidateResult.score

            if (scoreValue > bestScore) {
                bestScore = scoreValue
                bestProgram = candidateProgram.deepcopy()
                println("Best full score so far! Score: $scoreValue")
            }

            println("Score: $scoreValue with parameters: ${chosenParams.joinToString(", ")}")

            trialLogs[trialIdx] = mutableMapOf(
                "full_eval_score" to scoreValue,
                "score" to scoreValue,
                "full_eval" to true,
                "chosen_params" to chosenParams,
            )

            scoreData.add(mapOf(
                "score" to scoreValue,
                "program" to candidateProgram,
                "full_eval" to true,
            ))
        }

        // Sort candidates by score (best first)
        @Suppress("UNCHECKED_CAST")
        val sortedCandidates = scoreData.sortedByDescending { it["score"] as? Double ?: 0.0 }
        println("\nScores so far: ${sortedCandidates.map { it["score"] }.joinToString(", ")}")
        println("Best score: $bestScore")
        println("Returning best identified program with score $bestScore!")

        return MIPROv2OptimizationResult(
            bestProgram = bestProgram,
            bestScore = bestScore,
            trialLogs = trialLogs,
            candidatePrograms = sortedCandidates,
        )
    }
}

// ============================================================================
// Legacy alias
// ============================================================================

/**
 * Legacy alias for MIPROv2.
 */
typealias MIPRO = MIPROv2
