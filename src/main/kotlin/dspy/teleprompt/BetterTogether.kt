package dspy.teleprompt

import kotlin.collections.sortedByDescending
import dspy.evaluate.Evaluate
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings
import kotlin.ranges.until
import kotlin.random.Random

/**
 * BetterTogether meta-optimizer.
 *
 * Combines prompt optimization and weight optimization (fine-tuning) by applying
 * them in a configurable sequence, allowing a student program to iteratively
 * improve both its prompts and model parameters.
 *
 * The core insight is that prompt and weight optimization can complement each other:
 * prompt optimization can discover effective task decompositions and reasoning strategies,
 * while weight optimization can specialize the model to execute these patterns more
 * efficiently.
 *
 * Faithful port of `dspy/teleprompt/bettertogether.py`.
 * Note: Some features like GEPA, BootstrapFinetune, and launch_lms/kill_lms
 * are simplified since they depend on external infrastructure.
 */

// ============================================================================
// ANSI color codes
// ============================================================================

private const val YELLOW = "\u001b[93m"
private const val GREEN = "\u001b[92m"
private const val BLUE = "\u001b[94m"
private const val BOLD = "\u001b[1m"
private const val ENDC = "\u001b[0m"

// ============================================================================
// BetterTogether
// ============================================================================

class BetterTogether(
    val metric: (Example, Prediction, List<Any>?) -> Any?,
    optimizers: Map<String, Teleprompter> = emptyMap(),
) : Teleprompter() {
    companion object {
        const val STRAT_SEP = " -> "
    }

    val optimizers: Map<String, Teleprompter>

    init {
        // Set default optimizers if none provided
        if (optimizers.isEmpty()) {
            println(
                "No optimizers provided. Using defaults: " +
                    "BootstrapFewShotWithRandomSearch (p) and LabeledFewShot (w). " +
                    "You can use the letters p and w to specify the compile strategy. " +
                    "For example, to run weight optimization after prompt optimization, use strategy='p -> w'."
            )
            this.optimizers = mapOf(
                "p" to BootstrapFewShotWithRandomSearch(metric = this.metric),
                "w" to LabeledFewShot(k = 4),
            )
        } else {
            // Validate all optimizers are Teleprompter instances
            for ((key, optimizer) in optimizers) {
                require(optimizer is Teleprompter) {
                    "Optimizer '$key' must be a Teleprompter, got ${optimizer::class.simpleName}"
                }
            }
            this.optimizers = optimizers
        }
    }

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        println("\n${BOLD}==> BETTERTOGETHER COMPILATION STARTED <==${ENDC}")
        println("${BLUE}Strategy:${ENDC} p -> w -> p (default)")
        println("${BLUE}Trainset size:${ENDC} ${trainset.size}")

        val strategy = "p -> w -> p"
        val valsetRatio = 0.1

        // Prepare student and teacher
        val preparedStudent = student.deepcopy()
        val preparedTeacher = if (teacher != null) listOf(teacher.deepcopy()) else null

        // Prepare trainset and valset
        val datasets = prepareTrainsetAndValset(trainset, valset, valsetRatio)
        var effectiveTrainset = datasets.trainset.toMutableList()
        var effectiveValset = datasets.valset

        // Parse strategy
        val parsedStrategy = prepareStrategy(strategy)

        // Validate optimizer compile args (empty by default)
        val optimizerCompileArgs = mapOf<String, Map<String, Any?>>()

        // Run strategies
        var result = runStrategies(
            student = preparedStudent,
            trainset = effectiveTrainset,
            teacher = preparedTeacher,
            valset = effectiveValset,
            parsedStrategy = parsedStrategy,
            optimizerCompileArgs = optimizerCompileArgs,
        )

        // Mark as compiled
        result.program.compiled = true
        result.program._compiled = true

        println("\n${BOLD}${GREEN}==> BETTERTOGETHER COMPILATION COMPLETE <==${ENDC}")
        println("${GREEN}Best score achieved:${ENDC} ${result.bestScore}")
        val strategyDisplay = if (result.bestStrategy.isEmpty()) "original (no optimization)" else result.bestStrategy
        println("${GREEN}Best strategy:${ENDC} $strategyDisplay")

        return result.program
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    private data class DatasetPair(
        val trainset: List<Example>,
        val valset: List<Example>?,
    )

    private fun prepareTrainsetAndValset(
        trainset: List<Example>,
        valset: List<Example>?,
        valsetRatio: Double,
    ): DatasetPair {
        require(trainset.isNotEmpty()) { "trainset cannot be empty" }
        require(valsetRatio in 0.0..1.0) { "valsetRatio must be in range [0, 1], got $valsetRatio" }

        return if (valset != null) {
            println("${BLUE}Using provided validation set (${valset.size} examples). Ignoring valsetRatio.${ENDC}")
            DatasetPair(trainset.toList(), valset)
        } else if (valsetRatio == 0.0) {
            println("${YELLOW}No validation set provided and valsetRatio=0. No validation set created.${ENDC}")
            DatasetPair(trainset, null)
        } else {
            val numValExamples = (valsetRatio * trainset.size).toInt()
            val valExamples = trainset.subList(0, numValExamples)
            val trainExamples = trainset.subList(numValExamples, trainset.size)
            println("${BLUE}Sampling ${valsetRatio * 100}% of trainset as validation set.${ENDC}")
            println("${BLUE}Created validation set: ${valExamples.size} examples. Training set: ${trainExamples.size} examples.${ENDC}")
            DatasetPair(trainExamples, valExamples)
        }
    }

    private fun prepareStrategy(strategy: String): List<String> {
        require(strategy.isNotBlank()) { "strategy cannot be empty" }

        val parsed = strategy.split(STRAT_SEP)

        val invalidSteps = parsed.filter { it !in optimizers }
        require(invalidSteps.isEmpty()) {
            "Strategy contains invalid optimizer keys: $invalidSteps. " +
                "Valid keys are: ${optimizers.keys.toList()}"
        }

        return parsed
    }

    private data class OptimizationResult(
        val program: Module,
        val bestScore: Double?,
        val bestStrategy: String,
        val candidatePrograms: List<Map<String, Any?>>,
    )

    private suspend fun runStrategies(
        student: Module,
        trainset: MutableList<Example>,
        teacher: List<Module>?,
        valset: List<Example>?,
        parsedStrategy: List<String>,
        optimizerCompileArgs: Map<String, Map<String, Any?>>,
    ): OptimizationResult {
        val rng = Random(0)
        val candidatePrograms = mutableListOf<Map<String, Any?>>()
        var flagCompilationErrorOccurred = false

        // Evaluate original program
        println("\n${BOLD}==> BASELINE EVALUATION <==${ENDC}")
        println("Evaluating original program (no optimization applied)")

        val baselineScore = evaluateOnValset(student, valset, rng)
        addCandidate(candidatePrograms, student, strategy = "", score = baselineScore)
        println("${YELLOW}Baseline score:${ENDC} $baselineScore")

        var currentProgram = student

        // Apply each optimization step
        for ((ind, stepCode) in parsedStrategy.withIndex()) {
            val currentStrategy = parsedStrategy.subList(0, ind + 1).joinToString(STRAT_SEP)
            val optimizer = optimizers[stepCode]

            require(optimizer != null) { "Optimizer '$stepCode' not found in optimizers" }

            println("\n${BOLD}==> STEP ${ind + 1}/${parsedStrategy.size}: ${optimizer::class.simpleName?.uppercase()} <==${ENDC}")
            println("${BLUE}Current strategy:${ENDC} '$currentStrategy'")
            println("${BLUE}Optimizer:${ENDC} ${optimizer::class.simpleName?.uppercase()}")

            try {
                // Shuffle trainset between steps
                println("${BLUE}Shuffling trainset...${ENDC}")
                trainset.shuffle(rng)

                // Run optimizer
                currentProgram._compiled = false
                currentProgram.compiled = false
                println("${BLUE}Running ${optimizer::class.simpleName} with ${trainset.size} training examples...${ENDC}")

                currentProgram = optimizer.compile(
                    student = currentProgram,
                    trainset = trainset,
                    teacher = teacher?.firstOrNull(),
                    valset = valset,
                )

                // Evaluate optimized program
                val score = evaluateOnValset(currentProgram, valset, rng)
                addCandidate(candidatePrograms, currentProgram, currentStrategy, score)

                // Check if this is the best score so far
                val validScores = candidatePrograms.mapNotNull { it["score"] as? Double }
                val bestScoreSoFar = if (validScores.isNotEmpty()) validScores.max() else -Double.MAX_VALUE
                val isNewBest = score != null && score >= bestScoreSoFar

                if (isNewBest) {
                    println("${GREEN}New best score!${ENDC} $score (strategy: '$currentStrategy')")
                } else {
                    println("${YELLOW}Score after optimization:${ENDC} $score")
                }

            } catch (e: Exception) {
                flagCompilationErrorOccurred = true
                println("${YELLOW}Step ${ind + 1}/${parsedStrategy.size} failed with error: ${e::class.simpleName}: ${e.message}${ENDC}")
                println("${YELLOW}Stopping optimization early. Returning best program found so far from ${candidatePrograms.size} candidate(s).${ENDC}")
                e.printStackTrace()
                break
            }
        }

        // Sort candidates by score (best first), with earlier programs winning ties
        val candidatesWithIdx = candidatePrograms.mapIndexed { i, cp -> i to cp }
            .sortedByDescending { (_, cp) ->
                (cp["score"] as? Double) ?: -Double.MAX_VALUE
            }

        // Select best program
        val bestProgram = if (valset.isNullOrEmpty()) {
            // No valset: return the latest program (last in original order)
            candidatesWithIdx.last().second["program"] as? Module ?: currentProgram
        } else {
            // Valset provided: return highest score (first after sorting)
            candidatesWithIdx.first().second["program"] as? Module ?: currentProgram
        }

        val bestScore = candidatesWithIdx.first().second["score"] as? Double
        val bestStrategy = candidatesWithIdx.first().second["strategy"] as? String ?: ""

        return OptimizationResult(
            program = bestProgram,
            bestScore = bestScore,
            bestStrategy = bestStrategy,
            candidatePrograms = candidatesWithIdx.map { it.second },
        )
    }

    private suspend fun evaluateOnValset(
        program: Module,
        valset: List<Example>?,
        rng: Random,
    ): Double? {
        if (valset.isNullOrEmpty()) {
            println("${YELLOW}No validation set provided. Skipping evaluation.${ENDC}")
            return null
        }

        println("${BLUE}Evaluating on ${valset.size} validation examples...${ENDC}")
        val evaluate = Evaluate(
            devset = valset,
            metric = this.metric,
            displayTable = false,
            displayProgress = true,
        )
        val result = evaluate.__call__(program)
        return result.score
    }

    private fun addCandidate(
        candidatePrograms: MutableList<Map<String, Any?>>,
        student: Module,
        strategy: String,
        score: Double?,
    ) {
        candidatePrograms.add(mapOf(
            "score" to score,
            "program" to student.deepcopy(),
            "strategy" to strategy,
        ))
    }
}

// ============================================================================
// Legacy alias
// ============================================================================

/**
 * Legacy alias for BetterTogether.
 */
typealias BETTERTOGETHER = BetterTogether
