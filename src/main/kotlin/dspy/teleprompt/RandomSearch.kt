package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.evaluate.Evaluate
import dspy.evaluate.EvaluationResult
import dspy.utils.Settings
import kotlin.random.Random

/**
 * Bootstrap few-shot with random search teleprompter.
 *
 * Searches over multiple candidate programs created with different random seeds,
 * shuffles, and demo sizes, and selects the best performing one.
 *
 * Faithful port of `dspy/teleprompt/random_search.py`.
 */
class BootstrapFewShotWithRandomSearch(
    val metric: (Example, Prediction, List<Any>?) -> Any?,
    teacherSettings: Map<String, Any?>? = null,
    maxBootstrappedDemos: Int = 4,
    maxLabeledDemos: Int = 16,
    val maxRounds: Int = 1,
    numCandidatePrograms: Int = 16,
    val numThreads: Int? = null,
    maxErrors: Int? = null,
    val stopAtScore: Double? = null,
    val metricThreshold: Double? = null,
) : Teleprompter() {
    val teacherSettings: Map<String, Any?> = teacherSettings ?: emptyMap()
    val minNumSamples: Int = 1
    val maxNumSamples: Int = maxBootstrappedDemos
    var maxErrors: Int? = maxErrors
    val numCandidateSets: Int = numCandidatePrograms
    val maxLabeledDemos: Int = maxLabeledDemos

    init {
        println("Going to sample between $minNumSamples and $maxNumSamples traces per predictor.")
        println("Will attempt to bootstrap $numCandidateSets candidate sets.")
    }

    private lateinit var trainset: List<Example>
    private lateinit var valset: List<Example>

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        this.trainset = trainset
        this.valset = valset ?: this.trainset

        val effectiveMaxErrors = maxErrors ?: Settings.maxErrors

        val scores = mutableListOf<Double>()
        val allSubscores = mutableListOf<List<Any?>>()
        val scoreData = mutableListOf<Map<String, Any>>()

        var bestProgram: Module = student.resetCopy()

        for (seed in -3 until numCandidateSets) {
            val trainsetCopy = trainset.toMutableList()

            val program = when (seed) {
                -3 -> {
                    // Zero-shot
                    student.resetCopy()
                }
                -2 -> {
                    // Labels only
                    val teleprompter = LabeledFewShot(k = maxLabeledDemos)
                    teleprompter.compile(student, trainset = trainsetCopy)
                }
                -1 -> {
                    // Unshuffled few-shot
                    val optimizer = BootstrapFewShot(
                        metric = this.metric,
                        metricThreshold = this.metricThreshold,
                        maxBootstrappedDemos = maxNumSamples,
                        maxLabeledDemos = maxLabeledDemos,
                        teacherSettings = teacherSettings,
                        maxRounds = maxRounds,
                        maxErrors = effectiveMaxErrors,
                    )
                    optimizer.compile(student, trainset = trainsetCopy, teacher = teacher, valset = null)
                }
                else -> {
                    require(seed >= 0) { "Seed must be >= 0, got $seed" }
                    // Shuffled few-shot
                    val rng = Random(seed)
                    val shuffledTrainset = trainsetCopy.shuffled(rng)
                    // Python's randint(a, b) is inclusive on both ends
                    // Kotlin's nextInt(a, b) is exclusive on upper end, so use maxNumSamples + 1
                    val size = rng.nextInt(minNumSamples, maxNumSamples + 1)

                    val optimizer = BootstrapFewShot(
                        metric = this.metric,
                        metricThreshold = this.metricThreshold,
                        maxBootstrappedDemos = size,
                        maxLabeledDemos = maxLabeledDemos,
                        teacherSettings = teacherSettings,
                        maxRounds = maxRounds,
                        maxErrors = effectiveMaxErrors,
                    )
                    optimizer.compile(student, trainset = shuffledTrainset, teacher = teacher, valset = null)
                }
            }

            val evaluator = Evaluate(
                devset = valset ?: emptyList(),
                metric = this.metric,
                numThreads = numThreads,
                maxErrors = effectiveMaxErrors,
                displayTable = false,
                displayProgress = true,
            )

            val result = evaluator.__call__(program)

            val score = result.score
            val subscores = result.results.map { it.third }

            allSubscores.add(subscores)

            if (scores.isEmpty() || score > (scores.maxOrNull() ?: 0.0)) {
                println("New best score: $score for seed $seed")
                bestProgram = program
            }

            scores.add(score)
            println("Scores so far: $scores")
            println("Best score so far: ${scores.maxOrNull() ?: 0.0}")

            scoreData.add(mapOf(
                "score" to score,
                "subscores" to subscores,
                "seed" to seed,
                "program" to program,
            ))

            if (stopAtScore != null && score >= stopAtScore) {
                println("Stopping early because score $score is >= stop_at_score $stopAtScore")
                break
            }
        }

        // Attach all program candidates in decreasing average score
        @Suppress("UNCHECKED_CAST")
        val candidatePrograms = scoreData.sortedByDescending { it["score"] as? Double ?: 0.0 }
        bestProgram = bestProgram
        // Store candidate programs on the best program
        println("${candidatePrograms.size} candidate programs found.")

        return bestProgram
    }
}
