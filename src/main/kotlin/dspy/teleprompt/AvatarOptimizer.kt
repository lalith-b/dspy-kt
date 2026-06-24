package dspy.teleprompt

import dspy.predict.Predict
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.utils.Settings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlin.random.Random
import kotlin.reflect.full.memberProperties

/**
 * AvatarOptimizer: An optimizer that uses feedback-based instruction evolution.
 *
 * Compares positive and negative execution results, generates feedback via a Comparator
 * predictor, then iteratively refines the actor's instruction.
 *
 * Faithful port of `dspy/teleprompt/avatar_optimizer.py`.
 */

private const val DEFAULT_MAX_EXAMPLES = 10

/**
 * Result of evaluating a single example.
 */
data class EvalResult(
    val example: Map<String, Any?>,
    val score: Double,
    val actions: List<Any>? = null,
)

/**
 * Comparator signature: contrasts positive and negative results to generate feedback.
 */
class ComparatorSignature : Signature(
    instruction = """After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:

Task:
(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.
(2) Then, review the computational logic for any inconsistencies in the previous actions.
(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs.""",
    inputFields = listOf(
        InputField(name = "instruction", desc = "Instruction for the actor to execute the task"),
        InputField(name = "actions", desc = "Actions actor can take to complete the task"),
        InputField(name = "pos_input_with_metrics", desc = "Positive inputs along with their score on a evaluation metric and actions taken"),
        InputField(name = "neg_input_with_metrics", desc = "Negative inputs along with their score on a evaluation metric and actions taken"),
    ),
    outputFields = listOf(
        OutputField(name = "feedback", desc = "Feedback for the actor to improve the performance of negative inputs"),
    ),
)

/**
 * Feedback-based instruction generation signature.
 */
class FeedbackBasedInstructionSignature : Signature(
    instruction = """There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, and the inputs that did not do well are negative inputs.

You received feedback on how they can better use the tools to improve your performance on the negative inputs. You have been provided with the previous instruction, that they followed to use tools to complete the task, and the feedback on your performance.

Your task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve their performance on the task.

Make sure that the new instruction talks about how to use the tools effectively and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction.""",
    inputFields = listOf(
        InputField(name = "previous_instruction", desc = "Previous instruction for the actor to execute the task"),
        InputField(name = "feedback", desc = "Feedback for the actor to improve the performance of negative inputs"),
    ),
    outputFields = listOf(
        OutputField(name = "new_instruction", desc = "New instruction for the actor to execute the task"),
    ),
)

/**
 * Avatar optimizer that refines actor instructions through feedback.
 */
class AvatarOptimizer(
    val metric: (Example, Prediction) -> Double,
    val maxIters: Int = 10,
    val lowerBound: Double = 0.0,
    val upperBound: Double = 1.0,
    maxPositiveInputs: Int? = null,
    maxNegativeInputs: Int? = null,
    val optimizeFor: String = "max",
) : Teleprompter() {
    val maxPositiveInputs: Int = maxPositiveInputs ?: DEFAULT_MAX_EXAMPLES
    val maxNegativeInputs: Int = maxNegativeInputs ?: DEFAULT_MAX_EXAMPLES

    private val comparator: Predict
    private val feedbackInstruction: Predict

    init {
        require(metric != null) { "`metric` argument cannot be None. Please provide a metric function." }
        comparator = Predict(sig = ComparatorSignature())
        feedbackInstruction = Predict(sig = FeedbackBasedInstructionSignature())
    }

    private suspend fun processExample(
        actor: Module,
        example: Example,
        returnOutputs: Boolean,
    ): Any {
        val actorCopy = actor.deepcopy()

        return try {
            val inputs = example.inputs().toMap()
            val prediction = actorCopy(inputs)
            val score = runMetric(example, prediction)

            if (returnOutputs) {
                Triple(example, prediction, score)
            } else {
                score
            }
        } catch (e: Exception) {
            println(e.message)
            if (returnOutputs) {
                Triple(example, null, 0.0)
            } else {
                0.0
            }
        }
    }

    private fun runMetric(example: Example, prediction: Prediction?): Double {
        return try {
            if (prediction != null) {
                metric(example, prediction)
            } else {
                0.0
            }
        } catch (e: Exception) {
            0.0
        }
    }

    /**
     * Thread-safe evaluator that processes a devset in parallel.
     */
    suspend fun threadSafeEvaluator(
        devset: List<Example>,
        actor: Module,
        returnOutputs: Boolean = false,
        numThreads: Int? = null,
    ): Any {
        val totalExamples = devset.size
        var totalScore = 0.0
        val results = mutableListOf<Triple<Example, Prediction?, Double>>()

        val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        val deferreds = devset.map { example ->
            scope.async {
                processExample(actor, example, returnOutputs)
            }
        }

        val outcomes = deferreds.awaitAll()

        for (result in outcomes) {
            if (returnOutputs) {
                val triple = result as Triple<Example, Prediction?, Double>
                val (ex, pred, score) = triple
                totalScore += score
                results.add(triple)
            } else {
                @Suppress("UNCHECKED_CAST")
                totalScore += (result as Double)
            }
        }

        val avgMetric = if (totalExamples > 0) totalScore / totalExamples else 0.0

        return if (returnOutputs) {
            Pair(avgMetric, results)
        } else {
            avgMetric
        }
    }

    /**
     * Evaluate and split results into positive and negative buckets.
     */
    private suspend fun getPosNegResults(
        actor: Module,
        trainset: List<Example>,
    ): Triple<Double, List<EvalResult>, List<EvalResult>> {
        val posInputs = mutableListOf<EvalResult>()
        val negInputs = mutableListOf<EvalResult>()

        val result = threadSafeEvaluator(trainset, actor, returnOutputs = true)
        val (avgScore, results) = result as Pair<Double, List<Triple<Example, Prediction?, Double>>>

        println("Average Score: $avgScore")

        for ((example, prediction, score) in results) {
            if (score >= upperBound) {
                posInputs.add(EvalResult(
                    example = example.inputs().toMap(),
                    score = score,
                    actions = extractActions(prediction),
                ))
            } else if (score <= lowerBound) {
                negInputs.add(EvalResult(
                    example = example.inputs().toMap(),
                    score = score,
                    actions = extractActions(prediction),
                ))
            }
        }

        if (posInputs.isEmpty()) {
            throw IllegalArgumentException(
                "No positive examples found, try lowering the upper_bound or providing more training data"
            )
        }
        if (negInputs.isEmpty()) {
            throw IllegalArgumentException(
                "No negative examples found, try raising the lower_bound or providing more training data"
            )
        }

        return Triple(avgScore, posInputs, negInputs)
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractActions(prediction: Prediction?): List<Any>? {
        if (prediction == null) return null
        return prediction["actions"] as? List<Any>
    }

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        val bestActor = student.deepcopy()
        val bestScore = if (optimizeFor == "max") -999.0 else 999.0

        var currentBestScore = bestScore
        var currentBestActor: Module = bestActor

        for (i in 0 until maxIters) {
            println("=".repeat(20))
            println("Iteration ${i + 1}/$maxIters")

            val (score, posInputs, negInputs) = getPosNegResults(currentBestActor, trainset)
            println("Positive examples: ${posInputs.size}")
            println("Negative examples: ${negInputs.size}")
            println("Sampling $maxPositiveInputs positive examples and $maxNegativeInputs negative examples")

            val sampledPos = if (posInputs.size > maxPositiveInputs) {
                posInputs.shuffled(Random(0)).take(maxPositiveInputs)
            } else {
                posInputs
            }

            val sampledNeg = if (negInputs.size > maxNegativeInputs) {
                negInputs.shuffled(Random(0)).take(maxNegativeInputs)
            } else {
                negInputs
            }

            // Get current instruction from the actor
            val currentInstruction = getCurrentInstruction(currentBestActor)
            val actionsList = getCurrentActions(currentBestActor)

            // Get feedback from comparator
            val feedbackResult = comparator(
                mapOf(
                    "instruction" to currentInstruction,
                    "actions" to actionsList,
                    "pos_input_with_metrics" to sampledPos,
                    "neg_input_with_metrics" to sampledNeg,
                )
            )
            val feedback = feedbackResult["feedback"] as? String ?: ""

            // Generate new instruction
            val newInstructionResult = feedbackInstruction(
                mapOf(
                    "previous_instruction" to currentInstruction,
                    "feedback" to feedback,
                )
            )
            val newInstruction = newInstructionResult["new_instruction"] as? String ?: currentInstruction

            println("Generated new instruction: $newInstruction")

            // Update best actor if score improved
            if ((optimizeFor == "max" && currentBestScore < score) ||
                (optimizeFor == "min" && currentBestScore > score)) {
                currentBestScore = score
                updateActorInstruction(bestActor, newInstruction)
                currentBestActor = bestActor
            }
        }

        println("Best Actor: $bestActor")

        return bestActor
    }

    private fun getCurrentInstruction(actor: Module): String {
        // Try to get instruction from the actor's predictors
        val predictors = actor.namedPredictors()
        if (predictors.isNotEmpty()) {
            return predictors.first().second.signature.instructions
        }
        return ""
    }

    private fun getCurrentActions(actor: Module): List<String> {
        // Try to extract tool/action strings from the module
        // In the original, this accesses best_actor.tools
        return try {
            val tools = actor::class.memberProperties
                .find { it.name == "tools" }
                ?.let { prop -> prop.getter.call(actor) } as? List<Any> ?: emptyList()
            tools.map { it.toString() }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun updateActorInstruction(actor: Module, newInstruction: String) {
        // Update the instruction on the actor's predictor(s)
        for ((name, predictor) in actor.namedPredictors()) {
            predictor.signature = predictor.signature.withInstructions(newInstruction)
        }
    }
}
