package dspy.teleprompt

import dspy.clients.BaseLM
import dspy.predict.Predict
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.utils.Settings
import kotlin.math.exp

/**
 * Utility functions for SIMBA optimization.
 *
 * Faithful port of `dspy/teleprompt/simba_utils.py`.
 */

/**
 * OfferFeedback signature for generating module advice.
 */
class OfferFeedbackSignature : Signature(
    instruction = """
You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
similar inputs in the future.

The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

In your advice:
- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
- Ensure each unique module name appears exactly once as a key in the advice dictionary.
    """.trimIndent(),
    inputFields = listOf(
        InputField(name = "program_code", desc = "The code of the program that we are analyzing"),
        InputField(name = "modules_defn", desc = "The definition of each module in the program, including its I/O"),
        InputField(name = "program_inputs", desc = "The inputs to the program that we are analyzing"),
        InputField(name = "oracle_metadata", desc = "Any (hidden) metadata about the training set instance we're analyzing"),
        InputField(name = "worse_program_trajectory", desc = "The trajectory of the program's execution, showing each module's I/O"),
        InputField(name = "worse_program_outputs", desc = "The outputs of the program that we are analyzing"),
        InputField(name = "worse_reward_value", desc = "The reward value assigned to the program's outputs"),
        InputField(name = "worse_reward_info", desc = "Additional information that might be helpful to understanding the assigned reward value."),
        InputField(name = "better_program_trajectory", desc = "The trajectory of the program's execution, showing each module's I/O"),
        InputField(name = "better_program_outputs", desc = "The outputs of the program that we are analyzing"),
        InputField(name = "better_reward_value", desc = "The reward value assigned to the program's outputs"),
        InputField(name = "better_reward_info", desc = "Additional information that might be helpful to understanding the assigned reward value."),
        InputField(name = "module_names", desc = "The names of the modules in the program, for which we seek advice"),
    ),
    outputFields = listOf(
        OutputField(name = "discussion", desc = "Discussing blame of where each module went wrong, if it did"),
        OutputField(
            name = "module_advice",
            desc = """For each module, describe very concretely: If the module receives \$\{description of input or patterns
therein}, then it should \$\{description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act
like the successful trajectory rather than the lower-scoring trajectory."""
        ),
    ),
)

/**
 * Prepare distinct LMs for resampling from the baseline program.
 */
fun prepareModelsForResampling(
    program: Module,
    n: Int,
    teacherSettings: Map<String, Any?>? = null,
): List<BaseLM> {
    val lm = program.getLm() ?: Settings.lm()
        ?: throw IllegalStateException("No LM available for resampling")

    val startRolloutId = (lm.kwargs["rollout_id"] as? Int) ?: 0
    val rolloutIds = (startRolloutId until startRolloutId + n).toList()

    val models = mutableListOf<BaseLM>()
    var startRolloutIdx = 0

    // If we have a teacher model, use this as the first model
    if (teacherSettings != null) {
        val teacherLm = (teacherSettings["lm"] as? BaseLM) ?: lm
        teacherLm.kwargs["rollout_id"] = rolloutIds[startRolloutIdx]
        models.add(teacherLm)
        startRolloutIdx++
    }

    // The rest of the models are just copies of the base model
    for (i in startRolloutIdx until rolloutIds.size) {
        models.add(lm.copy("rollout_id" to rolloutIds[i], "temperature" to 1.0))
    }

    return models
}

/**
 * Wrap a program to capture trace, prediction, and score.
 */
fun wrapProgram(
    program: Module,
    metric: (Example, Prediction?) -> Any?,
): (Example) -> Map<String, Any?> {
    return { example ->
        var prediction: Prediction? = null
        var trace: List<Any>? = null
        var score: Double = 0.0
        val outputMetadata = mutableMapOf<String, Any?>()

        Settings.context(trace = mutableListOf()) {
            try {
                prediction = kotlinx.coroutines.runBlocking { program(example.inputs().toMap()) }
            } catch (e: Exception) {
                // Log warning
            }
            trace = Settings.trace?.toList()
        }

        try {
            val output = metric(example, prediction)
            score = when (output) {
                is Number -> output.toDouble()
                is Prediction -> {
                    if (!output.containsKey("score")) {
                        throw IllegalArgumentException(
                            "When `metric` returns a `Prediction`, it must contain a `score` field."
                        )
                    }
                    val s = output["score"] as Number
                    // Extract fields from the output Prediction, excluding `score`
                    for ((k, v) in output.items()) {
                        if (k != "score") outputMetadata[k] = v
                    }
                    s.toDouble()
                }
                else -> 0.0
            }
        } catch (e: Exception) {
            // Log warning
        }

        mapOf<String, Any?>(
            "prediction" to prediction,
            "trace" to trace,
            "score" to score,
            "example" to example,
            "output_metadata" to outputMetadata,
        )
    }
}

/**
 * Strategy function type for SIMBA.
 */
typealias SimbaStrategy = (
    bucket: List<Map<String, Any?>>,
    system: Module,
    predictor2name: Map<Int, String>,
    name2predictor: Map<String, Predict>,
    batch10pScore: Double,
    batch90pScore: Double,
    promptModel: BaseLM?,
) -> Boolean

/**
 * Create an append-a-demo strategy.
 */
fun appendADemo(demoInputFieldMaxlen: Int): SimbaStrategy {
    return { bucket, system, predictor2name, name2predictor, batch10pScore, batch90pScore, promptModel ->
        val good = bucket[0]

        val goodScore = (good["score"] as? Double) ?: 0.0
        if (goodScore <= batch10pScore) {
            println("Skipping appending a demo as good score ${good["score"]} is at or below the 10th percentile.")
            false
        } else {
            val trace = good["trace"] as? List<Triple<*, *, *>> ?: emptyList()
            val name2demo = mutableMapOf<String, Map<String, Any?>>()

            for (step in trace) {
                @Suppress("UNCHECKED_CAST")
                val triple = step as Triple<Any, Map<String, Any?>, Map<String, Any?>>
                val predictor = triple.first
                val inputs = triple.second.toMutableMap()
                val outputs = triple.third

                // Truncate long inputs
                for ((k, v) in inputs.toMap()) {
                    if (demoInputFieldMaxlen > 0 && str(v).length > demoInputFieldMaxlen) {
                        inputs[k] = "${str(v).take(demoInputFieldMaxlen)}\n\t\t... <TRUNCATED FOR BREVITY>"
                    }
                }

                val demo = mutableMapOf<String, Any?>()
                demo.putAll(inputs)
                demo.putAll(outputs)
                demo["augmented"] = true

                val name = predictor2name[System.identityHashCode(predictor)] ?: continue
                name2demo[name] = demo
            }

            for ((name, demo) in name2demo) {
                val predictor = name2predictor[name] ?: continue
                predictor.demos.add(demo)
            }

            println("Added ${name2demo.size} demos (one each) across all predictors.")
            true
        }
    }
}

/**
 * Append-a-rule strategy: generates rules from contrasting good and bad trajectories.
 */
fun appendARule(
    bucket: List<Map<String, Any?>>,
    system: Module,
    predictor2name: Map<Int, String>,
    name2predictor: Map<String, Predict>,
    batch10pScore: Double,
    batch90pScore: Double,
    promptModel: BaseLM?,
): Boolean {
    val good = bucket[0]
    val bad = bucket.last()

    val goodScore = (good["score"] as? Double) ?: 0.0
    val badScore = (bad["score"] as? Double) ?: 0.0

    if (goodScore <= batch10pScore || badScore >= batch90pScore) {
        println("Skipping rule generation as good score ${good["score"]} is at or below the 10th percentile " +
            "or bad score ${bad["score"]} is at or above the 90th percentile.")
        return false
    }

    val example = good["example"] as? Example ?: return false
    val moduleNames = system.namedPredictors().map { it.first }

    // Handle edge case where scores are equal
    val goodCopy = good.toMutableMap()
    val badCopy = bad.toMutableMap()

    if (goodScore <= badScore) {
        if (goodScore > batch90pScore) {
            badCopy["trace"] = emptyList<Any>()
            badCopy["score"] = "N/A"
            badCopy["prediction"] = mapOf("N/A" to "Prediction not available")
        } else {
            goodCopy["trace"] = emptyList<Any>()
            goodCopy["score"] = "N/A"
            goodCopy["prediction"] = mapOf("N/A" to "Prediction not available")
        }
    }

    val betterTrajectory = formatTrajectory(goodCopy["trace"] as? List<Triple<*, *, *>>, predictor2name)
    val worseTrajectory = formatTrajectory(badCopy["trace"] as? List<Triple<*, *, *>>, predictor2name)

    val kwargs = mapOf<String, Any?>(
        "program_code" to "",
        "modules_defn" to inspectModules(system),
        "program_inputs" to example.inputs().toMap(),
        "oracle_metadata" to example.labels().toMap(),
        "better_program_trajectory" to betterTrajectory,
        "better_program_outputs" to safeStringMap(good["prediction"]),
        "worse_program_trajectory" to worseTrajectory,
        "worse_program_outputs" to safeStringMap(bad["prediction"]),
        "worse_reward_value" to bad["score"],
        "better_reward_value" to good["score"],
        "worse_reward_info" to safeStringMap(bad["output_metadata"]),
        "better_reward_info" to safeStringMap(good["output_metadata"]),
        "module_names" to moduleNames,
    )

    // Convert non-string values to JSON-like strings
    val stringifiedKwargs: Map<String, Any?> = kwargs.entries.associate { (k, v) ->
        k to (if (v is String) v else toJsonLike(v))
    }

    val lmToUse = promptModel ?: Settings.lm()
    if (lmToUse != null) {
        Settings.context(trace = mutableListOf(), lm = lmToUse) {
            try {
                val adviceProgram = Predict(sig = OfferFeedbackSignature())
                val adviceResult = kotlinx.coroutines.runBlocking { adviceProgram(stringifiedKwargs) }

                @Suppress("UNCHECKED_CAST")
                val advice = adviceResult["module_advice"] as? Map<String, String> ?: return@context

                for ((name, predictor) in system.namedPredictors()) {
                    if (name in advice) {
                        println("Advice for $name: ${advice[name]}")
                        val newInstructions = predictor.signature.instructions + "\n\n" + advice[name]
                        predictor.signature = predictor.signature.withInstructions(newInstructions)
                    }
                }
            } catch (e: Exception) {
                println("Advice generation failed: ${e.message}")
            }
        }
    }

    return true
}

/**
 * Format a trajectory for display.
 */
private fun formatTrajectory(
    trace: List<Triple<*, *, *>>?,
    predictor2name: Map<Int, String>,
): String {
    if (trace.isNullOrEmpty()) return "[]"
    val trajectoryList = trace.map { step ->
        @Suppress("UNCHECKED_CAST")
        val triple = step as Triple<Any, Map<String, Any?>, Map<String, Any?>>
        val moduleName = predictor2name[System.identityHashCode(triple.first)] ?: "unknown"
        mapOf<String, Any?>(
            "module_name" to moduleName,
            "inputs" to triple.second,
            "outputs" to triple.third,
        )
    }
    return toJsonLike(trajectoryList)
}

/**
 * Inspect modules for their signatures.
 */
fun inspectModules(program: Module): String {
    val separator = "-".repeat(80)
    val output = mutableListOf(separator)

    for ((name, predictor) in program.namedPredictors()) {
        val sig = predictor.signature
        val instructions = sig.instructions.replace("\n", "\n\t\t")

        output.add("Module $name")
        output.add("\n\tInput Fields:")
        output.add("\t\t" + sig.inputFields.joinToString("\n\t\t") { f ->
            "${f.name}: ${f.desc}"
        })
        output.add("\tOutput Fields:")
        output.add("\t\t" + sig.outputFields.joinToString("\n\t\t") { f ->
            "${f.name}: ${f.desc}"
        })
        output.add("\tOriginal Instructions: $instructions")
        output.add(separator)
    }

    return output.joinToString("\n").replace(Regex("^\\n"), "")
}

/**
 * Safely convert any value to a Map<String, Any?>.
 */
@Suppress("UNCHECKED_CAST")
fun safeStringMap(value: Any?): Map<String, Any?> {
    return (value as? Map<*, *>)?.mapKeys { it.key.toString() }?.mapValues { it.value } ?: emptyMap()
}

/**
 * Convert an object to a JSON-like string representation.
 */
fun toJsonLike(o: Any?): String {
    return when (o) {
        null -> "null"
        is String -> o
        is Number -> o.toString()
        is Boolean -> o.toString()
        is Map<*, *> -> {
            "{\n" + o.entries.map { (k, v) ->
                "  \"${k.toString().replace("\"", "\\\"")}\": ${toJsonLike(v)}"
            }.joinToString(",\n") + "\n}"
        }
        is List<*> -> {
            "[" + o.joinToString(", ") { toJsonLike(it) } + "]"
        }
        is Set<*> -> {
            "[" + o.joinToString(", ") { toJsonLike(it) } + "]"
        }
        else -> "<non-serializable: ${o::class.simpleName}>"
    }
}

/**
 * Recursively mask non-serializable objects.
 */
fun recursiveMask(o: Any?): Any? {
    return when (o) {
        null -> null
        is String -> o
        is Number -> o
        is Boolean -> o
        is Map<*, *> -> o.mapKeys { it.key.toString() }.mapValues { recursiveMask(it.value) }
        is List<*> -> o.map { recursiveMask(it) }
        is Set<*> -> o.map { recursiveMask(it) }
        else -> "<non-serializable: ${o::class.simpleName}>"
    }
}

private fun str(v: Any?): String = v?.toString() ?: ""

/**
 * Compute softmax weights for a list of scores.
 */
fun softmax(scores: List<Double>, temperature: Double): List<Double> {
    if (scores.isEmpty()) return emptyList()

    val adjusted = scores.map { s -> exp(s / temperature) }
    val sum = adjusted.sum()
    if (sum <= 0.0) {
        // Uniform fallback
        return List(scores.size) { 1.0 / scores.size }
    }
    return adjusted.map { it / sum }
}

/**
 * Sample from a weighted distribution.
 */
fun weightedSample(rng: kotlin.random.Random, items: List<Any>, weights: List<Double>): Any {
    if (items.isEmpty()) throw IllegalArgumentException("No items to sample")
    val r = rng.nextDouble()
    var cumulative = 0.0
    for (i in items.indices) {
        cumulative += weights[i]
        if (r < cumulative) {
            return items[i]
        }
    }
    return items.last()
}
