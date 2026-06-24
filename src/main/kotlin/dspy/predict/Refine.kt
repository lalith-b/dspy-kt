package dspy.predict

import dspy.adapters.Adapter
import dspy.adapters.ChatAdapter
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.utils.Settings

/**
 * Signature for the OfferFeedback prediction.
 *
 * Used by Refine to generate feedback for modules that contributed to poor rewards.
 *
 * Port of `dspy/predict/refine.py` OfferFeedback signature
 */
fun offerFeedbackSignature(): Signature {
    return Signature(
        instruction = """
            In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
            any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
            it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
            The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
            to avoid the same mistake on the same or similar inputs.
        """.trimIndent(),
        inputFields = listOf(
            InputField(name = "program_code", desc = "The code of the program that we are analyzing"),
            InputField(name = "modules_defn", desc = "The definition of each module in the program, including its I/O"),
            InputField(name = "program_inputs", desc = "The inputs to the program that we are analyzing"),
            InputField(name = "program_trajectory", desc = "The trajectory of the program's execution, showing each module's I/O"),
            InputField(name = "program_outputs", desc = "The outputs of the program that we are analyzing"),
            InputField(name = "reward_code", desc = "The code of the reward function that we are analyzing"),
            InputField(name = "target_threshold", desc = "The target threshold for the reward function"),
            InputField(name = "reward_value", desc = "The reward value assigned to the program's outputs"),
            InputField(name = "module_names", desc = "The names of the modules in the program, for which we seek advice"),
        ),
        outputFields = listOf(
            OutputField(name = "discussion", desc = "Discussing blame of where each module went wrong, if it did"),
            OutputField(
                name = "advice",
                desc = "For each module, describe very concretely, in this order: the specific scenarios in which it has made " +
                    "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of " +
                    "scenario in the future. If the module is not to blame, write N/A."
            ),
        ),
    )
}

/**
 * Refines a module by running it up to N times with different rollout IDs at temperature=1.0
 * and returns the best prediction.
 *
 * This module runs the provided module multiple times with varying rollout identifiers and selects
 * either the first prediction that exceeds the specified threshold or the one with the highest reward.
 * If no prediction meets the threshold, it automatically generates feedback to improve future predictions.
 *
 * Port of `dspy/predict/refine.py`
 */
class Refine(
    private val module: Module,
    private val N: Int,
    private val rewardFn: (Map<String, Any?>, Prediction) -> Double,
    private val threshold: Double,
    failCount: Int? = null,
) : Module() {
    private var remainingFailCount: Int
    private val moduleCode: String
    private val rewardFnCode: String

    init {
        this.remainingFailCount = failCount ?: N
        // Capture source code using reflection (Kotlin equivalent of inspect.getsource)
        moduleCode = captureSourceCode(module)
        rewardFnCode = "<reward_function>"
    }

    /**
     * Forward pass: run the module up to N times with feedback-based refinement.
     */
    suspend fun forward(kwargs: Map<String, Any?>): Prediction? {
        val lm = module.getLm() ?: Settings.lm()
            ?: throw IllegalStateException("No LM configured. Call dspy.configure(lm=...) first.")

        val start = (lm.kwargs["rollout_id"] as? Int) ?: 0
        val rolloutIds = (start until start + N).toList()
        var bestPred: Prediction? = null
        var bestTrace: List<Triple<Any, Map<String, Any?>, Map<String, Any?>>>? = null
        var bestReward = -Double.MAX_VALUE
        var advice: Map<String, String>? = null
        var remainingFails = remainingFailCount

        for ((idx, rid) in rolloutIds.withIndex()) {
            val lmCopy = lm.copy("rollout_id" to rid, "temperature" to 1.0)
            val mod = module.deepcopy()
            mod.setLm(lmCopy)

            val predictor2Name = mod.namedPredictors().associate { (name, predictor) -> predictor to name }
            val signature2Name = mod.namedPredictors().associate { (name, predictor) -> predictor.signature to name }
            val moduleNames = mod.namedPredictors().map { it.first }

            var reward: Double? = null
            var pred: Prediction? = null
            var trace: List<Triple<Any, Map<String, Any?>, Map<String, Any?>>>? = null

            try {
                val savedTrace = Settings.trace
                Settings.trace = mutableListOf()
                try {
                    if (advice == null) {
                        pred = mod.invoke(kwargs = kwargs)
                    } else {
                        // Create a wrapper that injects the advice as a hint
                        val advisor = AdviceInjectorAdapter(Settings.adapter() ?: ChatAdapter(), advice, signature2Name)
                        Settings.setAdapter(advisor)
                        pred = mod.invoke(kwargs = kwargs)
                        Settings.setAdapter(null)
                    }
                    trace = Settings.trace?.toList()

                    // NOTE: Not including the trace of rewardFn
                    reward = rewardFn(kwargs, pred!!)
                } finally {
                    Settings.trace = savedTrace
                }
            } catch (e: Exception) {
                println("Refine: Attempt failed with rollout id $rid: ${e.message}")
                if (idx >= remainingFails) {
                    throw e
                }
                remainingFails--
                continue
            }

            if (reward != null && reward > bestReward) {
                bestReward = reward
                bestPred = pred
                bestTrace = trace
            }

            if (threshold != 0.0 && reward != null && reward >= threshold) {
                break
            }

            if (idx == N - 1) {
                break
            }

            // Generate advice for the next iteration
            val modulesDefn = inspectModules(mod)
            val trajectory = trace?.map { entry ->
                mapOf<String, Any?>(
                    "module_name" to (predictor2Name[entry.first] ?: "unknown"),
                    "inputs" to entry.second,
                    "outputs" to entry.third,
                )
            }

            // Build the advice request
            val adviseKwargs = mutableMapOf<String, Any?>().apply {
                put("program_code", moduleCode)
                put("modules_defn", modulesDefn)
                put("program_inputs", serializeToJson(kwargs))
                put("program_trajectory", serializeToJson(trajectory))
                put("program_outputs", serializeToJson(pred?.toDict() ?: emptyMap<String, Any?>()))
                put("reward_code", rewardFnCode)
                put("target_threshold", threshold)
                put("reward_value", reward!!)
                put("module_names", moduleNames)
            }

            // Generate advice
            advice = try {
                val offerSig = offerFeedbackSignature()
                val advicePred = Predict(sig = offerSig).invoke(kwargs = adviseKwargs)
                @Suppress("UNCHECKED_CAST")
                advicePred["advice"] as? Map<String, String>
            } catch (_: Exception) {
                null
            }
        }

        if (bestTrace != null) {
            Settings.trace?.addAll(bestTrace)
        }

        return bestPred
    }

    /**
     * Inspect the modules in a program and return a formatted string description.
     */
    private fun inspectModules(program: Module): String {
        val separator = "-".repeat(80)
        val output = mutableListOf(separator)

        for ((name, predictor) in program.namedPredictors()) {
            val sig = predictor.signature
            val instructions = sig.instructions
            val formattedInstructions = instructions.lines()
                .joinToString("\n") { "    $it" }

            output.add("Module $name")
            output.add("\n\tInput Fields:")
            output.add(sig.inputFields.joinToString("\n") { "    ${it.name}: ${it.desc}" })
            output.add("\tOutput Fields:")
            output.add(sig.outputFields.joinToString("\n") { "    ${it.name}: ${it.desc}" })
            output.add("\tOriginal Instructions: $formattedInstructions")
            output.add(separator)
        }

        return output.joinToString("\n") { it.trim() }
    }

    /**
     * Capture source code of a module using reflection.
     * This is a stub since Kotlin doesn't have inspect.getsource.
     */
    private fun captureSourceCode(module: Module): String {
        return "class ${module::class.simpleName} : Module()\n// Source code not available in Kotlin reflection"
    }

    /**
     * Serialize an object to JSON string.
     */
    private fun serializeToJson(obj: Any?): String {
        return recursiveMask(obj).toString()
    }

    /**
     * Recursively mask non-serializable objects.
     */
    private fun recursiveMask(o: Any?): Any? {
        return when (o) {
            null -> null
            is String, is Number, is Boolean -> o
            is Map<*, *> -> o.mapKeys { it.key.toString() }.mapValues { recursiveMask(it.value) }
            is List<*> -> o.map { recursiveMask(it) }
            is Array<*> -> o.map { recursiveMask(it) }
            is Triple<*, *, *> -> mapOf("first" to recursiveMask(o.first), "second" to recursiveMask(o.second), "third" to recursiveMask(o.third))
            else -> "<non-serializable: ${o::class.simpleName}>"
        }
    }

    override suspend operator fun invoke(kwargs: Map<String, Any?>): Prediction {
        return forward(kwargs) ?: throw IllegalStateException("Refine returned no prediction after $N attempts")
    }

    override fun deepcopy(): Module {
        return Refine(
            module = module.deepcopy(),
            N = N,
            rewardFn = rewardFn,
            threshold = threshold,
            failCount = remainingFailCount,
        )
    }
}

/**
 * Adapter wrapper that injects advice as hints.
 * Wraps a base adapter to inject advice from previous runs.
 */
class AdviceInjectorAdapter(
    private val baseAdapter: Adapter,
    private val advice: Map<String, String>,
    private val signature2Name: Map<Signature, String>,
) : Adapter(
    callbacks = baseAdapter.callbacks,
    useNativeFunctionCalling = baseAdapter.useNativeFunctionCalling,
    nativeResponseTypes = baseAdapter.nativeResponseTypes,
    parallelToolCalls = baseAdapter.parallelToolCalls,
) {
    override operator fun invoke(
        lm: dspy.clients.BaseLM,
        lmKwargs: MutableMap<String, Any?>,
        signature: Signature,
        demos: List<Map<String, Any?>>,
        inputs: Map<String, Any?>,
    ): List<Map<String, Any?>> {
        val name = signature2Name[signature]
        val hint = advice[name] ?: "N/A"
        val modifiedInputs = inputs.toMutableMap().apply {
            put("hint_", hint)
        }
        val modifiedSignature = signature.append(
            "hint_",
            InputField(desc = "A hint to the module from an earlier run")
        )
        return baseAdapter.invoke(lm, lmKwargs, modifiedSignature, demos, modifiedInputs)
    }

    override fun formatFieldDescription(signature: Signature): String {
        return baseAdapter.formatFieldDescription(signature)
    }

    override fun formatFieldStructure(signature: Signature): String {
        return baseAdapter.formatFieldStructure(signature)
    }

    override fun formatUserMessageContent(
        signature: Signature,
        inputs: Map<String, Any?>,
        prefix: String,
        suffix: String,
        mainRequest: Boolean,
    ): String {
        return baseAdapter.formatUserMessageContent(signature, inputs, prefix, suffix, mainRequest)
    }

    override fun formatAssistantMessageContent(
        signature: Signature,
        outputs: Map<String, Any?>,
        missingFieldMessage: String?,
    ): String {
        return baseAdapter.formatAssistantMessageContent(signature, outputs, missingFieldMessage)
    }

    override fun parse(signature: Signature, completion: String): Map<String, Any?> {
        return baseAdapter.parse(signature, completion)
    }
}
