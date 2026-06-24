package dspy.predict

import dspy.adapters.types.Tool
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.signatures.OutputField
import dspy.signatures.InputField

class ReAct(
    signature: String,
    toolsList: List<Tool>,
    maxIters: Int = 20,
) : Module() {
    private val react: Predict
    private val extract: ChainOfThought
    private val maxIters: Int = maxIters
    private val tools: MutableMap<String, Tool> = mutableMapOf()

    init {
        val sig = Signature.fromString(signature)

        // Build tools map
        for (tool in toolsList) {
            tools[tool.name ?: "unknown"] = tool
        }

        // Add finish tool
        tools["finish"] = Tool(
            func = { "Completed." },
            name = "finish",
            desc = "Marks the task as complete.",
        )

        // Build ReAct signature
        val inputs = sig.inputFields.joinToString(", ") { "${it.name}" }
        val outputs = sig.outputFields.joinToString(", ") { "${it.name}" }
        val instr = mutableListOf<String>().apply {
            if (sig.instruction.isNotEmpty()) {
                add(sig.instruction)
                add("")
            }
            add("You are an Agent. In each episode, you will be given the fields $inputs as input. And you can see your past trajectory so far.")
            add("Your goal is to use one or more of the supplied tools to collect any necessary information for producing $outputs.")
            add("To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.")
            add("After each tool call, you receive a resulting observation, which gets appended to your trajectory.")
            add("When writing next_thought, you may reason about the current situation and plan for future steps.")
            add("When selecting the next_tool_name and its next_tool_args, the tool must be one of:")
            for ((key, tool) in tools) {
                add("$key) $tool")
            }
            add("When providing next_tool_args, the value inside the field must be in JSON format")
        }

        val reactSig = Signature(
            instruction = instr.joinToString("\n"),
            inputFields = sig.inputFields + InputField(name = "trajectory", desc = ""),
            outputFields = listOf(
                OutputField(name = "next_thought", desc = ""),
                OutputField(name = "next_tool_name", desc = ""),
                OutputField(name = "next_tool_args", desc = ""),
            ),
        )

        val fallbackSig = Signature(
            instruction = sig.instruction,
            inputFields = sig.inputFields + InputField(name = "trajectory", desc = ""),
            outputFields = sig.outputFields,
        )

        this.react = Predict(sig = reactSig)
        this.extract = ChainOfThought(signature = fallbackSig)
    }

    suspend fun forward(example: Example): Prediction {
        val trajectory = mutableMapOf<String, Any?>()
        val maxIters = example.get("max_iters", maxIters) as Int

        for (idx in 0 until maxIters) {
            val pred = react.__call__(kwargs = example.toMap())

            trajectory["thought_${idx}"] = pred["next_thought"]
            trajectory["tool_name_${idx}"] = pred["next_tool_name"]
            trajectory["tool_args_${idx}"] = pred["next_tool_args"]

            try {
                trajectory["observation_${idx}"] = tools[pred["next_tool_name"] as String]?.invoke()
            } catch (e: Exception) {
                trajectory["observation_${idx}"] = "Execution error in ${pred["next_tool_name"]}: ${e.message}"
            }

            if (pred["next_tool_name"] == "finish") {
                break
            }
        }

        val extract = extract.forward(example)
        return Prediction(base = trajectory + extract.toMap())
    }
}
