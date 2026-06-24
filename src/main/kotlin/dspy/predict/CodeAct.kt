package dspy.predict

import dspy.adapters.types.Tool
import dspy.primitives.CodeInterpreter
import dspy.primitives.Example
import dspy.primitives.FinalOutput
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.signatures.SignatureUtils
import org.slf4j.LoggerFactory

private val logger = LoggerFactory.getLogger(CodeAct::class.java)

/**
 * CodeAct is a module that utilizes the Code Interpreter and predefined tools to solve the problem.
 *
 * Port of `dspy/predict/code_act.py`
 *
 * Example:
 * ```kotlin
 * fun factorial(n: Int): Int = if (n == 1) 1 else n * factorial(n - 1)
 * val act = CodeAct("n -> factorial", toolsList = listOf(Tool({ factorial(5) }, name = "factorial")), interpreter = myInterpreter)
 * val result = act.invoke(mapOf("n" to 5))
 * ```
 */
class CodeAct(
    signature: Any,
    toolsList: List<Tool>,
    private val maxIters: Int = 5,
    interpreter: CodeInterpreter? = null,
) : Module() {
    private val sig: Signature
    private val codeactTools: MutableMap<String, Tool> = mutableMapOf()
    private val codeact: Predict
    private val extractor: ChainOfThought
    private val codeInterpreter: CodeInterpreter

    init {
        sig = SignatureUtils.ensureSignature(signature)

        // Register tools
        for (tool in toolsList) {
            codeactTools[tool.name ?: "unknown"] = tool
        }

        val instructions = buildInstructions(sig, codeactTools)

        val codeactSignature = Signature(
            instruction = instructions.joinToString("\n"),
            inputFields = sig.inputFields + InputField(name = "trajectory", desc = ""),
            outputFields = listOf(
                OutputField(name = "generated_code", desc = "Python code that when executed, produces output relevant to answering the question"),
                OutputField(name = "finished", desc = "a boolean flag to determine if the process is done"),
            ),
        )

        val extractInputFields = mutableListOf<InputField>()
        extractInputFields.addAll(sig.inputFields)
        for (field in sig.outputFields) {
            // Output fields from original signature are used as inputs for extraction
            extractInputFields.add(InputField(name = field.name, desc = field.desc))
        }
        extractInputFields.add(InputField(name = "trajectory", desc = ""))

        val extractSignature = Signature(
            instruction = sig.instruction,
            inputFields = extractInputFields,
            outputFields = sig.outputFields,
        )

        this.codeact = Predict(sig = codeactSignature)
        this.extractor = ChainOfThought(signature = extractSignature)

        codeInterpreter = interpreter ?: throw IllegalArgumentException(
            "CodeInterpreter is required. Please provide one or ensure deno is available."
        )
    }

    private fun buildInstructions(
        signature: Signature,
        tools: Map<String, Tool>,
    ): List<String> {
        val instructions = mutableListOf<String>()

        if (signature.instruction.isNotEmpty()) {
            instructions.add("${signature.instruction}\n")
        }

        val inputs = signature.inputFields.joinToString(", ") { "`${it.name}`" }
        val outputs = signature.outputFields.joinToString(", ") { "`${it.name}`" }

        instructions.addAll(listOf(
            "You are an intelligent agent. For each episode, you will receive the fields $inputs as input.",
            "Your goal is to generate executable Python code that collects any necessary information for producing $outputs.",
            "For each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.",
            "Ensure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.",
            "When all information for producing the outputs ($outputs) are available to be extracted, mark `finished=True` besides the final Python code.",
            "You have access to the Python Standard Library and the following functions:",
        ))

        tools.entries.withIndex().forEach { (idx, entry) ->
            instructions.add("${idx + 1}) ${entry.value}")
        }

        return instructions
    }

    private fun parseCode(codeData: Map<String, Any?>): Pair<String, String?> {
        val code = codeData["generated_code"]?.toString()?.split("---", limit = 2)?.get(0)
            ?.split("\n\n\n", limit = 2)?.get(0) ?: ""

        val codeBlockMatch = Regex("```python[ \\n](.*?)[ \\n]```?", RegexOption.DOT_MATCHES_ALL).find(code)
        val codeBlock = codeBlockMatch?.groups?.get(1)?.value ?: code

        if (codeBlock.isBlank()) {
            return code to "Error: Empty code after parsing."
        }

        if ("\n" !in codeBlock && codeBlock.count { it == '=' } > 1) {
            return code to "Error: Code format is not correct."
        }

        val lines = codeBlock.split("\n")
        val lastLineMatch = Regex("^(\\w+)\\s*=").find(lines.last().trim())
        if (lastLineMatch != null && lines.size > 1) {
            return (codeBlock + "\n" + lastLineMatch.groupValues[1]) to null
        }

        return codeBlock to null
    }

    private fun executeCode(code: String): Pair<String?, String?> {
        if (code.isBlank()) {
            return null to "Error: Empty code before execution."
        }

        return run<Pair<String?, String?>> {
            try {
                val result: Any? = codeInterpreter.execute(code)
                val output: Any? = when (result) {
                    is FinalOutput -> result.output
                    else -> result
                }
                val escaped = (output?.toString() ?: "").replace("\\", "\\\\").replace("\"", "\\\"")
                val jsonStr: String = "{\"result\": \"$escaped\"}"
                jsonStr to null
            } catch (e: Exception) {
                null to (e.message ?: e.toString())
            }
        }
    }

    suspend fun forward(input: Map<String, Any?>): Prediction {
        val trajectory = mutableMapOf<String, Any?>()
        var currentMaxIters = input["max_iters"] as? Int ?: maxIters

        for (idx in 0 until currentMaxIters) {
            val codeData = codeact.forward((trajectory.toMutableMap() + input).toMutableMap())
            var code: String
            var error: String?

            val parseResult = parseCode(codeData.toMap())
            code = parseResult.first
            error = parseResult.second

            if (error != null) {
                trajectory["observation_${idx}"] = "Failed to parse the generated code: $error"
                continue
            }

            trajectory["generated_code_${idx}"] = code

            val execResult = executeCode(code)
            val output = execResult.first
            error = execResult.second

            if (error == null) {
                trajectory["code_output_${idx}"] = output
            } else {
                trajectory["observation_${idx}"] = "Failed to execute the generated code: $error"
            }

            if (codeData["finished"] as? Boolean == true) {
                break
            }
        }

        val extractResult = callWithPotentialTrajectoryTruncation(extractor, trajectory, input)
        codeInterpreter.shutdown()
        return Prediction(base = (trajectory + extractResult.toMap()))
    }

    /**
     * Helper method to call a module with potential trajectory truncation.
     * If the trajectory is too large, it gets truncated to fit within context limits.
     */
    private suspend fun callWithPotentialTrajectoryTruncation(
        module: ChainOfThought,
        trajectory: Map<String, Any?>,
        input: Map<String, Any?>,
    ): Prediction {
        // In Python, the trajectory is serialized as JSON and truncated if too long
        val trajectoryStr = buildSimpleJsonObject(trajectory)

        val example = Example(
            base = (input + mapOf("trajectory" to trajectoryStr)).toMutableMap(),
        )

        return module.forward(example)
    }

    private fun buildSimpleJsonObject(map: Map<String, Any?>): String {
        val entries = map.map { (k, v) ->
            val escapedK = k.replace("\\", "\\\\").replace("\"", "\\\"")
            val escapedV = (v?.toString() ?: "").replace("\\", "\\\\").replace("\"", "\\\"")
            "\"$escapedK\": \"$escapedV\""
        }
        return "{${entries.joinToString(", ")}}"
    }

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction = forward(kwargs)

    override fun namedParameters(): List<Pair<String, dspy.primitives.Parameter>> {
        return listOf(
            "codeact" to codeact,
            "extractor" to extractor,
        )
    }

    override fun toString(): String = "CodeAct($sig, tools=${codeactTools.keys})"
}
