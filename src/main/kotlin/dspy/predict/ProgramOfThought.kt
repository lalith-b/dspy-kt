package dspy.predict

import dspy.primitives.CodeInterpreter
import dspy.primitives.FinalOutput
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.signatures.SignatureUtils
import org.slf4j.LoggerFactory

private val logger = LoggerFactory.getLogger(ProgramOfThought::class.java)

/**
 * A DSPy module that runs Python programs to solve a problem.
 * This module requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
 *
 * Port of `dspy/predict/program_of_thought.py`
 *
 * Example:
 * ```kotlin
 * val pot = ProgramOfThought("question -> answer", interpreter = myInterpreter)
 * val result = pot.invoke(mapOf("question" to "what is 1+1?"))
 * ```
 */
class ProgramOfThought(
    signature: Any,
    private val maxIters: Int = 3,
    interpreter: CodeInterpreter? = null,
) : Module() {
    private val sig: Signature
    private val inputFields: List<InputField>
    private val outputFields: List<OutputField>
    private val codeGenerate: ChainOfThought
    private val codeRegenerate: ChainOfThought
    private val generateOutput: ChainOfThought
    private val codeInterpreter: CodeInterpreter

    init {
        sig = SignatureUtils.ensureSignature(signature)
        inputFields = sig.inputFields
        outputFields = sig.outputFields

        codeGenerate = ChainOfThought(
            Signature(
                instruction = generateInstruction("generate"),
                inputFields = generateSignature("generate").inputFields,
                outputFields = generateSignature("generate").outputFields,
            ),
        )
        codeRegenerate = ChainOfThought(
            Signature(
                instruction = generateInstruction("regenerate"),
                inputFields = generateSignature("regenerate").inputFields,
                outputFields = generateSignature("regenerate").outputFields,
            ),
        )
        generateOutput = ChainOfThought(
            Signature(
                instruction = generateInstruction("answer"),
                inputFields = generateSignature("answer").inputFields,
                outputFields = generateSignature("answer").outputFields,
            ),
        )

        codeInterpreter = interpreter ?: throw IllegalArgumentException(
            "CodeInterpreter is required. Please provide one or ensure deno is available."
        )
    }

    private fun generateSignature(mode: String): Signature {
        val inputFieldsList = inputFields.toMutableList()
        val outputFieldsList = mutableListOf<OutputField>()

        when (mode) {
            "generate" -> {
                outputFieldsList.add(
                    OutputField(name = "generated_code", desc = "python code that answers the question"),
                )
            }
            "regenerate" -> {
                inputFieldsList.add(
                    InputField(name = "previous_code", desc = "previously-generated python code that errored"),
                )
                inputFieldsList.add(
                    InputField(name = "error", desc = "error message from previously-generated python code"),
                )
                outputFieldsList.add(
                    OutputField(name = "generated_code", desc = "python code that answers the question"),
                )
            }
            "answer" -> {
                inputFieldsList.add(
                    InputField(name = "final_generated_code", desc = "python code that answers the question"),
                )
                inputFieldsList.add(
                    InputField(name = "code_output", desc = "output of previously-generated python code"),
                )
                outputFieldsList.addAll(outputFields)
            }
        }

        return Signature(
            instruction = "",
            inputFields = inputFieldsList,
            outputFields = outputFieldsList,
        )
    }

    private fun generateInstruction(mode: String): String {
        val modeSig = generateSignature(mode)
        val modeInputs = modeSig.inputFields.joinToString(", ") { "`${it.name}`" }
        val modeOutputs = modeSig.outputFields.joinToString(", ") { "`${it.name}`" }
        val finalOutputs = outputFields.joinToString(", ") { "`${it.name}`" }

        return when (mode) {
            "generate" -> listOf(
                "You will be given $modeInputs and you will respond with $modeOutputs.",
                "Generating executable Python code that programmatically computes the correct $modeOutputs.",
                "After you're done with the computation and think you have the final output, make sure to submit your output by calling the preloaded function `SUBMIT()`.",
                "You must structure your output in a dict, like {\"field_a\": value_a, ...}, with the correct value mapping for the field(s): $finalOutputs.",
            ).joinToString("\n")
            "regenerate" -> listOf(
                "You are given $modeInputs due to an error in previous code.",
                "Your task is to correct the error and provide the new `generated_code`.",
            ).joinToString("\n")
            else -> listOf(
                "Given the final code $modeInputs, provide the final $modeOutputs.",
            ).joinToString("\n")
        }
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
        val inputKwargs = inputFields.associate { it.name to input[it.name] }.toMutableMap()

        var codeData = codeGenerate.forward(dspy.primitives.Example(inputKwargs))
        var output: String? = null
        var code: String
        var error: String?

        val parseResult = parseCode(codeData.toMap())
        code = parseResult.first
        error = parseResult.second

        if (error == null) {
            val execResult = executeCode(code)
            output = execResult.first
            error = execResult.second
        }

        var hop = 1

        while (error != null) {
            logger.error("Error in code execution: $error")
            if (hop == maxIters) {
                codeInterpreter.shutdown()
                throw RuntimeException("Max hops reached. Failed to run ProgramOfThought: $error")
            }

            inputKwargs["previous_code"] = code
            inputKwargs["error"] = error
            codeData = codeRegenerate.forward(dspy.primitives.Example(inputKwargs))

            val parseResult2 = parseCode(codeData.toMap())
            code = parseResult2.first
            error = parseResult2.second

            if (error == null) {
                val execResult = executeCode(code)
                output = execResult.first
                error = execResult.second
            }

            hop++
        }

        inputKwargs["final_generated_code"] = code
        inputKwargs["code_output"] = output

        val outputGenResult = generateOutput.forward(dspy.primitives.Example(inputKwargs))
        codeInterpreter.shutdown()
        return outputGenResult
    }

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction = forward(kwargs)

    override fun namedParameters(): List<Pair<String, dspy.primitives.Parameter>> {
        return listOf(
            "codeGenerate" to codeGenerate,
            "codeRegenerate" to codeRegenerate,
            "generateOutput" to generateOutput,
        )
    }

    override fun toString(): String = "ProgramOfThought($sig)"
}
