package dspy.predict

import dspy.adapters.types.History
import dspy.adapters.types.Tool
import dspy.adapters.types.ToolCallResults
import dspy.adapters.types.ToolCalls
import dspy.core.AdapterParseError
import dspy.core.ContextWindowExceededError
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature
import dspy.signatures.SignatureUtils
import org.slf4j.LoggerFactory

/**
 * ReActV2 agent module.
 * Port of `dspy/predict/react_v2.py`.
 */
class ReActV2(
    signature: Any,
    toolsList: List<Any>,
    maxIters: Int = 20,
) : Module() {
    private val logger = LoggerFactory.getLogger(ReActV2::class.java)

    val signature: Signature
    val maxIters: Int = maxIters
    val tools: MutableMap<String, Tool> = mutableMapOf()
    private val react: Predict

    init {
        this.signature = SignatureUtils.ensureSignature(signature)
        val userTools = toolsList.map { tool ->
            if (tool is Tool) tool else Tool({ "" }, name = tool.toString())
        }
        for (tool in userTools) {
            if ((tool.name ?: "") == "submit") {
                throw IllegalArgumentException("`submit` is reserved by ReActV2 as the final-output tool.")
            }
            tools[tool.name ?: "unknown"] = tool
        }
        tools["submit"] = makeSubmitTool()
        this.react = Predict(sig = makeReactSignature())
    }

    private fun makeSubmitTool(): Tool {
        val argTypes = signature.outputFields.associate { it.name to it.annotation }
        return Tool(
            func = { mapOf<String, Any?>() },
            name = "submit",
            desc = "Submit the final outputs for the task.",
            args = emptyMap(),
            argTypes = argTypes,
        )
    }

    private fun makeReactSignature(): Signature {
        val inputFields = signature.inputFields.map { field ->
            InputField(
                name = field.name,
                desc = field.desc,
                prefix = field.prefix,
                annotation = field.annotation,
            )
        } + listOf(
            InputField(name = "history", desc = ""),
            InputField(name = "tools", desc = ""),
        )

        val outputFields = listOf(
            OutputField(name = "next_thought", desc = ""),
            OutputField(name = "tool_calls", desc = ""),
        )

        val backtick = "`"
        val inputs = signature.inputFields.joinToString(", ") { "$backtick${it.name}$backtick" }
        val outputs = signature.outputFields.joinToString(", ") { "$backtick${it.name}$backtick" }
        val toolNames = tools.keys.joinToString(", ") { "$backtick$it$backtick" }
        val instructions = buildString {
            if (signature.instructions.isNotBlank()) {
                append(signature.instructions)
                append("\n")
            }
            append("You are an Agent. Use the supplied tools to produce $outputs from $inputs.\n")
            append("Call tools when more information is needed.\n")
            append("When the final answer is ready, call $backtick submit$backtick with $outputs.\n")
            append("The available tools are: $toolNames.")
        }.trim()

        return Signature(
            instruction = instructions,
            inputFields = inputFields,
            outputFields = outputFields,
        )
    }

    suspend fun forward(kwargs: MutableMap<String, Any?> = mutableMapOf()): Prediction {
        val maxIters = (kwargs.remove("max_iters") as? Int) ?: this.maxIters
        var history = coerceHistory(kwargs.remove("history"))
        val pendingInputs = mutableMapOf<String, Any?>()
        for (field in signature.inputFields) {
            if (field.name in kwargs) {
                pendingInputs[field.name] = kwargs[field.name]
            }
        }

        var breakReason = "max_iters"
        for (turnIndex in 0 until maxIters) {
            try {
                val callKwargs = mutableMapOf<String, Any?>(
                    "history" to history,
                    "tools" to tools.values.toList(),
                )
                callKwargs.putAll(pendingInputs)
                val pred = react.__call__(kwargs = callKwargs)
                val toolCalls = coerceToolCalls(pred["tool_calls"])
                if (toolCalls.toolCalls.isEmpty()) {
                    breakReason = "empty_tool_calls"
                    break
                }

                val ensuredToolCalls = ensureToolCallIds(toolCalls, turnIndex)
                val result = executeToolCalls(ensuredToolCalls)
                val (toolCallResults, finalOutputs) = result

                val event = historyEvent(pendingInputs, pred, ensuredToolCalls, toolCallResults)
                if (finalOutputs != null) {
                    (event as MutableMap<String, Any?>).putAll(finalOutputs)
                }
                appendHistoryEvent(history, event)
                pendingInputs.clear()

                if (finalOutputs != null) {
                    val resultMap = mutableMapOf<String, Any?>()
                    resultMap.putAll(finalOutputs)
                    resultMap["history"] = history
                    resultMap["termination_reason"] = "submit"
                    return Prediction(base = resultMap)
                }
            } catch (e: AdapterParseError) {
                logger.warn("Ending ReActV2 loop after parse failure: {}", fmtExc(e))
                breakReason = "parse_error"
                break
            } catch (e: ContextWindowExceededError) {
                logger.warn("Ending ReActV2 loop after context window exceeded.")
                breakReason = "context_window_exceeded"
                break
            } catch (e: Exception) {
                logger.warn("Ending ReActV2 loop after error: {}", fmtExc(e))
                breakReason = "error"
                break
            }
        }

        return forcedSubmit(history, pendingInputs, breakReason, maxIters)
    }

    private fun executeToolCalls(
        toolCalls: ToolCalls
    ): Pair<ToolCallResults, Map<String, Any?>?> {
        val values = mutableListOf<Any?>()
        val isErrors = mutableListOf<Boolean>()
        var finalOutputs: Map<String, Any?>? = null

        for (toolCall in toolCalls.toolCalls) {
            if (toolCall.name !in tools) {
                values.add("Unknown tool: ${toolCall.name}")
                isErrors.add(true)
                continue
            }
            try {
                val value = tools[toolCall.name]?.invoke(toolCall.args as Map<String, Any>)
                values.add(value)
                isErrors.add(false)
                if (toolCall.name == "submit" && value is Map<*, *>) {
                    finalOutputs = value.mapKeys { it.key.toString() }.mapValues { it.value }
                }
            } catch (e: Exception) {
                values.add("Execution error in ${toolCall.name}: ${fmtExc(e)}")
                isErrors.add(true)
            }
        }

        return Pair(
            ToolCallResults.fromToolCallsAndValues(toolCalls, values, isErrors),
            finalOutputs,
        )
    }

    private fun historyEvent(
        pendingInputs: Map<String, Any?>,
        pred: Prediction,
        toolCalls: ToolCalls,
        toolCallResults: ToolCallResults,
    ): MutableMap<String, Any?> {
        val event = mutableMapOf<String, Any?>()
        event.putAll(pendingInputs)
        val nextThought = pred["next_thought"]
        if (nextThought != null) {
            event["next_thought"] = nextThought
        }
        if (toolCalls.toolCalls.isNotEmpty()) {
            if (toolCallResults.toolCallResults.isNotEmpty()) {
                val updated = toolCalls.copy(mapOf("tool_call_results" to toolCallResults))
                event["tool_calls"] = updated
            } else {
                event["tool_calls"] = toolCalls
            }
        }
        return event
    }

    private suspend fun forcedSubmit(
        history: History,
        pendingInputs: Map<String, Any?>,
        breakReason: String,
        turnIndex: Int,
    ): Prediction {
        try {
            val callKwargs = mutableMapOf<String, Any?>(
                "history" to history,
                "tools" to tools.values.toList(),
                "config" to mapOf<String, Any?>(
                    "tool_choice" to mapOf(
                        "type" to "function",
                        "function" to mapOf("name" to "submit"),
                    ),
                    "reasoning_effort" to null,
                ),
            )
            callKwargs.putAll(pendingInputs)
            val pred = react.__call__(kwargs = callKwargs)
            val ensuredToolCalls = ensureToolCallIds(
                coerceToolCalls(pred["tool_calls"]),
                turnIndex
            )
            val submitCalls = ToolCalls(
                toolCalls = ensuredToolCalls.toolCalls.filter { it.name == "submit" }
            )
            if (submitCalls.toolCalls.isEmpty()) {
                return Prediction(
                    base = mapOf("history" to history, "termination_reason" to (breakReason.ifEmpty { "failed" })),
                )
            }

            val result = executeToolCalls(submitCalls)
            val (toolCallResults, finalOutputs) = result
            val event = historyEvent(pendingInputs, pred, submitCalls, toolCallResults)
            if (finalOutputs != null) {
                event.putAll(finalOutputs)
            }
            appendHistoryEvent(history, event)

            if (finalOutputs != null) {
                val resultMap = mutableMapOf<String, Any?>()
                resultMap.putAll(finalOutputs)
                resultMap["history"] = history
                resultMap["termination_reason"] = "forced_submit"
                return Prediction(base = resultMap)
            }

            return Prediction(
                base = mapOf("history" to history, "termination_reason" to (breakReason.ifEmpty { "failed" })),
            )
        } catch (e: AdapterParseError) {
            logger.warn("Forced submit failed: {}", fmtExc(e))
            return Prediction(
                base = mapOf("history" to history, "termination_reason" to (breakReason.ifEmpty { "failed" })),
            )
        } catch (e: ContextWindowExceededError) {
            logger.warn("Forced submit failed: {}", fmtExc(e))
            return Prediction(
                base = mapOf("history" to history, "termination_reason" to (breakReason.ifEmpty { "failed" })),
            )
        } catch (e: Exception) {
            logger.warn("Forced submit failed: {}", fmtExc(e))
            return Prediction(
                base = mapOf("history" to history, "termination_reason" to (breakReason.ifEmpty { "failed" })),
            )
        }
    }

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction {
        return forward(kwargs.toMutableMap())
    }

    override fun toString(): String = "ReActV2($signature)"
}

// ---- Helper functions ----

fun coerceHistory(history: Any?): History {
    return when (history) {
        null -> History()
        is History -> history
        is Map<*, *> -> History(
            messages = (history["messages"] as? List<*>)
                ?.filterIsInstance<Map<String, Any?>>()
                ?.toMutableList()
                ?: mutableListOf()
        )
        else -> History()
    }
}

fun coerceToolCalls(toolCalls: Any?): ToolCalls {
    return when (toolCalls) {
        null -> ToolCalls()
        is ToolCalls -> toolCalls
        is List<*> -> {
            val dicts = toolCalls.filterIsInstance<Map<String, Any?>>()
            ToolCalls.fromDictList(dicts)
        }
        else -> ToolCalls()
    }
}

fun ensureToolCallIds(toolCalls: ToolCalls, turnIndex: Int): ToolCalls {
    val ensured = toolCalls.toolCalls.mapIndexed { callIndex, call ->
        if (call.id == null) {
            ToolCalls.ToolCall(
                id = "call_${turnIndex}_${callIndex}",
                name = call.name,
                args = call.args,
            )
        } else {
            call
        }
    }
    return ToolCalls(toolCalls = ensured)
}

fun appendHistoryEvent(history: History, event: Map<String, Any?>) {
    if (event.isNotEmpty()) {
        history.messages.add(event as Map<String, Any?>)
    }
}

fun fmtExc(error: Throwable, limit: Int = 5): String {
    return try {
        val sw = java.io.StringWriter()
        val pw = java.io.PrintWriter(sw)
        error.printStackTrace(pw)
        val lines = sw.toString().split("\n")
        "\n${lines.take(limit).joinToString("\n")}".trim()
    } catch (_: Exception) {
        error.message ?: error.toString()
    }
}
