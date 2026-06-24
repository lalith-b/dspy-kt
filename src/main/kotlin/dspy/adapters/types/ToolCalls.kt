package dspy.adapters.types

class ToolCalls(
    val toolCalls: List<ToolCall> = emptyList(),
    val toolCallResults: Any? = null,
) : Type() {
    override fun format(): Any {
        return toolCalls.map { it.format() }
    }

    data class ToolCall(
        val id: String? = null,
        val name: String,
        val args: Map<String, Any?> = emptyMap(),
    ) : Type() {
        override fun format(): Map<String, Any?> {
            return mapOf("name" to name, "args" to args)
        }
    }

    fun copy(update: Map<String, Any?>): ToolCalls {
        return ToolCalls(
            toolCalls = toolCalls,
            toolCallResults = update["tool_call_results"] ?: toolCallResults,
        )
    }

    companion object {
        fun fromDictList(dicts: List<Map<String, Any?>>): ToolCalls {
            val calls = dicts.mapNotNull { d ->
                val name = (d["name"] as? String)?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
                val id = d["id"] as? String
                val args = (d["args"] as? Map<String, Any?>) ?: emptyMap()
                ToolCall(id = id, name = name, args = args)
            }
            return ToolCalls(toolCalls = calls)
        }
    }
}

/**
 * Results from executing tool calls.
 */
data class ToolCallResults(
    val toolCallResults: List<ToolCallResult> = emptyList(),
) {
    data class ToolCallResult(
        val id: String? = null,
        val name: String,
        val value: Any?,
        val isError: Boolean = false,
    ) : Type() {
        override fun format(): Map<String, Any?> {
            return mapOf("name" to name, "value" to value, "isError" to isError)
        }
    }

    companion object {
        fun fromToolCallsAndValues(
            toolCalls: ToolCalls,
            values: List<Any?>,
            isErrors: List<Boolean>
        ): ToolCallResults {
            val results = toolCalls.toolCalls.mapIndexed { index, call ->
                ToolCallResult(
                    id = call.id,
                    name = call.name,
                    value = values.getOrElse(index) { null },
                    isError = isErrors.getOrElse(index) { false },
                )
            }
            return ToolCallResults(toolCallResults = results)
        }
    }
}
