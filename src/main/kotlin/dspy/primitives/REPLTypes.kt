package dspy.primitives

/**
 * REPL (Read-Eval-Print Loop) types for code execution.
 */
sealed class REPLType {
    abstract val name: String
}

data class REPLCode(override val name: String = "code", val content: String) : REPLType()

data class REPLOutput(override val name: String = "output", val content: String) : REPLType()

data class REPLInput(override val name: String = "input", val content: String) : REPLType()

data class REPLPrompt(
    override val name: String = "prompt",
    val content: String,
    val history: List<Any> = emptyList()
) : REPLType()

/**
 * REPL (Read-Eval-Print Loop) history entry.
 */
data class REPLEntry(
    val reasoning: String = "",
    val code: String = "",
    val output: String = ""
) {
    fun toMap(): Map<String, Any?> = mapOf(
        "reasoning" to reasoning,
        "code" to code,
        "output" to output
    )

    companion object {
        fun formatOutput(output: String, maxChars: Int): String {
            return if (output.length > maxChars) {
                output.substring(0, maxChars) + "...\n[truncated - ${(output.length - maxChars)} chars omitted]"
            } else {
                output
            }
        }

        fun fromMap(map: Map<*, *>): REPLEntry = REPLEntry(
            reasoning = map["reasoning"]?.toString() ?: "",
            code = map["code"]?.toString() ?: "",
            output = map["output"]?.toString() ?: ""
        )
    }
}

/**
 * Container for REPL interaction history.
 *
 * Immutable: append() returns a new instance with the entry added.
 *
 * Port of `dspy/primitives/repl_types.py` - `REPLHistory`
 */
data class REPLHistory(
    val entries: List<REPLEntry> = emptyList(),
    val maxOutputChars: Int = 10_000
) {
    fun append(reasoning: String, code: String, output: String): REPLHistory {
        return REPLHistory(
            entries = entries + REPLEntry(reasoning, code, output),
            maxOutputChars = maxOutputChars
        )
    }

    fun format(): String {
        if (entries.isEmpty()) {
            return "You have not interacted with the REPL environment yet."
        }
        return entries.joinToString("\n") { entry ->
            entry.reasoning + "\n" + entry.code + "\n" + REPLEntry.formatOutput(entry.output, maxOutputChars)
        }
    }
}

/**
 * Generic code interpreter for code execution.
 * @deprecated Use [dspy.primitives.CodeInterpreter] interface instead.
 */
@Deprecated("Use dspy.primitives.CodeInterpreter interface instead", replaceWith = ReplaceWith("CodeInterpreter", "dspy.primitives.CodeInterpreter"))
class GenericCodeInterpreter {
    fun invoke(code: String, language: String = "python", timeout: Int = 30): String {
        return when (language) {
            "python" -> throw NotImplementedError("Python interpreter requires JPython bridge or subprocess")
            else -> throw IllegalArgumentException("Unsupported language: $language")
        }
    }
}

/**
 * Sandbox for safe code execution.
 */
class Sandbox {
    fun execute(code: String, timeout: Int = 30): Pair<Boolean, String> {
        return try {
            // In production, this would use a Python subprocess or JPython bridge
            throw NotImplementedError("Sandbox execution requires JPython bridge or subprocess")
        } catch (e: Exception) {
            Pair(false, e.message ?: "Execution failed")
        }
    }
}
