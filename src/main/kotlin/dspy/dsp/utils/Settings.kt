package dspy.dsp.utils

/**
 * DSP utility settings.
 */
object Settings {
    @Volatile
    var lm: Any? = null
    @Volatile
    var rm: Any? = null
    @Volatile
    var adapter: Any? = null
    @Volatile
    var trace: List<Map<String, Any?>>? = null
    @Volatile
    var config: MutableMap<String, Any?> = mutableMapOf()
}

/**
 * General DSP utility functions.
 */
object Utils {
    fun dedent(s: String): String {
        val lines = s.lines()
        if (lines.isEmpty()) return s
        val nonEmptyLines = lines.filter { it.trim().isNotEmpty() }
        if (nonEmptyLines.isEmpty()) return s
        val minIndent = nonEmptyLines.minOfOrNull { line ->
            line.takeWhile { it == ' ' || it == '\t' }.length
} ?: 0
        val indent = " ".repeat(minIndent)
        return lines.joinToString("\n") { line ->
            if (line.startsWith(indent)) line.removePrefix(indent) else line
        }
    }
}
