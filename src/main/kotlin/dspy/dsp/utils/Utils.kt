package dspy.dsp.utils

/**
 * DSP utility functions.
 */
object DSPUtils {
    /**
     * Dedent multi-line strings.
     */
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
            if (line.startsWith(indent)) line.substring(indent.length) else line
        }
    }

    /**
     * Normalize whitespace in a string.
     */
    fun normalizeWhitespace(s: String): String {
        return s.replace(Regex("\\s+"), " ").trim()
    }

    /**
     * Truncate string to max characters.
     */
    fun truncate(s: String, maxChars: Int, suffix: String = "..."): String {
        return if (s.length <= maxChars) s else s.substring(0, maxChars - suffix.length) + suffix
    }

    /**
     * Word wrap a string.
     */
    fun wordWrap(s: String, width: Int = 80): String {
        val words = s.split(Regex("\\s+"))
        val lines = mutableListOf<String>()
        var currentLine = ""
        for (word in words) {
            if ((currentLine + " " + word).length <= width) {
                currentLine = if (currentLine.isEmpty()) word else "$currentLine $word"
            } else {
                if (currentLine.isNotEmpty()) lines.add(currentLine)
                currentLine = word
            }
        }
        if (currentLine.isNotEmpty()) lines.add(currentLine)
        return lines.joinToString("\n")
    }
}
