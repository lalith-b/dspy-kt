package dspy.utils

import dspy.primitives.Module
import java.io.File

/**
 * Utility functions for saving and loading DSPy modules.
 *
 * Port of `dspy/utils/saving.py`.
 */
object Saving {

    /**
     * Get dependency version information.
     */
    fun getDependencyVersions(): Map<String, String> {
        return mapOf(
            "kotlin" to System.getProperty("kotlin.version", "unknown"),
            "dspy" to "0.0.1",
        )
    }

    /**
     * Save a DSPy module to the given directory.
     */
    fun save(program: Module, path: String) {
        val dir = File(path)
        dir.mkdirs()

        val metadata = mapOf(
            "dependency_versions" to getDependencyVersions(),
            "module_class" to program::class.java.name,
        )
        File(dir, "metadata.json").writeText(mapToJson(metadata))

        val state = program.dumpState()
        File(dir, "state.json").writeText(mapToJson(state))
    }

    /**
     * Load a saved DSPy module from the given directory.
     */
    fun load(path: String): Map<String, Any?> {
        val dir = File(path)
        if (!dir.exists()) {
            throw IllegalArgumentException("The path '$path' does not exist.")
        }

        val metadataFile = File(dir, "metadata.json")
        if (!metadataFile.exists()) {
            throw IllegalArgumentException("Metadata file not found: ${metadataFile.absolutePath}")
        }

        val metadata = jsonToMap(metadataFile.readText())
        val savedDepVersions = (metadata["dependency_versions"] as? Map<*, *>)
            ?.mapKeys { it.key.toString() }
            ?.mapValues { it.value?.toString() ?: "" }
            ?: emptyMap<String, String>()

        val currentVersions = getDependencyVersions()
        for ((key, savedVersion) in savedDepVersions) {
            val currentVersion = currentVersions[key]
            if (currentVersion != null && currentVersion != savedVersion) {
                println(
                    "There is a mismatch of $key version between saved model and current environment. " +
                        "You saved with `$key==$savedVersion`, but now you have `$key==$currentVersion`. " +
                        "This might cause errors or performance downgrade on the loaded model."
                )
            }
        }

        val stateFile = File(dir, "state.json")
        val state = if (stateFile.exists()) jsonToMap(stateFile.readText()) else emptyMap<String, Any>()

        return mapOf("metadata" to metadata, "state" to state)
    }

    // ---- Simple JSON helpers (no kotlinx.serialization dependency) ----

    fun mapToJson(m: Map<String, Any?>): String {
        return m.entries.joinToString(", ", "{", "}") { (k, v) ->
            "\"$k\": ${valueToJson(v)}"
        }
    }

    private fun valueToJson(v: Any?): String {
        return when (v) {
            null -> "null"
            is String -> "\"${v.replace("\\", "\\\\").replace("\"", "\\\"")}\""
            is Number -> v.toString()
            is Boolean -> v.toString()
            is Map<*, *> -> mapToJson(v.mapKeys { it.key.toString() }.mapValues { it.value })
            is List<*> -> v.joinToString(", ", "[", "]") { valueToJson(it) }
            else -> "\"${v.toString().replace("\"", "\\\"")}\""
        }
    }

    fun jsonToMap(s: String): Map<String, Any?> {
        return parseValue(s.trim()) as? Map<String, Any?> ?: emptyMap()
    }

    private fun parseValue(s: String): Any? {
        val t = s.trim()
        return when {
            t == "null" -> null
            t == "true" -> true
            t == "false" -> false
            t.startsWith('"') && t.endsWith('"') -> t.substring(1, t.length - 1)
            t.startsWith('{') -> {
                val inner = t.substring(1, t.length - 1)
                val result = mutableMapOf<String, Any?>()
                if (inner.trim().isNotEmpty()) {
                    for (pair in splitEntries(inner)) {
                        val idx = pair.indexOf(':')
                        if (idx > 0) {
                            val k = pair.substring(0, idx).trim().removeSurrounding("\"")
                            val v = parseValue(pair.substring(idx + 1).trim())
                            result[k] = v
                        }
                    }
                }
                result
            }
            t.startsWith('[') -> {
                val inner = t.substring(1, t.length - 1)
                if (inner.trim().isEmpty()) emptyList()
                else splitEntries(inner).map { parseValue(it.trim()) }
            }
            else -> try { t.toDouble() } catch (_: NumberFormatException) { t }
        }
    }

    private fun splitEntries(s: String): List<String> {
        val entries = mutableListOf<String>()
        var depth = 0
        var current = StringBuilder()
        for (c in s) {
            when (c) {
                '{', '[' -> { depth++; current.append(c) }
                '}', ']' -> { depth--; current.append(c) }
                ',' -> {
                    if (depth == 0) {
                        entries.add(current.toString().trim())
                        current = StringBuilder()
                    } else {
                        current.append(c)
                    }
                }
                else -> current.append(c)
            }
        }
        if (current.toString().trim().isNotEmpty()) {
            entries.add(current.toString().trim())
        }
        return entries
    }
}
