package dspy.utils

/**
 * Pretty-print history of LM interactions.
 *
 * Faithful port of `dspy/utils/inspect_history.py`.
 */

/**
 * ANSI color codes.
 */
private const val ANSI_GREEN = "\u001b[32m"
private const val ANSI_RED = "\u001b[31m"
private const val ANSI_BLUE = "\u001b[34m"
private const val ANSI_RESET = "\u001b[0m"

/**
 * Print the last n prompts and their completions from the history.
 *
 * Args:
 *     history: The history list to print from.
 *     n: Number of recent entries to display. Defaults to 1.
 *     useColors: Whether to use ANSI colors. Auto-disabled when writing to a file.
 */
fun prettyPrintHistory(history: List<Map<String, Any?>>, n: Int = 1, useColors: Boolean = true) {
    val items = history.takeLast(n)

    for (item in items) {
        val messages = (item["messages"] as? List<Map<String, Any?>>)
            ?: listOf(mapOf("role" to "user", "content" to (item["prompt"] ?: "")))
        val outputs = item["outputs"] as? List<Any?> ?: emptyList()
        val timestamp = item["timestamp"] as? String ?: "Unknown time"

        if (useColors) println("\n\n\n${ANSI_BLUE}[$timestamp]${ANSI_RESET}")
        else println("\n\n\n[$timestamp]")

        for (msg in messages) {
            val role = (msg["role"] as? String ?: "unknown").capitalize()
            if (useColors) println("${ANSI_RED}$role message:${ANSI_RESET}")
            else println("$role message:")

            val content = msg["content"]
            when (content) {
                is String -> println(content.trim())
                is List<*> -> {
                    for (c in content) {
                        if (c is Map<*, *>) {
                            val type = c["type"] as? String
                            when (type) {
                                "text" -> println((c["text"] as? String ?: "").trim())
                                "image_url" -> {
                                    val imageUrl = (c["image_url"] as? Map<*, *>)?.get("url") as? String ?: ""
                                    val display = if (imageUrl.contains("base64,")) {
                                        val prefix = imageUrl.split("base64,")[0]
                                        val b64Len = imageUrl.split("base64,")[1].length
                                        "<${prefix}base64,<IMAGE BASE 64 ENCODED($b64Len)>]"
                                    } else {
                                        "<image_url: $imageUrl>"
                                    }
                                    if (useColors) println("${ANSI_BLUE}${display.trim()}${ANSI_RESET}")
                                    else println(display.trim())
                                }
                                "input_audio" -> {
                                    val audio = c["input_audio"] as? Map<*, *>
                                    val format = audio?.get("format") as? String ?: "unknown"
                                    val data = audio?.get("data") as? String ?: ""
                                    val audioStr = "<audio format='$format' base64-encoded, length=${data.length}>"
                                    if (useColors) println("${ANSI_BLUE}${audioStr.trim()}${ANSI_RESET}")
                                    else println(audioStr.trim())
                                }
                                "file", "input_file" -> {
                                    val file = c["file"] as? Map<*, *> ?: (c["input_file"] as? Map<*, *>)
                                    val filename = file?.get("filename") as? String ?: ""
                                    val fileId = file?.get("file_id") as? String ?: ""
                                    val fileData = file?.get("file_data") as? String ?: ""
                                    val fileStr = "<file: name:$filename, id:$fileId, data_length:${fileData.length}>"
                                    if (useColors) println("${ANSI_BLUE}${fileStr.trim()}${ANSI_RESET}")
                                    else println(fileStr.trim())
                                }
                            }
                        }
                    }
                }
            }

            // Print tool calls
            printToolCalls(msg["tool_calls"] as? List<Map<String, Any?>>, useColors)
            println()
        }

        // Print outputs
        if (outputs.isNotEmpty()) {
            if (outputs[0] is Map<*, *>) {
                val output = outputs[0] as Map<*, *>
                val text = output["text"] as? String
                if (text != null) {
                    if (useColors) println("${ANSI_RED}Response:${ANSI_RESET}")
                    else println("Response:")
                    if (useColors) println("${ANSI_GREEN}${text.trim()}${ANSI_RESET}")
                    else println(text.trim())
                }
                printToolCalls(output["tool_calls"] as? List<Map<String, Any?>>, useColors)
            } else {
                if (useColors) println("${ANSI_RED}Response:${ANSI_RESET}")
                else println("Response:")
                if (useColors) println("${ANSI_GREEN}${outputs[0]?.toString()?.trim()}${ANSI_RESET}")
                else println(outputs[0]?.toString()?.trim())
            }

            if (outputs.size > 1) {
                val choicesText = " \t (and ${outputs.size - 1} other completions)"
                if (useColors) print("${ANSI_RED}$choicesText${ANSI_RESET}")
                else print(choicesText)
            }
        }
    }

    println("\n\n\n")
}

/**
 * Print tool calls from a message or output.
 */
private fun printToolCalls(toolCalls: List<Map<String, Any?>>?, useColors: Boolean) {
    if (toolCalls.isNullOrEmpty()) return

    if (useColors) println("${ANSI_RED}Tool calls:${ANSI_RESET}")
    else println("Tool calls:")

    for (toolCall in toolCalls) {
        val function = toolCall["function"] as? Map<String, Any?>
        val name = function?.get("name") as? String ?: toolCall["name"] as? String ?: "<unknown>"

        var arguments = function?.get("arguments") ?: toolCall["args"] ?: toolCall["arguments"] ?: ""
        if (arguments is String) {
            arguments = try {
                kotlinx.serialization.json.Json.decodeFromString<kotlinx.serialization.json.JsonElement>(arguments)
            } catch (_: Exception) {
                arguments
            }
        }

        val argStr = when (arguments) {
            is Map<*, *> -> arguments.toString()
            is List<*> -> arguments.toString()
            else -> arguments.toString()
        }

        if (useColors) println("${ANSI_GREEN}${name}: ${argStr}${ANSI_RESET}")
        else println("$name: $argStr")
    }
}
