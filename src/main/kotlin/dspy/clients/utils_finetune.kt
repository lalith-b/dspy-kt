package dspy.clients.utils_finetune

import java.io.File
import java.security.MessageDigest
import kotlinx.serialization.json.Json

val DSPY_CACHEDIR by lazy {
    System.getenv("DSPY_CACHEDIR") ?: "${System.getProperty("user.home")}/.cache/dspy"
}

object Hasher {
    fun hash(data: Any?): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(data.toString().toByteArray())
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

/**
 * Training status for fine-tuning jobs.
 */
enum class TrainingStatus(val value: String) {
    not_started("not_started"),
    pending("pending"),
    running("running"),
    succeeded("succeeded"),
    failed("failed"),
    cancelled("cancelled");

    companion object {
        fun fromString(value: String): TrainingStatus? {
            return entries.find { it.value == value }
        }
    }
}

/**
 * Training data format types.
 */
enum class TrainDataFormat(val value: String) {
    CHAT("chat"),
    COMPLETION("completion"),
    GRPO_CHAT("grpo_chat");

    companion object {
        fun fromString(value: String): TrainDataFormat {
            return when (value.lowercase()) {
                "chat" -> CHAT
                "completion" -> COMPLETION
                "grpo_chat" -> GRPO_CHAT
                else -> throw IllegalArgumentException("Unknown data format: $value")
            }
        }
    }
}

/**
 * GRPO (Group Relative Policy Optimization) data structures.
 */
data class GRPOChatData(
    val messages: List<Map<String, Any?>>,
    val completion: Map<String, Any?>,
    val reward: Double
)

data class GRPOGroup(
    val batchId: Int?,
    val group: List<GRPOChatData>
)

data class GRPOStatus(
    val jobId: String,
    val status: String?,
    val currentModel: String,
    val checkpoints: Map<String, String>,
    val lastCheckpoint: String?,
    val pendingBatchIds: List<Int> = emptyList()
)

/**
 * Get the finetune directory.
 */
fun getFinetuneDirectory(): String {
    val defaultFinetuneDir = File(DSPY_CACHEDIR, "finetune").absolutePath
    val finetuneDir = System.getenv("DSPY_FINETUNEDIR") ?: defaultFinetuneDir
    File(finetuneDir).mkdirs()
    return finetuneDir
}

/**
 * Save training data to a JSONL file.
 */
fun saveData(data: List<Map<String, Any?>>): String {
    val hash = Hasher.hash(data)
    val fileName = "$hash.jsonl"
    val finetuneDir = getFinetuneDirectory()
    val filePath = File(finetuneDir, fileName).absolutePath

    File(filePath).bufferedWriter().use { writer ->
        for (item in data) {
            writer.write(serializeMapToJson(item) + "\n")
        }
    }
    return filePath
}

/**
 * Write lines to a file.
 */
fun writeLines(filePath: String, data: List<Map<String, Any?>>) {
    File(filePath).bufferedWriter().use { writer ->
        for (item in data) {
            writer.write(serializeMapToJson(item) + "\n")
        }
    }
}

private fun serializeMapToJson(map: Map<String, Any?>): String {
    val obj = kotlinx.serialization.json.buildJsonObject {
        for ((k, v) in map) put(k, when (v) {
            null -> kotlinx.serialization.json.JsonNull
            is String -> kotlinx.serialization.json.JsonPrimitive(v)
            is Number -> kotlinx.serialization.json.JsonPrimitive(v)
            is Boolean -> kotlinx.serialization.json.JsonPrimitive(v)
            else -> kotlinx.serialization.json.JsonPrimitive(v.toString())
        })
    }
    return obj.toString()
}

/**
 * Validate the data format.
 */
fun validateDataFormat(data: List<Map<String, Any?>>, dataFormat: TrainDataFormat) {
    when (dataFormat) {
        TrainDataFormat.CHAT -> {
            for (item in data) {
                if (!item.containsKey("messages")) {
                    throw IllegalArgumentException("Each item must have 'messages' key for CHAT format")
                }
                val messages = item["messages"] as? List<*>
                if (messages == null) {
                    throw IllegalArgumentException("'messages' must be a list")
                }
                for (msg in messages) {
                    if (msg is Map<*, *>) {
                        if (!msg.containsKey("role") || !msg.containsKey("content")) {
                            throw IllegalArgumentException("Each message must have 'role' and 'content' keys")
                        }
                    }
                }
            }
        }
        TrainDataFormat.COMPLETION -> {
            for (item in data) {
                if (!item.containsKey("prompt")) {
                    throw IllegalArgumentException("Each item must have 'prompt' key for COMPLETION format")
                }
            }
        }
        TrainDataFormat.GRPO_CHAT -> {
            for (item in data) {
                if (!item.containsKey("messages") || !item.containsKey("completion") || !item.containsKey("reward")) {
                    throw IllegalArgumentException("Each item must have 'messages', 'completion', and 'reward' keys for GRPO_CHAT format")
                }
            }
        }
    }
}
