package dspy.adapters.types

import kotlinx.serialization.Serializable

/**
 * A file input type for DSPy.
 *
 * The file_data field should be a data URI with the format:
 *     data:<mime_type>;base64,<base64_encoded_data>
 */
@Serializable
data class File(
    val fileData: String? = null,
    val fileId: String? = null,
    val filename: String? = null
) : Type() {
    override fun format(): List<Map<String, Any?>> {
        val fileDict = mutableMapOf<String, Any?>()
        fileData?.let { fileDict["file_data"] = it }
        fileId?.let { fileDict["file_id"] = it }
        filename?.let { fileDict["filename"] = it }
        return listOf(mapOf("type" to "file", "file" to fileDict))
    }

    companion object {
        fun fromPath(filePath: String, filename: String? = null, mimeType: String? = null): File {
            throw NotImplementedError("File.fromPath not yet fully implemented in Kotlin")
        }

        fun fromBytes(fileBytes: ByteArray, filename: String? = null, mimeType: String = "application/octet-stream"): File {
            val encodedData = java.util.Base64.getEncoder().encodeToString(fileBytes)
            val fileData = "data:$mimeType;base64,$encodedData"
            return File(fileData = fileData, filename = filename)
        }

        fun fromFileId(fileId: String, filename: String? = null): File {
            return File(fileId = fileId, filename = filename)
        }
    }

    override fun toString(): String {
        val parts = mutableListOf<String>()
        fileData?.let {
            if (it.startsWith("data:")) {
                val mimeType = it.split(";")[0].split(":")[1]
                val lenData = if ("base64," in it) it.split("base64,")[1].length else it.length
                parts.add("file_data=<DATA_URI($mimeType, $lenData chars)>")
            } else {
                parts.add("file_data=<DATA(${it.length} chars)>")
            }
        }
        fileId?.let { parts.add("file_id='$it'") }
        filename?.let { parts.add("filename='$it'") }
        return "File(${parts.joinToString(", ")})"
    }
}

fun encodeFileToDict(fileInput: Any): Map<String, String?> {
    return when (fileInput) {
        is File -> {
            mutableMapOf<String, String?>().apply {
                fileInput.fileData?.let { this["file_data"] = it }
                fileInput.fileId?.let { this["file_id"] = it }
                fileInput.filename?.let { this["filename"] = it }
            }
        }
        is String -> {
            val fileObj = File.fromPath(fileInput)
            mapOf("file_data" to fileObj.fileData, "filename" to fileObj.filename)
        }
        is ByteArray -> {
            val fileObj = File.fromBytes(fileInput)
            mapOf("file_data" to fileObj.fileData)
        }
        else -> throw IllegalArgumentException("Unsupported file input type: ${fileInput::class}")
    }
}
