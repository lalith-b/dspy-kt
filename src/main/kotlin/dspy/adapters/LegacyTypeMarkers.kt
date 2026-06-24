package dspy.adapters

import dspy.adapters.types.History
import dspy.adapters.types.Type
import dspy.core.types.LMAudioPart
import dspy.core.types.LMBinaryPart
import dspy.core.types.LMDocumentPart
import dspy.core.types.LMImagePart
import dspy.core.types.LMMessage
import dspy.core.types.LMPart
import dspy.core.types.LMTextPart
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

const val CUSTOM_TYPE_START_IDENTIFIER = "<<DSPY_CUSTOM_TYPE_START>>"
const val CUSTOM_TYPE_END_IDENTIFIER = "<<DSPY_CUSTOM_TYPE_END>>"

private val markerPattern by lazy {
    Regex("$CUSTOM_TYPE_START_IDENTIFIER(.*?)$CUSTOM_TYPE_END_IDENTIFIER", RegexOption.DOT_MATCHES_ALL)
}

fun expandLegacyCustomTypeMarkersInChatMessage(message: Map<String, Any?>): Map<String, Any?> {
    if (message["role"] != "user" || message["content"] !is String) return message
    val content = message["content"] as String
    if (CUSTOM_TYPE_START_IDENTIFIER !in content) return message
    return message.toMutableMap().apply {
        this["content"] = splitLegacyCustomTypeTextToBlocks(content)
    }
}

fun expandLegacyCustomTypeMarkersInLmMessage(message: LMMessage): LMMessage {
    if (message.role != "user") return message
    var changed = false
    val expandedParts = mutableListOf<LMPart>()
    for (part in message.parts) {
        if (part !is LMTextPart || CUSTOM_TYPE_START_IDENTIFIER !in part.text) {
            expandedParts.add(part)
            continue
        }
        changed = true
        expandedParts.addAll(splitLegacyCustomTypeTextToParts(part.text))
    }
    return if (changed) message.copy(parts = expandedParts) else message
}

private fun splitLegacyCustomTypeTextToBlocks(text: String): List<Map<String, Any?>> {
    val blocks = mutableListOf<Map<String, Any?>>()
    var lastEnd = 0
    for (match in markerPattern.findAll(text)) {
        val start = match.range.first
        val end = match.range.last
        if (start > lastEnd) {
            blocks.add(mapOf("type" to "text", "text" to text.substring(lastEnd, start)))
        }
        blocks.addAll(legacyCustomTypePayloadToBlocks(match.groupValues[1].trim()))
        lastEnd = end
    }
    if (lastEnd < text.length) {
        blocks.add(mapOf("type" to "text", "text" to text.substring(lastEnd)))
    }
    return blocks
}

private fun splitLegacyCustomTypeTextToParts(text: String): List<LMPart> {
    val parts = mutableListOf<LMPart>()
    var lastEnd = 0
    for (match in markerPattern.findAll(text)) {
        val start = match.range.first
        val end = match.range.last
        if (start > lastEnd) {
            parts.add(LMTextPart(text = text.substring(lastEnd, start)))
        }
        parts.addAll(legacyCustomTypePayloadToParts(match.groupValues[1].trim()))
        lastEnd = end
    }
    if (lastEnd < text.length) {
        parts.add(LMTextPart(text = text.substring(lastEnd)))
    }
    return parts
}

private fun legacyCustomTypePayloadToBlocks(payload: String): List<Map<String, Any?>> {
    val parsed = parseLegacyPayload(payload)
    if (parsed is List<*>) {
        return parsed.map { if (it is Map<*, *>) it as Map<String, Any?> else mapOf("type" to "text", "text" to it.toString()) }
    }
    return listOf(mapOf("type" to "text", "text" to payload))
}

private fun legacyCustomTypePayloadToParts(payload: String): List<LMPart> {
    val parsed = parseLegacyPayload(payload)
    if (parsed !is List<*>) {
        return listOf(LMTextPart(text = payload))
    }
    return parsed.map { legacyContentBlockToLmPart(it) }
}

private fun parseLegacyPayload(payload: String): Any? {
    return try {
        Json { ignoreUnknownKeys = true }.parseToJsonElement(payload).let { it }
    } catch (_: Exception) {
        try {
            parseDoublyQuotedJson(payload)
        } catch (_: Exception) {
            null
        }
    }
}

private fun parseDoublyQuotedJson(value: String): JsonElement {
    return Json { ignoreUnknownKeys = true }.parseToJsonElement("\"$value\"")
}

private fun legacyContentBlockToLmPart(block: Any?): LMPart {
    if (block !is Map<*, *>) return LMTextPart(text = block.toString())
    val blockType = block["type"] as? String
    return when (blockType) {
        "text" -> LMTextPart(text = (block["text"] as? String) ?: "")
        "image_url" -> {
            val imageUrl = (block["image_url"] as? Map<*, *>) ?: emptyMap<String, Any?>()
            val source = (imageUrl["url"] as? String) ?: imageUrl.toString()
            if (source.startsWith("data:") && "," in source) {
                val (mediaType, data) = splitDataUri(source)
                LMImagePart(data = data, mediaType = mediaType)
            } else {
                LMImagePart(url = source)
            }
        }
        "input_audio" -> {
            val audio = (block["input_audio"] as? Map<*, *>) ?: emptyMap<String, Any?>()
            LMAudioPart(
                data = audio["data"] as? String ?: "",
                mediaType = "audio/${audio["format"] as? String ?: "wav"}"
            )
        }
        "file" -> {
            val file = (block["file"] as? Map<*, *>) ?: emptyMap<String, Any?>()
            if (file["file_data"] != null) {
                val (mediaType, data) = splitDataUri(file["file_data"] as String)
                LMBinaryPart(
                    data = data, mediaType = mediaType, filename = file["filename"] as? String,
                    metadata = mapOf("legacy_content_block" to block)
                )
            } else {
                LMBinaryPart(
                    fileId = file["file_id"] as? String ?: "",
                    filename = file["filename"] as? String,
                    metadata = mapOf("legacy_content_block" to block)
                )
            }
        }
        "document" -> {
            val source = block["source"]
            LMDocumentPart(
                source = if (source is Map<*, *>) source as Map<String, Any?> else mapOf("type" to "text", "data" to source.toString()),
                citations = (block["citations"] as? Map<String, Any?>) ?: mapOf("enabled" to true),
                title = block["title"] as? String,
                context = block["context"] as? String
            )
        }
        else -> LMTextPart(text = "", metadata = mapOf("legacy_content_block" to block))
    }
}

private fun splitDataUri(value: String): Pair<String, String> {
    if (value.startsWith("data:") && "," in value) {
        val (header, data) = value.split(",", limit = 2)
        val mediaType = header.removePrefix("data:").split(";")[0]
        return mediaType to data
    }
    return "application/octet-stream" to value
}
