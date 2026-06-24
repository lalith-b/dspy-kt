package dspy.adapters.types

import kotlinx.serialization.Serializable

/**
 * A document type for providing content that can be cited by language models.
 *
 * This type represents documents that can be passed to language models for citation-enabled
 * responses, particularly useful with Anthropic's Citations API.
 */
@Serializable
data class Document(
    val data: String,
    val title: String? = null,
    val mediaType: String = "text/plain",
    val context: String? = null
) : Type() {
    override fun format(): List<Map<String, Any?>> {
        val documentBlock = mutableMapOf<String, Any?>(
            "type" to "document",
            "source" to mapOf(
                "type" to "text",
                "media_type" to mediaType,
                "data" to data
            ),
            "citations" to mapOf("enabled" to true)
        )
        title?.let { documentBlock["title"] = it }
        context?.let { documentBlock["context"] = it }
        return listOf(documentBlock)
    }

    companion object {
        fun description(): String {
            return "A document containing text content that can be referenced and cited. Include the full text content and optionally a title for proper referencing."
        }
    }

    override fun toString(): String {
        val titlePart = if (title != null) "'$title': " else ""
        return "Document($titlePart${data.length} chars)"
    }
}
