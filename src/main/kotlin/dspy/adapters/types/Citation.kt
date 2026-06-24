package dspy.adapters.types

import kotlinx.serialization.Serializable

/**
 * Individual citation with character location information.
 */
@Serializable
data class CitationItem(
    val type: String = "char_location",
    val citedText: String,
    val documentIndex: Int,
    val documentTitle: String? = null,
    val startCharIndex: Int,
    val endCharIndex: Int,
    val supportedText: String? = null
) {
    fun format(): Map<String, Any?> {
        val citationDict = mapOf<String, Any?>(
            "type" to type,
            "cited_text" to citedText,
            "document_index" to documentIndex,
            "start_char_index" to startCharIndex,
            "end_char_index" to endCharIndex
        ).toMutableMap()
        documentTitle?.let { citationDict["document_title"] = it }
        supportedText?.let { citationDict["supported_text"] = it }
        return citationDict
    }
}

/**
 * Citations extracted from an LM response with source references.
 *
 * This type represents citations returned by language models that support
 * citation extraction, particularly Anthropic's Citations API through LiteLLM.
 */
@Serializable
data class Citations(
    val citations: List<CitationItem> = emptyList()
) : Type(), Iterable<CitationItem> {

    companion object {
        fun fromDictList(citationsDicts: List<Map<String, Any?>>): Citations {
            val citations = citationsDicts.map { item ->
                CitationItem(
                    citedText = item["cited_text"] as String,
                    documentIndex = item["document_index"] as Int,
                    documentTitle = item["document_title"] as? String,
                    startCharIndex = item["start_char_index"] as Int,
                    endCharIndex = item["end_char_index"] as Int,
                    supportedText = item["supported_text"] as? String
                )
            }
            return Citations(citations = citations)
        }

        fun description(): String {
            return "Citations with quoted text and source references. Include the exact text being cited and information about its source."
        }

        fun adaptToNativeLmFeature(signature: Any, fieldName: String, lm: Any, lmKwargs: MutableMap<String, Any?>): Any {
            val model = (lm as? Any)?.let { (it as? Map<*, *>)?.get("model") } as? String
            if (model != null && model.startsWith("anthropic/")) {
                return signature // delete field
            }
            return signature
        }

        fun isStreamable(): Boolean = true

        fun parseLmResponse(response: Any): Citations? {
            if (response is Map<*, *>) {
                val citationsData = response["citations"] as? List<*>
                if (citationsData != null) {
                    return fromDictList(citationsData.map { it as Map<String, Any?> })
                }
            }
            return null
        }
    }

    override fun format(): List<Map<String, Any?>> {
        return citations.map { it.format() }
    }

    override fun iterator(): Iterator<CitationItem> = citations.iterator()

    operator fun get(index: Int): CitationItem = citations[index]

    override fun toString(): String {
        return "Citations(${citations.size} citations)"
    }
}
