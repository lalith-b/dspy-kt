package dspy.experimental

/**
 * Citations type for experimental citation support in DSPy.
 */
data class ExperimentalCitations(
    val items: List<Citation> = emptyList()
) {
    companion object {
        fun fromList(items: List<Citation>): ExperimentalCitations = ExperimentalCitations(items)
    }
}

data class Citation(
    val text: String,
    val title: String? = null,
    val url: String? = null,
    val authors: List<String>? = null,
    val year: Int? = null,
    val type: String? = null
)

/**
 * Document type for experimental citation support.
 */
data class CitationDocument(
    val title: String,
    val content: String,
    val metadata: Map<String, Any?> = emptyMap(),
    val citations: List<Citation> = emptyList()
)
