package dspy

data class Image(
    val url: String? = null,
    val path: String? = null,
    val data: String? = null,
)

data class History(
    val messages: List<Map<String, Any>> = emptyList(),
) {
    fun toDict(): Map<String, Any> = mapOf("messages" to messages)
}
