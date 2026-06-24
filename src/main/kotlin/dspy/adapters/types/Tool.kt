package dspy.adapters.types

class Tool(
    val func: () -> Any,
    var name: String? = null,
    var desc: String? = null,
    var args: Map<String, Any>? = null,
    var argTypes: Map<String, Any>? = null,
    var hasKwargs: Boolean = false,
) : Type() {
    operator fun invoke(kwargs: Map<String, Any> = emptyMap()): Any {
        return func()
    }

    fun formatAsLiteLLMFunctionCall(): Map<String, Any?> {
        return mapOf(
            "type" to "function",
            "function" to mapOf(
                "name" to (name ?: "unknown"),
                "description" to (desc ?: ""),
                "parameters" to (args ?: emptyMap<String, Any>())
            )
        )
    }

    override fun format(): String = toString()

    override fun toString(): String = "Tool($name, $desc)"
}
