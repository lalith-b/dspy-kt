package dspy.adapters.types

/**
 * BAML adapter type marker for DSPy.
 */
class BamlAdapterType(
    val name: String,
    val schema: Map<String, Any?> = emptyMap()
) {
    override fun toString(): String = "BamlAdapterType($name)"
}
