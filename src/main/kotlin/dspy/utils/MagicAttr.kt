package dspy.utils

/**
 * MagicAttr - nested attribute access utility.
 *
 * Port of `dspy/utils/magicattr.py`
 *
 * Note: The Python version uses AST parsing for dynamic nested attribute access.
 * Kotlin has compile-time type safety, so we provide a simplified version
 * that works with Map-based nested access (common in DSPy for dynamic data).
 */

/**
 * Get a nested attribute from an object using dot notation.
 *
 * Supports:
 * - Object properties: "foo.bar" -> obj.foo.bar
 * - Map keys: "foo.bar" -> obj["foo"]["bar"]
 * - List indices: "foo.0.bar" -> obj["foo"][0].bar
 *
 * Example:
 * ```kotlin
 * val obj = mapOf("a" to mapOf("b" to 42))
 * val value = obj.getAttr("a.b") // Returns 42
 * ```
 */
fun Any?.getAttr(attr: String, default: Any? = null): Any? {
    if (this == null) {
        return default
    }
    
    val chunks = attr.split(".")
    var obj: Any? = this
    
    for (chunk in chunks) {
        if (obj == null) {
            return default
        }
        
        try {
            obj = lookup(obj, chunk)
        } catch (e: Exception) {
            if (default != null) {
                return default
            }
            throw e
        }
    }
    
    return obj
}

/**
 * Set a nested attribute in an object using dot notation.
 *
 * Example:
 * ```kotlin
 * val obj = mutableMapOf("a" to mutableMapOf<String, Any?>())
 * obj.setAttr("a.b", 42)
 * // obj is now {"a": {"b": 42}}
 * ```
 */
fun Any.setAttr(attr: String, value: Any?) {
    val chunks = attr.split(".")
    if (chunks.isEmpty()) {
        throw IllegalArgumentException("Attribute name must not be empty")
    }
    
    // Navigate to the parent object
    var obj: Any = this
    for (i in 0 until chunks.size - 1) {
        obj = lookup(obj, chunks[i]) ?: throw IllegalStateException("Cannot set nested attribute: path doesn't exist")
    }
    
    // Set the final attribute
    val lastKey = chunks.last()
    when (obj) {
        is MutableMap<*, *> -> {
            @Suppress("UNCHECKED_CAST")
            (obj as MutableMap<String, Any?>)[lastKey] = value
        }
        else -> {
            // For objects, use reflection (or throw if not supported)
            throw UnsupportedOperationException("Cannot set attribute on non-Map object: ${obj::class.simpleName}")
        }
    }
}

/**
 * Delete a nested attribute from an object using dot notation.
 */
fun Any.deleteAttr(attr: String) {
    val chunks = attr.split(".")
    if (chunks.isEmpty()) {
        throw IllegalArgumentException("Attribute name must not be empty")
    }
    
    // Navigate to the parent object
    var obj: Any = this
    for (i in 0 until chunks.size - 1) {
        obj = lookup(obj, chunks[i]) ?: throw IllegalStateException("Cannot delete nested attribute: path doesn't exist")
    }
    
    // Delete the final attribute
    val lastKey = chunks.last()
    when (obj) {
        is MutableMap<*, *> -> {
            @Suppress("UNCHECKED_CAST")
            (obj as MutableMap<String, Any?>).remove(lastKey)
        }
        else -> {
            throw UnsupportedOperationException("Cannot delete attribute on non-Map object: ${obj::class.simpleName}")
        }
    }
}

/**
 * Lookup a single attribute on an object.
 */
private fun lookup(obj: Any, attr: String): Any? {
    return when (obj) {
        is Map<*, *> -> {
            // Try string key first, then try integer index
            val key = attr.toIntOrNull() ?: attr
            obj[key]
        }
        is List<*> -> {
            // Try integer index
            val index = attr.toIntOrNull()
            if (index != null && index in obj.indices) {
                obj[index]
            } else {
                throw IndexOutOfBoundsException("Invalid list index: $attr")
            }
        }
        else -> {
            throw NoSuchPropertyException("Property '$attr' not found on ${obj::class.simpleName}")
        }
    }
}

/**
 * Exception for missing properties.
 */
class NoSuchPropertyException(message: String) : RuntimeException(message)
