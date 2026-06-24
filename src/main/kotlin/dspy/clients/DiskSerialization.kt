package dspy.clients

/**
 * Restricted pickle deserialization for disk cache.
 *
 * Provides a restricted unpickler that only allows trusted module prefixes
 * and explicit safe_types.
 */
object DiskSerialization {
    private val trustedModulePrefixes = listOf("litellm.types.", "openai.types.")
    private val numpyAllowed = setOf(
        "numpy.dtype",
        "numpy.ndarray",
        "numpy._core.numeric._frombuffer",
        "numpy.core.numeric._frombuffer",
        "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray._reconstruct",
        "_codecs.encode"
    )

    /**
     * Check if a type is in the safe_types allowlist.
     */
    fun isTypeAllowed(module: String, name: String, allowed: Set<Pair<String, String>>): Boolean {
        if (trustedModulePrefixes.any { module.startsWith(it) }) return true
        if ("$module.$name" in numpyAllowed) return true
        return Pair(module, name) in allowed
    }

    /**
     * Deserialize a cached value with restricted unpickling.
     */
    fun restrictedLoad(data: ByteArray, allowed: Set<Pair<String, String>> = emptySet()): Any? {
        // Kotlin equivalent of restricted pickle deserialization
        // In practice, this would use kotlinx.serialization instead of pickle
        try {
            return kotlinx.serialization.json.Json.decodeFromString<kotlinx.serialization.json.JsonElement>(
                data.decodeToString()
            )
        } catch (e: Exception) {
            throw DeserializationError("Corrupt cache entry: ${e.message}")
        }
    }
}

/**
 * Raised when a cached value cannot be deserialized.
 */
class DeserializationError(message: String) : Exception(message)
