package dspy.utils

import java.security.MessageDigest

/**
 * Hasher that accepts Kotlin objects as inputs.
 *
 * Vendored from the `datasets` package from Hugging Face (Apache 2.0).
 * Replaced xxhash with SHA-256 to remove C-extension dep.
 */
class Hasher {
    private val m: MessageDigest = MessageDigest.getInstance("SHA-256")

    @Suppress("unused")
    companion object {
        /** Return a hex digest for one or more byte chunks. */
        fun hashBytes(value: List<ByteArray>): String {
            val md = MessageDigest.getInstance("SHA-256")
            for (chunk in value) {
                md.update(chunk)
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        /** Return a hex digest for a single byte array. */
        fun hashBytes(value: ByteArray): String {
            val md = MessageDigest.getInstance("SHA-256")
            md.update(value)
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }

    /** Update the running digest with a typed object payload. */
    fun update(value: Any?) {
        val header = "==${value?.javaClass?.name ?: "null"}=="
        val valueHash = hash(value)
        m.update(header.toByteArray(Charsets.UTF_8))
        m.update(valueHash.toByteArray(Charsets.UTF_8))
    }

    /** Return the hexadecimal digest of the current hasher state. */
    fun hexdigest(): String {
        return m.digest().joinToString("") { "%02x".format(it) }
    }

    private fun hash(value: Any?): String {
        // Simple string-based serialization for hashing
        val serialized = when (value) {
            null -> "null"
            is String -> value
            is Number -> value.toString()
            is Boolean -> value.toString()
            is Collection<*> -> value.joinToString(",") { it?.toString() ?: "null" }
            is Map<*, *> -> value.entries.map { "${it.key}=${it.value}" }.joinToString(",")
            else -> value.toString()
        }
        return hashBytes(serialized.toByteArray(Charsets.UTF_8))
    }
}
