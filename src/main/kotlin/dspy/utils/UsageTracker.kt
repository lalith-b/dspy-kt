package dspy.utils

import dspy.signatures.Signature

/**
 * Tracks LM usage data within a context.
 */
class UsageTracker {
    // Map of LM name to list of usage entries
    val usageData: MutableMap<String, MutableList<Map<String, Any?>>> = mutableMapOf()

    fun addUsage(lm: String, usageEntry: Map<String, Any?>) {
        if (usageEntry.isNotEmpty()) {
            usageData.getOrPut(lm) { mutableListOf() }.add(usageEntry)
        }
    }

    /** Calculate total tokens from all tracked usage. */
    fun getTotalTokens(): Map<String, Map<String, Any?>> {
        val totalUsageByLm = mutableMapOf<String, Map<String, Any?>>()
        for ((lm, usageEntries) in usageData) {
            var totalUsage = emptyMap<String, Any?>()
            for (entry in usageEntries) {
                totalUsage = mergeUsageEntries(totalUsage, entry)
            }
            totalUsageByLm[lm] = totalUsage
        }
        return totalUsageByLm
    }

    private fun mergeUsageEntries(
        entry1: Map<String, Any?>?,
        entry2: Map<String, Any?>?
    ): Map<String, Any?> {
        if (entry1.isNullOrEmpty()) return entry2?.toMutableMap() ?: emptyMap()
        if (entry2.isNullOrEmpty()) return entry1.toMutableMap()

        val result = entry2.toMutableMap()
        for ((k, v) in entry1) {
            val current = result[k]
            if (v is Map<*, *> || current is Map<*, *>) {
                @Suppress("UNCHECKED_CAST")
                result[k] = mergeUsageEntries(
                    current as? Map<String, Any?>,
                    v as? Map<String, Any?>
                )
            } else if (current != null || v != null) {
                val num1 = (current as? Number)?.toDouble() ?: 0.0
                val num2 = (v as? Number)?.toDouble() ?: 0.0
                result[k] = num1 + num2
            }
        }
        return result
    }
}

/**
 * Context manager equivalent for tracking LM usage.
 * Returns a UsageTracker and sets it in settings.
 */
inline fun <R> trackUsage(block: UsageTracker.() -> R): UsageTracker {
    val tracker = UsageTracker()
    // Note: In Kotlin we use inline functions with receiver instead of context managers
    tracker.block()
    return tracker
}
