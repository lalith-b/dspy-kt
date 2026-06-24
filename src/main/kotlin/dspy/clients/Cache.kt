package dspy.clients

import dspy.core.types.LMResponse
import dspy.utils.Settings
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import java.io.File
import java.security.MessageDigest

class Cache(
    enableDiskCache: Boolean = false,
    enableMemoryCache: Boolean = true,
    diskCacheDir: String? = null,
    diskSizeLimitBytes: Long = 1024 * 1024 * 10,
    memoryMaxEntries: Int = 1_000_000,
) {
    private val enableDiskCache: Boolean = enableDiskCache
    private val enableMemoryCache: Boolean = enableMemoryCache
    private val diskCacheDir: String? = diskCacheDir
    private val memoryCache: MutableMap<String, Any> =
        if (enableMemoryCache) LRUMap(memoryMaxEntries) else mutableMapOf()
    private val diskCache: MutableMap<String, String> = mutableMapOf()
    private val lock = Mutex()

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        isLenient = true
    }

    init {
        if (enableDiskCache && diskCacheDir != null) {
            File(diskCacheDir).mkdirs()
        }
    }

    fun cacheKey(request: Map<String, Any?>, ignoredArgsForCacheKey: List<String> = emptyList()): String {
        val params = request.filterKeys { it !in ignoredArgsForCacheKey }
        val transformed = params.mapValues { (_, v) -> transformValue(v) }
        val jsonStr = transformed.toString()
        val md = MessageDigest.getInstance("SHA-256")
        return md.digest(jsonStr.toByteArray()).joinToString("") { "%02x".format(it) }
    }

    suspend fun get(
        request: Map<String, Any?>,
        ignoredArgsForCacheKey: List<String> = emptyList(),
    ): Any? {
        if (!enableMemoryCache && !enableDiskCache) return null
        return try {
            lock.withLock {
                val key = cacheKey(request, ignoredArgsForCacheKey)
                if (enableMemoryCache) {
                    memoryCache[key]?.let { return@let prepareCachedResponse(it) }
                }
                if (enableDiskCache) {
                    diskCache[key]?.let { diskValue ->
                        val response = deserializeFromDisk(diskValue)
                        if (response != null) {
                            if (enableMemoryCache) memoryCache[key] = response
                            return@let prepareCachedResponse(response)
                        }
                    }
                }
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    suspend fun put(
        request: Map<String, Any?>,
        value: Any,
        ignoredArgsForCacheKey: List<String> = emptyList(),
        enableMemoryCacheAtCallTime: Boolean = true,
    ) {
        val effectiveMemoryCache = enableMemoryCache && enableMemoryCacheAtCallTime
        if (!effectiveMemoryCache && !enableDiskCache) return
        try {
            lock.withLock {
                val key = cacheKey(request, ignoredArgsForCacheKey)
                if (effectiveMemoryCache) memoryCache[key] = value
                if (enableDiskCache) {
                    try {
                        val serialized = serializeToDisk(value)
                        diskCache[key] = serialized
                        diskCacheDir?.let { dir ->
                            File(dir, key).writeText(serialized)
                        }
                    } catch (e: Exception) { /* ignore */ }
                }
            }
        } catch (e: Exception) { /* ignore */ }
    }

    suspend fun resetMemoryCache() {
        if (!enableMemoryCache) return
        lock.withLock { memoryCache.clear() }
    }

    private fun prepareCachedResponse(response: Any): Any {
        return if (response is LMResponse) response.copy(cacheHit = true) else response
    }

    private fun transformValue(value: Any?): Any? {
        return when (value) {
            is Map<*, *> -> value.mapValues { (_, v) -> transformValue(v) }
            is List<*> -> value.map { transformValue(it) }
            is Enum<*> -> value.name
            null -> null
            else -> value
        }
    }

    private fun serializeToDisk(value: Any): String = value.toString()

    private fun deserializeFromDisk(serialized: String): Any? {
        return try {
            json.decodeFromString<JsonElement>(serialized)
        } catch (e: Exception) {
            null
        }
    }
}

private class LRUMap<V>(private val maxSize: Int) : LinkedHashMap<String, V>(16, 0.75f, true) {
    override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, V>?): Boolean = size > maxSize
}

fun <R> requestCache(
    cacheArgName: String? = null,
    ignoredArgsForCacheKey: List<String> = listOf("api_key", "api_base", "base_url"),
    enableMemoryCache: Boolean = true,
): (suspend (Map<String, Any?>) -> R) -> (suspend (Map<String, Any?>) -> R) {
    return { fn ->
        suspend fun wrapper(request: Map<String, Any?>): R {
            val cache = Settings.get("cache") as? Cache ?: Cache()
            val cachedResult = cache.get(request, ignoredArgsForCacheKey)
            if (cachedResult != null) {
                @Suppress("UNCHECKED_CAST")
                return cachedResult as R
            }
            val result = fn(request)
            cache.put(request, result as Any, ignoredArgsForCacheKey, enableMemoryCache)
            return result
        }
        ::wrapper
    }
}
