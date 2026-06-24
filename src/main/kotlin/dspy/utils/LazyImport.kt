package dspy.utils

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock

/**
 * Lazy import helper for optional dependencies.
 *
 * Port of `dspy/utils/lazy_import.py`
 *
 * Note: Kotlin/JVM has a different module system than Python.
 * This provides lazy class loading via reflection for optional
 * dependencies that may not be on the classpath.
 *
 * Example:
 * ```kotlin
 * val numpy = require("numpy")
 * if (numpy.isAvailable) {
 *     val clazz = numpy.load()
 *     // Use clazz via reflection
 * }
 * ```
 */

/**
 * Detect if DSPy distribution is available.
 */
private fun detectDspyDist(): String {
    return "dspy-kt"
}

/**
 * Install hints for optional dependencies.
 */
private val INSTALL_HINTS = mapOf(
    "optuna" to "optuna",
    "mcp" to "mcp",
    "langchain_core" to "langchain",
    "weaviate" to "weaviate",
    "anthropic" to "anthropic",
    "numpy" to "numpy",
    "litellm" to "litellm"
)

/**
 * Per-module locks for thread-safe lazy loading.
 */
private val lazyModuleLocks = ConcurrentHashMap.newKeySet<String>()

/**
 * Check if a module is available on the classpath.
 */
fun isAvailable(module: String): Boolean {
    return try {
        Class.forName(module) != null
    } catch (e: ClassNotFoundException) {
        false
    }
}

/**
 * Lazy module proxy.
 */
class LazyModule(private val moduleName: String, private val feature: String? = null) {
    private val lock = ReentrantLock()
    private var loaded: Class<*>? = null
    
    /**
     * Check if the module is available.
     */
    val isAvailable: Boolean
        get() = try {
            Class.forName(moduleName) != null
        } catch (e: ClassNotFoundException) {
            false
        }
    
    /**
     * Load the module class.
     */
    fun load(): Class<*> {
        lock.lock()
        try {
            if (loaded != null) {
                return loaded!!
            }
            val clazz = Class.forName(moduleName)
            loaded = clazz
            return clazz
        } catch (e: ClassNotFoundException) {
            val top = moduleName.split(".").first()
            val feat = feature ?: "this feature"
            val ext = INSTALL_HINTS[top] ?: top
            val dist = detectDspyDist()
            throw ImportError(
                "$top is required to use $feat. " +
                "Install with `pip install ${dist}[$ext]` or `pip install $top`."
            )
        } finally {
            lock.unlock()
        }
    }
    
    /**
     * Get a field from the loaded module.
     */
    fun getField(fieldName: String): Any {
        val clazz = load()
        val field = clazz.getField(fieldName)
        return field.get(null)
    }
    
    /**
     * Call a static method on the loaded module.
     */
    fun callStatic(methodName: String, vararg args: Any?): Any {
        val clazz = load()
        val method = clazz.getMethod(methodName, *args.map { it!!::class.java }.toTypedArray())
        return method.invoke(null, *args)
    }
}

/**
 * Require a module (lazy loading).
 */
fun require(moduleName: String, extra: String? = null, feature: String? = null): LazyModule {
    return LazyModule(moduleName, feature)
}

/**
 * Exception for missing modules.
 */
class ImportError(message: String) : RuntimeException(message)
