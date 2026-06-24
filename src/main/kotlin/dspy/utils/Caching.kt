package dspy.utils

import java.io.File

/** Default cache directory: ~/.dspy_cache */
val DEFAULT_CACHE_DIR = File(System.getProperty("user.home"), ".dspy_cache").absolutePath

/** DSPy cache directory from env or default. */
val DSPY_CACHEDIR = System.getenv("DSPY_CACHEDIR") ?: DEFAULT_CACHE_DIR

/** Create a subdirectory in the DSPy cache directory. */
fun createSubdirInCachedir(subdir: String): String {
    val path = File(DSPY_CACHEDIR, subdir).absolutePath
    File(path).mkdirs()
    return path
}
