package dspy.utils

/**
 * Experimental API decorator.
 *
 * Marks APIs as experimental by adding a notice to their documentation.
 * Faithful port of `dspy/utils/annotation.py`.
 */

/**
 * Annotation to mark a class or function as experimental.
 *
 * Example:
 * ```kotlin
 * @Experimental("3.0.0")
 * class MyClass { }
 * ```
 */
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
@Retention(AnnotationRetention.BINARY)
annotation class Experimental(
    /**
     * The version in which the API was introduced as experimental.
     */
    val version: String = "",
)

/**
 * Get the experimental notice text for a class or function.
 */
fun getExperimentalNotice(apiName: String, apiType: String, version: String? = null): String {
    val versionText = if (version?.isNotBlank() == true) " (introduced in v$version)" else ""
    return "Experimental: This $apiType may change or be removed in a future release without warning.$versionText"
}

/**
 * Check if an annotation is experimental.
 */
fun isExperimental(annotation: Annotation): Boolean {
    return annotation is Experimental
}

/**
 * Get the experimental version from an annotation.
 */
fun getExperimentalVersion(annotation: Annotation): String? {
    return if (annotation is Experimental) annotation.version.takeIf { it.isNotBlank() } else null
}

/**
 * Helper to get the minimum indent of a docstring.
 *
 * Based on the assumption that the closing triple quote for multiline comments
 * must be on a new line (ruff rule D209).
 */
fun getMinIndentOfDocstring(docstringStr: String): String {
    if (docstringStr.isBlank() || !docstringStr.contains('\n')) {
        return ""
    }
    val lines = docstringStr.split('\n')
    val lastLine = lines.last()
    val match = Regex("^\\s*").find(lastLine)
    return match?.value ?: ""
}
