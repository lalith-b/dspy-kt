package dspy.adapters.types

import kotlin.reflect.KClass
import kotlinx.serialization.Serializable

/**
 * Code type in DSPy.
 *
 * This type is useful for code generation and code analysis.
 */
@Serializable
data class Code(
    val code: String,
    val language: String = "python"
) : Type() {
    override fun format(): String {
        return code
    }

    companion object {
        fun description(language: String = "python"): String {
            return """Code represented in a string, specified in the `code` field. If this is an output field, the code
field should follow the markdown code block format, e.g. 
```${language.lowercase()}
{code}
```
Programming language: $language""".trimIndent()
        }

        fun createWithLanguage(language: String): KClass<out Code> {
            return Code::class // Kotlin generics don't support runtime parameterization
        }
    }

    override fun toString(): String {
        return code
    }
}

private val codeBlockRegex = Regex("````?[^\\n]*\\n(.*?)```", RegexOption.DOT_MATCHES_ALL)
private val simpleCodeBlockRegex = Regex("```(.*?)```", RegexOption.DOT_MATCHES_ALL)

internal fun filterCode(code: String): String {
    val match = codeBlockRegex.find(code)
    if (match != null) return match.groupValues[1].trim()
    val simpleMatch = simpleCodeBlockRegex.find(code)
    if (simpleMatch != null) return simpleMatch.groupValues[1].trim()
    return code
}
