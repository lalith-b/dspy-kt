package dspy.signatures

import kotlin.reflect.KClass

sealed class FieldInfo {
    abstract val name: String
    open val prefix: String? = null
    open val desc: String? = null
    open val example: Any? = null
    open val format: ((Any?) -> Any?)? = null
    open val parser: ((String) -> Any?)? = null
    open val constraints: List<String> = emptyList()
    open val isTypeUndefined: Boolean = false
    open val annotation: KClass<*> = String::class
}

data class InputField(
    override val name: String = "",
    override val prefix: String? = null,
    override val desc: String? = null,
    override val example: Any? = null,
    override val format: ((Any?) -> Any?)? = null,
    override val parser: ((String) -> Any?)? = null,
    override val constraints: List<String> = emptyList(),
    override val isTypeUndefined: Boolean = false,
    override val annotation: KClass<*> = String::class,
) : FieldInfo()

data class OutputField(
    override val name: String = "",
    override val prefix: String? = null,
    override val desc: String? = null,
    override val example: Any? = null,
    override val format: ((Any?) -> Any?)? = null,
    override val parser: ((String) -> Any?)? = null,
    override val constraints: List<String> = emptyList(),
    override val isTypeUndefined: Boolean = false,
    override val annotation: KClass<*> = String::class,
) : FieldInfo()

fun InputField(desc: String? = null, prefix: String? = null, format: ((Any?) -> Any?)? = null, parser: ((String) -> Any?)? = null, gt: Int? = null, ge: Int? = null, lt: Int? = null, le: Int? = null, min_length: Int? = null, max_length: Int? = null): InputField {
    val constraints = mutableListOf<String>()
    if (gt != null) constraints.add("greater than: $gt")
    if (ge != null) constraints.add("greater than or equal to: $ge")
    if (lt != null) constraints.add("less than: $lt")
    if (le != null) constraints.add("less than or equal to: $le")
    if (min_length != null) constraints.add("minimum length: $min_length")
    if (max_length != null) constraints.add("maximum length: $max_length")
    return InputField(name = "", prefix = prefix, desc = desc, format = format, parser = parser, constraints = constraints)
}

fun OutputField(desc: String? = null, prefix: String? = null, format: ((Any?) -> Any?)? = null, parser: ((String) -> Any?)? = null, gt: Int? = null, ge: Int? = null, lt: Int? = null, le: Int? = null, min_length: Int? = null, max_length: Int? = null): OutputField {
    val constraints = mutableListOf<String>()
    if (gt != null) constraints.add("greater than: $gt")
    if (ge != null) constraints.add("greater than or equal to: $ge")
    if (lt != null) constraints.add("less than: $lt")
    if (le != null) constraints.add("less than or equal to: $le")
    if (min_length != null) constraints.add("minimum length: $min_length")
    if (max_length != null) constraints.add("maximum length: $max_length")
    return OutputField(name = "", prefix = prefix, desc = desc, format = format, parser = parser, constraints = constraints)
}
