package dspy.adapters.utils

import dspy.adapters.types.Type
import kotlin.reflect.full.isSubclassOf
import dspy.signatures.FieldInfo
import dspy.signatures.InputField
import dspy.signatures.OutputField
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlin.reflect.KClass

private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
    isLenient = true
    prettyPrint = false
}

/**
 * Formats the specified value so that it can be serialized as a JSON string.
 */
fun serializeForJson(value: Any?): JsonElement {
    return when (value) {
        null -> JsonNull
        is String -> JsonPrimitive(value)
        is Number -> JsonPrimitive(value)
        is Boolean -> JsonPrimitive(value)
        is Enum<*> -> JsonPrimitive(value.name)
        is List<*> -> kotlinx.serialization.json.buildJsonArray {
            for (item in value) add(serializeForJson(item))
        }
        is Map<*, *> -> buildJsonObject {
            for ((k, v) in value) put(k.toString(), serializeForJson(v))
        }
        is Set<*> -> kotlinx.serialization.json.buildJsonArray {
            for (item in value) add(serializeForJson(item))
        }
        else -> JsonPrimitive(value.toString())
    }
}

/**
 * Deserializes a JsonElement back to a Kotlin value.
 */
fun deserializeFromJson(element: JsonElement): Any? {
    return when (element) {
        is JsonNull -> null
        is JsonPrimitive -> {
            if (element.isString) element.content
            else {
                when (element.content) {
                    "true" -> true
                    "false" -> false
                    else -> element.content.toIntOrNull() ?: element.content.toDoubleOrNull() ?: element.content
                }
            }
        }
        is kotlinx.serialization.json.JsonArray -> element.map { deserializeFromJson(it) }
        is kotlinx.serialization.json.JsonObject -> element.mapValues { (_, v) -> deserializeFromJson(v) }
    }
}

/**
 * Formats the value of the specified field according to the field's DSPy type and annotation.
 */
fun formatFieldValue(fieldInfo: FieldInfo, value: Any?): String {
    var stringValue: String? = null
    if (value is List<*> && fieldInfo.annotation == String::class) {
        stringValue = _formatInputListFieldValue(value)
    } else {
        val jsonableValue = serializeForJson(value)
        stringValue = when (jsonableValue) {
            is Map<*, *>, is List<*> -> jsonableValue.toString()
            else -> jsonableValue.toString()
        }
    }
    return stringValue ?: ""
}

/**
 * Translates a field type into a descriptive string for the field structure section.
 * Takes only the FieldInfo parameter (field name is inferred from field.name).
 */
fun translateFieldType(field: FieldInfo): String {
    val fieldType = field.annotation
    val fieldTypeStr = getAnnotationName(fieldType)
    return "{$field.name}: Type: $fieldTypeStr"
}

/**
 * Finds the enum member corresponding to the specified identifier.
 */
fun <T : Enum<T>> findEnumMember(enumClass: KClass<T>, identifier: String): T? {
    enumClass.java.enumConstants?.firstOrNull { it.name == identifier }?.let { return it }
    enumClass.java.enumConstants?.firstOrNull { it.toString() == identifier }?.let { return it }
    return null
}

/**
 * Parses a string value based on the given type annotation.
 */
fun parseValue(value: String, annotation: KClass<*>): Any? {
    val trimmedValue = value.trim()
    return when (annotation) {
        String::class -> trimmedValue
        Int::class -> trimmedValue.toIntOrNull()
        Long::class -> trimmedValue.toLongOrNull()
        Double::class -> trimmedValue.toDoubleOrNull()
        Float::class -> trimmedValue.toFloatOrNull()
        Boolean::class -> trimmedValue.lowercase().let {
            if (it == "true") true else if (it == "false") false else null
        }
        else -> try {
            json.parseToJsonElement(trimmedValue)
        } catch (e: Exception) {
            trimmedValue
        }
    }
}

/**
 * Gets a human-readable name for the given type annotation.
 */
fun getAnnotationName(annotation: KClass<*>): String {
    return when {
        annotation == Type::class -> "str"
        annotation.isSubclassOf(kotlin.Enum::class) -> annotation.simpleName ?: "Enum"
        else -> annotation.simpleName ?: annotation.qualifiedName ?: "Any"
    }
}

/**
 * Gets a description string for the given list of fields.
 */
fun getFieldDescriptionString(fields: List<out FieldInfo>): String {
    return fields.mapIndexed { idx, field ->
        val fieldMessage = buildString {
            append("${idx + 1}. `${field.name}`")
            append(" (${getAnnotationName(field.annotation)})")
            val desc = field.desc?.takeIf { it.isNotEmpty() }?.let { ": $it" } ?: ""
            append(desc)
            if (field.constraints.isNotEmpty()) {
                append("\nConstraints: ${field.constraints.joinToString("; ")}")
            }
        }
        fieldMessage
    }.joinToString("\n")
}

/**
 * Formats the value of an input field of type list.
 */
private fun _formatInputListFieldValue(value: List<*>): String {
    if (value.isEmpty()) return "N/A"
    if (value.size == 1) return _formatBlob(value[0]?.toString() ?: "")
    return value.mapIndexed { idx, txt ->
        "[$idx] ${_formatBlob(txt?.toString() ?: "")}"
    }.joinToString("\n")
}

/**
 * Formats the specified text blobs so that an LM can parse it correctly within a list
 * of multiple text blobs.
 */
private fun _formatBlob(blob: String): String {
    if (!blob.contains("\n") && !blob.contains("«") && !blob.contains("»")) {
        return "«$blob»"
    }
    val modifiedBlob = blob.replace("\n", "\n    ")
    return "«««\n    $modifiedBlob\n»»»"
}

/**
 * Returns the DSPy field type ("input" or "output") for the given field.
 */
fun getDspyFieldType(field: FieldInfo): String {
    return if (field is InputField) "input" else "output"
}

/**
 * Returns the specified string quoted for inclusion in a literal type annotation.
 */
fun quotedStringForLiteralTypeAnnotation(s: String): String {
    val hasSingle = s.contains('\'')
    val hasDouble = s.contains('"')
    return when {
        hasSingle && !hasDouble -> "\"$s\""
        hasDouble && !hasSingle -> "'$s'"
        hasSingle && hasDouble -> "'${s.replace("'", "\\'")}'"
        else -> "'$s'"
    }
}
