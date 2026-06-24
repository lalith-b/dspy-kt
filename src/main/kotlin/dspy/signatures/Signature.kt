package dspy.signatures

import kotlin.reflect.KClass

open class Signature(
    val instruction: String = "",
    val inputFields: List<InputField> = emptyList(),
    val outputFields: List<OutputField> = emptyList(),
) {
    val instructions: String get() = instruction

    override fun toString(): String {
        return "Signature(instruction='$instruction', inputFields=${inputFields.map { it.name }}, outputFields=${outputFields.map { it.name }})"
    }

    /**
     * Create a new Signature with updated instructions.
     */
    fun withInstructions(instructions: String): Signature {
        return Signature(
            instruction = instructions,
            inputFields = this.inputFields,
            outputFields = this.outputFields,
        )
    }

    /**
     * Create a copy of this Signature.
     */
    fun copy(
        instruction: String = this.instruction,
        inputFields: List<InputField> = this.inputFields,
        outputFields: List<OutputField> = this.outputFields,
    ): Signature {
        return Signature(instruction, inputFields, outputFields)
    }

    fun dumpState(): Map<String, Any?> {
        return mapOf(
            "instructions" to instruction,
            "fields" to (inputFields + outputFields).map { field ->
                mapOf(
                    "prefix" to (field.prefix ?: "${field.name.replace("_", " ")}:"),
                    "description" to (field.desc ?: "${field.name}"),
                )
            },
        )
    }

    fun loadState(state: Map<String, Any?>) {
    }

    /**
     * Append an input field to this signature, returning a new Signature.
     */
    fun append(fieldName: String, field: InputField): Signature {
        val updated = if (field.name.isEmpty()) field.copy(name = fieldName) else field
        return Signature(
            instruction = this.instruction,
            inputFields = this.inputFields + updated,
            outputFields = this.outputFields,
        )
    }

    /**
     * Prepend an output field to this signature, returning a new Signature.
     */
    fun prepend(fieldName: String, field: OutputField): Signature {
        val updated = if (field.name.isEmpty()) field.copy(name = fieldName) else field
        return Signature(
            instruction = this.instruction,
            inputFields = this.inputFields,
            outputFields = listOf(updated) + this.outputFields,
        )
    }

    companion object {
        fun makeSignature(
            fields: Map<String, Pair<KClass<*>, Any>>,
            instructions: String = ""
        ): Signature {
            val inputFields = mutableListOf<InputField>()
            val outputFields = mutableListOf<OutputField>()
            for ((name, pair) in fields) {
                val (annotation, fieldAnnotation) = pair
                val fieldInfo = when (fieldAnnotation) {
                    is InputField -> InputField(name = name, annotation = annotation)
                    is OutputField -> OutputField(name = name, annotation = annotation)
                    else -> InputField(name = name, annotation = annotation)
                }
                if (fieldInfo is InputField) {
                    inputFields.add(fieldInfo)
                } else {
                    outputFields.add(fieldInfo as OutputField)
                }
            }
            return Signature(instruction = instructions, inputFields = inputFields, outputFields = outputFields)
        }

        fun fromString(spec: String): Signature {
            val parts = spec.split("->")
            require(parts.size == 2) { "Signature format: input1,input2 -> output1,output2" }
            val inputs = parts[0].split(",").map { it.trim() }.filter { it.isNotEmpty() }.map {
                InputField(name = it, prefix = "${it.replace("_", " ")}:", desc = it)
            }
            val outputs = parts[1].split(",").map { it.trim() }.filter { it.isNotEmpty() }.map {
                OutputField(name = it, prefix = "${it.replace("_", " ")}:", desc = it)
            }
            return Signature(
                instruction = "Given inputs, produce outputs.",
                inputFields = inputs,
                outputFields = outputs,
            )
        }
    }
}

fun inferPrefix(name: String): String {
    return name
        .replace("_", " ")
        .replace(Regex("([a-z])([A-Z])")) { "${it.groupValues[1]} ${it.groupValues[2]}" }
        .replace(Regex("([A-Z]+)([A-Z][a-z])")) { "${it.groupValues[1]} ${it.groupValues[2]}" }
        .split(" ").joinToString(" ") { it.lowercase().replaceFirstChar { c -> c.uppercaseChar() } }
}
