package dspy.primitives

/**
 * Abstract base class for RLM sandbox-serializable types.
 *
 * Types that subclass SandboxSerializable can be injected into the REPL
 * environment used by RLM. Subclasses implement four abstract methods
 * describing how a value enters and appears inside the sandbox.
 */
abstract class SandboxSerializable {
    /**
     * Python statements (usually imports) executed once in the sandbox.
     * The returned text is also surfaced to the LLM in the variable
     * description, so the model knows which names are in scope.
     */
    abstract fun sandboxSetup(): String

    /**
     * Serialize the value to text bytes or binary bytes.
     */
    abstract fun toSandbox(): String

    /**
     * The assignment expression in the sandbox.
     */
    abstract fun sandboxAssignment(name: String): String

    /**
     * The description of the variable for the LLM.
     */
    abstract fun variableDescription(name: String): String

    /**
     * Build a REPL variable from this serializable.
     */
    fun toREPLVariable(name: String): REPLVariable {
        return REPLVariable(
            name = name,
            type = this::class.simpleName ?: "Any",
            value = toSandbox(),
            setup = sandboxSetup(),
            assignment = sandboxAssignment(name),
            description = variableDescription(name)
        )
    }
}

/**
 * Build a REPL variable without subclassing SandboxSerializable.
 */
fun buildREPLVariable(
    name: String,
    type: String,
    value: String,
    setup: String = "",
    assignment: String? = null,
    description: String? = null
): REPLVariable {
    return REPLVariable(
        name = name,
        type = type,
        value = value,
        setup = setup,
        assignment = assignment ?: "$name = $value",
        description = description ?: "Variable $name of type $type"
    )
}

/**
 * REPL variable for sandbox serialization.
 */
data class REPLVariable(
    val name: String,
    val type: String,
    val value: String,
    val setup: String,
    val assignment: String,
    val description: String
)
