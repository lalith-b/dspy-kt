package dspy.primitives

/**
 * Error raised during code interpretation.
 *
 * This exception covers two distinct failure modes:
 * 1. **Execution errors**: The sandbox ran user code that failed (NameError, TypeError, etc.)
 * 2. **Protocol errors**: Communication between host and sandbox failed (malformed JSON, etc.)
 *
 * Port of `dspy/primitives/code_interpreter.py`
 */
open class CodeInterpreterError(message: String) : RuntimeException(message)

/**
 * Returned by [CodeInterpreter.execute] when SUBMIT() is called.
 *
 * Signals that the code execution loop should terminate and return
 * the contained output to the caller.
 */
class FinalOutput(
    val output: Any?
) {
    override fun toString(): String = "FinalOutput(${output?.let { it.toString().replace("\"", "\\\"") } ?: "null"})"

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FinalOutput) return false
        return output == other.output
    }

    override fun hashCode(): Int = output?.hashCode() ?: 0
}

/**
 * Protocol for code execution environments (interpreters).
 *
 * Implementations must provide:
 * - [tools]: Tools available for interpreter code to call
 * - [start()]: Initialize the interpreter and allocate resources
 * - [execute()]: Run code and return results
 * - [shutdown()]: Clean up resources
 *
 * The interpreter maintains state across [execute] calls within a session,
 * allowing variables defined in one call to be used in subsequent calls.
 *
 * Lifecycle:
 * 1. Create instance (config only, no resources allocated)
 * 2. [start] - Initialize interpreter (explicit) or let [execute] do it (lazy)
 * 3. [execute] - Run code (can be called many times)
 * 4. [shutdown] - Release resources
 *
 * Example implementations:
 * - LocalInterpreter: Deno/Pyodide WASM interpreter (local)
 * - MockInterpreter: Scriptable responses for testing
 *
 * For interpreter pooling, call [start] to pre-warm instances, then
 * distribute [execute] calls across the pool.
 */
interface CodeInterpreter {
    /**
     * Tools available for interpreter code to call.
     *
     * Tools are host-side functions that can be invoked from within the
     * interpreter. Each tool accepts keyword arguments and returns a string.
     */
    val tools: Map<String, (Map<String, Any?>) -> String>

    /**
     * Initialize the interpreter and allocate resources.
     *
     * This method prepares the interpreter for code execution. It can be called
     * explicitly to pre-warm the interpreter, or implementations may call it
     * lazily on first [execute].
     *
     * Calling [start] multiple times should be safe (idempotent).
     */
    fun start()

    /**
     * Execute Python code and return the result.
     *
     * @param code Python code to execute
     * @param variables Variables to inject into the namespace before execution
     * @return One of:
     *   - [FinalOutput]: If SUBMIT() was called in code
     *   - String: Captured stdout from print() statements
     *   - List: Multiple output lines
     *   - null: If no output was produced
     * @throws CodeInterpreterError On runtime errors (undefined vars, tool failures, etc.)
     * @throws IllegalArgumentException On invalid Python syntax
     *
     * State persists across calls. Variables defined in one execute()
     * call are available in subsequent calls until shutdown().
     */
    fun execute(code: String, variables: Map<String, Any?>? = null): Any?

    /**
     * Release resources and terminate the interpreter session.
     *
     * After shutdown(), the interpreter should not be used again.
     * A new instance should be created for a fresh session.
     */
    fun shutdown()
}
