package dspy.primitives

/**
 * Base Module for DSPy programs.
 *
 * All DSPy modules inherit from this class to provide parameter tracking
 * and consistent invocation semantics.
 */
abstract class BaseModule : Module() {
    private val _children: MutableList<Module> = mutableListOf()

    /**
     * Add a child module to this module.
     */
    fun addChild(child: Module) {
        _children.add(child)
    }

    /**
     * Get all child modules.
     */
    fun children(): List<Module> = _children.toList()

    /**
     * Get all parameters (including from children).
     */
    override fun parameters(): List<Parameter> {
        return (namedParameters() as List<Parameter>) + (_children.flatMap { it.parameters() } as List<Parameter>)
    }
}
