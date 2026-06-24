package dspy.utils

import dspy.adapters.types.Tool

/**
 * Build a DSPy tool from a LangChain tool.
 *
 * This function converts a LangChain tool (either created with @tool decorator
 * or by subclassing BaseTool) into a DSPy Tool.
 *
 * Port of `dspy/utils/langchain_tool.py`
 *
 * Note: This is a stub because LangChain is a Python library with no Kotlin equivalent.
 * The function signature is preserved for interoperability.
 */
fun convertLangChainTool(tool: Any): Tool {
    // LangChain is Python-only; stub raises NotImplementedError
    throw NotImplementedError(
        "convertLangChainTool requires a LangChain BaseTool instance. " +
            "LangChain is a Python library and has no Kotlin equivalent. " +
            "To integrate external tools, use dspy.Tool directly."
    )
}

/**
 * Stub interface for LangChain BaseTool reference.
 * LangChain is a Python library; this interface preserves the API shape.
 */
interface LangChainBaseTool {
    val name: String
    val description: String

    suspend fun invoke(args: Map<String, Any?>): Any?
}
