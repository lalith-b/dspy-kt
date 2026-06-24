package dspy.utils

import dspy.adapters.types.Tool

/**
 * Convert MCP CallToolResult to a usable return value.
 *
 * Port of `dspy/utils/mcp.py` - `_convert_mcp_tool_result`
 *
 * MCP (Model Context Protocol) is a Python/TypeScript library with no Kotlin equivalent.
 * This module provides stub implementations preserving the API shape.
 */

/**
 * Stub for MCP CallToolResult content types.
 */
sealed class McpContent {
    data class Text(val text: String) : McpContent()
    data class Image(val data: String, val mimeType: String) : McpContent()
    data class Resource(val uri: String, val mimeType: String? = null, val text: String? = null) : McpContent()
}

/**
 * Stub for MCP CallToolResult.
 */
data class McpCallToolResult(
    val content: List<McpContent> = emptyList(),
    val isError: Boolean = false,
)

/**
 * Stub for MCP Tool definition.
 */
data class McpToolDefinition(
    val name: String,
    val description: String? = null,
    val inputSchema: Map<String, Any>? = null,
)

/**
 * Stub for MCP ClientSession.
 */
interface McpClientSession {
    suspend fun callTool(name: String, arguments: Map<String, Any?>): McpCallToolResult
}

/**
 * Convert an MCP CallToolResult to a string or list.
 *
 * If the result contains a single TextContent, returns that text as a String.
 * If multiple TextContent items, returns a list of text strings.
 * If isError, raises RuntimeError.
 * Otherwise returns non-text contents.
 */
fun convertMcpToolResult(result: McpCallToolResult): Any {
    val textContents = result.content.filterIsInstance<McpContent.Text>()
    val nonTextContents = result.content - textContents.toSet()

    val toolContent = textContents.map { it.text }

    if (result.isError) {
        val errorMessage = if (toolContent.size == 1) toolContent[0] else toolContent.toString()
        throw RuntimeException("Failed to call a MCP tool: $errorMessage")
    }

    return if (toolContent.isNotEmpty()) {
        if (toolContent.size == 1) toolContent[0] else toolContent
    } else {
        nonTextContents
    }
}

/**
 * Build a DSPy tool from an MCP tool definition.
 *
 * Port of `dspy/utils/mcp.py` - `convert_mcp_tool`
 *
 * @param session The MCP client session to use.
 * @param tool The MCP tool definition.
 * @return A dspy Tool that wraps the MCP tool.
 */
fun convertMcpTool(session: McpClientSession, tool: McpToolDefinition): Tool {
    val args = tool.inputSchema ?: emptyMap<String, Any>()

    return Tool(
        func = { mapOf<String, Any?>() },
        name = tool.name,
        desc = tool.description,
        args = args,
    )
}
