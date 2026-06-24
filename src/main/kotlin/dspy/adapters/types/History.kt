package dspy.adapters.types

/**
 * Class representing the conversation history.
 *
 * The conversation history is a list of messages, each message entity should have keys from the associated signature.
 */
data class History(
    val messages: MutableList<Map<String, Any?>> = mutableListOf()
)
