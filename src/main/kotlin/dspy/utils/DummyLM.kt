package dspy.utils

import dspy.clients.BaseLM

/**
 * Dummy language model for unit testing purposes.
 *
 * Cycles through a list of predetermined responses.
 * Port of `dspy/utils/dummies.py` DummyLM.
 */
class DummyLM(
    private val answers: List<Map<String, Any?>>,
    modelName: String = "dummy",
) : BaseLM(
    model = modelName,
    modelType = "chat",
    temperature = null,
    maxTokens = null,
    cache = false,
    callbacks = null,
    numRetries = 1,
) {
    private var index: Int = 0

    override suspend fun forward(
        prompt: String?,
        messages: List<Map<String, Any>>?,
        extraKwargs: Map<String, Any?>,
    ): Any {
        val answer = if (index < answers.size) answers[index] else mapOf<String, Any?>(
            "answer" to "No more responses"
        )
        index++

        // Format the answer with [[ ## field_name ## ]] headers
        val formatted = formatAnswer(answer)

        // Return an OpenAI-like response structure
        return mapOf(
            "choices" to listOf(
                mapOf(
                    "message" to mapOf("content" to formatted),
                    "finish_reason" to "stop"
                )
            ),
            "usage" to mapOf(
                "prompt_tokens" to 0,
                "completion_tokens" to 0,
                "total_tokens" to 0
            ),
            "model" to model
        )
    }

    private fun formatAnswer(answer: Map<String, Any?>): String {
        val nl = "\n"
        return answer.entries.joinToString("$nl$nl") { (name, value) ->
            "[[ ## ${name} ## ]]" + nl + (value ?: "")
        }
    }

    fun reset() {
        index = 0
    }
}
