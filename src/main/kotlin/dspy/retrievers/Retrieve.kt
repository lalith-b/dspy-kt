package dspy.retrievers

import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings

/**
 * Port of `dspy/retrievers/retrieve.py`.
 */

fun singleQueryPassage(passages: List<Map<String, Any?>>): Prediction {
    val passagesDict = mutableMapOf<String, MutableList<Any?>>()
    passages.firstOrNull()?.keys?.forEach { key ->
        passagesDict[key] = mutableListOf()
    }
    for (docs in passages) {
        for ((key, value) in docs) {
            passagesDict.getOrPut(key) { mutableListOf() }.add(value)
        }
    }
    val result = passagesDict.mapValues { it.value }
    val final = if ("long_text" in result) {
        result.toMutableMap().apply {
            put("passages", remove("long_text") ?: mutableListOf())
        }
    } else {
        result
    }
    return Prediction(base = final)
}

/**
 * Basic retrieve module that uses the configured RM (retrieval model) to fetch passages.
 */
class Retrieve(
    private val k: Int = 3,
) : Module() {

    companion object {
        const val NAME = "Search"
        const val INPUT_VARIABLE = "query"
        const val DESC = "takes a search query and returns one or more potentially relevant passages from a corpus"
    }

    fun reset() {
        // No-op
    }

    fun forward(
        query: String,
        kOverride: Int? = null,
    ): Prediction {
        val effectiveK = kOverride ?: k

        if (Settings.rm == null) {
            throw AssertionError("No RM is loaded.")
        }

        val retriever = Settings.rm
        val passages = mutableListOf<String>()

        if (retriever is Retriever) {
            val results = retriever.query(query, effectiveK)
            for (result in results) {
                val text = result["long_text"]?.toString() ?: result["text"]?.toString()
                if (text != null) {
                    passages.add(text)
                }
            }
        }

        return Prediction(base = mapOf("passages" to passages))
    }

    operator fun invoke(query: String, kOverride: Int? = null): Prediction {
        return forward(query, kOverride)
    }
}
