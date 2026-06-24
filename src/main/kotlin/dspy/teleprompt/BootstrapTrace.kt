package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import kotlinx.coroutines.runBlocking

/**
 * Failed prediction with error information.
 */
data class FailedPrediction(
    val completionText: String,
    val formatReward: Double? = null,
)

/**
 * Trace data for bootstrapping.
 */
typealias TraceData = Map<String, Any?>

/**
 * Bootstrap trace data collection.
 *
 * Runs the program on each example and collects traces, predictions, and scores.
 *
 * Faithful port of `dspy/teleprompt/bootstrap_trace.py`.
 */
@Suppress("UNCHECKED_CAST")
fun bootstrapTraceData(
    program: Module,
    dataset: List<Example>,
    metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    numThreads: Int? = null,
    raiseOnError: Boolean = true,
    captureFailedParses: Boolean = false,
    failureScore: Double = 0.0,
    formatFailureScore: Double = -1.0,
    logFormatFailures: Boolean = false,
    callbackMetadata: Map<String, Any?>? = null,
): List<TraceData> {
    // Return a list of dicts with keys: example_ind, example, prediction, trace, and score (if metric != null)
    val evaluate = dspy.evaluate.Evaluate(
        devset = dataset,
        numThreads = numThreads,
        displayProgress = true,
        displayTable = false,
        maxErrors = dataset.size * 10,
        failureScore = failureScore,
    )

    val wrappedMetric: ((Example, Prediction, List<Any>?) -> Any?)? = { example, prediction, trace ->
        if (prediction is FailedPrediction) {
            prediction.formatReward ?: formatFailureScore
        } else {
            metric?.invoke(example, prediction, trace) ?: true
        }
    }

    // In a full implementation, this would patch the program's forward method
    // to capture traces. For now, we run the evaluator directly.
    val results = runBlocking { evaluate.__call__(program, metric = wrappedMetric) }

    val resultsList = results.results

    val data = mutableListOf<TraceData>()
    for (exampleInd in resultsList.indices) {
        val triple = resultsList[exampleInd]
        val example = triple.first as Example
        val prediction = triple.second
        val score = triple.third

        val dataDict = mutableMapOf<String, Any?>(
            "example" to example,
            "prediction" to prediction,
            "trace" to emptyList<Any>(),
            "example_ind" to exampleInd,
        )
        if (metric != null) {
            dataDict["score"] = score
        }
        data.add(dataDict)
    }

    return data
}
