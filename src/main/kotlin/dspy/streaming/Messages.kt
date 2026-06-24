package dspy.utils

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector

/**
 * Stream response for streaming output from a predictor.
 */
data class StreamResponse(
    val predictName: String?,
    val signatureFieldName: String,
    val chunk: String,
    val isLastChunk: Boolean,
)

/**
 * Status message for status streaming.
 */
data class StatusMessage(
    val message: String,
)

/**
 * Provides customizable status message streaming for DSPy programs.
 */
open class StatusMessageProvider {
    /** Status message before a Tool is called. */
    open fun toolStartStatusMessage(instance: Any, inputs: Map<String, Any?>): String? {
        return "Calling tool ${instance::class.simpleName}..."
    }

    /** Status message after a Tool is called. */
    open fun toolEndStatusMessage(outputs: Any?): String? {
        return "Tool calling finished! Querying the LLM with tool calling results..."
    }

    /** Status message before a Module or Predict is called. */
    open fun moduleStartStatusMessage(instance: Any, inputs: Map<String, Any?>): String? {
        return null
    }

    /** Status message after a Module or Predict is called. */
    open fun moduleEndStatusMessage(outputs: Any?): String? {
        return null
    }

    /** Status message before a LM is called. */
    open fun lmStartStatusMessage(instance: Any, inputs: Map<String, Any?>): String? {
        return null
    }

    /** Status message after a LM is called. */
    open fun lmEndStatusMessage(outputs: Any?): String? {
        return null
    }
}

/**
 * Base class for streaming callbacks.
 */
abstract class StreamingCallback {
    open suspend fun onToolStart(callId: String, instance: Any, inputs: Map<String, Any?>): Unit = Unit
    open suspend fun onToolEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?): Unit = Unit
    open suspend fun onLmStart(callId: String, instance: Any, inputs: Map<String, Any?>): Unit = Unit
    open suspend fun onLmEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?): Unit = Unit
    open suspend fun onModuleStart(callId: String, instance: Any, inputs: Map<String, Any?>): Unit = Unit
    open suspend fun onModuleEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?): Unit = Unit
}

/**
 * Streaming callback that sends status messages to a Flow.
 */
class StatusStreamingCallback(
    private val statusProvider: StatusMessageProvider = StatusMessageProvider(),
) : StreamingCallback() {

    private var streamFlow: FlowCollector<StatusMessage>? = null

    fun setStream(flow: FlowCollector<StatusMessage>) {
        streamFlow = flow
    }

    override suspend fun onToolStart(callId: String, instance: Any, inputs: Map<String, Any?>) {
        val msg = statusProvider.toolStartStatusMessage(instance, inputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }

    override suspend fun onToolEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?) {
        val msg = statusProvider.toolEndStatusMessage(outputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }

    override suspend fun onLmStart(callId: String, instance: Any, inputs: Map<String, Any?>) {
        val msg = statusProvider.lmStartStatusMessage(instance, inputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }

    override suspend fun onLmEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?) {
        val msg = statusProvider.lmEndStatusMessage(outputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }

    override suspend fun onModuleStart(callId: String, instance: Any, inputs: Map<String, Any?>) {
        val msg = statusProvider.moduleStartStatusMessage(instance, inputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }

    override suspend fun onModuleEnd(callId: String, outputs: Map<String, Any?>?, exception: Throwable?) {
        val msg = statusProvider.moduleEndStatusMessage(outputs)
        msg?.let { streamFlow?.emit(StatusMessage(it)) }
    }
}
