package dspy.utils

import dspy.adapters.Adapter
import dspy.adapters.types.Tool
import dspy.clients.BaseLM
import dspy.evaluate.Evaluate
import dspy.primitives.Module
import java.util.UUID

typealias CallbackHandler = BaseCallback

val ACTIVE_CALL_ID: ThreadLocal<String?> = ThreadLocal.withInitial { null }

typealias StartHandler = (String, Any, Map<String, Any?>) -> Unit

typealias EndHandler = (String, Any?, Throwable?) -> Unit

abstract class BaseCallback {

    open fun onModuleStart(callId: String, instance: Module, inputs: Map<String, Any?>) {
    }

    open fun onModuleEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }

    open fun onLmStart(callId: String, instance: BaseLM, inputs: Map<String, Any?>) {
    }

    open fun onLmEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }

    open fun onToolStart(callId: String, instance: Tool, inputs: Map<String, Any?>) {
    }

    open fun onToolEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }

    open fun onAdapterFormatStart(callId: String, instance: Adapter, inputs: Map<String, Any?>) {
    }

    open fun onAdapterFormatEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }

    open fun onAdapterParseStart(callId: String, instance: Adapter, inputs: Map<String, Any?>) {
    }

    open fun onAdapterParseEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }

    open fun onEvaluateStart(callId: String, instance: Evaluate, inputs: Map<String, Any?>) {
    }

    open fun onEvaluateEnd(callId: String, outputs: Any?, exception: Throwable?) {
    }
}

fun getOnStartHandler(callback: BaseCallback, instance: Any, fnName: String = ""): StartHandler {
    return when {
        instance is BaseLM -> { _: String, _: Any, inputs: Map<String, Any?> -> callback.onLmStart("", instance, inputs) }
        instance is Evaluate -> { _: String, _: Any, inputs: Map<String, Any?> -> callback.onEvaluateStart("", instance, inputs) }
        instance is Adapter -> when (fnName) {
            "format" -> { _: String, _: Any, inputs: Map<String, Any?> -> callback.onAdapterFormatStart("", instance, inputs) }
            "parse" -> { _: String, _: Any, inputs: Map<String, Any?> -> callback.onAdapterParseStart("", instance, inputs) }
            else -> throw IllegalArgumentException("Unsupported adapter method")
        }
        instance is Tool -> { _: String, _: Any, inputs: Map<String, Any?> -> callback.onToolStart("", instance, inputs) }
        else -> { _: String, inst: Any, inputs: Map<String, Any?> -> callback.onModuleStart("", inst as Module, inputs) }
    }
}

fun getOnEndHandler(callback: BaseCallback, instance: Any, fnName: String = ""): EndHandler {
    return when {
        instance is BaseLM -> { _: String, outputs: Any?, exception: Throwable? -> callback.onLmEnd("", outputs, exception) }
        instance is Evaluate -> { _: String, outputs: Any?, exception: Throwable? -> callback.onEvaluateEnd("", outputs, exception) }
        instance is Adapter -> when (fnName) {
            "format" -> { _: String, outputs: Any?, exception: Throwable? -> callback.onAdapterFormatEnd("", outputs, exception) }
            "parse" -> { _: String, outputs: Any?, exception: Throwable? -> callback.onAdapterParseEnd("", outputs, exception) }
            else -> throw IllegalArgumentException("Unsupported adapter method")
        }
        instance is Tool -> { _: String, outputs: Any?, exception: Throwable? -> callback.onToolEnd("", outputs, exception) }
        else -> { _: String, outputs: Any?, exception: Throwable? -> callback.onModuleEnd("", outputs, exception) }
    }
}

fun generateCallId(): String {
    return UUID.randomUUID().toString()
}

fun <T> withCallback(callback: BaseCallback, instance: Any, inputs: Map<String, Any?>, fnName: String, block: () -> T): T {
    val callId = generateCallId()
    val startHandler = getOnStartHandler(callback, instance, fnName)
    val endHandler = getOnEndHandler(callback, instance, fnName)
    try {
        startHandler(callId, instance, inputs)
        return block()
    } catch (e: Exception) {
        endHandler(callId, null, e)
        throw e
    }
}
