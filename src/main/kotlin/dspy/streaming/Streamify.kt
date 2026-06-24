package dspy.streaming

import dspy.primitives.Module
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

fun streamify(program: Module, inputs: Map<String, Any?>): Flow<String> = flow {
    // Placeholder: emit chunks of output
    emit("streaming_placeholder")
}
