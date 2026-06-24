package dspy.predict.avatar

import dspy.signatures.Signature
import dspy.signatures.InputField
import dspy.signatures.OutputField
import kotlin.reflect.KClass

/**
 * Avatar signatures for reasoning steps.
 */
object AvatarSignatures {
    fun createReasoningSignature(): Signature {
        return Signature.makeSignature(mapOf(
            "input" to Pair(String::class, InputField()),
            "context" to Pair(String::class, InputField()),
            "thought" to Pair(String::class, OutputField()),
            "action" to Pair(String::class, OutputField()),
        ), "Think about the problem and decide on the next action")
    }

    fun createFinalSignature(): Signature {
        return Signature.makeSignature(mapOf(
            "input" to Pair(String::class, InputField()),
            "reasoning" to Pair(String::class, InputField()),
            "answer" to Pair(String::class, OutputField()),
        ), "Provide the final answer based on the reasoning")
    }
}
