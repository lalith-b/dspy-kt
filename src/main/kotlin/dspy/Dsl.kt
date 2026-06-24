package dspy

import dspy.adapters.types.Tool
import dspy.primitives.Example
import dspy.primitives.Prediction

object dspySettings {
    var adapter: dspy.adapters.Adapter? = null
}

fun tool(
    func: () -> Any,
    name: String? = null,
    desc: String? = null,
    args: Map<String, Any>? = null,
): Tool = Tool(func, name, desc, args)

fun Signature(
    inputFields: Map<String, dspy.signatures.InputField>,
    instruction: String = "",
) = dspy.signatures.Signature(
    instruction = instruction,
    inputFields = inputFields.values.toList(),
)
