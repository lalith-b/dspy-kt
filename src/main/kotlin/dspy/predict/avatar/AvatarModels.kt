package dspy.predict.avatar

/**
 * Avatar models for advanced reasoning.
 */
data class AvatarModel(
    val name: String,
    val capabilities: List<String>,
    val temperature: Float = 0.7f
)

data class AvatarAction(
    val type: String,
    val parameters: Map<String, Any?> = emptyMap()
)

data class AvatarObservation(
    val action: AvatarAction,
    val result: String,
    val success: Boolean
)
