package dspy.clients

open class Provider(
    val name: String = "default",
    var finetunable: Boolean = false,
) {
    open fun isProviderModel(model: String): Boolean = false
}
