package dspy.primitives

open class Prediction(
    base: Map<String, Any?>? = null,
    extra: Map<String, Any?> = emptyMap(),
) : Example(base, extra), Parameter {
    var lmUsage: dspy.core.types.LMUsage? = null
    private var _completions: Completions? = null

    val completions: Completions? get() = _completions

    companion object {
        fun fromCompletions(listOrDict: Any, signature: dspy.signatures.Signature? = null): Prediction {
            val obj = Prediction()
            obj._completions = Completions(listOrDict, signature)
            val store = mutableMapOf<String, Any?>()
            for ((k, v) in obj._completions?.items() ?: emptyMap()) {
                store[k] = (v as List<*>)[0]
            }
            obj._store.putAll(store)
            return obj
        }
    }

    override fun toString(): String = "Prediction(${_store.toString()})"

    fun toFloat(): Double {
        require("score" in _store) { "Prediction object does not have a 'score' field to convert to float." }
        return (_store["score"] as Number).toDouble()
    }

    override fun dumpState(): Map<String, Any?> = _store.toMap()
    override fun loadState(state: Any?) {
        if (state is Map<*, *>) {
            for ((k, v) in state) {
                _store[k.toString()] = v
            }
        }
    }
}

class Completions(
    listOrDict: Any,
    val signature: dspy.signatures.Signature? = null,
) {
    private val _completions: Map<String, List<Any>>

    init {
        if (listOrDict is List<*>) {
            val kwargs = mutableMapOf<String, MutableList<Any>>()
            for (item in listOrDict) {
                if (item is Map<*, *>) {
                    for ((k, v) in item) {
                        kwargs.getOrPut(k.toString()) { mutableListOf() }.add(v as Any)
                    }
                }
            }
            _completions = kwargs
        } else if (listOrDict is Map<*, *>) {
            _completions = listOrDict.mapKeys { it.key.toString() }.mapValues { it.value as List<Any> }
        } else {
            throw IllegalArgumentException("list_or_dict must be a list or dict")
        }
        require(_completions.values.all { it is List<*> }) { "All values must be lists" }
        if (_completions.isNotEmpty()) {
            val length = _completions.values.first().size
            require(_completions.values.all { it.size == length }) { "All lists must have the same length" }
        }
    }

    fun items(): Map<String, List<Any>> = _completions

    operator fun get(key: String): List<Any> = _completions[key]!!

    operator fun get(index: Int): Prediction {
        require(index >= 0 && index < size) { "Index out of range" }
        val store = mutableMapOf<String, Any?>()
        for ((k, v) in _completions) {
            store[k] = v[index]
        }
        return Prediction(store)
    }

    val size: Int
        get() {
            if (_completions.isEmpty()) return 0
            return _completions.values.first().size
        }

    operator fun contains(key: String): Boolean = key in _completions

    override fun toString(): String = "Completions(${_completions.toString()})"
}
