package dspy.primitives

open class Example(
    base: Map<String, Any?>? = null,
    extra: Map<String, Any?> = emptyMap(),
) {
    protected val _store: MutableMap<String, Any?> = mutableMapOf()
    protected var _inputKeys: Set<String>? = null

    init {
        if (base != null) {
            _store.putAll(base)
        }
        _store.putAll(extra)
    }

    operator fun get(key: String): Any? = _store[key]
    operator fun set(key: String, value: Any?) {
        _store[key] = value
    }

    fun get(key: String, default: Any?): Any? = _store.getOrElse(key) { default }

    fun remove(key: String) {
        _store.remove(key)
    }

    val size: Int
        get() = _store.size

    fun keys(): Set<String> = _store.keys
    fun values(): Collection<Any?> = _store.values
    fun items(): Set<Pair<String, Any?>> = _store.entries.map { it.toPair() }.toSet()

    fun withInputKeys(vararg keys: String): Example {
        _inputKeys = keys.toSet()
        return this
    }

    fun withInputs(vararg keys: String): Example {
        _inputKeys = keys.toSet()
        return this
    }

    fun inputKeys(): Set<String> = _inputKeys ?: _store.keys

    fun inputs(): Example {
        val keys = inputKeys()
        val copy = Example()
        for (key in keys) {
            copy._store[key] = _store[key]
        }
        copy._inputKeys = keys
        return copy
    }

    fun labels(): Example {
        val keys = _store.keys - inputKeys()
        val copy = Example()
        for (key in keys) {
            copy._store[key] = _store[key]
        }
        return copy
    }

    fun copy(extra: Map<String, Any?> = emptyMap()): Example {
        val copy = Example(base = _store.toMap(), extra = extra)
        copy._inputKeys = _inputKeys
        return copy
    }

    fun without(key: String): Example {
        val copy = Example(base = _store.filterKeys { it != key })
        copy._inputKeys = _inputKeys
        return copy
    }

    fun toMap(): Map<String, Any?> = _store.toMap()
    fun toDict(): Map<String, Any?> = _store.toMap()

    fun containsKey(key: String): Boolean = _store.containsKey(key)
    fun getOrDefault(key: String, default: Any?): Any? = _store.getOrElse(key) { default }

    override fun equals(other: Any?): Boolean = other is Example && _store == other._store
    override fun hashCode(): Int = _store.hashCode()
    override fun toString(): String = "Example(${_store.toString()}) (input_keys=${_inputKeys?.toString() ?: "None"})"
}
