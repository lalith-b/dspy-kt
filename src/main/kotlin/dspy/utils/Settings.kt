package dspy.utils

import dspy.adapters.Adapter
import dspy.clients.BaseLM

typealias TraceEntry = Triple<Any, Map<String, Any?>, Map<String, Any?>>

object Settings {
    private var _lm: BaseLM? = null
    private var _adapter: Adapter? = null
    private var _rm: Any? = null
    var trace: MutableList<TraceEntry>? = null
    var maxErrors: Int? = null
    var numThreads: Int = 1
    const val maxTraceSize: Int = 1000
    const val maxHistorySize: Int = 100
    private val _settings: MutableMap<String, Any?> = mutableMapOf()

    fun configure(model: String, apiKey: String? = null, baseUrl: String = "https://api.openai.com/v1") {
        _lm = dspy.clients.LM(model = model, apiKey = apiKey, baseUrl = baseUrl)
    }

    fun lm(): BaseLM? = _lm
    fun setLm(lm: BaseLM?) { _lm = lm }
    fun adapter(): Adapter? = _adapter
    fun setAdapter(adapter: Adapter?) { _adapter = adapter }
    val rm: Any? get() = _rm
    fun setRm(rm: Any?) { _rm = rm }
    fun get(key: String, default: Any? = null): Any? = _settings[key] ?: default
    fun set(key: String, value: Any?) { _settings[key] = value }

    fun <T> context(
        trace: MutableList<TraceEntry>? = null,
        lm: BaseLM? = null,
        adapter: Adapter? = null,
        maxErrors: Int? = null,
        settings: Map<String, Any?>? = null,
        block: () -> T,
    ): T {
        val savedTrace = this.trace
        val savedLm = _lm
        val savedAdapter = _adapter
        val savedMaxErrors = this.maxErrors
        val savedGenSettings = settings?.keys?.associateWith { _settings[it] } ?: emptyMap()

        if (trace != null) this.trace = trace
        if (lm != null) _lm = lm
        if (adapter != null) _adapter = adapter
        if (maxErrors != null) this.maxErrors = maxErrors
        settings?.forEach { (k, v) -> _settings[k] = v }

        try {
            return block()
        } finally {
            this.trace = savedTrace
            _lm = savedLm
            _adapter = savedAdapter
            this.maxErrors = savedMaxErrors
            for (k in savedGenSettings.keys) {
                _settings[k] = savedGenSettings[k]
            }
        }
    }
}
