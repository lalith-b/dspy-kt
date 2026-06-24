package dspy.primitives

import dspy.core.types.LMHistoryEntry
import dspy.predict.Predict
import kotlin.reflect.KMutableProperty1
import kotlin.reflect.full.memberProperties

open class Module {
    var compiled: Boolean = false
        set(value) {
            field = value
            _compiled = value
        }
    var _compiled: Boolean = false
    val callbacks: MutableList<dspy.utils.CallbackHandler> = mutableListOf()
    val history: MutableList<LMHistoryEntry> = mutableListOf()
    
    // Optimization result metadata (set by teleprompters)
    var candidatePrograms: Any? = null
    var trialLogs: Any? = null
    var detailedResults: Any? = null
    var totalCalls: Int? = null
    var resultsBest: Any? = null
    var resultsLatest: Any? = null
    var simbaIdx: Int? = null
    
    // Generic attribute storage for dynamic attributes
    private val _extraAttrs = mutableMapOf<String, Any?>()
    
    fun setAttribute(name: String, value: Any?) {
        // Try to set as a named property first
        val prop = this::class.memberProperties.find { it.name == name }
        if (prop is KMutableProperty1) {
            prop.setter.call(this, value)
        } else {
            _extraAttrs[name] = value
        }
    }
    
    fun getAttribute(name: String): Any? {
        val prop = this::class.memberProperties.find { it.name == name }
        if (prop != null) return prop.getter.call(this)
        return _extraAttrs[name]
    }

    fun addCallback(callback: dspy.utils.CallbackHandler) {
        callbacks.add(callback)
    }

    open fun dumpState(): Map<String, Any?> {
        return namedParameters().associate { (name, param) ->
            name to (if (param is Parameter) param.dumpState() else param)
        }
    }

    open fun loadState(state: Map<String, Any?>) {
        for ((name, param) in namedParameters()) {
            if (param is Parameter) {
                param.loadState(state[name])
            }
        }
    }

    open fun namedParameters(): List<Pair<String, Parameter>> {
        return emptyList()
    }

    open fun parameters(): List<Parameter> {
        return namedParameters().map { it.second }
    }

    fun namedPredictors(): List<Pair<String, Predict>> {
        return namedParameters().filter { (_, param) -> param is Predict }
            .map { (name, param) -> name to (param as Predict) }
    }

    fun predictors(): List<Predict> {
        return namedPredictors().map { it.second }
    }

    open fun deepcopy(): Module {
        throw NotImplementedError("deepcopy not implemented for ${this::class.simpleName}")
    }

    open fun resetCopy(): Module {
        throw NotImplementedError("resetCopy not implemented for ${this::class.simpleName}")
    }

    suspend open operator fun invoke(kwargs: Map<String, Any?>): Prediction {
        throw NotImplementedError("invoke not implemented for ${this::class.simpleName}")
    }

    fun setLm(lm: dspy.clients.BaseLM?) {
        for ((_, param) in namedPredictors()) {
            param._lm = lm
        }
    }

    fun getLm(): dspy.clients.BaseLM {
        val allUsedLms = namedPredictors().mapNotNull { it.second._lm }
        if (allUsedLms.toSet().size == 1) {
            return allUsedLms.first()
        }
        throw ModuleValueError("Multiple LMs are being used in the module. There's no unique LM to return.")
    }

    fun mapNamedPredictors(func: (Predict) -> Predict): Module {
        for ((name, predictor) in namedPredictors().toList()) {
            setAttributeByName(this, name, func(predictor))
        }
        return this
    }

    override fun toString(): String {
        return namedPredictors().joinToString("\n") { (name, param) -> "$name = $param" }
    }
}

internal fun setAttributeByName(obj: Any, name: String, value: Any) {
    if (!name.contains(".")) {
        val prop = obj::class.memberProperties.find { it.name == name }
        if (prop is KMutableProperty1) {
            prop.setter.call(obj, value)
        }
        return
    }
    val parts = name.split(".")
    var current: Any = obj
    for (i in parts.indices) {
        val part = parts[i]
        val p = current::class.memberProperties.find { it.name == part }
        if (p != null) {
            if (i == parts.size - 1) {
                if (p is KMutableProperty1) {
                    p.setter.call(current, value)
                }
            } else {
                val next = p.getter.call(current)
                if (next != null) current = next
            }
        }
    }
}

class ModuleValueError(message: String) : IllegalArgumentException(message)

interface Parameter {
    fun dumpState(): Map<String, Any?>
    fun loadState(state: Any?)
}
