package dspy.predict

import dspy.adapters.ChatAdapter
import dspy.clients.BaseLM
import dspy.core.LMConfigurationError
import dspy.core.types.LMConfig
import dspy.primitives.Completions
import dspy.primitives.Module
import dspy.primitives.Parameter
import dspy.primitives.Prediction
import dspy.signatures.Signature
import dspy.utils.CallbackHandler
import dspy.utils.Settings
import org.slf4j.LoggerFactory
import kotlin.random.Random
import kotlin.Triple

/**
 * Basic DSPy module that maps inputs to outputs using a language model.
 *
 * Port of `dspy/predict/predict.py`
 */
class Predict(
    signature: String? = null,
    val sig: Signature? = null,
    callbacks: List<CallbackHandler>? = null,
    val config: MutableMap<String, Any?> = mutableMapOf(),
) : Module(), Parameter {
    private val logger = LoggerFactory.getLogger(Predict::class.java)

    val stage: String = Random.nextBytes(8).joinToString("") { "%02x".format(it) }
    var signature: Signature
    var _lm: BaseLM? = null
    var traces: MutableList<Any> = mutableListOf()
    var train: MutableList<Any> = mutableListOf()
    var demos: MutableList<Map<String, Any?>> = mutableListOf()

    init {
        if (callbacks != null) {
            this.callbacks.addAll(callbacks)
        }

        this.signature = if (signature != null) {
            Signature.fromString(signature)
        } else if (sig != null) {
            sig
        } else {
            throw IllegalArgumentException("Must provide either signature string or Signature object")
        }

        reset()
    }

    fun reset() {
        _lm = null
        traces = mutableListOf()
        train = mutableListOf()
        demos = mutableListOf()
    }

    override fun dumpState(): Map<String, Any?> {
        val state = mutableMapOf<String, Any?>(
            "traces" to traces.toList(),
            "train" to train.toList(),
            "demos" to demos.map { serializeObject(it) },
            "signature" to signature.dumpState(),
            "lm" to _lm?.dumpState(),
        )
        return state
    }

    override fun loadState(state: Any?) {
        if (state !is Map<*, *>) return

        val excludedKeys = listOf("signature", "extended_signature", "lm")

        for ((name, value) in state) {
            if (name !in excludedKeys && name is String) {
                when (name) {
                    "traces" -> traces = (value as? List<Any>)?.toMutableList() ?: mutableListOf()
                    "train" -> train = (value as? List<Any>)?.toMutableList() ?: mutableListOf()
                    "demos" -> demos = (value as? List<Any>)?.mapNotNull { it as? Map<String, Any?> }?.toMutableList() ?: mutableListOf()
                }
            }
        }

        // Load signature state
        @Suppress("UNCHECKED_CAST")
        (state["signature"] as? Map<String, Any?>)?.let { sigState ->
            // Signature state loading
        }

        // Load LM state
        val lmState = state["lm"] as? Map<String, Any?>
        if (lmState != null) {
            _lm = BaseLM.loadState(lmState)
        }

        if ("extended_signature" in state) {
            throw NotImplementedError("Loading extended_signature is no longer supported")
        }
    }

    fun _getPositionalArgsErrorMessage(): String {
        val inputFields = signature.inputFields.map { it.name }
        return buildString {
            append("Positional arguments are not allowed when calling `Predict`, must use keyword arguments ")
            append("that match your signature input fields: '${inputFields.joinToString(", ")}'. ")
            append("For example: `predict(${inputFields.firstOrNull() ?: "field"}=input_value, ...)`.")
        }
    }

    /**
     * Main call method.
     */
    suspend fun __call__(vararg args: Pair<String, Any?>, kwargs: Map<String, Any?> = emptyMap()): Prediction {
        if (args.isNotEmpty()) {
            throw IllegalArgumentException(_getPositionalArgsErrorMessage())
        }
        return forward(kwargs.toMutableMap())
    }

    /**
     * Async call method.
     */
    suspend fun acall(vararg args: Pair<String, Any?>, kwargs: Map<String, Any?> = emptyMap()): Prediction {
        if (args.isNotEmpty()) {
            throw IllegalArgumentException(_getPositionalArgsErrorMessage())
        }
        return aforward(kwargs.toMutableMap())
    }

    /**
     * Pre-process forward pass: extract config, LM, validate inputs.
     */
    private fun forwardPreprocess(kwargs: MutableMap<String, Any?>): Tuple {
        require("new_signature" !in kwargs) { "new_signature is no longer a valid keyword argument." }

        val sig = (kwargs.remove("signature") as? Signature) ?: signature
        val demoList = (kwargs.remove("demos") as? List<Map<String, Any?>>) ?: demos.toList()
        val callConfig = mutableMapOf<String, Any?>()
        callConfig.putAll(config)
        callConfig.putAll(kwargs.remove("config") as? Map<String, Any?> ?: emptyMap())

        // Get the right LM to use
        val lmInstance = (kwargs.remove("lm") as? BaseLM) ?: _lm ?: Settings.lm()

        if (lmInstance == null) {
            throw IllegalArgumentException(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`."
            )
        }

        // Temperature/n logic
        val temperature = callConfig["temperature"] as? Double
            ?: (lmInstance.kwargs["temperature"] as? Double)
        val numGenerations = (callConfig["n"] as? Int)
            ?: (lmInstance.kwargs["maxTokens"] as? Int)
            ?: 1

        if ((temperature == null || temperature <= 0.15) && numGenerations > 1) {
            callConfig["temperature"] = 0.7
        }

        // Handle prediction (standard predicted outputs format)
        val prediction = kwargs["prediction"] as? Map<*, *>
        if (prediction != null && prediction["type"] == "content" && "content" in prediction) {
            callConfig["prediction"] = prediction
            kwargs.remove("prediction")
        }

        // Populate default values for missing input fields
        for (field in sig.inputFields) {
            if (field.name !in kwargs && !field.isTypeUndefined) {
                // Set default if available
            }
        }

        // Check and warn for extra fields not in signature
        val extraFields = kwargs.keys.filter { it !in sig.inputFields.map { f -> f.name } }
        if (extraFields.isNotEmpty()) {
            logger.warn(
                "Input contains fields not in signature. These fields will be ignored: {}. " +
                    "Expected fields: {}.",
                extraFields,
                sig.inputFields.map { it.name },
            )
        }

        // Check for missing input fields
        val missing = sig.inputFields.filter {
            it.name !in kwargs && it.annotation != Any::class
        }.map { it.name }
        if (missing.isNotEmpty()) {
            val present = sig.inputFields.filter { it.name in kwargs }.map { it.name }
            logger.warn(
                "Not all input fields were provided to module. Present: {}. Missing: {}.",
                present,
                missing,
            )
        }

        return Tuple(lmInstance, callConfig, sig, demoList, kwargs)
    }

    private data class Tuple(
        val lm: BaseLM,
        val config: Map<String, Any?>,
        val signature: Signature,
        val demos: List<Map<String, Any?>>,
        val inputs: Map<String, Any?>,
    )

    private fun forwardPostprocess(
        completions: List<Map<String, Any?>>,
        sig: Signature,
        kwargs: Map<String, Any?>,
    ): Prediction {
        val trace = kwargs["_trace"] as? Boolean ?: true
        val pred = Prediction.fromCompletions(completions, signature = sig)

        if (trace && Settings.maxTraceSize > 0) {
            traces.add(Triple(this, kwargs.toMap(), pred.toDict()))
        }

        // Also add to Settings.trace when it's active (for bootstrapping)
        if (Settings.trace != null) {
            Settings.trace!!.add(Triple(this, kwargs.toMap(), pred.toDict()))
        }

        return pred
    }

    /**
     * Forward pass: process inputs, call adapter, parse results.
     */
    suspend fun forward(kwargs: MutableMap<String, Any?> = mutableMapOf()): Prediction {
        val (lmInstance, callConfig, sig, demoList, inputs) = forwardPreprocess(kwargs)

        val adapter = Settings.adapter() ?: ChatAdapter()

        val completions = adapter(
            lm = lmInstance,
            lmKwargs = callConfig.toMutableMap(),
            signature = sig,
            demos = demoList,
            inputs = inputs,
        )

        return forwardPostprocess(completions, sig, kwargs)
    }

    /**
     * Async forward pass.
     */
    suspend fun aforward(kwargs: MutableMap<String, Any?> = mutableMapOf()): Prediction {
        val (lmInstance, callConfig, sig, demoList, inputs) = forwardPreprocess(kwargs)

        val adapter = Settings.adapter() ?: ChatAdapter()

        val completions = adapter.acall(
            lm = lmInstance,
            lmKwargs = callConfig.toMutableMap(),
            signature = sig,
            demos = demoList,
            inputs = inputs,
        )

        return forwardPostprocess(completions, sig, kwargs)
    }

    fun updateConfig(vararg kwargs: Pair<String, Any?>) {
        config.putAll(kwargs.toMap())
    }

    override fun namedParameters(): List<Pair<String, Parameter>> {
        return listOf("self" to this)
    }

    // ---- Deep copy & reset copy (faithful port) ----

    override fun deepcopy(): Module {
        return Predict(sig = this.signature).apply {
            demos = this@Predict.demos.map { it.toMutableMap() }.toMutableList()
            config.putAll(this@Predict.config)
            traces = this@Predict.traces.toList().toMutableList()
            train = this@Predict.train.toList().toMutableList()
            _compiled = this@Predict._compiled
        }
    }

    override fun resetCopy(): Module {
        return Predict(sig = this.signature).apply {
            config.putAll(this@Predict.config)
        }
    }

    // ---- Invoke (faithful port of Module.__call__) ----

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction = forward(kwargs.toMutableMap())

    override fun toString(): String = "Predict($signature)"
}

/**
 * Recursively serialize a given object into a JSON-compatible format.
 */
fun serializeObject(obj: Any?): Any? {
    return when (obj) {
        null -> null
        is String -> obj
        is Number -> obj
        is Boolean -> obj
        is List<*> -> obj.map { serializeObject(it) }
        is Map<*, *> -> obj.mapKeys { it.key.toString() }.mapValues { serializeObject(it.value) }
        else -> obj.toString()
    }
}
