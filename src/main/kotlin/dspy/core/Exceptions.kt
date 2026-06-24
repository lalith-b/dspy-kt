package dspy.core

open class DSPyException(message: String) : RuntimeException(message)

// ===== LM Errors =====

open class LMError(
    message: String,
    open val model: String? = null,
    open val provider: String? = null,
    open val providerCode: String? = null,
    open val status: Int? = null,
    open val requestId: String? = null,
    open val retryAfter: Double? = null,
) : DSPyException(message)

class LMConfigurationError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMNotConfiguredError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMAuthError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMTransportError(message: String) : LMError(message)

class LMTimeoutError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMRateLimitError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMBillingError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMInvalidRequestError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMUnsupportedModelError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMUnsupportedFeatureError(
    message: String,
    model: String? = null,
    provider: String? = null,
    val features: List<String> = emptyList(),
) : LMError(message, model = model, provider = provider)

class LMServerError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMProviderError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class ContextWindowExceededError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

class LMUnexpectedError(message: String, model: String? = null, provider: String? = null) : LMError(message, model = model, provider = provider)

// ===== Adapter Errors =====

class AdapterParseError(
    message: String,
    val adapterName: String = "",
    val signature: dspy.signatures.Signature? = null,
    val lmResponse: String? = null,
    val parsedResult: Map<String, Any?>? = null,
) : DSPyException(message)
