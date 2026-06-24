package dspy.utils

import dspy.signatures.Signature

/**
 * Port of `dspy/utils/exceptions.py`.
 *
 * Base class for DSPy errors with structured metadata.
 */
open class DSPyException(
    override val message: String,
    open val code: String? = null,
    open val model: String? = null,
    open val provider: String? = null,
    open val providerCode: String? = null,
    open val status: Int? = null,
    open val requestId: String? = null,
    open val retryAfter: Double? = null,
) : RuntimeException(message)

/**
 * Base class for language model errors.
 */
open class LMError(
    message: String = "",
    code: String? = null,
    model: String? = null,
    provider: String? = null,
    providerCode: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : DSPyException(message, code ?: "lm_error", model, provider, providerCode, status, requestId, retryAfter)

/**
 * The LM request failed before the provider returned a response.
 */
class LMTransportError(message: String = "") : LMError(message, code = "transport")

/**
 * The LM or provider client is not configured correctly.
 */
open class LMConfigurationError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
) : LMError(message, code = "configuration", model = model, provider = provider)

/**
 * The LM is missing required provider configuration or credentials.
 */
class LMNotConfiguredError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
) : LMConfigurationError(message, model = model, provider = provider)

/**
 * The LM, provider, or DSPy provider wrapper does not support a requested feature.
 */
class LMUnsupportedFeatureError(
    message: String = "",
    val features: List<String> = emptyList(),
    val issues: List<String> = emptyList(),
    code: String? = null,
    model: String? = null,
    provider: String? = null,
    providerCode: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMError(message, code ?: "unsupported_feature", model, provider, providerCode, status, requestId, retryAfter)

/**
 * The provider returned an error response.
 */
open class LMProviderError(
    message: String = "",
    code: String? = null,
    model: String? = null,
    provider: String? = null,
    providerCode: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMError(message, code ?: "provider", model, provider, providerCode, status, requestId, retryAfter)

/**
 * An unexpected failure occurred at the LM provider boundary.
 */
class LMUnexpectedError(message: String = "", model: String? = null, provider: String? = null) :
    LMError(message, code = "unexpected", model = model, provider = provider)

/**
 * The provider rejected the request because authentication failed.
 */
class LMAuthError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code = "auth", model = model, provider = provider, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * The provider rejected the request because billing or quota failed.
 */
class LMBillingError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code = "billing", model = model, provider = provider, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * The provider rate-limited the request.
 */
class LMRateLimitError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code = "rate_limit", model = model, provider = provider, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * The provider rejected the request shape or resource.
 */
open class LMInvalidRequestError(
    message: String = "",
    code: String? = null,
    model: String? = null,
    provider: String? = null,
    providerCode: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code ?: "invalid_request", model, provider, providerCode, status, requestId, retryAfter)

/**
 * Raised when the prompt exceeds the model's context window.
 */
class ContextWindowExceededError(
    message: String = "Context window exceeded",
    model: String? = null,
    provider: String? = null,
    providerCode: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMInvalidRequestError(message, code = "context_window_exceeded", model = model, provider = provider, providerCode = providerCode, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * The requested model is unavailable or unsupported by the provider.
 */
class LMUnsupportedModelError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
) : LMInvalidRequestError(message, code = "unsupported_model", model = model, provider = provider)

/**
 * The provider request timed out.
 */
class LMTimeoutError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code = "timeout", model = model, provider = provider, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * The provider failed while handling the request.
 */
class LMServerError(
    message: String = "",
    model: String? = null,
    provider: String? = null,
    status: Int? = null,
    requestId: String? = null,
    retryAfter: Double? = null,
) : LMProviderError(message, code = "server", model = model, provider = provider, status = status, requestId = requestId, retryAfter = retryAfter)

/**
 * Return whether an LM error is generally safe to retry.
 */
fun isRetryableLmError(error: Exception): Boolean {
    return error is LMRateLimitError || error is LMTimeoutError ||
           error is LMServerError || error is LMTransportError
}

/**
 * Raised when an adapter cannot parse an LM response into signature outputs.
 */
class AdapterParseError(
    val adapterName: String,
    val signature: Signature,
    val lmResponse: String,
    message: String? = null,
    val parsedResult: Map<String, Any?>? = null,
) : DSPyException(
    message = buildString {
        if (message != null) append("$message\n\n")
        append("Adapter $adapterName failed to parse the LM response. \n\n")
        append("LM Response: $lmResponse \n\n")
        val outputFieldNames = signature.outputFields.joinToString(", ") { it.name }
        append("Expected to find output fields in the LM response: [$outputFieldNames] \n\n")
        if (parsedResult != null) {
            val parsedKeys = parsedResult.keys.joinToString(", ")
            append("Actual output fields parsed from the LM response: [$parsedKeys] \n\n")
        }
    }.trim(),
    code = "adapter_parse_error",
)
