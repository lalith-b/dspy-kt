package dspy.clients

import dspy.core.LMAuthError
import dspy.core.LMTransportError
import dspy.core.LMUnexpectedError
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.Json
import kotlinx.coroutines.delay
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.io.IOException

class LM(
    model: String,
    modelType: String = "chat",
    temperature: Double? = null,
    maxTokens: Int? = null,
    cache: Boolean = true,
    numRetries: Int = 3,
    val apiKey: String? = null,
    val baseUrl: String = "https://api.openai.com/v1",
    val orgId: String? = null,
    val timeout: Long = 30000,
    vararg extraKwargs: Pair<String, Any?>,
) : BaseLM(
    model = model,
    modelType = modelType,
    temperature = temperature,
    maxTokens = maxTokens,
    cache = cache,
    numRetries = numRetries,
    extraKwargs = extraKwargs,
) {
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                encodeDefaults = true
                isLenient = true
            })
        }
        install(HttpTimeout) {
            requestTimeoutMillis = timeout
            socketTimeoutMillis = timeout
        }
    }

    private val endpoint: String
        get() = if (modelType == "text") "$baseUrl/completions" else "$baseUrl/chat/completions"

    override suspend fun forward(
        prompt: String?,
        messages: List<Map<String, Any>>?,
        extraKwargs: Map<String, Any?>,
    ): Any {
        val mergedKwargs = (this.kwargs + extraKwargs).filterValues { it != null }
        val body = buildRequestBody(prompt, messages, mergedKwargs)
        return callApi(endpoint, body)
    }

    private fun buildRequestBody(
        prompt: String?,
        messages: List<Map<String, Any>>?,
        kwargs: Map<String, Any?>,
    ): String {
        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"model\":\"${model}\"")
        if (messages != null && messages.isNotEmpty()) {
            sb.append(",\"messages\":[")
            val parts = messages.map { msg ->
                val role = (msg["role"] as? String ?: "user").escape()
                val content = (msg["content"] as? String ?: "").escape()
                "{\"role\":\"$role\",\"content\":\"$content\"}"
            }
            sb.append(parts.joinToString(","))
            sb.append("]")
        } else if (prompt != null) {
            sb.append(",\"prompt\":\"${prompt.escape()}\"")
        }
        (kwargs["temperature"] as? Double)?.let { sb.append(",\"temperature\":$it") }
        (kwargs["maxTokens"] as? Int)?.let { sb.append(",\"max_tokens\":$it") }
        (kwargs["topP"] as? Double)?.let { sb.append(",\"top_p\":$it") }
        (kwargs["n"] as? Int)?.let { sb.append(",\"n\":$it") }
        sb.append("}")
        return sb.toString()
    }

    private suspend fun callApi(url: String, body: String): Any {
        var lastError: Throwable? = null
        for (attempt in 0 until numRetries) {
            try {
                val response = client.post(url) {
                    header("Authorization", "Bearer ${apiKey ?: System.getenv("OPENAI_API_KEY") ?: ""}")
                    contentType(ContentType.Application.Json)
                    if (orgId != null) header("OpenAI-Organization", orgId)
                    setBody(body)
                }
                when (val status = response.status) {
                    in HttpStatusCode.Unauthorized..HttpStatusCode.Forbidden -> throw LMAuthError("Auth failed")
                    HttpStatusCode.TooManyRequests -> { delay(1000L * (1L shl attempt)); continue }
                    in HttpStatusCode.InternalServerError..HttpStatusCode.GatewayTimeout -> {
                        throw dspy.core.LMServerError("Server error: $status")
                    }
                    else -> if (!status.isSuccess()) throw LMTransportError("HTTP error: $status")
                }
                return response.bodyAsText()
            } catch (e: SocketTimeoutException) { lastError = e; delay(1000L * (1L shl attempt)); continue }
            catch (e: ConnectException) { lastError = e; delay(1000L * (1L shl attempt)); continue }
            catch (e: IOException) { lastError = e; delay(1000L * (1L shl attempt)); continue }
            catch (e: Exception) { throw LMUnexpectedError(e.message ?: "Unknown error") }
        }
        throw lastError ?: LMTransportError("All retries exhausted")
    }

    override val supportsFunctionCalling: Boolean = true

    fun close() { client.close() }
}

private fun String.escape(): String = replace("\\", "\\\\").replace("\"", "\\\"")
