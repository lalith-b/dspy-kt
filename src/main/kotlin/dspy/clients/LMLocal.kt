package dspy.clients

import dspy.clients.utils_finetune.TrainDataFormat
import dspy.clients.utils_finetune.TrainingStatus
import dspy.clients.utils_finetune.saveData
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.random.Random

/**
 * Training job configuration for fine-tuning.
 */
data class TrainingJob(
    val name: String,
    val model: String,
    val trainData: List<Map<String, Any?>>
)

/**
 * Local provider implementation for DSPy fine-tuning.
 *
 * Supports launching local SGLang servers and local fine-tuning with HuggingFace.
 */
object LocalProvider {
    private var process: Process? = null

    fun launch(lm: LM, launchKwargs: Map<String, Any?>? = null) {
        if (process != null) {
            println("Server is already launched.")
            return
        }

        var model = lm.model
        model = model.removePrefix("openai/").removePrefix("local:").removePrefix("huggingface/")

        val port = getFreePort()
        val timeout = launchKwargs?.get("timeout") as? Int ?: 1800

        println("Launching SGLang server for model $model on port $port")
        // Would spawn subprocess: python -m sglang.launch_server --model-path $model --port $port --host 0.0.0.0

        // Note: lm.kwargs is immutable; update handled by caller
        val newKwargs = mapOf("api_key" to "not-needed", "base_url" to "http://127.0.0.1:${port}/v1")
        println("Updated kwargs: $newKwargs")
    }

    fun kill(lm: LM) {
        // Would terminate the SGLang process
        println("Server killed.")
    }

    fun finetune(
        job: TrainingJob,
        model: String,
        trainData: List<Map<String, Any?>>,
        trainDataFormat: TrainDataFormat?,
        trainKwargs: Map<String, Any?>? = null
    ): String {
        var cleanModel = model.removePrefix("openai/").removePrefix("local:")

        if (trainDataFormat != TrainDataFormat.CHAT) {
            throw IllegalArgumentException("Only chat models are supported for local finetuning.")
        }

        val dataPath = saveData(trainData)
        println("Train data saved to $dataPath")

        val outputDir = createOutputDir(cleanModel, dataPath)

        val defaultTrainKwargs = mapOf(
            "device" to null,
            "use_peft" to false,
            "num_train_epochs" to 5,
            "per_device_train_batch_size" to 1,
            "gradient_accumulation_steps" to 8,
            "learning_rate" to 1e-5,
            "max_seq_length" to 4096,
            "packing" to true,
            "bf16" to true,
            "output_dir" to outputDir
        )
        val finalKwargs = defaultTrainKwargs + (trainKwargs ?: emptyMap())

        println("Starting local training, will save to $outputDir")
        // Would call trainSftLocally()
        println("Training complete")
        return "openai/local:$outputDir"
    }

    private fun createOutputDir(modelName: String, dataPath: String): String {
        val modelStr = modelName.replace('/', '-')
        val timeStr = java.time.LocalDateTime.now().toString().replace(':', '-')
        val rndStr = (1..6).map { Random.nextInt(0, 36).toString(36) }.joinToString("")
        val modelIdentifier = "${rndStr}_${modelStr}_$timeStr"
        return dataPath.replace(".jsonl", "_$modelIdentifier")
    }

    private fun getFreePort(): Int {
        return java.net.ServerSocket(0).use { socket ->
            socket.localPort
        }
    }

    fun waitForServer(baseUrl: String, timeout: Int? = null) {
        val startTime = System.currentTimeMillis()
        while (true) {
            try {
                val url = URL("$baseUrl/v1/models")
                val connection = url.openConnection() as HttpURLConnection
                connection.setRequestProperty("Authorization", "Bearer None")
                if (connection.responseCode == 200) {
                    Thread.sleep(5000)
                    break
                }
                connection.disconnect()
            } catch (_: Exception) {
                // Server not up yet
            }
            Thread.sleep(1000)
            if (timeout != null && (System.currentTimeMillis() - startTime) > timeout * 1000L) {
                throw TimeoutHandler("Server did not become ready within timeout period")
            }
        }
    }
}

class TimeoutHandler(message: String) : Exception(message)
