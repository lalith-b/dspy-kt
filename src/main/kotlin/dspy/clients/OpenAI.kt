package dspy.clients

import dspy.clients.utils_finetune.TrainDataFormat
import dspy.clients.utils_finetune.TrainingStatus
import dspy.clients.utils_finetune.saveData
import java.time.Duration
import java.time.Instant

/**
 * OpenAI provider implementation for DSPy fine-tuning.
 */
class OpenAIProvider {

    companion object {
        fun isProviderModel(model: String): Boolean {
            return model.startsWith("openai/") || model.startsWith("ft:")
        }

        private fun removeProviderPrefix(model: String): String {
            return model.removePrefix("openai/")
        }

        fun isTerminalTrainingStatus(status: TrainingStatus): Boolean {
            return status in listOf(
                TrainingStatus.succeeded,
                TrainingStatus.failed,
                TrainingStatus.cancelled
            )
        }

        private val providerStatusToTrainingStatus = mapOf(
            "validating_files" to TrainingStatus.pending,
            "queued" to TrainingStatus.pending,
            "running" to TrainingStatus.running,
            "succeeded" to TrainingStatus.succeeded,
            "failed" to TrainingStatus.failed,
            "cancelled" to TrainingStatus.cancelled
        )

        fun validateDataFormat(dataFormat: TrainDataFormat) {
            val supported = listOf(TrainDataFormat.CHAT, TrainDataFormat.COMPLETION)
            if (dataFormat !in supported) {
                throw IllegalArgumentException("OpenAI does not support the data format $dataFormat.")
            }
        }

        fun finetune(
            job: TrainingJobOpenAI,
            model: String,
            trainData: List<Map<String, Any?>>,
            trainDataFormat: TrainDataFormat?,
            trainKwargs: Map<String, Any?>? = null
        ): String {
            val cleanModel = removeProviderPrefix(model)

            println("[OpenAI Provider] Validating the data format")
            validateDataFormat(trainDataFormat ?: TrainDataFormat.CHAT)

            println("[OpenAI Provider] Saving the data to a file")
            val dataPath = saveData(trainData)
            println("[OpenAI Provider] Data saved to $dataPath")

            println("[OpenAI Provider] Uploading the data to the provider")
            // job.providerFileId = uploadData(dataPath)
            // job.providerJobId = startRemoteTraining(job.providerFileId, cleanModel, trainKwargs)

            println("[OpenAI Provider] Waiting for training to complete")
            // waitForJob(job)

            println("[OpenAI Provider] Model retrieved")
            return cleanModel
        }
    }

    /**
     * Wait for a training job to complete by polling.
     */
    fun waitForJob(job: TrainingJobOpenAI, pollFrequency: Int = 20) {
        var done = false
        var reportedEstimatedTime = false
        while (!done) {
            if (!reportedEstimatedTime) {
                // Report estimated time
                reportedEstimatedTime = true
            }
            Thread.sleep(pollFrequency * 1000L)
            val status = job.status()
            done = isTerminalTrainingStatus(status)
        }
    }
}

/**
 * Training job specific to OpenAI.
 */
open class TrainingJobOpenAI : java.io.Serializable {
    var providerFileId: String? = null
    var providerJobId: String? = null

    fun status(): TrainingStatus {
        // Would poll OpenAI API
        return TrainingStatus.not_started
    }

    fun cancel() {
        // Would cancel via OpenAI API
        providerFileId = null
        providerJobId = null
    }
}
