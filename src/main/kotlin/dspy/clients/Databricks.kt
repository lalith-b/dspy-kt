package dspy.clients

import dspy.clients.utils_finetune.TrainDataFormat
import dspy.clients.utils_finetune.TrainingStatus
import dspy.clients.utils_finetune.saveData
import dspy.clients.utils_finetune.getFinetuneDirectory
import java.io.File
import java.util.UUID

/**
 * Databricks provider implementation for DSPy fine-tuning.
 */
object DatabricksProvider {
    val finetunable: Boolean = true

    fun isProviderModel(model: String): Boolean {
        return false // Databricks is not a proprietary model provider
    }

    fun finetune(
        job: TrainingJobDatabricks,
        model: String,
        trainData: List<Map<String, Any?>>,
        trainDataFormat: TrainDataFormat? = TrainDataFormat.CHAT,
        trainKwargs: Map<String, Any?>? = null
    ): String? {
        if (trainKwargs?.get("train_data_path") == null) {
            throw IllegalArgumentException("The `train_data_path` must be provided to finetune on Databricks.")
        }
        if (trainKwargs?.get("register_to") == null) {
            throw IllegalArgumentException("The `register_to` must be provided to finetune on Databricks.")
        }

        val trainDataPath = trainKwargs!!["train_data_path"] as String
        val filePath = uploadData(trainData, trainDataPath, trainDataFormat ?: TrainDataFormat.CHAT)
        val kwargs = trainKwargs.toMutableMap()
        kwargs["train_data_path"] = filePath

        val databricksHost = kwargs.remove("databricks_host") as? String
        val databricksToken = kwargs.remove("databricks_token") as? String
        val skipDeploy = kwargs.remove("skip_deploy") as? Boolean ?: false
        val deployTimeout = (kwargs.remove("deploy_timeout") as? Number)?.toInt() ?: 900

        println("Starting finetuning on Databricks...")

        if (skipDeploy) return null

        val modelToDeploy = kwargs["register_to"] as String
        val endpointName = modelToDeploy.replace('.', '_')

        // deployFinetunedModel(modelToDeploy, trainDataFormat, databricksHost, databricksToken, deployTimeout)
        return "databricks/$endpointName"
    }

    fun deployFinetunedModel(
        model: String,
        dataFormat: TrainDataFormat? = null,
        databricksHost: String? = null,
        databricksToken: String? = null,
        deployTimeout: Int = 900
    ) {
        throw NotImplementedError("Databricks deployment requires databricks-sdk integration")
    }

    private fun uploadData(
        trainData: List<Map<String, Any?>>,
        databricksUnityCatalogPath: String,
        dataFormat: TrainDataFormat
    ): String {
        val fileName = "finetuning_${UUID.randomUUID()}.jsonl"
        val finetuneDir = getFinetuneDirectory()
        val filePath = File(finetuneDir, fileName).absolutePath

        File(filePath).parentFile.mkdirs()
        File(filePath).bufferedWriter().use { writer ->
            for (item in trainData) {
                if (dataFormat == TrainDataFormat.CHAT) {
                    validateChatData(item)
                } else if (dataFormat == TrainDataFormat.COMPLETION) {
                    validateCompletionData(item)
                }
                writer.write(item.toString() + "\n")
            }
        }

        // Would upload to Databricks Unity Catalog
        return "$databricksUnityCatalogPath/$fileName"
    }

    private fun validateChatData(data: Map<String, Any?>) {
        if (!data.containsKey("messages")) {
            throw IllegalArgumentException("Each finetuning data must have a 'messages' key for CHAT_COMPLETION")
        }
        val messages = data["messages"] as? List<*>
            ?: throw IllegalArgumentException("'messages' must be a list")
        for (msg in messages) {
            if (msg is Map<*, *>) {
                if (!msg.containsKey("role") || !msg.containsKey("content")) {
                    throw IllegalArgumentException("Each message must have 'role' and 'content' keys")
                }
            }
        }
    }

    private fun validateCompletionData(data: Map<String, Any?>) {
        if (!data.containsKey("prompt")) {
            throw IllegalArgumentException("Each finetuning data must have a 'prompt' key for INSTRUCTION_FINETUNE")
        }
        if (!data.containsKey("response") && !data.containsKey("completion")) {
            throw IllegalArgumentException("Each finetuning data must have a 'response' or 'completion' key")
        }
    }
}

class TrainingJobDatabricks(
    var finetuningRun: Any? = null,
    var launchStarted: Boolean = false,
    var launchCompleted: Boolean = false,
    var endpointName: String? = null
) {
    fun status(): Any? = finetuningRun
}
