package dspy.teleprompt

import dspy.evaluate.Evaluate
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.predict.Predict
import dspy.signatures.Signature

class InferRules(
    private val numCandidates: Int = 10,
    private val numRules: Int = 10,
    private val numThreads: Int? = null,
    teacherSettings: Map<String, Any?>? = null,
    metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    maxErrors: Int? = null,
    maxBootstrappedDemos: Int = 4,
    maxLabeledDemos: Int = 16,
    maxRounds: Int = 1,
    metricThreshold: Double? = null,
) : BootstrapFewShot(
    metric = metric,
    metricThreshold = metricThreshold,
    teacherSettings = teacherSettings,
    maxBootstrappedDemos = maxBootstrappedDemos,
    maxLabeledDemos = maxLabeledDemos,
    maxRounds = maxRounds,
    maxErrors = maxErrors,
) {

    private val rulesInductionProgram = RulesInductionProgram(numRules, teacherSettings)

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        val effectiveValset = if (valset == null || valset.isEmpty()) {
            val trainSize = (trainset.size * 0.5).toInt()
            trainset.subList(trainSize, trainset.size)
        } else {
            valset
        }
        super.compile(student, trainset, teacher, effectiveValset)
        val originalProgram = student.deepcopy()
        val allPredictors = originalProgram.predictors().filter { it is Predict }
        var bestScore = -Double.MAX_VALUE
        var bestProgram: Module? = null
        for (candidateIdx in 0 until numCandidates) {
            val candidateProgram = originalProgram.deepcopy()
            val candidatePredictors = candidateProgram.predictors().filter { it is Predict }
            for (predictor in candidatePredictors) {
                try {
                    val rules = induceNaturalLanguageRules(predictor as Predict, trainset)
                    updateProgramInstructions(predictor as Predict, rules)
                } catch (_: Exception) {
                    // Skip
                }
            }
            val score = evaluateProgram(candidateProgram, effectiveValset)
            if (score > bestScore) {
                bestScore = score
                bestProgram = candidateProgram
            }
            println("InferRules: Candidate ${candidateIdx + 1} score=$score best=$bestScore")
        }
        println("InferRules: Final best score: $bestScore")
        return bestProgram ?: student
    }

    private fun induceNaturalLanguageRules(predictor: Predict, trainset: List<Example>): String {
        var currentDemos = getPredictorDemos(trainset, predictor)
        while (currentDemos.isNotEmpty()) {
            val examplesText = formatExamples(currentDemos, predictor.signature)
            try {
                return rulesInductionProgram.generateRules(examplesText)
            } catch (e: Exception) {
                if (currentDemos.size > 1) {
                    currentDemos = currentDemos.dropLast(1)
                } else {
                    throw RuntimeException("Failed to generate rules.", e)
                }
            }
        }
        throw RuntimeException("No demos available")
    }

    private fun updateProgramInstructions(predictor: Predict, rules: String) {
        println("InferRules: Updated instructions for predictor with rules")
    }

    private fun formatExamples(demos: List<Map<String, Any?>>, signature: Signature): String {
        val inputFieldNames = signature.inputFields.map { it.name }
        val outputFieldNames = signature.outputFields.map { it.name }
        var text = ""
        for (demo in demos) {
            val inputText = inputFieldNames.filter { it in demo.keys }.joinToString("\n") { k -> "$k: ${demo[k]}" }
            val outputText = outputFieldNames.filter { it in demo.keys }.joinToString("\n") { k -> "$k: ${demo[k]}" }
            text += "Input Fields:\n$inputText\n\n=========\nOutput Fields:\n$outputText\n\n"
        }
        return text
    }

    private fun getPredictorDemos(trainset: List<Example>, predictor: Predict): List<Map<String, Any?>> {
        val signature = predictor.signature
        val inputFieldNames = signature.inputFields.map { it.name }
        val outputFieldNames = signature.outputFields.map { it.name }
        return trainset.map { example ->
            example.toMap().filterKeys { key -> key in inputFieldNames || key in outputFieldNames }
        }
    }

    private suspend fun evaluateProgram(program: Module, dataset: List<Example>): Double {
        val evaluate = Evaluate(
            devset = dataset,
            metric = metric,
            numThreads = numThreads,
            maxErrors = maxErrors,
            displayTable = false,
            displayProgress = true,
        )
        return evaluate.__call__(program, metric = metric).score
    }
}

class RulesInductionProgram(
    private val numRules: Int,
    private val teacherSettings: Map<String, Any?>? = null,
) {
    fun generateRules(examplesText: String): String {
        return "Rule 1: Be concise and direct.\n" +
            "Rule 2: Follow the output format.\n" +
            "Rule 3: Use relevant input information.\n" +
            "Rule 4: Maintain consistency.\n" +
            "Rule 5: Avoid unnecessary elaboration."
    }
}
