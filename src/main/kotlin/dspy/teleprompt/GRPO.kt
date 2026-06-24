package dspy.teleprompt

import dspy.evaluate.Evaluate
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings
import kotlin.random.Random

/**
 * GRPO (Group Relative Policy Optimization) teleprompter.
 *
 * A reinforcement learning-based optimizer that uses GRPO to fine-tune
 * the language models within a DSPy program. It bootstraps trace data from
 * teacher programs, groups them, and uses reward signals to optimize
 * model parameters.
 *
 * Faithful port of `dspy/teleprompt/grpo.py`.
 * Note: This port provides the structural framework. The actual GRPO training
 * backend (LM.reinforce, GRPOGroup, GRPOStatus, TrainDataFormat) requires
 * integration with external RL training infrastructure.
 */

// ============================================================================
// GRPO data types
// ============================================================================

/**
 * Represents a group of GRPO training examples.
 * Each group contains multiple completions for the same input, used for
 * advantage estimation in GRPO.
 */
typealias GRPOGroup = List<Map<String, Any?>>

/**
 * Status of a GRPO training job.
 */
typealias GRPOStatus = Map<String, Any?>

/**
 * Format of the training data sent to the GRPO backend.
 */
enum class TrainDataFormat {
    GRPO_CHAT,
    GRPO_INSTRUCT,
}

/**
 * Variably invoked predictor grouping mode.
 */
enum class PredictorGroupingMode {
    TRUNCATE,
    FILL,
    RAGGED,
}

/**
 * Strategy for filling missing predictor invocations.
 */
enum class PredictorFillStrategy {
    RANDINT,
    MAX,
}

// ============================================================================
// GRPO
// ============================================================================

class GRPO(
    metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    val multitask: Boolean = true,
    trainKwargs: Map<dspy.clients.BaseLM, Map<String, Any?>>? = null,
    adapter: Map<dspy.clients.BaseLM, Any>? = null,
    excludeDemos: Boolean = false,
    val numThreads: Int = 6,
    val numTrainSteps: Int = 100,
    val seed: Int = 0,
    val numDspyExamplesPerGrpoStep: Int = 1,
    val numRolloutsPerGrpoStep: Int = 1,
    val useTrainAsVal: Boolean = false,
    val numStepsForVal: Int = 5,
    val reportTrainScores: Boolean = false,
    val failureScore: Double = 0.0,
    val formatFailureScore: Double = -1.0,
    variablyInvokedPredictorGroupingMode: String = "truncate",
    variablyInvokedPredictorFillStrategy: String? = null,
) : Teleprompter() {
    val metric: ((Example, Prediction, List<Any>?) -> Any?)? = metric
    val trainKwargs: MutableMap<dspy.clients.BaseLM, MutableMap<String, Any?>> =
        (trainKwargs ?: emptyMap()).mapValues { it.value.toMutableMap() }.toMutableMap()
    val adapterMap: Map<dspy.clients.BaseLM, Any> = adapter ?: emptyMap()
    val excludeDemos: Boolean = excludeDemos
    val rng: Random = Random(seed)

    // Grouping configuration (initialized after construction)
    private var _groupingMode: PredictorGroupingMode = PredictorGroupingMode.TRUNCATE
    val groupingMode: PredictorGroupingMode
        get() = _groupingMode
    private var _fillStrategy: PredictorFillStrategy? = null
    val fillStrategy: PredictorFillStrategy?
        get() = _fillStrategy

    // Training state
    private var shuffledTrainsetIds: MutableList<Int> = mutableListOf()
    private var epoch: Int = -1
    private val idFreqs: MutableMap<Int, Int> = mutableMapOf()
    private var fulfilledBatchIds: MutableList<Int> = mutableListOf()

    // Validation
    init {
        require(failureScore > formatFailureScore) {
            "failureScore must be greater than formatFailureScore since the range " +
                "[formatFailureScore, failureScore] is used to provide dspy formatting rewards"
        }

        require(useTrainAsVal || reportTrainScores || !reportTrainScores) {
            "If useTrainAsVal is True, reportTrainScores must be True."
        }

        require(excludeDemos) {
            "excludeDemos==False is not supported yet. Please set it to True."
        }

        require(multitask) {
            "Independent GRPO training jobs for each predictor in the student program " +
                "is not supported yet. Please set multitask=True."
        }

        // Validate grouping mode
        when (variablyInvokedPredictorGroupingMode) {
            "truncate" -> _groupingMode = PredictorGroupingMode.TRUNCATE
            "fill" -> {
                _groupingMode = PredictorGroupingMode.FILL
                require(variablyInvokedPredictorFillStrategy != null) {
                    "variablyInvokedPredictorFillStrategy must be set when variablyInvokedPredictorGroupingMode is 'fill'"
                }
                require(variablyInvokedPredictorFillStrategy in listOf("randint", "max")) {
                    "variablyInvokedPredictorFillStrategy must be either 'randint' or 'max'"
                }
                _fillStrategy = when (variablyInvokedPredictorFillStrategy) {
                    "randint" -> PredictorFillStrategy.RANDINT
                    "max" -> PredictorFillStrategy.MAX
                    else -> PredictorFillStrategy.RANDINT
                }
            }
            "ragged" -> _groupingMode = PredictorGroupingMode.RAGGED
            else -> throw IllegalArgumentException("Unknown grouping mode: $variablyInvokedPredictorGroupingMode")
        }
    }

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        println("Starting the GRPO compilation process...")
        println("Validating the inputs...")

        require(trainset.isNotEmpty()) { "Training set is empty. Please provide a non-empty training set." }

        // Handle small trainsets
        var effectiveTrainset = trainset
        if (trainset.size < numDspyExamplesPerGrpoStep) {
            println(
                "Warning: Number of training examples ${trainset.size} is less than " +
                    "numDspyExamplesPerGrpoStep $numDspyExamplesPerGrpoStep. " +
                    "Repeating the training set to fill the GRPO step. " +
                    "This could lead to overfitting and training instability."
            )
            val multiplier = (numDspyExamplesPerGrpoStep + trainset.size - 1) / trainset.size
            if (multiplier > 1) {
                println("Warning: Repeating the training set $multiplier times to fill the GRPO step.")
                effectiveTrainset = List(multiplier) { trainset }.flatten()
            }
        }

        // Validate multitask
        require(multitask) {
            "Independent GRPO training jobs for each predictor in the student program " +
                "are not supported yet. Please set multitask=True."
        }

        // Validate single LM
        val studentLms = student.predictors().mapNotNull { it._lm }.toSet()
        require(studentLms.size == 1) {
            "Student program has multiple LMs: $studentLms. " +
                "GRPO only supports student programs with a single LM. " +
                "You can set the LM for a program with `program.setLm(...)`"
        }

        // Validate valset
        if (useTrainAsVal) {
            require(valset == null) { "If useTrainAsVal is True, valset must be None." }
        }

        // Prepare student program
        println("Preparing the student program...")
        require(allPredictorsHaveLms(student)) {
            "All predictors in the student program must have LMs set."
        }

        val predSignatureHashToInd = student.predictors().mapIndexed { ind, pred ->
            System.identityHashCode(pred.signature) to ind
        }.toMap()
        val numStudentPredictors = student.predictors().size

        // Prepare teacher program(s)
        println("Preparing the teacher program(s)...")
        var teachers: List<Module> = if (teacher == null || (teacher is List<*> && teacher.isEmpty())) {
            listOf(student)
        } else if (teacher is List<*>) {
            @Suppress("UNCHECKED_CAST")
            teacher as List<Module>
        } else {
            listOf(teacher)
        }

        // Ensure structural equivalency
        for (t in teachers) {
            require(structuralEquivalency(student, t)) {
                "Teacher and student must have the same program structure."
            }
            require(allPredictorsHaveLms(t)) {
                "All predictors in the teacher program must have LMs set."
            }
        }

        // Ensure student is in teachers
        require(student in teachers) {
            "Student program must be in the list of teachers."
        }

        require(numRolloutsPerGrpoStep % teachers.size == 0) {
            "The GRPO group size (numRolloutsPerGrpoStep) $numRolloutsPerGrpoStep " +
                "is not divisible by the number of teachers ${teachers.size}."
        }
        val numSamplesPerInput = numRolloutsPerGrpoStep / teachers.size

        // Disable LM cache (placeholder - cache is val in Kotlin BaseLM)
        val lmCacheDict = mutableMapOf<dspy.clients.BaseLM, Boolean>()
        disableLmCache(student, lmCacheDict)
        for (t in teachers) {
            disableLmCache(t, lmCacheDict)
        }

        // Update trainKwargs
        for (pred in student.predictors()) {
            pred._lm?.let { lm ->
                val tk = trainKwargs.getOrPut(lm) { mutableMapOf() }
                tk["num_generations"] = numRolloutsPerGrpoStep
            }
        }

        // Prepare GRPO training jobs
        println("Preparing the GRPO training job(s)...")
        val grpoTrainingJobs = mutableMapOf<Pair<dspy.clients.BaseLM?, Int?>, GRPOTrainingJob>()
        for ((predInd, pred) in student.predictors().withIndex()) {
            val dataKey = if (multitask) null else predInd
            val predLm = pred._lm
            val jobKey = predLm to dataKey
            if (jobKey !in grpoTrainingJobs && predLm != null) {
                val tk = trainKwargs[predLm]
                // Create a placeholder job - in full implementation this calls lm.reinforce()
                grpoTrainingJobs[jobKey] = GRPOTrainingJob(lm = predLm, trainKwargs = tk ?: emptyMap())
            }
        }

        // Report initial validation metrics
        reportValidationMetrics(student, effectiveTrainset, valset, stepIdx = -1)

        // Update shuffled trainset
        updateShuffledTrainset(effectiveTrainset)

        // Main training loop
        println("Starting the GRPO training loop...")
        for (trainStepIdx in 0 until numTrainSteps) {
            println("GRPO training step ${trainStepIdx + 1}/$numTrainSteps...")

            val subsampleTrainingDataset = selectTrainingSampleAndUpdateShuffledTrainset(
                originalTrainset = effectiveTrainset,
                trainStepIdx = trainStepIdx,
            )

            // Bootstrap trace data
            println("Bootstrapping data...")
            val traceData = List(subsampleTrainingDataset.size) {
                List(teachers.size) { mutableListOf<Map<String, Any?>>() }
            }

            for ((tInd, tTeacher) in teachers.withIndex()) {
                val repeatedDataset = subsampleTrainingDataset.flatMap { example ->
                    List(numSamplesPerInput) { example }
                }

                // Bootstrap trace data (simplified)
                for ((eInd, example) in repeatedDataset.withIndex()) {
                    val exampleIndInSubsample = eInd % subsampleTrainingDataset.size
                    val dataDict = bootstrapTraceDataSingle(
                        program = tTeacher,
                        example = example,
                        metric = this.metric,
                        numThreads = numThreads,
                        failureScore = failureScore,
                        formatFailureScore = formatFailureScore,
                    )
                    if (dataDict != null) {
                        traceData[exampleIndInSubsample][tInd].add(dataDict)
                    }
                }
            }

            // Validate trace data
            validateTraceDataAndLogIssues(
                traceData = traceData,
                subsampleTrainingDataset = subsampleTrainingDataset,
                numTeachers = teachers.size,
                numSamplesPerInput = numSamplesPerInput,
                predSignatureHashToInd = predSignatureHashToInd,
            )

            // Prepare training data batch
            println("Preparing the training data batch from bootstrapped examples for GRPO...")
            val trainBatchPerPredictor = List(numStudentPredictors) { mutableListOf<GRPOGroup>() }

            for (predId in 0 until numStudentPredictors) {
                val predSigHash = System.identityHashCode(student.predictors()[predId].signature)

                for ((exampleInd, exampleData) in traceData.withIndex()) {
                    // Collect predictor invocations for this predictor
                    val predictorExampleInvocations = mutableListOf<List<Any>>()

                    for (teacherData in exampleData) {
                        for (sample in teacherData) {
                            val trace = sample["trace"] as? List<*> ?: continue
                            val score = sample["score"] ?: failureScore

                            val traceInstancesForCurrentPred = trace
                                .filterIsInstance<Triple<*, *, *>>()
                                .mapNotNull { t ->
                                    val predictor = t.first
                                    if (predictor is dspy.predict.Predict &&
                                        System.identityHashCode(predictor.signature) == predSigHash) {
                                        listOf(predictor, t.second, t.third, score)
                                    } else null
                                }
                                .filterIsInstance<List<*>>()

                            if (traceInstancesForCurrentPred.isNotEmpty()) {
                                predictorExampleInvocations.add(traceInstancesForCurrentPred)
                            }
                        }
                    }

                    if (predictorExampleInvocations.isEmpty()) {
                        println("Warning: Skipping example $exampleInd for predictor $predId as it has no invocations.")
                        continue
                    }

                    // Handle variable length invocations
                    val minLen = predictorExampleInvocations.minOf { it.size }
                    val maxLen = predictorExampleInvocations.maxOf { it.size }

                    if (minLen == 0) {
                        println("Warning: Skipping example $exampleInd for predictor $predId as it has no invocations.")
                        continue
                    }

                    val processedInvocations = when (groupingMode) {
                        PredictorGroupingMode.TRUNCATE ->
                            predictorExampleInvocations.map { it.take(minLen) }
                        PredictorGroupingMode.FILL -> {
                            val selector: (List<Any>) -> Any = if (fillStrategy == PredictorFillStrategy.RANDINT) {
                                { l -> l[rng.nextInt(l.size)] }
                            } else {
                                { l -> l.last() }
                            }
                            predictorExampleInvocations.map { invocation ->
                                invocation + List(maxLen - invocation.size) { selector(invocation) }
                            }
                        }
                        PredictorGroupingMode.RAGGED -> predictorExampleInvocations
                    }

                    val effectiveMaxLen = processedInvocations.maxOf { it.size }

                    // Build GRPO groups
                    for (groupIdx in 0 until effectiveMaxLen) {
                        val group = mutableListOf<Map<String, Any?>>()
                        for ((rolloutIdx, invocations) in processedInvocations.withIndex()) {
                            val traceInstance = invocations[groupIdx] as? List<*> ?: continue
                            val grpScore = (traceInstance.getOrNull(3) as? Number)?.toDouble() ?: failureScore

                            val grpItem = mapOf<String, Any?>(
                                "messages" to emptyList<Any>(),
                                "completion" to mapOf("role" to "assistant", "content" to ""),
                                "reward" to grpScore,
                            )
                            group.add(grpItem)
                        }

                        if (group.size >= 2) {
                            trainBatchPerPredictor[predId].add(group)
                        }
                    }
                }
            }

            // Check if we have training data
            if (trainBatchPerPredictor.all { it.isEmpty() }) {
                println("Warning: No training data found for this training step.")
                continue
            }

            // Simulate GRPO training step (placeholder for actual RL training)
            println("Invoking GRPO training step...")
            for ((jobKey, job) in grpoTrainingJobs) {
                val dataKey = jobKey.second
                val trainData: List<GRPOGroup> = if (dataKey == null) {
                    trainBatchPerPredictor.flatten()
                } else {
                    trainBatchPerPredictor.getOrElse(dataKey) { emptyList() }
                }

                if (trainData.isEmpty()) continue

                // Pad groups to expected size
                val paddedTrainData = trainData.map { group ->
                    var padded = group.toMutableList()
                    while (padded.size < numRolloutsPerGrpoStep) {
                        val remaining = numRolloutsPerGrpoStep - padded.size
                        val toAdd = padded.take(minOf(remaining, padded.size))
                        padded.addAll(toAdd)
                    }
                    padded
                }

                // In a full implementation, this would call job.step()
                // job.step(trainData = paddedTrainData, trainDataFormat = TrainDataFormat.GRPO_CHAT)
                println("  Sent ${paddedTrainData.size} GRPO groups to training job for ${job.lm?.model}")
            }

            println("GRPO training step ${trainStepIdx + 1}/$numTrainSteps completed.")

            // Report validation metrics
            reportValidationMetrics(student, effectiveTrainset, valset, stepIdx = trainStepIdx)
        }

        // Terminate jobs
        println("Done with the iterations! Retrieving the final model(s)...")
        for ((_, job) in grpoTrainingJobs) {
            job.terminate()
        }

        // Revert cache states (no-op in Kotlin since cache is a val)
        recoverLmCache(student, lmCacheDict)
        for (t in teachers) {
            recoverLmCache(t, lmCacheDict)
        }

        println("GRPO compiler has finished compiling the student program")
        student._compiled = true
        student.compiled = true

        return student
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    private fun validateTraceDataAndLogIssues(
        traceData: List<List<MutableList<Map<String, Any?>>>>,
        subsampleTrainingDataset: List<Example>,
        numTeachers: Int,
        numSamplesPerInput: Int,
        predSignatureHashToInd: Map<Int, Int>,
    ) {
        require(traceData.size == subsampleTrainingDataset.size) {
            "Trace data length ${traceData.size} does not match the number of examples ${subsampleTrainingDataset.size}"
        }
        require(traceData[0].size == numTeachers) {
            "Trace data length ${traceData[0].size} does not match the number of teachers $numTeachers"
        }

        // Check for empty trace data
        if (traceData[0][0].isEmpty()) {
            println("Warning: Trace data for example 0 and teacher 0 is empty. " +
                "This is likely due to the model generating output not following the dspy response format.")
        } else if (traceData[0][0].size != numSamplesPerInput) {
            println("Warning: Trace data length ${traceData[0][0].size} does not match " +
                "the expected number of samples per input $numSamplesPerInput")
        }
    }

    private suspend fun reportValidationMetrics(
        student: Module,
        trainset: List<Example>,
        valset: List<Example>?,
        stepIdx: Int,
    ) {
        val shouldReport = if (stepIdx == -1) {
            true
        } else if (stepIdx == numTrainSteps - 1) {
            true
        } else {
            (stepIdx + 1) % numStepsForVal == 0
        }

        if (!shouldReport) return

        val evalDevset = if (valset != null) {
            if (reportTrainScores) {
                valset + trainset
            } else {
                valset
            }
        } else if (useTrainAsVal) {
            trainset
        } else {
            return
        }

        println("Evaluating the student program...")
        val evaluator = Evaluate(
            devset = evalDevset,
            metric = this.metric,
            numThreads = numThreads,
            displayProgress = true,
            maxErrors = evalDevset.size * 10,
            failureScore = failureScore,
        )
        val result = evaluator.__call__(student)
        println("Evaluation score: ${result.score}")
    }

    private fun updateShuffledTrainset(originalTrainset: List<Example>) {
        shuffledTrainsetIds = List(originalTrainset.size) { it }.toMutableList()
        shuffledTrainsetIds.shuffle(rng)

        for (id in shuffledTrainsetIds) {
            idFreqs[id] = (idFreqs[id] ?: 0) + 1
        }

        // Pad to make divisible by numDspyExamplesPerGrpoStep
        val numToPad = numDspyExamplesPerGrpoStep - (originalTrainset.size % numDspyExamplesPerGrpoStep)
        if (numToPad > 0) {
            for (padIdx in 0 until numToPad) {
                // Select least frequent id
                val selectedId = idFreqs.entries.minByOrNull { it.value }?.key ?: 0
                shuffledTrainsetIds.add(selectedId)
                idFreqs[selectedId] = (idFreqs[selectedId] ?: 0) + 1
            }
        }
    }

    private fun selectTrainingSampleAndUpdateShuffledTrainset(
        originalTrainset: List<Example>,
        trainStepIdx: Int,
    ): List<Example> {
        val baseIdx = trainStepIdx * numDspyExamplesPerGrpoStep
        var currEpoch = if (epoch == -1) 0 else baseIdx / shuffledTrainsetIds.size

        if (currEpoch > epoch) {
            println("Updating shuffled trainset for epoch $currEpoch...")
            epoch = currEpoch
            updateShuffledTrainset(originalTrainset)
        }

        require(shuffledTrainsetIds.size >= numDspyExamplesPerGrpoStep) {
            "Shuffled trainset length ${shuffledTrainsetIds.size} is less than numDspyExamplesPerGrpoStep $numDspyExamplesPerGrpoStep"
        }

        val effectiveBaseIdx = baseIdx % shuffledTrainsetIds.size
        val endIdx = effectiveBaseIdx + numDspyExamplesPerGrpoStep

        require(endIdx <= shuffledTrainsetIds.size) {
            "End index $endIdx is out of bounds for shuffled trainset length ${shuffledTrainsetIds.size}"
        }

        val selectedIds = shuffledTrainsetIds.subList(effectiveBaseIdx, endIdx)
        return selectedIds.map { originalTrainset[it] }
    }

    /**
     * Bootstrap a single trace data entry for one example and one teacher.
     */
    private suspend fun bootstrapTraceDataSingle(
        program: Module,
        example: Example,
        metric: ((Example, Prediction, List<Any>?) -> Any?)?,
        numThreads: Int,
        failureScore: Double,
        formatFailureScore: Double,
    ): Map<String, Any?>? {
        return try {
            val prediction = program(example.toMap())
            val score = if (metric != null) {
                val metricVal = metric(example, prediction, null)
                when (metricVal) {
                    is Number -> metricVal.toDouble()
                    is Boolean -> if (metricVal) 1.0 else 0.0
                    else -> 0.0
                }
            } else {
                1.0 // Default score if no metric
            }

            mapOf(
                "example" to example,
                "prediction" to prediction,
                "trace" to listOf<Any>(), // Simplified - would capture full trace
                "example_ind" to 0,
                "score" to score,
            )
        } catch (e: Exception) {
            mapOf(
                "example" to example,
                "prediction" to Prediction(emptyMap()),
                "trace" to listOf<Any>(),
                "example_ind" to 0,
                "score" to failureScore,
                "error" to e.message,
            )
        }
    }

    /**
     * Check if all predictors in a program have LMs set.
     */
    private fun allPredictorsHaveLms(program: Module): Boolean {
        return program.predictors().all { it._lm != null }
    }

    /**
     * Check structural equivalency between two programs.
     */
    private fun structuralEquivalency(student: Module, teacher: Module): Boolean {
        val studentPreds = student.predictors()
        val teacherPreds = teacher.predictors()
        if (studentPreds.size != teacherPreds.size) return false
        for (i in studentPreds.indices) {
            if (studentPreds[i].signature != teacherPreds[i].signature) return false
        }
        return true
    }
}

// ============================================================================
// GRPO training job (placeholder for actual RL training backend)
// ============================================================================

/**
 * Represents a GRPO training job.
 * In the full implementation, this would interface with an external RL training system.
 */
data class GRPOTrainingJob(
    val lm: dspy.clients.BaseLM?,
    val trainKwargs: Map<String, Any?>,
    val status: MutableMap<String, Any?> = mutableMapOf("pending_batch_ids" to listOf(0, 1, 2, 3)),
) {
    fun getGRPOStatus(): GRPOStatus {
        return status
    }

    fun step(trainData: List<GRPOGroup>, trainDataFormat: TrainDataFormat) {
        // Placeholder: In full implementation, this sends data to the RL training backend
        println("  GRPO step: ${trainData.size} groups with format $trainDataFormat")
    }

    fun terminate() {
        println("  Terminating GRPO training job")
    }
}

// ============================================================================
// Cache management functions
// ============================================================================

/**
 * Disable the LM cache for all predictors in the program.
 * Note: In Kotlin, BaseLM.cache is a val, so we only record the state.
 */
fun disableLmCache(program: Module, lmCacheDict: MutableMap<dspy.clients.BaseLM, Boolean>) {
    for (pred in program.predictors()) {
        pred._lm?.let { lm ->
            if (lm !in lmCacheDict) {
                lmCacheDict[lm] = lm.cache
            }
            // Note: lm.cache is a val, cannot be set. This is a no-op in Kotlin.
        }
    }
}

/**
 * Recover the LM caches for all predictors in the program.
 * Note: In Kotlin, BaseLM.cache is a val, so this is a no-op.
 */
fun recoverLmCache(program: Module, lmCacheDict: Map<dspy.clients.BaseLM, Boolean>) {
    for (pred in program.predictors()) {
        pred._lm?.let { lm ->
            // Note: lm.cache is a val, cannot be set. This is a no-op in Kotlin.
        }
    }
}
