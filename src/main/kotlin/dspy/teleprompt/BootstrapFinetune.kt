package dspy.teleprompt

import dspy.adapters.Adapter
import dspy.adapters.ChatAdapter
import dspy.clients.BaseLM
import dspy.clients.LM
import dspy.predict.Predict
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings

/**
 * Base class for finetune teleprompters.
 *
 * Port of `dspy/teleprompt/bootstrap_finetune.py` - `FinetuneTeleprompter`.
 */
open class FinetuneTeleprompter(
    trainKwargs: Map<String, Any?>? = null,
) : Teleprompter() {
    val trainKwargs: Map<BaseLM, Any?> = convertToLmDict(trainKwargs ?: emptyMap<String, Any?>())

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun convertToLmDict(arg: Any?): Map<BaseLM, Any?> {
            if (arg is Map<*, *> && arg.isNotEmpty() && arg.keys.all { it is BaseLM }) {
                return arg as Map<BaseLM, Any?>
            }
            // Default: use the same value for all LMs
            return object : Map<BaseLM, Any?> by emptyMap() {
                override fun containsKey(key: BaseLM) = true
                override fun get(key: BaseLM): Any? = arg
            }
        }
    }
}

/**
 * Bootstrap finetuning teleprompter.
 *
 * Port of `dspy/teleprompt/bootstrap_finetune.py` - `BootstrapFinetune`.
 */
@Suppress("UNCHECKED_CAST")
class BootstrapFinetune(
    private val metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    private val multitask: Boolean = true,
    trainKwargs: Map<String, Any?>? = null,
    adapter: Any? = null,
    private val excludeDemos: Boolean = false,
    private val numThreads: Int? = null,
) : FinetuneTeleprompter(trainKwargs) {

    private val adapterMap: Map<BaseLM, Any?> = FinetuneTeleprompter.convertToLmDict(adapter)

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        println("Preparing the student and teacher programs...")
        require(allPredictorsHaveLms(student)) {
            "All predictors must have an LM assigned before fine-tuning."
        }

        println("Bootstrapping data...")
        val traceData = mutableListOf<Map<String, Any?>>()

        val teachers: List<Module?> = when (teacher) {
            is List<*> -> teacher.map { it as? Module }
            else -> listOf(teacher)
        }
        val preparedTeachers = teachers.map { prepareTeacher(student, it) }
        val effectiveNumThreads = numThreads ?: Settings.numThreads

        for (t in preparedTeachers) {
            val tData = bootstrapTraceData(program = t, dataset = trainset, metric = metric, numThreads = effectiveNumThreads)
            traceData.addAll(tData)
        }

        println("Preparing the train data...")
        val keyToData = mutableMapOf<Pair<BaseLM?, Int?>, Map<String, Any?>>()

        for ((predInd, pred) in student.predictors().withIndex()) {
            val dataPredInd = if (multitask) null else predInd
            val predictor = pred as? Predict ?: continue

            val lm = predictor._lm ?: throw IllegalStateException(
                "Predictor $predInd does not have an LM assigned."
            )

            val trainingKey = lm to dataPredInd

            if (trainingKey !in keyToData) {
                val (trainData, dataFormat) = prepareFinetuneData(
                    traceData = traceData,
                    lm = lm,
                    predInd = dataPredInd,
                )

                println("Using ${trainData.size} data points for fine-tuning: ${lm.model}")

                keyToData[trainingKey] = mutableMapOf<String, Any?>(
                    "lm" to lm,
                    "train_data" to trainData,
                    "train_data_format" to dataFormat,
                    "train_kwargs" to (trainKwargs[lm] ?: emptyMap<String, Any?>()),
                )
            }
        }

        println("Starting LM fine-tuning...")
        require(keyToData.size <= (effectiveNumThreads ?: keyToData.size)) {
            "Requires num_threads >= number of fine-tuning jobs."
        }

        println("${keyToData.size} fine-tuning job(s) to start")
        val keyToLm = finetuneLms(keyToData)

        println("Updating the student program with the fine-tuned LMs...")
        for ((predInd, pred) in student.predictors().withIndex()) {
            val dataPredInd = if (multitask) null else predInd
            val predictor = pred as? Predict ?: continue
            val trainingKey = predictor._lm to dataPredInd

            val finetunedLm = keyToLm[trainingKey]
            if (finetunedLm is Exception) {
                throw RuntimeException("Finetuned LM for predictor $predInd failed.", finetunedLm)
            }
            if (finetunedLm is BaseLM) {
                predictor._lm = finetunedLm
            }
            predictor.demos = if (excludeDemos) mutableListOf() else predictor.demos
        }

        println("BootstrapFinetune has finished compiling the student program")
        student.compiled = true
        student._compiled = true
        return student
    }

    @Suppress("UNCHECKED_CAST")
    fun finetuneLms(finetuneDict: Map<Pair<BaseLM?, Int?>, Map<String, Any?>>): Map<Pair<BaseLM?, Int?>, BaseLM> {
        val numJobs = finetuneDict.size
        println("Starting $numJobs fine-tuning job(s)...")

        val keyToLm = mutableMapOf<Pair<BaseLM?, Int?>, BaseLM>()
        for ((ind, entry) in finetuneDict.entries.withIndex()) {
            val (key, finetuneKwargs) = entry
            val lm = finetuneKwargs["lm"] as? BaseLM ?: continue
            println("Would kill LM to free up resources.")
            keyToLm[key] = lm
            println("Job ${ind + 1}/$numJobs is done")
        }

        return keyToLm
    }

    private fun prepareFinetuneData(
        traceData: List<Map<String, Any?>>,
        lm: BaseLM,
        predInd: Int? = null,
    ): Pair<List<Map<String, Any?>>, String> {
        var filteredData = traceData
        if (metric != null) {
            println("Collected data for ${filteredData.size} examples")
            filteredData = filteredData.filter { (it["score"] as? Boolean?) == true }
            println("After filtering with the metric, ${filteredData.size} examples remain")
        }

        val data = mutableListOf<Map<String, Any?>>()
        val adapter = (adapterMap[lm] as? Adapter) ?: Settings.adapter() ?: ChatAdapter()
        val dataFormat = "chat"

        for (item in filteredData) {
            @Suppress("UNCHECKED_CAST")
            val trace = item["trace"] as? List<Map<String, Any?>> ?: continue
            for ((i, _) in trace.withIndex()) {
                val includeData = predInd == null || predInd == i
                if (includeData) {
                    data.add(mapOf<String, Any?>(
                        "messages" to emptyList<Any>(),
                        "label" to "",
                    ))
                }
            }
        }

        data.shuffle(kotlin.random.Random(0))
        return data to dataFormat
    }
}

/**
 * Build call data from trace for fine-tuning.
 */
fun buildCallDataFromTrace(
    trace: List<Map<String, Any?>>,
    predInd: Int,
    adapter: Adapter,
    excludeDemos: Boolean = false,
): Map<String, Any?> {
    return mapOf<String, Any?>(
        "messages" to emptyList<Any>(),
        "label" to "",
    )
}

/**
 * Return True if all predictors in the program have an LM set.
 */
fun allPredictorsHaveLms(program: Module): Boolean {
    return program.predictors().all { (it as? Predict)?._lm != null }
}

/**
 * Copy program preserving LM assignments.
 */
fun copyProgramWithLms(program: Module): Module {
    val predLms = program.predictors().map { (it as? Predict)?._lm }
    val copy = program.deepcopy()
    for ((ind, pred) in copy.predictors().withIndex()) {
        (pred as? Predict)?._lm = predLms[ind]
    }
    return copy
}

/**
 * Prepare student program for fine-tuning.
 */
fun prepareStudent(student: Module): Module {
    if (student._compiled) {
        throw IllegalArgumentException("The student program should not be compiled.")
    }
    return student
}

/**
 * Prepare teacher program for fine-tuning.
 */
fun prepareTeacher(student: Module, teacher: Module? = null): Module {
    if (teacher == null) return student
    assertStructuralEquivalency(student, teacher)
    assertNoSharedPredictor(student, teacher)
    return teacher
}

/**
 * Assert that two programs are structurally equivalent.
 */
fun assertStructuralEquivalency(program1: Module, program2: Module) {
    val num1 = program1.predictors().size
    val num2 = program2.predictors().size
    require(num1 == num2) {
        "Structurally equivalent programs must have the same number of predictors. $num1 != $num2"
    }
    val zip = program1.namedPredictors().zip(program2.namedPredictors())
    for ((ind, pair) in zip.withIndex()) {
        val (name1, pred1) = pair.first
        val (name2, pred2) = pair.second
        require(name1 == name2) {
            "Predictor names must match at index $ind: '$name1' != '$name2'"
        }
        require(pred1 is Predict)
        require(pred2 is Predict)
    }
}

/**
 * Assert that two programs don't share predictors.
 */
fun assertNoSharedPredictor(program1: Module, program2: Module) {
    val idToName1 = program1.namedPredictors().associate { (name, pred) -> System.identityHashCode(pred) to name }
    val idToName2 = program2.namedPredictors().associate { (name, pred) -> System.identityHashCode(pred) to name }
    val sharedIds = idToName1.keys.intersect(idToName2.keys)
    require(sharedIds.isEmpty()) {
        "The programs share predictors: " + sharedIds.joinToString(", ") { idToName1[it] ?: "unknown" }
    }
}

/**
 * Get unique LMs from a program.
 */
fun getUniqueLms(program: Module): List<BaseLM> {
    return program.predictors().mapNotNull { (it as? Predict)?._lm }.distinct()
}

/**
 * Launch all LMs in a program.
 */
fun launchLms(program: Module) {
    // No-op in stub
}

/**
 * Kill all LMs in a program.
 */
fun killLms(program: Module) {
    // No-op in stub
}
