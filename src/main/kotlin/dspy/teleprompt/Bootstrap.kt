package dspy.teleprompt

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings
import dspy.utils.TraceEntry
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

/**
 * Bootstrap few-shot teleprompter.
 *
 * Composes a set of demos from a combination of labeled examples in the training
 * set and bootstrapped demos (obtained by running a teacher program and tracing).
 *
 * Each bootstrap round copies the LM with a new ``rollout_id`` at ``temperature=1.0`` to
 * bypass caches and gather diverse traces.
 *
 * Faithful port of `dspy/teleprompt/bootstrap.py`.
 */
open class BootstrapFewShot(
    val metric: ((Example, Prediction, List<Any>?) -> Any?)? = null,
    val metricThreshold: Double? = null,
    teacherSettings: Map<String, Any?>? = null,
    val maxBootstrappedDemos: Int = 4,
    val maxLabeledDemos: Int = 16,
    val maxRounds: Int = 1,
    maxErrors: Int? = null,
) : Teleprompter() {
    val teacherSettings: Map<String, Any?> = teacherSettings ?: emptyMap()
    var maxErrors: Int? = maxErrors
    private var errorCount: Int = 0
    private val errorLock = Mutex()

    private lateinit var student: Module
    private lateinit var teacher: Module
    private lateinit var trainset: List<Example>
    private lateinit var validation: List<Example>
    private lateinit var name2Predictor: Map<String, Module?>
    private lateinit var predictor2Name: Map<Int, String>
    private lateinit var name2Traces: Map<String, MutableList<Map<String, Any?>>>

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        this.trainset = trainset

        prepareStudentAndTeacher(student, teacher)
        preparePredictorMappings()
        bootstrap()

        this.student = train()
        this.student.compiled = true
        this.student._compiled = true

        return this.student
    }

    private suspend fun prepareStudentAndTeacher(student: Module, teacher: Module?) {
        this.student = student.resetCopy()

        // Deep copy instead of reset copy for the student-as-teacher
        this.teacher = teacher?.deepcopy() ?: student.deepcopy()

        require(!this.student.compiled) { "Student must be uncompiled." }

        if (maxLabeledDemos > 0 && !this.teacher.compiled) {
            val teleprompter = LabeledFewShot(k = maxLabeledDemos)
            this.teacher = teleprompter.compile(this.teacher.resetCopy(), trainset = this.trainset)
        }
    }

    private fun preparePredictorMappings() {
        val name2Predictor: MutableMap<String, Module?> = mutableMapOf()
        val predictor2Name: MutableMap<Int, String> = mutableMapOf()
        val student = this.student
        val teacher = this.teacher

        require(student.predictors().size == teacher.predictors().size) {
            "Student and teacher must have the same number of predictors."
        }

        val studentNamed = student.namedPredictors()
        val teacherNamed = teacher.namedPredictors()

        for (i in studentNamed.indices) {
            val (name1, predictor1) = studentNamed[i]
            val (name2, predictor2) = teacherNamed[i]
            require(name1 == name2) { "Student and teacher must have the same program structure." }

            // Check signatures match
            val sig1 = predictor1.signature
            val sig2 = predictor2.signature
            require(sig1 == sig2) {
                "Student and teacher must have the same signatures. " +
                    "${sig1::class} != ${sig2::class}"
            }
            require(System.identityHashCode(predictor1) != System.identityHashCode(predictor2)) {
                "Student and teacher must be different objects."
            }

            name2Predictor[name1] = null
            predictor2Name[System.identityHashCode(predictor1)] = name1
            predictor2Name[System.identityHashCode(predictor2)] = name2
        }

        this.name2Predictor = name2Predictor
        this.predictor2Name = predictor2Name
    }

    private suspend fun bootstrap(maxBootstraps: Int? = null) {
        val max = maxBootstraps ?: maxBootstrappedDemos
        var bootstrapAttempts = 0
        val bootstrapped = mutableMapOf<Int, Boolean>()

        name2Traces = name2Predictor.keys.associateWith { mutableListOf<Map<String, Any?>>() }

        for ((exampleIdx, example) in trainset.withIndex()) {
            if (bootstrapped.size >= max) break

            for (roundIdx in 0 until maxRounds) {
                bootstrapAttempts++
                if (bootstrapOneExample(example, roundIdx)) {
                    bootstrapped[exampleIdx] = true
                    break
                }
            }
        }

        println(
            "Bootstrapped ${bootstrapped.size} full traces after " +
                "${trainset.lastIndex} examples for up to $maxRounds rounds, " +
                "amounting to $bootstrapAttempts attempts."
        )

        // Unbootstrapped training examples
        validation = trainset.filterIndexed { idx, _ -> idx !in bootstrapped }
        validation = validation.shuffled(Random(0))
    }

    private suspend fun bootstrapOneExample(example: Example, roundIdx: Int = 0): Boolean {
        val name2Traces: MutableMap<String, MutableList<Map<String, Any?>>> = mutableMapOf()
        val teacher = this.teacher
        val predictorCache: MutableMap<String, List<Map<String, Any?>>> = mutableMapOf()

        var success = false

        try {
            Settings.context(trace = mutableListOf()) {
                val lm = Settings.lm()
                val effectiveLm = if (roundIdx > 0 && lm != null) {
                    lm.copy("rollout_id" to roundIdx, "temperature" to 1.0)
                } else {
                    lm
                }
                val newSettings = if (roundIdx > 0 && effectiveLm != null) {
                    mapOf("lm" to effectiveLm)
                } else {
                    emptyMap()
                }

                if (newSettings.isNotEmpty()) {
                    Settings.context(lm = effectiveLm) {
                        for ((name, predictor) in teacher.namedPredictors()) {
                            predictorCache[name] = predictor.demos.toList()
                            predictor.demos = predictor.demos.filter { demo ->
                                demo != example.toDict()
                            }.toMutableList()
                        }

                        val prediction = runBlocking { teacher(example.inputs().toMap()) }

                        for ((name, predictor) in teacher.namedPredictors()) {
                            predictor.demos = predictorCache[name]?.toMutableList() ?: mutableListOf()
                        }

                        val trace = Settings.trace ?: mutableListOf()
                        success = runBlocking { evaluateMetric(example, trace) }
                    }
                } else {
                    for ((name, predictor) in teacher.namedPredictors()) {
                        predictorCache[name] = predictor.demos.toList()
                        predictor.demos = predictor.demos.filter { demo ->
                            demo != example.toDict()
                        }.toMutableList()
                    }

                    val prediction = runBlocking { teacher(example.inputs().toMap()) }

                    for ((name, predictor) in teacher.namedPredictors()) {
                        predictor.demos = predictorCache[name]?.toMutableList() ?: mutableListOf()
                    }

                    val trace = Settings.trace ?: mutableListOf()
                    success = runBlocking { evaluateMetric(example, trace) }
                }
            }
        } catch (e: Exception) {
            runBlocking { handleException(e, example) }
            success = false
        }

        if (success) {
            val trace = Settings.trace ?: mutableListOf()
            for (step in trace) {
                val stepTriple = step as Triple<*, *, *>
                val predictor = stepTriple.first
                val inputs = stepTriple.second as Map<String, Any?>
                val outputs = stepTriple.third as Map<String, Any?>
                val demo = mutableMapOf<String, Any?>()
                demo.putAll(inputs)
                demo.putAll(outputs)
                demo["augmented"] = true

                val predictorName = predictor2Name[System.identityHashCode(predictor)]
                    ?: continue

                name2Traces.getOrPut(predictorName) { mutableListOf() }.add(demo)
            }

            // Update traces
            for ((name, demos) in name2Traces) {
                if (demos.size > 1) {
                    val rng = Random(demos.hashCode().toLong())
                    val selected = if (rng.nextDouble() < 0.5) {
                        demos.random(rng)
                    } else {
                        demos.last()
                    }
                    name2Traces[name]?.add(selected)
                } else {
                    name2Traces[name]?.addAll(demos)
                }
            }
        }

        return success
    }

    private suspend fun evaluateMetric(example: Example, trace: List<Any>): Boolean {
        return if (metric != null) {
            val metricVal = metric!!.invoke(example, Prediction(emptyMap()), null)
            if (metricThreshold != null) {
                (metricVal as? Number)?.toDouble()?.let { it >= metricThreshold } ?: false
            } else {
                when (metricVal) {
                    is Boolean -> metricVal
                    is Number -> metricVal.toDouble() > 0
                    else -> metricVal != null
                }
            }
        } else {
            true
        }
    }

    private suspend fun handleException(e: Exception, example: Example) {
        val currentErrorCount: Int
        errorLock.withLock {
            errorCount++
            currentErrorCount = errorCount
        }
        val effectiveMaxErrors = maxErrors ?: Settings.maxErrors
        if (currentErrorCount >= (effectiveMaxErrors ?: Int.MAX_VALUE)) {
            throw e
        }
    }

    private fun train(): Module {
        val rng = Random(0)
        var rawDemos = validation

        for ((name, predictor) in student.namedPredictors()) {
            val augmentedDemos = name2Traces[name]?.take(maxBootstrappedDemos) ?: emptyList()

            var sampleSize = minOf(maxLabeledDemos - augmentedDemos.size, rawDemos.size)
            sampleSize = maxOf(0, sampleSize)

            rawDemos = rawDemos.shuffled(rng).take(sampleSize)
            val finalDemos = (augmentedDemos + rawDemos.map { it.toDict() })
            predictor.demos = finalDemos.toMutableList()
        }

        return student
    }
}
