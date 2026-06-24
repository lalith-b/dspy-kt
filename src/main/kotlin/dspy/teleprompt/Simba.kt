package dspy.teleprompt

import dspy.clients.BaseLM
import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.predict.Parallel
import dspy.predict.Predict
import dspy.utils.Settings
import java.util.logging.Logger
import kotlin.math.exp
import kotlin.random.Random

/**
 * SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer for DSPy.
 *
 * Faithful port of `dspy/teleprompt/simba.py`.
 */
class SIMBA(
    val metric: (Example, Prediction?) -> Any?,
    val bsize: Int = 32,
    val numCandidates: Int = 6,
    val maxSteps: Int = 8,
    val maxDemos: Int = 4,
    promptModel: BaseLM? = null,
    teacherSettings: Map<String, Any?>? = null,
    val demoInputFieldMaxlen: Int = 100_000,
    val numThreads: Int? = null,
    val temperatureForSampling: Double = 0.2,
    val temperatureForCandidates: Double = 0.2,
) : Teleprompter() {

    private val logger: Logger = Logger.getLogger(SIMBA::class.java.name)
    val promptModel: BaseLM? = promptModel ?: Settings.lm()
    val teacherSettings: Map<String, Any?>? = teacherSettings

    private val _strategies: List<SimbaStrategy>

    init {
        _strategies = if (maxDemos > 0) {
            listOf(
                appendADemo(demoInputFieldMaxlen),
                { bucket, system, predictor2name, name2predictor, batch10pScore, batch90pScore, pm ->
                    appendARule(bucket, system, predictor2name, name2predictor, batch10pScore, batch90pScore, pm)
                }
            )
        } else {
            listOf(
                { bucket, system, predictor2name, name2predictor, batch10pScore, batch90pScore, pm ->
                    appendARule(bucket, system, predictor2name, name2predictor, batch10pScore, batch90pScore, pm)
                }
            )
        }
    }

    override suspend fun compile(
        student: Module,
        trainset: List<Example>,
        teacher: Module?,
        valset: List<Example>?,
    ): Module {
        require(trainset.size >= bsize) { "Trainset too small: ${trainset.size} < $bsize" }

        val rng = Random(0)

        data class ProgEntry(val module: Module, val idx: Int, val scores: MutableList<Double>)

        val programs = mutableListOf<ProgEntry>()
        var nextIdx = 0

        fun calcAvg(progIdx: Int): Double {
            return programs.find { it.idx == progIdx }?.scores?.average() ?: 0.0
        }

        fun topK(k: Int): List<Int> {
            val scored = programs.sortedByDescending { calcAvg(it.idx) }
            var top = scored.take(k).map { it.idx }.toMutableList()
            if (top.none { it == 0 } && top.isNotEmpty()) {
                top[top.size - 1] = 0
            }
            return top.distinct()
        }

        fun softSample(programIdxs: List<Int>, temp: Double): Int {
            if (programIdxs.isEmpty()) throw IllegalStateException("No programs.")
            val sc = programIdxs.map { calcAvg(it) }
            val adj = sc.map { exp(it / temp) }
            val s = adj.sum()
            if (s <= 0.0) return programIdxs[rng.nextInt(programIdxs.size)]
            val probs = adj.map { it / s }
            val r = rng.nextDouble()
            var c = 0.0
            for (i in probs.indices) {
                c += probs[i]
                if (r < c) return programIdxs[i]
            }
            return programIdxs.last()
        }

        fun register(prog: Module, scores: List<Double>): Int {
            nextIdx++
            programs.add(ProgEntry(prog, nextIdx, scores.toMutableList()))
            return nextIdx
        }

        // Init baseline
        val studentCopy = student.deepcopy()
        programs.add(ProgEntry(studentCopy, 0, mutableListOf()))
        val winners = mutableListOf<Module>(studentCopy)

        val dataIdx = trainset.indices.toMutableList()
        dataIdx.shuffle(rng)
        var instIdx = 0
        val par = Parallel(accessExamples = false, numThreads = numThreads)
        val trialLogs = mutableMapOf<Int, MutableMap<String, Any>>()

        for (bIdx in 0 until maxSteps) {
            trialLogs[bIdx] = mutableMapOf()

            if (instIdx + bsize > trainset.size) {
                dataIdx.shuffle(rng)
                instIdx = 0
            }

            val endInst = minOf(instIdx + bsize, trainset.size)
            val batch = dataIdx.subList(instIdx, endInst).map { trainset[it] }
            instIdx += bsize

            val models = prepareModelsForResampling(programs[0].module, numCandidates, teacherSettings)
            val topProgs = topK(numCandidates)

            val execPairs = mutableListOf<Pair<(Example) -> Map<String, Any?>, Example>>()
            val pred2name = mutableMapOf<Int, String>()

            for (model in models) {
                for (example in batch) {
                    val chosen = softSample(topProgs, temperatureForSampling)
                    val candSys = programs[chosen].module.deepcopy()
                    candSys.setLm(model)
                    for ((name, pred) in candSys.namedPredictors()) {
                        pred2name[System.identityHashCode(pred)] = name
                    }
                    execPairs.add(Pair(wrapProgram(candSys, metric), example))
                }
            }

            @Suppress("UNCHECKED_CAST")
            val outputs = par(execPairs) as? List<Map<String, Any?>> ?: emptyList()

            val allScores = outputs.map { (it["score"] as? Double) ?: 0.0 }
            val sorted = allScores.sorted()
            val p10 = percentile(sorted, 10.0)
            val p90 = percentile(sorted, 90.0)

            val buckets = mutableListOf<Pair<List<Map<String, Any?>>, Triple<Double, Double, Double>>>()
            for (i in batch.indices) {
                val bucket = mutableListOf<Map<String, Any?>>()
                for (j in i until outputs.size step batch.size) bucket.add(outputs[j])
                bucket.sortByDescending { (it["score"] as? Double) ?: 0.0 }
                val mx = (bucket.first()["score"] as? Double) ?: 0.0
                val mn = (bucket.last()["score"] as? Double) ?: 0.0
                val av = bucket.map { (it["score"] as? Double) ?: 0.0 }.average()
                buckets.add(Pair(bucket, Triple(mx - mn, mx, mx - av)))
            }
            buckets.sortByDescending { it.second.first }

            // Build candidates
            val sysCands = mutableListOf<Module>()
            for (entry in buckets) {
                val (bucket, stats) = entry
                val srcIdx = softSample(topK(numCandidates), temperatureForCandidates)
                val srcProgram = programs.find { it.idx == srcIdx } ?: programs[0]
                val sysCand = srcProgram.module.deepcopy()

                val n2p = mutableMapOf<String, Predict>()
                for ((name, pred) in sysCand.namedPredictors()) n2p[name] = pred

                val maxDTmp = if (maxDemos > 0) maxDemos else 3
                val numDemos = n2p.values.map { it.demos.size }.maxOrNull() ?: 0
                var dropCount = maxOf(
                    poissonRandom(rng, numDemos.toDouble() / maxOf(maxDTmp.toDouble(), 0.01)).toInt(),
                    if (numDemos >= maxDTmp) 1 else 0
                )
                dropCount = minOf(dropCount, numDemos)
                val dropSet = (0 until dropCount).map { rng.nextInt(numDemos) }.toSet()

                for ((_, pred) in n2p) {
                    pred.demos = pred.demos.mapIndexed { i, d -> if (i !in dropSet) d else null }.filterNotNull().toMutableList()
                }

                val strat = _strategies[rng.nextInt(_strategies.size)]
                try {
                    strat(bucket, sysCand, pred2name, n2p, p10, p90, promptModel)
                } catch (e: Exception) {
                    logger.warning("Strategy failed: ${e.message}")
                    continue
                }
                sysCands.add(sysCand)
                if (sysCands.size >= numCandidates + 1) break
            }

            // Evaluate
            val evalPairs = sysCands.flatMap { sys ->
                batch.map { ex -> Pair(wrapProgram(sys, metric), ex) }
            }
            @Suppress("UNCHECKED_CAST")
            val evalOut = par(evalPairs) as? List<Map<String, Any?>> ?: emptyList()

            val candScores = sysCands.indices.map { ci ->
                val s = ci * batch.size
                val e = (ci + 1) * batch.size
                val sc = (s until e).map { (evalOut[it]["score"] as? Double) ?: 0.0 }
                sc.average()
            }

            if (candScores.isNotEmpty()) {
                val bestI = candScores.indices.maxByOrNull { candScores[it] }!!
                winners.add(sysCands[bestI].deepcopy())
            }
            for (ci in sysCands.indices) {
                val s = ci * batch.size
                val e = (ci + 1) * batch.size
                val sc = (s until e).map { (evalOut[it]["score"] as? Double) ?: 0.0 }
                register(sysCands[ci], sc)
            }
        }

        // Final validation
        val M = maxOf(winners.size - 1, 0)
        val N = numCandidates + 1
        val progIdxs = if (M < 1) {
            List(N) { 0 }
        } else {
            (0 until N).map { (it * M / (N - 1.0)).toInt() }.distinct()
        }

        val candProgs = progIdxs.map { winners[it].deepcopy() }
        val valPairs = candProgs.flatMap { sys ->
            trainset.map { ex -> Pair(wrapProgram(sys, metric), ex) }
        }
        @Suppress("UNCHECKED_CAST")
        val valOut = par(valPairs) as? List<Map<String, Any?>> ?: emptyList()

        val scores = candProgs.indices.map { pi ->
            val s = pi * trainset.size
            val e = (pi + 1) * trainset.size
            val sc = (s until e).map { (valOut[it]["score"] as? Double) ?: 0.0 }
            sc.average()
        }

        val candData = scores.mapIndexed { i, s -> mapOf("score" to s, "program" to candProgs[i]) }
            .sortedByDescending { it["score"] as Double }

        val bestIdx = scores.indices.maxByOrNull { scores[it] } ?: 0
        val bestProg = candProgs[bestIdx].deepcopy()

        bestProg.setAttribute("candidate_programs", candData)
        bestProg.setAttribute("trial_logs", trialLogs)

        return bestProg
    }

    private fun percentile(sorted: List<Double>, p: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val k = (p / 100.0) * (sorted.size - 1)
        val f = k.toInt()
        val c = (f + 1).coerceAtMost(sorted.size - 1)
        return sorted[f] + (k - f) * (sorted[c] - sorted[f])
    }

    private fun poissonRandom(rng: Random, lambda: Double): Int {
        if (lambda <= 0.0) return 0
        val L = exp(-lambda)
        var k = 0
        var p = 1.0
        do { k++; p *= rng.nextDouble() } while (p > L)
        return k - 1
    }
}
