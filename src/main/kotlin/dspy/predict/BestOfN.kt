package dspy.predict

import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.utils.Settings

/**
 * BestOfN module.
 *
 * Runs a module up to N times with different rollout IDs at temperature=1.0 and
 * returns the best prediction out of N attempts or the first prediction that passes the threshold.
 *
 * Port of `dspy/predict/best_of_n.py`
 */
class BestOfN(
    private val module: Module,
    private val N: Int,
    private val rewardFn: (Map<String, Any?>, Prediction) -> Double,
    private val threshold: Double,
    failCount: Int? = null,
) : Module() {
    private var remainingFailCount: Int

    init {
        this.remainingFailCount = failCount ?: N
    }

    /**
     * Forward pass: run the module up to N times and return the best prediction.
     */
    suspend fun forward(kwargs: Map<String, Any?>): Prediction? {
        val lm = module.getLm() ?: Settings.lm()
            ?: throw IllegalStateException("No LM configured. Call dspy.configure(lm=...) first.")

        val start = (lm.kwargs["rollout_id"] as? Int) ?: 0
        val rolloutIds = (start until start + N).toList()
        var bestPred: Prediction? = null
        var bestTrace: List<Triple<Any, Map<String, Any?>, Map<String, Any?>>>? = null
        var bestReward = -Double.MAX_VALUE
        var localFailCount = remainingFailCount

        for ((idx, rid) in rolloutIds.withIndex()) {
            val lmCopy = lm.copy("rollout_id" to rid, "temperature" to 1.0)
            val mod = module.deepcopy()
            mod.setLm(lmCopy)

            var reward: Double? = null
            var pred: Prediction? = null
            var trace: List<Triple<Any, Map<String, Any?>, Map<String, Any?>>>? = null

            try {
                val savedTrace = Settings.trace
                Settings.trace = mutableListOf()
                try {
                    pred = mod.invoke(kwargs = kwargs)
                    trace = Settings.trace?.toList()

                    // NOTE: Not including the trace of rewardFn
                    reward = rewardFn(kwargs, pred!!)
                } finally {
                    Settings.trace = savedTrace
                }
            } catch (e: Exception) {
                println("BestOfN: Attempt ${idx + 1} failed with rollout id $rid: ${e.message}")
                if (idx >= localFailCount) {
                    throw e
                }
                localFailCount--
                continue
            }

            if (reward != null && reward > bestReward) {
                bestReward = reward
                bestPred = pred
                bestTrace = trace
            }

            if (reward != null && reward >= threshold) {
                break
            }
        }

        if (bestTrace != null) {
            Settings.trace?.addAll(bestTrace)
        }

        return bestPred
    }

    override suspend operator fun invoke(kwargs: Map<String, Any?>): Prediction {
        return forward(kwargs) ?: throw IllegalStateException("BestOfN returned no prediction after $N attempts")
    }

    override fun deepcopy(): Module {
        return BestOfN(
            module = module.deepcopy(),
            N = N,
            rewardFn = rewardFn,
            threshold = threshold,
            failCount = remainingFailCount,
        )
    }
}
