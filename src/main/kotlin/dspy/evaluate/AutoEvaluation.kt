package dspy.evaluate

import dspy.primitives.Example
import dspy.primitives.Module
import dspy.primitives.Prediction
import dspy.predict.ChainOfThought
import dspy.signatures.InputField
import dspy.signatures.OutputField
import dspy.signatures.Signature

/**
 * Signature for computing semantic recall and precision.
 *
 * Compare a system's response to the ground truth to compute its recall and precision.
 * If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::SemanticRecallPrecision`
 */
object SemanticRecallPrecision {
    private val instance = Signature(
        instruction = "",
        inputFields = listOf(
            InputField(name = "question", desc = ""),
            InputField(name = "ground_truth", desc = ""),
            InputField(name = "system_response", desc = ""),
        ),
        outputFields = listOf(
            OutputField(name = "recall", desc = "fraction (out of 1.0) of ground truth covered by the system response"),
            OutputField(name = "precision", desc = "fraction (out of 1.0) of system response covered by the ground truth"),
        ),
    )

    fun signature(): Signature = instance
}

/**
 * Signature for decompositional semantic recall and precision.
 *
 * Compare a system's response to the ground truth to compute recall and precision of key ideas.
 * You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::DecompositionalSemanticRecallPrecision`
 */
object DecompositionalSemanticRecallPrecision {
    private val instance = Signature(
        instruction = "",
        inputFields = listOf(
            InputField(name = "question", desc = ""),
            InputField(name = "ground_truth", desc = ""),
            InputField(name = "system_response", desc = ""),
        ),
        outputFields = listOf(
            OutputField(name = "ground_truth_key_ideas", desc = "enumeration of key ideas in the ground truth"),
            OutputField(name = "system_response_key_ideas", desc = "enumeration of key ideas in the system response"),
            OutputField(name = "discussion", desc = "discussion of the overlap between ground truth and system response"),
            OutputField(name = "recall", desc = "fraction (out of 1.0) of ground truth covered by the system response"),
            OutputField(name = "precision", desc = "fraction (out of 1.0) of system response covered by the ground truth"),
        ),
    )

    fun signature(): Signature = instance
}

/**
 * Compute the F1 score from precision and recall, clamping both to [0, 1].
 *
 * Port of `dspy/evaluate/auto_evaluation.py::f1_score`
 */
fun f1Score(precision: Double, recall: Double): Double {
    val p = precision.coerceIn(0.0, 1.0)
    val r = recall.coerceIn(0.0, 1.0)
    return if (p + r == 0.0) 0.0 else 2.0 * (p * r) / (p + r)
}

/**
 * Computes semantic F1 between a prediction and ground truth via LLM-based precision/recall.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::SemanticF1`
 *
 * @param threshold Minimum F1 score to accept during optimization. Defaults to 0.66.
 * @param decompositional If true, uses DecompositionalSemanticRecallPrecision. Defaults to false.
 */
class SemanticF1(
    threshold: Double = 0.66,
    decompositional: Boolean = false,
) : Module() {
    val threshold: Double = threshold
    private val module: ChainOfThought

    init {
        val sig = if (decompositional) {
            DecompositionalSemanticRecallPrecision.signature()
        } else {
            SemanticRecallPrecision.signature()
        }
        module = ChainOfThought(signature = sig)
    }

    suspend fun forward(example: Example, pred: Prediction, trace: Any? = null): Prediction {
        val scores = module.forward(
            Example(
                base = mapOf(
                    "question" to example["question"],
                    "ground_truth" to example["response"],
                    "system_response" to pred["response"],
                ),
            ),
        )

        val precisionVal = scores["precision"] as? Double ?: 0.0
        val recallVal = scores["recall"] as? Double ?: 0.0
        val score = f1Score(precisionVal, recallVal)

        return if (trace != null) {
            Prediction(base = mapOf("score" to (score >= threshold)))
        } else {
            Prediction(base = mapOf("score" to score))
        }
    }

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction {
        val example = kwargs["example"] as? Example ?: Example(base = kwargs)
        val pred = kwargs["pred"] as? Prediction ?: Prediction(base = kwargs)
        val trace = kwargs["trace"]
        return forward(example, pred, trace)
    }

    override fun namedParameters(): List<Pair<String, dspy.primitives.Parameter>> {
        return listOf("module" to module)
    }
}

/**
 * Signature for estimating answer completeness.
 *
 * Estimate the completeness of a system's responses, against the ground truth.
 * You will first enumerate key ideas in each response, discuss their overlap, and then report completeness.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::AnswerCompleteness`
 */
object AnswerCompleteness {
    private val instance = Signature(
        instruction = "",
        inputFields = listOf(
            InputField(name = "question", desc = ""),
            InputField(name = "ground_truth", desc = ""),
            InputField(name = "system_response", desc = ""),
        ),
        outputFields = listOf(
            OutputField(name = "ground_truth_key_ideas", desc = "enumeration of key ideas in the ground truth"),
            OutputField(name = "system_response_key_ideas", desc = "enumeration of key ideas in the system response"),
            OutputField(name = "discussion", desc = "discussion of the overlap between ground truth and system response"),
            OutputField(name = "completeness", desc = "fraction (out of 1.0) of ground truth covered by the system response"),
        ),
    )

    fun signature(): Signature = instance
}

/**
 * Signature for estimating answer groundedness.
 *
 * Estimate the groundedness of a system's responses, against real retrieved documents written by people.
 * You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then
 * discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::AnswerGroundedness`
 */
object AnswerGroundedness {
    private val instance = Signature(
        instruction = "",
        inputFields = listOf(
            InputField(name = "question", desc = ""),
            InputField(name = "retrieved_context", desc = ""),
            InputField(name = "system_response", desc = ""),
        ),
        outputFields = listOf(
            OutputField(name = "system_response_claims", desc = "enumeration of non-trivial or check-worthy claims in the system response"),
            OutputField(name = "discussion", desc = "discussion of how supported the claims are by the retrieved context"),
            OutputField(name = "groundedness", desc = "fraction (out of 1.0) of system response supported by the retrieved context"),
        ),
    )

    fun signature(): Signature = instance
}

/**
 * Combines answer completeness and groundedness into a single score.
 *
 * Port of `dspy/evaluate/auto_evaluation.py::CompleteAndGrounded`
 *
 * @param threshold Minimum score to accept during optimization. Defaults to 0.66.
 */
class CompleteAndGrounded(
    threshold: Double = 0.66,
) : Module() {
    val threshold: Double = threshold
    private val completenessModule: ChainOfThought
    private val groundednessModule: ChainOfThought

    init {
        completenessModule = ChainOfThought(signature = AnswerCompleteness.signature())
        groundednessModule = ChainOfThought(signature = AnswerGroundedness.signature())
    }

    suspend fun forward(example: Example, pred: Prediction, trace: Any? = null): Prediction {
        val completeness = completenessModule.forward(
            Example(
                base = mapOf(
                    "question" to example["question"],
                    "ground_truth" to example["response"],
                    "system_response" to pred["response"],
                ),
            ),
        )

        val groundedness = groundednessModule.forward(
            Example(
                base = mapOf(
                    "question" to example["question"],
                    "retrieved_context" to pred["context"],
                    "system_response" to pred["response"],
                ),
            ),
        )

        val groundednessScore = groundedness["groundedness"] as? Double ?: 0.0
        val completenessScore = completeness["completeness"] as? Double ?: 0.0
        val score = f1Score(groundednessScore, completenessScore)

        return if (trace != null) {
            Prediction(base = mapOf("score" to (score >= threshold)))
        } else {
            Prediction(base = mapOf("score" to score))
        }
    }

    override suspend fun invoke(kwargs: Map<String, Any?>): Prediction {
        val example = kwargs["example"] as? Example ?: Example(base = kwargs)
        val pred = kwargs["pred"] as? Prediction ?: Prediction(base = kwargs)
        val trace = kwargs["trace"]
        return forward(example, pred, trace)
    }

    override fun namedParameters(): List<Pair<String, dspy.primitives.Parameter>> {
        return listOf(
            "completenessModule" to completenessModule,
            "groundednessModule" to groundednessModule,
        )
    }
}
