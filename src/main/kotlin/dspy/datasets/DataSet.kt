package dspy.datasets

import dspy.primitives.Example
import java.util.UUID
import kotlin.random.Random

/**
 * Base class for DSPy datasets.
 *
 * Provides lazy-loading of train/dev/test splits with shuffling and sampling.
 *
 * Faithful port of `dspy/datasets/dataset.py`.
 */
open class DataSet(
    private val trainSeed: Int = 0,
    private var trainSize: Int? = null,
    private val evalSeed: Int = 0,
    private var devSize: Int? = null,
    private var testSize: Int? = null,
    private val inputKeys: List<String>? = null,
) {
    var doShuffle: Boolean = true
    val name: String = this::class.simpleName ?: "DataSet"

    private var _trainData: Any? = null
    private var _devData: Any? = null
    private var _testData: Any? = null
    private var _cachedTrain: List<Example>? = null
    private var _cachedDev: List<Example>? = null
    private var _cachedTest: List<Example>? = null

    /**
     * Subclasses should set this to the raw training data.
     */
    open fun _train(): Iterable<Map<String, Any?>> = emptyList()

    /**
     * Subclasses should set this to the raw dev data.
     */
    open fun _dev(): Iterable<Map<String, Any?>> = emptyList()

    /**
     * Subclasses should set this to the raw test data.
     */
    open fun _test(): Iterable<Map<String, Any?>> = emptyList()

    /**
     * Reset seeds and optionally sizes. Invalidates cached splits.
     */
    fun resetSeeds(
        trainSeed: Int? = null,
        trainSize: Int? = null,
        evalSeed: Int? = null,
        devSize: Int? = null,
        testSize: Int? = null,
    ) {
        this.trainSize = trainSize ?: this.trainSize
        this.devSize = devSize ?: this.devSize
        this.testSize = testSize ?: this.testSize

        _cachedTrain = null
        _cachedDev = null
        _cachedTest = null
    }

    /**
     * Lazily-loaded training examples.
     */
    val train: List<Example>
        get() {
            if (_cachedTrain == null) {
                _cachedTrain = shuffleAndSample("train", _train(), trainSize, trainSeed)
            }
            return _cachedTrain!!
        }

    /**
     * Lazily-loaded dev examples.
     */
    val dev: List<Example>
        get() {
            if (_cachedDev == null) {
                _cachedDev = shuffleAndSample("dev", _dev(), devSize, evalSeed)
            }
            return _cachedDev!!
        }

    /**
     * Lazily-loaded test examples.
     */
    val test: List<Example>
        get() {
            if (_cachedTest == null) {
                _cachedTest = shuffleAndSample("test", _test(), testSize, evalSeed)
            }
            return _cachedTest!!
        }

    /**
     * Shuffle and sample data for a split.
     */
    private fun shuffleAndSample(
        split: String,
        data: Iterable<Map<String, Any?>>,
        size: Int?,
        seed: Int = 0,
    ): List<Example> {
        val dataList = data.toList().toMutableList()

        val baseRng = Random(seed)

        val shuffled = if (doShuffle) {
            dataList.shuffle(baseRng)
            dataList
        } else {
            dataList
        }

        val sliced = if (size != null && size < shuffled.size) {
            shuffled.take(size)
        } else {
            shuffled
        }

        return sliced.map { example ->
            val store = mutableMapOf<String, Any?>()
            store.putAll(example)
            store["dspy_uuid"] = UUID.randomUUID().toString()
            store["dspy_split"] = split

            val exampleObj = Example(store.toMap())
            inputKeys?.let { keys ->
                if (keys.isNotEmpty()) {
                    exampleObj.withInputs(*keys.toTypedArray())
                }
            }
            exampleObj
        }
    }

    /**
     * Prepare datasets split by multiple train seeds.
     *
     * Usage:
     * ```kotlin
     * val datasets = MyDataSet.prepareBySeed(
     *     trainSeeds = listOf(1, 2, 3, 4, 5),
     *     trainSize = 16,
     *     devSize = 1000,
     * )
     * // datasets.train_sets and datasets.eval_sets
     * ```
     */
    companion object {
        /**
         * Prepare datasets split by multiple train seeds.
         */
        fun prepareBySeed(
            dataset: DataSet,
            trainSeeds: List<Int> = listOf(1, 2, 3, 4, 5),
            trainSize: Int = 16,
            devSize: Int = 1000,
            divideEvalPerSeed: Boolean = true,
            evalSeed: Int = 2023,
            extraArgs: Map<String, Any?> = emptyMap(),
        ): DotDict {
            dataset.resetSeeds(
                trainSeed = trainSeeds.firstOrNull() ?: 0,
                trainSize = trainSize,
                evalSeed = evalSeed,
                devSize = devSize,
            )

            val evalSet = dataset.dev
            val evalSets = mutableListOf<List<Example>>()
            val trainSets = mutableListOf<List<Example>>()

            val examplesPerSeed = if (divideEvalPerSeed) devSize / trainSeeds.size else devSize
            var evalOffset = 0

            for (trainSeed in trainSeeds) {
                dataset.resetSeeds(
                    trainSeed = trainSeed,
                    trainSize = trainSize,
                    evalSeed = evalSeed,
                    devSize = devSize,
                )

                val evalSlice = evalSet.subList(
                    evalOffset,
                    minOf(evalOffset + examplesPerSeed, evalSet.size)
                )
                evalSets.add(evalSlice)
                trainSets.add(dataset.train)

                require(evalSlice.size == examplesPerSeed) {
                    "Expected $examplesPerSeed eval examples, got ${evalSlice.size}"
                }
                require(dataset.train.size == trainSize) {
                    "Expected $trainSize train examples, got ${dataset.train.size}"
                }

                if (divideEvalPerSeed) {
                    evalOffset += examplesPerSeed
                }
            }

            return DotDict(
                mapOf(
                    "train_sets" to trainSets,
                    "eval_sets" to evalSets,
                )
            )
        }
    }
}

/**
 * Simple dotdict for dynamic property access.
 *
 * Faithful port of `dspy.dsp.utils.dotdict`.
 */
class DotDict(private val data: Map<String, Any?> = emptyMap()) : Map<String, Any?> by data {
    constructor(vararg pairs: Pair<String, Any?>) : this(pairs.toMap())

    fun getByKey(key: String): Any? = data[key]

    override fun toString(): String = data.toString()
}
