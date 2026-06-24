package dspy.predict

/**
 * Marker interface for DSPy parameters.
 *
 * In Python, `Parameter` is an empty class used as a base class for
 * `Predict`. In Kotlin, we delegate to the existing `Parameter` interface
 * from `dspy.primitives.Parameter`.
 *
 * This file exists as a faithful port of `dspy/predict/parameter.py`.
 */
typealias Parameter = dspy.primitives.Parameter
