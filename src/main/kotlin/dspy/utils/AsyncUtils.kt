package dspy.utils

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * Async utilities for DSPy.
 * 
 * Port of dspy/utils/asyncify.py
 */

/**
 * Convert a coroutine block to a callback-based function.
 */
suspend fun <T> asyncify(block: suspend () -> T): T = block()

/**
 * Launch a coroutine in the background.
 */
fun launchCoroutine(block: suspend CoroutineScope.() -> Unit): Job {
    val scope = CoroutineScope(Dispatchers.Default)
    return scope.launch { block() }
}

/**
 * Run a coroutine block.
 */
fun <T> run(block: suspend CoroutineScope.() -> T): T = runBlocking { block() }

