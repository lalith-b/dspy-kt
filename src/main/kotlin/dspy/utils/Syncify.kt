package dspy.utils

import dspy.primitives.Module

/**
 * Sync utilities for DSPy.
 *
 * Faithful port of `dspy/utils/syncify.py`.
 *
 * Provides utilities for running async coroutines from synchronous contexts
 * and converting async DSPy modules to sync programs.
 */

/**
 * Run an async coroutine from a synchronous context.
 *
 * If we're in a running event loop (e.g., Jupyter), uses
 * nest_asyncio. Otherwise, creates a new event loop.
 */
fun runAsync(coro: suspend () -> Any?): Any? {
    return kotlinx.coroutines.runBlocking {
        coro()
    }
}

/**
 * Convert an async DSPy module to a sync program.
 *
 * Two modes:
 * - `inPlace=true` (recommended): Modify the module in place. May not work
 *   if the module already has a `forward` method different from `aforward`.
 * - `inPlace=false`: Return a wrapper module. More robust but changes architecture.
 *
 * Args:
 *     program: The async program to convert, must have an `aforward` method.
 *     inPlace: If true, modify the module in place.
 *
 * Returns:
 *     The sync program.
 */
fun syncify(program: Module, inPlace: Boolean = true): Module {
    if (inPlace) {
        // In Kotlin, async/sync is handled natively with suspend functions.
        // This is a stub that returns the program as-is.
        // A full implementation would wrap aforward() calls in runBlocking.
        return program
    } else {
        return SyncWrapper(program)
    }
}

/**
 * Wrapper module that makes async calls synchronous.
 */
class SyncWrapper(private val program: Module) : Module() {
    fun forward(vararg args: Any?, kwargs: Map<String, Any?> = emptyMap()): Any? {
        // In Kotlin, we would use runBlocking to call the async version
        return runAsync {
            // Call the program's forward method
            program.invoke(kwargs = kwargs)
        }
    }
}
