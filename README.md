# DSPy-Kotlin

A Kotlin port of [DSPy](https://github.com/stanfordnlp/dspy) — the programming framework for LLM-powered applications.

## Design

- **Pure Ktor HTTP clients** — no langchain4j or external LLM frameworks
- **Kotlin Coroutines** for async/concurrency (better than Python async)
- **kotlinx.serialization** for JSON request/response handling
- **Gradle** build system (JVM 17+)

## Architecture

```
dspy/
├── core/types/          # Normalized LM types (Messages, Parts, Config, Tools)
├── signatures/          # Input/output field definitions (builder pattern)
├── primitives/          # Module, Example, Prediction (composable building blocks)
├── clients/             # LM client (Ktor HTTP), cache (2-level LRU + disk)
├── adapters/            # ChatAdapter, JSONAdapter, XmlAdapter, TwoStepAdapter
├── predict/             # Predict, ChainOfThought, BestOfN, Retry
├── teleprompt/          # BootstrapFewShot, RandomSearch, Ensemble
├── evaluate/            # Evaluate runner, metrics (exactMatch, contains)
├── streaming/           # Kotlin Flow-based streaming
├── datasets/            # DataLoader, built-in dataset loaders
└── utils/               # Settings, Callback, Hasher, Saving
```

## Quick Start

```kotlin
import dspy.*

// Configure the LM
configure("gpt-4o", apiKey = System.getenv("OPENAI_API_KEY"))

// Create a predictor
val predict = Predict("question -> answer")

// Run it (suspend function)
val result = predict(question = "What is 2+2?")
println(result["answer"]) // "4"

// Chain of Thought
val cot = ChainOfThought("question -> reasoning, answer")

// Evaluation
val devset = listOf(
    Example(question = "What is 2+2?", answer = "4").withInputs("question"),
    Example(question = "What is 3+3?", answer = "6").withInputs("question"),
)

val result = evaluate(predict, devset, ::exactMatch)
println(result) // EvaluationResult(total=2, passed=2, accuracy=100%)
```

## Status

Scaffolding phase — core types and API surface in place.

## Differences from Python DSPy

| Python Feature | Kotlin Equivalent |
|---|---|
| `Signature("q -> a")` metaclass | `signature("q -> a")` factory function + `SignatureBuilder` DSL |
| `dspy.Predict("q -> a")` | `Predict("q -> a")` — same API |
| `Example(...).with_inputs()` | `Example(...).withInputs()` — same API |
| `module(**kwargs)` | `module(q = "What?", a = "4")` — named args |
| `async def` | `suspend fun` — Kotlin coroutines |
| `Streaming` via generators | `Flow<String>` — Kotlin flows |
| `Optuna` for optimization | No Kotlin equivalent — grid/random search first |
| `diskcache` + `cloudpickle` | `LinkedHashMap` LRU + `kotlinx.io` disk |

## Building

```bash
./gradle/wrapper/gradlew build
./gradle/wrapper/gradlew test
```

## License

Apache 2.0 (same as Python DSPy)
