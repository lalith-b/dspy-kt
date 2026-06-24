package dspy.primitives

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe

class ExampleInitializationTest : FunSpec({
    test("example initialization with named args") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        example["a"] shouldBe 1
        example["b"] shouldBe 2
    }

    test("example initialization from base Example") {
        val base = Example(mapOf("a" to 1, "b" to 2))
        val example = Example(base = base.toMap(), extra = mapOf("c" to 3))
        example["a"] shouldBe 1
        example["b"] shouldBe 2
        example["c"] shouldBe 3
    }

    test("example initialization from dict") {
        val baseDict = mapOf("a" to 1, "b" to 2)
        val example = Example(base = baseDict, extra = mapOf("c" to 3))
        example["a"] shouldBe 1
        example["b"] shouldBe 2
        example["c"] shouldBe 3
    }
})

class ExampleSetGetTest : FunSpec({
    test("example set and get item") {
        val example = Example()
        example["a"] = 1
        example["a"] shouldBe 1
    }
})

class ExampleDeletionTest : FunSpec({
    test("example deletion returns null") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        example.remove("a")
        example["a"] shouldBe null
    }
})

class ExampleLenTest : FunSpec({
    test("example len") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        example.size shouldBe 2
    }
})

class ExampleReprTest : FunSpec({
    test("example repr str") {
        val example = Example(base = mapOf("a" to 1))
        example.toString().contains("Example") shouldBe true
    }
})

class ExampleEqTest : FunSpec({
    test("example equality") {
        val example1 = Example(base = mapOf("a" to 1, "b" to 2))
        val example2 = Example(base = mapOf("a" to 1, "b" to 2))
        example1 shouldBe example2
        example1 shouldNotBe ""
    }

    test("example hash") {
        val example1 = Example(base = mapOf("a" to 1, "b" to 2))
        val example2 = Example(base = mapOf("a" to 1, "b" to 2))
        example1.hashCode() shouldBe example2.hashCode()
    }
})

class ExampleKeysValuesItemsTest : FunSpec({
    test("example keys values items") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        setOf("a", "b") shouldBe example.keys()
        example.values().contains(1) shouldBe true
        example.items().contains(Pair("b", 2)) shouldBe true
    }
})

class ExampleGetTest : FunSpec({
    test("example get with default") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        example.get("a") shouldBe 1
        example.get("c", "default") shouldBe "default"
    }
})

class ExampleWithInputsTest : FunSpec({
    test("example with inputs") {
        val example = Example(base = mapOf("a" to 1, "b" to 2)).withInputs("a")
        example.inputKeys() shouldBe setOf("a")
    }
})

class ExampleInputsLabelsTest : FunSpec({
    test("example inputs labels") {
        val example = Example(base = mapOf("a" to 1, "b" to 2)).withInputs("a")
        val inputs = example.inputs()
        inputs.toDict() shouldBe mapOf("a" to 1)
        val labels = example.labels()
        labels.toDict() shouldBe mapOf("b" to 2)
    }
})

class ExampleCopyWithoutTest : FunSpec({
    test("example copy without") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        val copied = example.copy(extra = mapOf("c" to 3))
        copied["a"] shouldBe 1
        copied["c"] shouldBe 3
        val withoutA = copied.without("a")
        withoutA["a"] shouldBe null
    }
})

class ExampleToDictTest : FunSpec({
    test("example to dict") {
        val example = Example(base = mapOf("a" to 1, "b" to 2))
        example.toDict() shouldBe mapOf("a" to 1, "b" to 2)
    }
})
