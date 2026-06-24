plugins {
    kotlin("jvm") version "2.2.0"
    kotlin("plugin.serialization") version "2.2.0"
    `maven-publish`
}

group = "com.stanfordnlp"
version = "0.1.0-SNAPSHOT"
description = "DSPy Kotlin - Declarational AI with LLMs"

repositories {
    mavenCentral()
}

val ktorVersion = "3.0.3"
val coroutinesVersion = "1.8.1"
val serializationVersion = "1.7.1"
val kotestVersion = "5.9.1"
val slf4jVersion = "2.0.16"
val logbackVersion = "1.5.12"

dependencies {
    // Ktor HTTP client
    implementation("io.ktor:ktor-client-core:$ktorVersion")
    implementation("io.ktor:ktor-client-cio:$ktorVersion")
    implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-client-auth:$ktorVersion")
    implementation("io.ktor:ktor-client-logging:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("io.ktor:ktor-server-core:$ktorVersion") // for SSE client support

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")

    // Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:$serializationVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")
    implementation("org.jetbrains.kotlinx:kotlinx-io:0.1.16")

    // Logging
    implementation("org.slf4j:slf4j-api:$slf4jVersion")
    implementation("ch.qos.logback:logback-classic:$logbackVersion")

    // Testing
    testImplementation(kotlin("test"))
    testImplementation("io.kotest:kotest-runner-junit5:$kotestVersion")
    testImplementation("io.kotest:kotest-assertions-core:$kotestVersion")
    testImplementation("io.kotest:kotest-property:$kotestVersion")
    testImplementation("io.ktor:ktor-client-mock:$ktorVersion")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:$coroutinesVersion")
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        freeCompilerArgs.add("-Xno-call-assertions")
        freeCompilerArgs.add("-Xno-receiver-assertions")
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

tasks.test {
    useJUnitPlatform()
}

// Source JAR
tasks.register<Jar>("sourcesJar") {
    archiveClassifier.set("sources")
    from(kotlin.sourceSets["main"].kotlin)
    dependsOn(tasks.classes)
}

// Javadoc JAR (empty for Kotlin)
tasks.register<Jar>("javadocJar") {
    archiveClassifier.set("javadoc")
    from(layout.buildDirectory.map { it.dir("classes/kotlin/main") })
    dependsOn(tasks.classes)
    mustRunAfter(tasks.compileTestKotlin)
}

// Shadow JAR (fat JAR with dependencies) - disabled for now due to Gradle 9.1 compatibility
// tasks.shadowJar {
//     archiveClassifier.set("all")
//     manifest {
//         attributes(
//             "Implementation-Title" to project.name,
//             "Implementation-Version" to project.version,
//             "Implementation-Vendor" to "StanfordNLP",
//         )
//     }
//     // Exclude unnecessary files
//     exclude("META-INF/*.SF", "META-INF/*.DSA", "META-INF/*.RSA")
//     dependsOn(tasks.jar)
// }

// Maven Publishing
publishing {
    publications {
        register<MavenPublication>("dspyKt") {
            groupId = "com.stanfordnlp"
            artifactId = "dspy-kt"
            version = project.version.toString()

            from(components["kotlin"])

            // Attach source and javadoc JARs
            artifact(tasks["sourcesJar"])
            artifact(tasks["javadocJar"])

            // Shadow JAR as additional artifact - disabled
            // artifact(tasks["shadowJar"]) {
            //     classifier = "all"
            // }

            pom {
                name.set("DSPy Kotlin")
                description.set("Kotlin port of DSPy - Declarational AI with LLMs")
                url.set("https://github.com/stanfordnlp/dspy-kt")
                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
                developers {
                    developer {
                        id.set("dspy-team")
                        name.set("DSPy Team")
                        email.set("dspy@stanford.edu")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/stanfordnlp/dspy-kt.git")
                    developerConnection.set("scm:git:ssh://github.com/stanfordnlp/dspy-kt.git")
                    url.set("https://github.com/stanfordnlp/dspy-kt")
                }
            }
        }
    }

    repositories {
        maven {
            name = "local"
            url = uri("${layout.buildDirectory.get()}/repo")
        }
    }
}

// Signing (optional, for release)
// signing {
//     sign(publishing.publications["dspyKt"])
// }
