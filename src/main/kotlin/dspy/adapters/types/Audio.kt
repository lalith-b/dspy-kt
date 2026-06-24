package dspy.adapters.types

/**
 * Audio type in DSPy.
 *
 * Represents audio content that can be passed as input to language models.
 */
data class Audio(
    val data: String,
    val audioFormat: String
) : Type() {
    override fun format(): List<Map<String, Any?>> {
        return listOf(mapOf(
            "type" to "input_audio",
            "input_audio" to mapOf("data" to data, "format" to audioFormat)
        ))
    }

    companion object {
        fun fromUrl(url: String): Audio {
            // Download from URL and encode as base64
            throw NotImplementedError("Audio.fromUrl not yet implemented in Kotlin")
        }

        fun fromFile(filePath: String): Audio {
            throw NotImplementedError("Audio.fromFile not yet implemented in Kotlin")
        }

        fun fromArray(array: Any, samplingRate: Int, format: String = "wav"): Audio {
            throw NotImplementedError("Audio.fromArray not yet implemented in Kotlin (requires soundfile)")
        }
    }

    override fun toString(): String {
        return "Audio(data=<AUDIO_BASE_64_ENCODED(${data.length})>, audio_format='$audioFormat')"
    }
}

fun normalizeAudioFormat(audioFormat: String): String {
    return audioFormat.removePrefix("x-")
}

fun encodeAudio(audio: Any, samplingRate: Int = 16000, format: String = "wav"): Map<String, String> {
    return when (audio) {
        is Map<*, *> -> {
            val m = audio as Map<String, Any?>
            if ("data" in m && "audio_format" in m) {
                mapOf("data" to (m["data"] as String), "audio_format" to (m["audio_format"] as String))
            } else throw IllegalArgumentException("Invalid audio dict")
        }
        is Audio -> mapOf("data" to audio.data, "audio_format" to audio.audioFormat)
        is String -> {
            if (audio.startsWith("data:audio/")) {
                val parts = audio.split(",", limit = 2)
                val mime = parts[0].split(";")[0].split(":")[1]
                val audioFormat = normalizeAudioFormat(mime.split("/")[1])
                mapOf("data" to parts[1], "audio_format" to audioFormat)
            } else if (audio.startsWith("http")) {
                val a = Audio.fromUrl(audio)
                mapOf("data" to a.data, "audio_format" to a.audioFormat)
            } else {
                throw IllegalArgumentException("Unrecognized audio string: $audio")
            }
        }
        is ByteArray -> {
            val encoded = java.util.Base64.getEncoder().encodeToString(audio)
            mapOf("data" to encoded, "audio_format" to format)
        }
        else -> throw IllegalArgumentException("Unsupported type for encode_audio: ${audio::class}")
    }
}
