package dspy.adapters.types

import java.io.File
import java.net.URL
import kotlinx.serialization.Serializable

/**
 * Image type in DSPy.
 *
 * Represents image content that can be passed as input to language models.
 */
@Serializable
data class Image(
    val url: String
) : Type() {
    override fun format(): List<Map<String, Any?>> {
        val imageUrl = encodeImage(url)
        return listOf(mapOf("type" to "image_url", "image_url" to mapOf("url" to imageUrl)))
    }

    companion object {
        @Deprecated("Use Image(url) instead")
        fun fromUrl(url: String): Image {
            return Image(url)
        }

        @Deprecated("Use Image(file_path) instead")
        fun fromFile(filePath: String): Image {
            return Image(filePath)
        }
    }

    override fun toString(): String {
        return if ("base64" in url) {
            val base64Part = url.split("base64,")[1]
            val imageType = url.split(";")[0].split("/").last()
            "Image(url=data:image/$imageType;base64,<IMAGE_BASE_64_ENCODED(${base64Part.length})>)"
        } else {
            "Image(url='$url')"
        }
    }
}

private fun isUrlString(string: String): Boolean {
    return try {
        val url = URL(string)
        listOf("http", "https", "gs").contains(url.protocol) && url.host != null
    } catch (_: Exception) {
        false
    }
}

fun encodeImage(image: Any, downloadImages: Boolean = false, verify: Boolean = true): String {
    return when (image) {
        is Map<*, *> -> (image as? Map<String, Any?>)?.get("url") as? String ?: throw IllegalArgumentException("Invalid image dict")
        is String -> {
            if (image.startsWith("data:")) {
                image // Already a data URI
            } else if (File(image).exists()) {
                encodeImageFromFile(image)
            } else if (isUrlString(image)) {
                if (downloadImages) {
                    encodeImageFromUrl(image, verify = verify)
                } else {
                    image
                }
            } else {
                throw IllegalArgumentException("Unrecognized file string: $image")
            }
        }
        is ByteArray -> {
            val mimeType = "image/png" // Default for raw bytes
            val encoded = java.util.Base64.getEncoder().encodeToString(image)
            "data:$mimeType;base64,$encoded"
        }
        is Image -> image.url
        else -> throw IllegalArgumentException("Unsupported image type: ${image::class}")
    }
}

private fun encodeImageFromFile(filePath: String): String {
    val file = File(filePath)
    val fileData = file.readBytes()
    val mimeType = java.net.URLConnection.guessContentTypeFromName(filePath)
        ?: throw IllegalArgumentException("Could not determine MIME type for file: $filePath")
    val encodedData = java.util.Base64.getEncoder().encodeToString(fileData)
    return "data:$mimeType;base64,$encodedData"
}

private fun encodeImageFromUrl(imageUrl: String, verify: Boolean = true): String {
    val connection = URL(imageUrl).openConnection()
    // Note: SSL verification control not available in standard Java URL
    val content = connection.getInputStream().readBytes()
    val contentType = connection.contentType ?: "image/png"
    val encoded = java.util.Base64.getEncoder().encodeToString(content)
    return "data:$contentType;base64,$encoded"
}

fun isImage(obj: Any): Boolean {
    if (obj is String) {
        return obj.startsWith("data:") || File(obj).exists() || isUrlString(obj)
    }
    return false
}
