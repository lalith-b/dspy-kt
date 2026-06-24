package dspy.utils

import java.text.SimpleDateFormat
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger

class DSPyLoggingStream {
    var enabled: Boolean = true

    fun write(text: String) {
        if (enabled) {
            System.err.print(text)
        }
    }

    fun flush() {
        if (enabled) {
            System.err.flush()
        }
    }
}

val DSPY_LOGGING_STREAM = DSPyLoggingStream()

fun disableLogging() {
    DSPY_LOGGING_STREAM.enabled = false
}

fun enableLogging() {
    DSPY_LOGGING_STREAM.enabled = true
}

fun formatRecord(record: LogRecord): String {
    val timestamp = SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(java.util.Date(record.millis))
    val level = record.level.name
    val name = record.loggerName
    val message = record.message
    return "$timestamp $level $name: $message\n"
}

fun configureDspyLoggers(rootModuleName: String) {
    val logger = Logger.getLogger(rootModuleName)
    logger.level = Level.INFO
    logger.useParentHandlers = false

    val handlers = logger.handlers.toList()
    for (h in handlers) {
        logger.removeHandler(h)
    }

    val logHandler = object : Handler() {
        init {
            level = Level.ALL
        }

        override fun publish(record: LogRecord) {
            if (DSPY_LOGGING_STREAM.enabled) {
                val text = formatRecord(record)
                DSPY_LOGGING_STREAM.write(text)
                DSPY_LOGGING_STREAM.flush()
            }
        }

        override fun flush() {
            DSPY_LOGGING_STREAM.flush()
        }

        override fun close() {
        }
    }

    logger.addHandler(logHandler)
}
