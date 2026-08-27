package cn.tetraploid.rtwind.sensor.log

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Ring-buffer file log for field debugging (copy/share after crashes).
 * Initialized from [cn.tetraploid.rtwind.sensor.RtwindSensorApp] before other components run.
 */
object AppLog {
    private const val FILE_NAME = "rtwind_app.log"
    private const val MAX_BYTES = 512 * 1024

    private val ready = AtomicBoolean(false)
    private val lock = Any()
    private var logFile: File? = null
    private val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun init(context: Context) {
        synchronized(lock) {
            logFile = File(context.applicationContext.filesDir, FILE_NAME)
            ready.set(true)
        }
    }

    fun i(tag: String, message: String) = write("I", tag, message, null)

    fun w(tag: String, message: String, error: Throwable? = null) = write("W", tag, message, error)

    fun e(tag: String, message: String, error: Throwable? = null) = write("E", tag, message, error)

    fun logCrash(thread: Thread, error: Throwable) {
        write(
            "F",
            "CRASH",
            "Uncaught on ${thread.name}: ${error.message}",
            error,
        )
    }

    fun readText(): String = synchronized(lock) {
        val file = logFile
        if (file == null || !file.exists()) return ""
        runCatching { file.readText() }.getOrDefault("")
    }

    fun lineCount(): Int = readText().lines().count { it.isNotBlank() }

    fun byteSize(): Long = synchronized(lock) {
        val file = logFile
        if (file == null || !file.exists()) 0L else file.length()
    }

    fun clear() = synchronized(lock) {
        logFile?.writeText("")
    }

    private fun write(level: String, tag: String, message: String, error: Throwable?) {
        val line = buildString {
            append(ts.format(Date()))
            append(' ')
            append(level)
            append('/')
            append(tag)
            append(": ")
            append(message)
            if (error != null) {
                append('\n')
                append(Log.getStackTraceString(error))
            }
        }

        when (level) {
            "E", "F" -> Log.e(tag, message, error)
            "W" -> Log.w(tag, message, error)
            else -> Log.i(tag, message)
        }

        if (!ready.get()) return
        synchronized(lock) {
            val file = logFile ?: return
            runCatching {
                file.appendText(line + "\n")
                trimIfNeeded(file)
            }
        }
    }

    private fun trimIfNeeded(file: File) {
        if (file.length() <= MAX_BYTES) return
        val text = file.readText()
        val keep = text.takeLast(MAX_BYTES / 2)
        val idx = keep.indexOf('\n')
        file.writeText(
            if (idx >= 0) keep.substring(idx + 1) else keep,
        )
    }
}
