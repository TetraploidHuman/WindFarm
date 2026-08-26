package cn.tetraploid.rtwind.sensor.data

import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.webSocketSession
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.request.url
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.websocket.Frame
import io.ktor.websocket.CloseReason
import io.ktor.websocket.close
import io.ktor.websocket.readText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.concurrent.atomic.AtomicReference
import javax.inject.Inject
import javax.inject.Singleton

@Serializable
data class WsIngestEnvelope(
    val type: String = "ingest",
    val frame: TelemetryPayload,
)

@Singleton
class RtwindApi @Inject constructor(
    private val client: HttpClient,
    private val json: Json,
) {
    private val outbound = MutableSharedFlow<TelemetryPayload>(
        replay = 0,
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    /** Latest frame for HTTP coalesce (always overwrite). */
    private val latestHttp = AtomicReference<TelemetryPayload?>(null)
    private data class CameraUpload(val jpeg: ByteArray, val vehicleId: String)
    private val latestCamera = AtomicReference<CameraUpload?>(null)
    private var transportJob: Job? = null
    private val connected = MutableStateFlow(false)
    private val sendMutex = Mutex()

    fun isWsConnected(): Boolean = connected.value

    suspend fun health(baseUrl: String): Result<String> = runCatching {
        val resp = client.get("$baseUrl/api/health")
        val text = resp.bodyAsText()
        if (!resp.status.isSuccess()) error("HTTP ${resp.status.value}: $text")
        text
    }

    private var onCameraUploadResult: ((success: Boolean, frameKb: Int, error: String?) -> Unit)? = null

    fun setCameraUploadListener(listener: (success: Boolean, frameKb: Int, error: String?) -> Unit) {
        onCameraUploadResult = listener
    }

    /** Non-blocking camera JPEG publish (coalesced HTTP upload). */
    fun publishCamera(jpeg: ByteArray, vehicleId: String) {
        latestCamera.set(CameraUpload(jpeg, vehicleId))
    }

    /** Non-blocking publish: WS queue + HTTP latest coalesce. */
    fun publish(payload: TelemetryPayload) {
        outbound.tryEmit(payload)
        latestHttp.set(payload)
    }

    fun startTransport(scope: CoroutineScope, baseUrl: String) {
        if (transportJob?.isActive == true) return
        val wsUrl = toWsUrl(baseUrl)
        transportJob = scope.launch {
            // HTTP coalesce worker: never blocks producers; always sends newest pending frame
            val httpWorker = launch {
                while (isActive) {
                    if (connected.value) {
                        delay(50)
                        continue
                    }
                    val payload = latestHttp.getAndSet(null)
                    if (payload == null) {
                        delay(40)
                        continue
                    }
                    runCatching {
                        val resp = client.post("$baseUrl/api/ingest") {
                            contentType(ContentType.Application.Json)
                            setBody(payload)
                        }
                        if (!resp.status.isSuccess()) error("HTTP ${resp.status.value}")
                    }
                }
            }

            val cameraWorker = launch {
                while (isActive) {
                    val upload = latestCamera.getAndSet(null)
                    if (upload == null) {
                        delay(8)
                        continue
                    }
                    val frameKb = (upload.jpeg.size + 1023) / 1024
                    runCatching {
                        val resp = client.post("$baseUrl/api/camera/upload?vehicle_id=${upload.vehicleId}") {
                            contentType(ContentType.Image.JPEG)
                            setBody(upload.jpeg)
                        }
                        if (!resp.status.isSuccess()) error("HTTP ${resp.status.value}")
                    }.onSuccess {
                        onCameraUploadResult?.invoke(true, frameKb, null)
                    }.onFailure { e ->
                        onCameraUploadResult?.invoke(false, frameKb, e.message ?: "upload failed")
                    }
                }
            }

            // WS worker with fast reconnect
            val wsWorker = launch {
                while (isActive) {
                    try {
                        val session = client.webSocketSession { url(wsUrl) }
                        connected.value = true
                        val sender = launch {
                            outbound.collect { payload ->
                                sendMutex.withLock {
                                    session.send(
                                        Frame.Text(json.encodeToString(WsIngestEnvelope(frame = payload))),
                                    )
                                }
                            }
                        }
                        val reader = launch {
                            try {
                                for (frame in session.incoming) {
                                    if (frame is Frame.Text) frame.readText()
                                }
                            } catch (_: Exception) {
                            }
                        }
                        reader.join()
                        sender.cancel()
                        runCatching { session.close(CloseReason(CloseReason.Codes.NORMAL, "bye")) }
                    } catch (_: Exception) {
                        // fall through to reconnect
                    } finally {
                        connected.value = false
                    }
                    delay(300)
                }
            }

            httpWorker.join()
            cameraWorker.cancel()
            wsWorker.cancel()
        }
    }

    fun stopTransport() {
        transportJob?.cancel()
        transportJob = null
        connected.value = false
        latestHttp.set(null)
        latestCamera.set(null)
    }

    // Back-compat names used by aggregator
    fun enqueueWs(payload: TelemetryPayload) = publish(payload)
    fun startWebSocket(scope: CoroutineScope, baseUrl: String) = startTransport(scope, baseUrl)
    fun stopWebSocket() = stopTransport()

    companion object {
        fun toWsUrl(baseUrl: String): String {
            val b = baseUrl.trim().trimEnd('/')
            val withScheme = when {
                b.startsWith("https://", ignoreCase = true) ->
                    "wss://" + b.substringAfter("://")
                b.startsWith("http://", ignoreCase = true) ->
                    "ws://" + b.substringAfter("://")
                b.startsWith("wss://", ignoreCase = true) || b.startsWith("ws://", ignoreCase = true) -> b
                else -> "ws://$b"
            }
            return when {
                withScheme.endsWith("/api/ws/ingest") -> withScheme
                withScheme.endsWith("/api/ws") -> withScheme.replace("/api/ws", "/api/ws/ingest")
                else -> "$withScheme/api/ws/ingest"
            }
        }
    }
}
