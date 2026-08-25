package cn.tetraploid.rtwind.sensor.data

import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
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
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private var wsJob: Job? = null
    @Volatile private var wsConnected: Boolean = false

    fun isWsConnected(): Boolean = wsConnected

    suspend fun health(baseUrl: String): Result<String> = runCatching {
        val resp = client.get("$baseUrl/api/health")
        val text = resp.bodyAsText()
        if (!resp.status.isSuccess()) error("HTTP ${resp.status.value}: $text")
        text
    }

    suspend fun ingestHttp(baseUrl: String, payload: TelemetryPayload): Result<Unit> = runCatching {
        val resp = client.post("$baseUrl/api/ingest") {
            contentType(ContentType.Application.Json)
            setBody(payload)
        }
        if (!resp.status.isSuccess()) {
            error("HTTP ${resp.status.value}: ${resp.bodyAsText()}")
        }
    }

    fun enqueueWs(payload: TelemetryPayload) {
        outbound.tryEmit(payload)
    }

    fun startWebSocket(scope: CoroutineScope, baseUrl: String) {
        if (wsJob?.isActive == true) return
        val wsUrl = toWsUrl(baseUrl)
        wsJob = scope.launch {
            while (isActive) {
                try {
                    client.webSocket(urlString = wsUrl) {
                        wsConnected = true
                        val sender = launch {
                            outbound.collect { payload ->
                                send(Frame.Text(json.encodeToString(WsIngestEnvelope(frame = payload))))
                            }
                        }
                        try {
                            for (frame in incoming) {
                                if (frame is Frame.Text) {
                                    frame.readText() // drain server events
                                }
                            }
                        } finally {
                            sender.cancel()
                        }
                    }
                } catch (_: Exception) {
                    // reconnect below
                } finally {
                    wsConnected = false
                }
                kotlinx.coroutines.delay(800)
            }
        }
    }

    fun stopWebSocket() {
        wsJob?.cancel()
        wsJob = null
        wsConnected = false
    }

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
            return if (withScheme.endsWith("/api/ws/ingest")) withScheme
            else if (withScheme.endsWith("/api/ws")) withScheme.replace("/api/ws", "/api/ws/ingest")
            else "$withScheme/api/ws/ingest"
        }
    }
}
