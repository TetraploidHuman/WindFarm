package cn.tetraploid.rtwind.sensor.data

import io.ktor.client.HttpClient
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class RtwindApi @Inject constructor(
    private val client: HttpClient,
) {
    suspend fun health(baseUrl: String): Result<String> = runCatching {
        val resp = client.get("$baseUrl/api/health")
        val text = resp.bodyAsText()
        if (!resp.status.isSuccess()) error("HTTP ${resp.status.value}: $text")
        text
    }

    suspend fun ingest(baseUrl: String, payload: TelemetryPayload): Result<Unit> = runCatching {
        val resp = client.post("$baseUrl/api/ingest") {
            contentType(ContentType.Application.Json)
            setBody(payload)
        }
        if (!resp.status.isSuccess()) {
            error("HTTP ${resp.status.value}: ${resp.bodyAsText()}")
        }
    }
}
