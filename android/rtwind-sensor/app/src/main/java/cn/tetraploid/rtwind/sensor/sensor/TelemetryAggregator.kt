package cn.tetraploid.rtwind.sensor.sensor

import android.content.Context
import android.content.IntentFilter
import android.os.BatteryManager
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.data.RtwindApi
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import cn.tetraploid.rtwind.sensor.data.TelemetryPayload
import cn.tetraploid.rtwind.sensor.data.TelemetrySnapshot
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TelemetryAggregator @Inject constructor(
    @ApplicationContext private val context: Context,
    private val locationTracker: LocationTracker,
    private val imuTracker: ImuTracker,
    private val settingsRepository: SettingsRepository,
    private val api: RtwindApi,
) {
    private val _snapshot = MutableStateFlow(TelemetrySnapshot())
    val snapshot: StateFlow<TelemetrySnapshot> = _snapshot.asStateFlow()

    private var gps: GpsReading? = null
    private var imu: ImuReading = ImuReading(0.0, 0.0, 0.0)
    private var prevAlt: Double? = null
    private var prevAltTimeMs: Long = 0L

    private var sensorJobs: List<Job> = emptyList()
    private var uploadJob: Job? = null
    private var linkJob: Job? = null

    fun start(scope: CoroutineScope) {
        if (sensorJobs.isNotEmpty()) return
        _snapshot.update {
            it.copy(
                serviceRunning = true,
                lastUploadError = null,
                locationDiag = locationTracker.diagnose(),
            )
        }
        publishLocal()

        linkJob = scope.launch {
            val settings = settingsRepository.settings.first()
            api.startWebSocket(scope, settings.serverBaseUrl)
        }

        sensorJobs = listOf(
            scope.launch {
                try {
                    locationTracker.readings().collect { reading ->
                        gps = reading
                        _snapshot.update {
                            it.copy(
                                locationDiag = "已定位(${reading.provider}) " + locationTracker.diagnose(),
                            )
                        }
                        publishLocal()
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    _snapshot.update {
                        it.copy(locationDiag = "定位失败: ${e.message} | ${locationTracker.diagnose()}")
                    }
                }
            },
            scope.launch {
                try {
                    imuTracker.readings().collect { reading ->
                        imu = reading
                        publishLocal()
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (_: Exception) {
                }
            },
        )

        uploadJob = scope.launch {
            while (isActive) {
                val settings = settingsRepository.settings.first()
                uploadOnce(settings)
                delay((1000L / settings.uploadHz.coerceAtLeast(1)))
            }
        }
    }

    fun stop() {
        sensorJobs.forEach { it.cancel() }
        sensorJobs = emptyList()
        uploadJob?.cancel()
        uploadJob = null
        linkJob?.cancel()
        linkJob = null
        api.stopWebSocket()
        _snapshot.update {
            it.copy(
                serviceRunning = false,
                lastUploadError = null,
                locationDiag = "已停止 | ${locationTracker.diagnose()}",
            )
        }
    }

    private fun publishLocal() {
        val g = gps
        val climb = g?.let { computeClimbRate(it.altMsl) } ?: 0.0
        _snapshot.update {
            it.copy(
                lat = g?.lat,
                lon = g?.lon,
                altMsl = g?.altMsl,
                heading = when {
                    g != null && g.speedMps > 0.5 -> g.bearing
                    else -> imu.yawDeg
                },
                roll = imu.rollDeg,
                pitch = imu.pitchDeg,
                yaw = imu.yawDeg,
                groundspeed = g?.speedMps ?: 0.0,
                climbRate = climb,
                battery = readBatteryPct(),
                gpsAccuracyM = g?.accuracyM,
                gpsProvider = g?.provider,
            )
        }
    }

    private suspend fun uploadOnce(settings: AppSettings) {
        val g = gps
        if (g == null) {
            _snapshot.update {
                it.copy(
                    lastUploadOk = false,
                    lastUploadError = "等待定位…",
                    locationDiag = locationTracker.diagnose(),
                )
            }
            return
        }

        val payload = TelemetryPayload(
            lat = g.lat,
            lon = g.lon,
            altMsl = g.altMsl,
            heading = if (g.speedMps > 0.5) g.bearing else imu.yawDeg,
            roll = if (settings.enableImu) imu.rollDeg else 0.0,
            pitch = if (settings.enableImu) imu.pitchDeg else 0.0,
            yaw = if (settings.enableImu) imu.yawDeg else null,
            airspeed = g.speedMps,
            groundspeed = g.speedMps,
            climbRate = computeClimbRate(g.altMsl),
            battery = readBatteryPct(),
            vehicleId = settings.vehicleId,
            t = TelemetryPayload.nowIso(),
        )

        if (api.isWsConnected()) {
            api.enqueueWs(payload)
            _snapshot.update {
                it.copy(
                    lastUploadOk = true,
                    lastUploadError = null,
                    uploadChannel = "WebSocket",
                    uploadsTotal = it.uploadsTotal + 1,
                )
            }
            return
        }

        // WS 未连上时退回 HTTP，避免完全断流
        api.ingestHttp(settings.serverBaseUrl, payload)
            .onSuccess {
                _snapshot.update {
                    it.copy(
                        lastUploadOk = true,
                        lastUploadError = null,
                        uploadChannel = "HTTP",
                        uploadsTotal = it.uploadsTotal + 1,
                    )
                }
            }
            .onFailure { err ->
                _snapshot.update {
                    it.copy(
                        lastUploadOk = false,
                        lastUploadError = err.message ?: "上传失败",
                        uploadChannel = "HTTP",
                        uploadsFailed = it.uploadsFailed + 1,
                    )
                }
            }
    }

    private fun computeClimbRate(altMsl: Double): Double {
        val now = System.currentTimeMillis()
        val prev = prevAlt
        val dt = (now - prevAltTimeMs) / 1000.0
        prevAlt = altMsl
        prevAltTimeMs = now
        if (prev == null || dt <= 0.0) return 0.0
        return (altMsl - prev) / dt
    }

    private fun readBatteryPct(): Double? {
        val intent = context.registerReceiver(null, IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
            ?: return null
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (level < 0 || scale <= 0) return null
        return level * 100.0 / scale
    }
}
