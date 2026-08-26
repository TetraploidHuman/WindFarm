package cn.tetraploid.rtwind.sensor.sensor

import android.content.Context
import android.content.IntentFilter
import android.os.BatteryManager
import cn.tetraploid.rtwind.sensor.camera.CameraController
import cn.tetraploid.rtwind.sensor.camera.CameraStreamConfig
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
import androidx.lifecycle.LifecycleOwner
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
    private val cameraController: CameraController,
) {
    private val _snapshot = MutableStateFlow(TelemetrySnapshot())
    val snapshot: StateFlow<TelemetrySnapshot> = _snapshot.asStateFlow()

    private var gps: GpsReading? = null
    private var imu: ImuReading = ImuReading(0.0, 0.0, 0.0)
    private var prevAlt: Double? = null
    private var prevAltTimeMs: Long = 0L
    private var lastPushMs: Long = 0L

    private var sensorJobs: List<Job> = emptyList()
    private var uploadJob: Job? = null
    private var linkJob: Job? = null
    private var cameraSettingsJob: Job? = null
    private var cachedSettings: AppSettings = AppSettings()

    fun start(scope: CoroutineScope, lifecycleOwner: LifecycleOwner) {
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
            cachedSettings = settingsRepository.settings.first()
            api.setCameraUploadListener { success, frameKb, error ->
                _snapshot.update {
                    if (success) {
                        it.copy(
                            cameraUploadsTotal = it.cameraUploadsTotal + 1,
                            cameraLastFrameKb = frameKb,
                            cameraLastError = null,
                        )
                    } else {
                        it.copy(
                            cameraUploadsFailed = it.cameraUploadsFailed + 1,
                            cameraLastError = error,
                        )
                    }
                }
            }
            api.startTransport(scope, cachedSettings.serverBaseUrl)
            settingsRepository.settings.collect { s ->
                cachedSettings = s
            }
        }

        cameraSettingsJob = scope.launch {
            val initial = settingsRepository.settings.first()
            cachedSettings = initial
            if (initial.enableCamera) {
                startCameraStream(lifecycleOwner, initial)
            }
            settingsRepository.settings.collect { s ->
                cachedSettings = s
                if (!s.enableCamera) {
                    cameraController.stopStream()
                } else if (!cameraController.isStreaming) {
                    startCameraStream(lifecycleOwner, s)
                } else {
                    launch {
                        cameraController.updateConfig(CameraStreamConfig.fromSettings(s))
                    }
                }
            }
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
                        pushFrame()
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
                        // 姿态变化也立刻推（之前只改本地 UI，网页要等 GPS/定时器 → 像卡了几秒）
                        pushFrame()
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (_: Exception) {
                }
            },
        )

        uploadJob = scope.launch {
            while (isActive) {
                pushFrame()
                delay((1000L / cachedSettings.uploadHz.coerceAtLeast(1)))
                _snapshot.update {
                    it.copy(uploadChannel = if (api.isWsConnected()) "WebSocket" else "HTTP")
                }
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
        cameraSettingsJob?.cancel()
        cameraSettingsJob = null
        cameraController.stopStream()
        api.stopTransport()
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

    private fun pushFrame() {
        val g = gps ?: run {
            _snapshot.update {
                it.copy(lastUploadOk = false, lastUploadError = "等待定位…")
            }
            return
        }

        val minIntervalMs = (1000L / cachedSettings.uploadHz.coerceIn(1, 20)).coerceAtLeast(50L)
        val now = System.currentTimeMillis()
        if (now - lastPushMs < minIntervalMs) return
        lastPushMs = now

        val payload = TelemetryPayload(
            lat = g.lat,
            lon = g.lon,
            altMsl = g.altMsl,
            heading = if (g.speedMps > 0.5) g.bearing else imu.yawDeg,
            roll = if (cachedSettings.enableImu) imu.rollDeg else 0.0,
            pitch = if (cachedSettings.enableImu) imu.pitchDeg else 0.0,
            yaw = if (cachedSettings.enableImu) imu.yawDeg else null,
            airspeed = g.speedMps,
            groundspeed = g.speedMps,
            climbRate = computeClimbRate(g.altMsl),
            battery = readBatteryPct(),
            vehicleId = cachedSettings.vehicleId,
            t = TelemetryPayload.nowIso(),
            clientTs = now / 1000.0,
        )

        // 完全非阻塞：内部 WS 队列 + HTTP 合并发送
        api.publish(payload)
        _snapshot.update {
            it.copy(
                lastUploadOk = true,
                lastUploadError = null,
                uploadChannel = if (api.isWsConnected()) "WebSocket" else "HTTP",
                uploadsTotal = it.uploadsTotal + 1,
            )
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

    private suspend fun startCameraStream(lifecycleOwner: LifecycleOwner, settings: AppSettings) {
        val config = CameraStreamConfig.fromSettings(settings)
        cameraController.startStream(lifecycleOwner, config) { jpeg ->
            api.publishCamera(jpeg, settings.vehicleId)
        }
    }
}
