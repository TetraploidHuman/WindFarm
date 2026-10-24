package cn.tetraploid.rtwind.sensor.sensor

import android.content.Context
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import androidx.lifecycle.LifecycleOwner
import cn.tetraploid.rtwind.sensor.camera.CameraController
import cn.tetraploid.rtwind.sensor.camera.CameraStreamConfig
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.data.RtwindApi
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import cn.tetraploid.rtwind.sensor.data.TelemetryPayload
import cn.tetraploid.rtwind.sensor.data.TelemetrySnapshot
import cn.tetraploid.rtwind.sensor.log.AppLog
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
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.Dispatchers
import java.util.Locale
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
    private companion object {
        private const val TAG = "Telemetry"
    }

    private val _snapshot = MutableStateFlow(TelemetrySnapshot())
    val snapshot: StateFlow<TelemetrySnapshot> = _snapshot.asStateFlow()

    private var gps: GpsReading? = null
    private var imu: ImuReading = ImuReading(0.0, 0.0, 0.0)
    private var prevAlt: Double? = null
    private var prevAltTimeMs: Long = 0L
    private var lastPushMs: Long = 0L

    /** 校准完成（成功或失败）前不推送任何帧，避免上传未零点化的姿态。 */
    @Volatile
    private var attitudeZeroReady = false

    private var sensorJobs: List<Job> = emptyList()
    private var uploadJob: Job? = null
    private var linkJob: Job? = null
    @Volatile
    private var cachedSettings: AppSettings = AppSettings()

    fun start(scope: CoroutineScope) {
        if (sensorJobs.isNotEmpty()) return
        attitudeZeroReady = false
        // 上一架次的姿态必须先作废，否则校准期间可能把旧姿态继续推给 RTWind。
        imu = ImuReading(0.0, 0.0, 0.0)
        _snapshot.update {
            it.copy(
                serviceRunning = true,
                lastUploadError = null,
                locationDiag = locationTracker.diagnose(),
                attitudeZeroStatus = "姿态零点校准中…（请保持静止）",
                roll = 0.0,
                pitch = 0.0,
                yaw = 0.0,
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
                // 停止上传会取消本协程：CancellationException 必须继续抛出，不能当成校准失败。
                val calibration: Result<AttitudeZero> = try {
                    Result.success(imuTracker.calibrateZero())
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Result.failure(e)
                }
                calibration
                    .onSuccess { z ->
                        attitudeZeroReady = true
                        _snapshot.update { it.copy(attitudeZeroStatus = formatZeroStatus(z)) }
                        AppLog.i(
                            TAG,
                            "attitude zero calibrated roll=${z.rollDeg} pitch=${z.pitchDeg} " +
                                "yaw=${z.yawDeg} samples=${z.samples} spread=${z.spreadDeg}",
                        )
                    }
                    .onFailure { e ->
                        attitudeZeroReady = true // 仍推送绝对姿态，绝不永久阻塞遥测
                        _snapshot.update {
                            it.copy(attitudeZeroStatus = "校准失败: ${e.message ?: "未知错误"}")
                        }
                        AppLog.w(TAG, "attitude zero calibration failed", e)
                    }
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

    /** 相机绑在前台 Service 上，息屏后 Activity STOP 也不会拆掉 ImageAnalysis。 */
    suspend fun bindCamera(owner: LifecycleOwner) {
        val settings = settingsRepository.settings.first()
        cachedSettings = settings
        if (!settings.enableCamera) {
            cameraController.stopStream()
            return
        }
        val config = CameraStreamConfig.fromSettings(settings)
        cameraController.startStream(owner, config) { jpeg ->
            api.publishCamera(jpeg, cachedSettings.vehicleId)
        }
    }

    fun stop() {
        sensorJobs.forEach { it.cancel() }
        sensorJobs = emptyList()
        uploadJob?.cancel()
        uploadJob = null
        linkJob?.cancel()
        linkJob = null
        runBlocking(Dispatchers.Main.immediate) {
            runCatching { cameraController.stopStream() }
        }
        api.stopTransport()
        attitudeZeroReady = false
        imuTracker.clearZero()
        _snapshot.update {
            it.copy(
                serviceRunning = false,
                lastUploadError = null,
                locationDiag = "已停止 | ${locationTracker.diagnose()}",
                attitudeZeroStatus = "未校准",
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
        // 姿态零点校准完成前不上传，避免把未零点化的姿态发到 RTWind。
        if (!attitudeZeroReady) return
        val minIntervalMs = (1000L / cachedSettings.uploadHz.coerceIn(1, 20)).coerceAtLeast(50L)
        val now = System.currentTimeMillis()
        if (now - lastPushMs < minIntervalMs) return
        lastPushMs = now

        val g = gps
        val payload = TelemetryPayload(
            lat = g?.lat,
            lon = g?.lon,
            altMsl = g?.altMsl,
            heading = when {
                g != null && g.speedMps > 0.5 -> g.bearing
                else -> imu.yawDeg
            },
            roll = if (cachedSettings.enableImu) imu.rollDeg else 0.0,
            pitch = if (cachedSettings.enableImu) imu.pitchDeg else 0.0,
            yaw = if (cachedSettings.enableImu) imu.yawDeg else null,
            airspeed = g?.speedMps ?: 0.0,
            groundspeed = g?.speedMps ?: 0.0,
            climbRate = g?.let { computeClimbRate(it.altMsl) } ?: 0.0,
            battery = readBatteryPct(),
            vehicleId = cachedSettings.vehicleId,
            t = TelemetryPayload.nowIso(),
            clientTs = now / 1000.0,
        )

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

    private fun formatZeroStatus(z: AttitudeZero): String {
        val base = String.format(
            Locale.US,
            "已校准 · R %.1f° · P %.1f° · Y %.1f° · %d 样本",
            z.rollDeg,
            z.pitchDeg,
            z.yawDeg,
            z.samples,
        )
        return if (z.spreadDeg > 5.0) "$base · 晃动较大，建议静止重试" else base
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
        val filter = IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED)
        val intent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(null, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            context.registerReceiver(null, filter)
        } ?: return null
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (level < 0 || scale <= 0) return null
        return level * 100.0 / scale
    }

}
