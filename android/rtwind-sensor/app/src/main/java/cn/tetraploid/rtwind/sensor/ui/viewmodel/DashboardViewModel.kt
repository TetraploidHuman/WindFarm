package cn.tetraploid.rtwind.sensor.ui.viewmodel

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import cn.tetraploid.rtwind.sensor.camera.CameraController
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.data.RtwindApi
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import cn.tetraploid.rtwind.sensor.data.TelemetrySnapshot
import cn.tetraploid.rtwind.sensor.sensor.TelemetryAggregator
import cn.tetraploid.rtwind.sensor.log.AppLog
import cn.tetraploid.rtwind.sensor.service.TelemetryService
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class DashboardViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    aggregator: TelemetryAggregator,
    private val settingsRepository: SettingsRepository,
    private val api: RtwindApi,
    val cameraController: CameraController,
) : ViewModel() {

    val snapshot: StateFlow<TelemetrySnapshot> = aggregator.snapshot
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), TelemetrySnapshot())

    val settings = settingsRepository.settings
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AppSettings())

    fun startTelemetry() {
        viewModelScope.launch {
            AppLog.i("Dashboard", "startTelemetry enableCamera=${settings.value.enableCamera}")
            runCatching { cameraController.unbind() }
                .onFailure { AppLog.w("Dashboard", "unbind before start failed", it) }
            runCatching {
                TelemetryService.start(context, settings.value.enableCamera)
            }.onFailure { AppLog.e("Dashboard", "startTelemetry failed", it) }
            promptIgnoreBatteryOptimizations()
        }
    }

    /** 国产系统息屏后会冻结未加入白名单的进程，即使前台服务也一样。 */
    private fun promptIgnoreBatteryOptimizations() {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        if (pm.isIgnoringBatteryOptimizations(context.packageName)) return
        val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
            data = Uri.parse("package:${context.packageName}")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        runCatching { context.startActivity(intent) }
            .onFailure { AppLog.w("Dashboard", "battery optimization prompt failed", it) }
    }

    fun stopTelemetry() {
        viewModelScope.launch {
            AppLog.i("Dashboard", "stopTelemetry")
            runCatching { cameraController.stopStream() }
                .onFailure { AppLog.w("Dashboard", "stopStream failed", it) }
            TelemetryService.stop(context)
        }
    }

    suspend fun attachCameraPreview(previewView: PreviewView) {
        if (!settings.value.enableCamera) return
        cameraController.attachPreview(previewView)
    }

    suspend fun bindCameraPreviewOnly(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        if (settings.value.enableCamera) {
            cameraController.bindPreview(lifecycleOwner, previewView)
        }
    }

    fun setKeepScreenOn(enabled: Boolean) {
        viewModelScope.launch { settingsRepository.updateKeepScreenOn(enabled) }
    }

    fun pingServer(onResult: (String) -> Unit) {
        viewModelScope.launch {
            val base = settings.value?.serverBaseUrl ?: return@launch
            api.health(base)
                .onSuccess { onResult("服务器正常: $it") }
                .onFailure { onResult("连接失败: ${it.message}") }
        }
    }
}
