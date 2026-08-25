package cn.tetraploid.rtwind.sensor.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import cn.tetraploid.rtwind.sensor.camera.CameraController
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.data.RtwindApi
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import cn.tetraploid.rtwind.sensor.data.TelemetrySnapshot
import cn.tetraploid.rtwind.sensor.sensor.TelemetryAggregator
import cn.tetraploid.rtwind.sensor.service.TelemetryService
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import android.content.Context
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
        TelemetryService.start(context)
    }

    fun stopTelemetry() {
        TelemetryService.stop(context)
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
