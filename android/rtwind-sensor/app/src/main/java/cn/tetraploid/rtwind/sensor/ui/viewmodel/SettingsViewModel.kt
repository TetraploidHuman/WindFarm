package cn.tetraploid.rtwind.sensor.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val repository: SettingsRepository,
) : ViewModel() {

    val settings = repository.settings
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AppSettings())

    fun saveServer(url: String) = viewModelScope.launch { repository.updateServer(url) }
    fun saveVehicle(id: String) = viewModelScope.launch { repository.updateVehicle(id) }
    fun saveHz(hz: Int) = viewModelScope.launch { repository.updateHz(hz) }
    fun saveCamera(enabled: Boolean) = viewModelScope.launch { repository.updateCamera(enabled) }
    fun saveImu(enabled: Boolean) = viewModelScope.launch { repository.updateImu(enabled) }
    fun saveCameraHeight(height: Int) = viewModelScope.launch { repository.updateCameraHeight(height) }
    fun saveCameraUploadHz(hz: Int) = viewModelScope.launch { repository.updateCameraUploadHz(hz) }
    fun saveCameraJpegQuality(q: Int) = viewModelScope.launch { repository.updateCameraJpegQuality(q) }
}
