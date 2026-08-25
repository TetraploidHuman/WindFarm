package cn.tetraploid.rtwind.sensor.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import cn.tetraploid.rtwind.sensor.data.AppSettings
import cn.tetraploid.rtwind.sensor.ui.viewmodel.SettingsViewModel

@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val settings by viewModel.settings.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("连接设置", style = MaterialTheme.typography.headlineSmall)
        settings?.let { s -> SettingsForm(s, viewModel) } ?: SettingsForm(AppSettings(), viewModel)
    }
}

@Composable
private fun SettingsForm(settings: AppSettings, viewModel: SettingsViewModel) {
    var server by remember(settings.serverBaseUrl) { mutableStateOf(settings.serverBaseUrl) }
    var vehicle by remember(settings.vehicleId) { mutableStateOf(settings.vehicleId) }
    var hz by remember(settings.uploadHz) { mutableStateOf(settings.uploadHz.toFloat()) }

    OutlinedTextField(
        value = server,
        onValueChange = {
            server = it
            viewModel.saveServer(it)
        },
        label = { Text("RTWind 服务器地址") },
        supportingText = { Text("例如 ${AppSettings.DEFAULT_SERVER}") },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
    )

    OutlinedTextField(
        value = vehicle,
        onValueChange = {
            vehicle = it
            viewModel.saveVehicle(it)
        },
        label = { Text("飞行器 ID (vehicle_id)") },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
    )

    Text("上传频率: ${hz.toInt()} Hz")
    Slider(
        value = hz,
        onValueChange = {
            hz = it
            viewModel.saveHz(it.toInt())
        },
        valueRange = 1f..20f,
        steps = 18,
        modifier = Modifier.fillMaxWidth(),
    )

    ToggleRow(
        label = "启用 IMU（Roll/Pitch/Yaw）",
        checked = settings.enableImu,
        onCheckedChange = viewModel::saveImu,
    )
    ToggleRow(
        label = "启用摄像头预览",
        checked = settings.enableCamera,
        onCheckedChange = viewModel::saveCamera,
    )

    Text(
        "使用前请在 RTWind 网页端将数据源切换为 Live，否则 ingest 数据不会显示在地图上。",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
    )
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    androidx.compose.foundation.layout.Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, modifier = Modifier.weight(1f).padding(end = 8.dp))
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
