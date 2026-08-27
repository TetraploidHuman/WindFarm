package cn.tetraploid.rtwind.sensor.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import cn.tetraploid.rtwind.sensor.data.TelemetrySnapshot
import cn.tetraploid.rtwind.sensor.ui.viewmodel.DashboardViewModel
import java.util.Locale
import kotlinx.coroutines.launch

@Composable
fun DashboardScreen(
    viewModel: DashboardViewModel = hiltViewModel(),
) {
    val context = LocalContext.current
    val snapshot by viewModel.snapshot.collectAsStateWithLifecycle()
    val settings by viewModel.settings.collectAsStateWithLifecycle()
    var pingMsg by remember { mutableStateOf<String?>(null) }
    var permissionsReady by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        val hasLocation = grants[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
            grants[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        val hasCamera = grants[Manifest.permission.CAMERA] == true ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        val needCamera = settings.enableCamera
        permissionsReady = hasLocation && (!needCamera || hasCamera)
        when {
            !hasLocation ->
                Toast.makeText(context, "需要定位权限才能上传遥测（请选「精确位置」）", Toast.LENGTH_LONG).show()
            needCamera && !hasCamera ->
                Toast.makeText(context, "已启用摄像头上传，需授予相机权限；或在设置里关闭摄像头", Toast.LENGTH_LONG).show()
            grants[Manifest.permission.ACCESS_FINE_LOCATION] != true ->
                Toast.makeText(context, "建议授予「精确位置」，否则定位可能很慢或不准", Toast.LENGTH_LONG).show()
        }
    }

    LaunchedEffect(settings.enableCamera) {
        val perms = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            add(Manifest.permission.ACCESS_COARSE_LOCATION)
            add(Manifest.permission.POST_NOTIFICATIONS)
            if (settings.enableCamera) add(Manifest.permission.CAMERA)
        }
        permissionLauncher.launch(perms.toTypedArray())
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("RTWind 机载传感器", style = MaterialTheme.typography.headlineSmall)
        Text(
            "将手机固定在无人机上，采集 GPS / IMU 并推送到 RTWind Live 仪表盘。",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
        )

        if (settings.enableCamera && permissionsReady) {
            CameraPreviewCard(viewModel, snapshot.serviceRunning)
        }

        TelemetryCard(snapshot)

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { viewModel.startTelemetry() },
                enabled = permissionsReady && !snapshot.serviceRunning,
                modifier = Modifier.weight(1f),
            ) {
                Text("开始上传")
            }
            OutlinedButton(
                onClick = { viewModel.stopTelemetry() },
                enabled = snapshot.serviceRunning,
                modifier = Modifier.weight(1f),
            ) {
                Text("停止")
            }
        }

        OutlinedButton(
            onClick = { viewModel.pingServer { pingMsg = it } },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("测试服务器连接")
        }
        pingMsg?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun CameraPreviewCard(viewModel: DashboardViewModel, serviceRunning: Boolean) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    var previewView by remember { mutableStateOf<PreviewView?>(null) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text("机载摄像头预览", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).also { previewView = it }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
            )
        }
    }

    LaunchedEffect(previewView, serviceRunning) {
        val view = previewView ?: return@LaunchedEffect
        runCatching {
            if (serviceRunning) {
                viewModel.cameraController.attachPreview(lifecycleOwner, view)
            } else {
                viewModel.cameraController.bindPreview(lifecycleOwner, view)
            }
        }
    }

    DisposableEffect(serviceRunning) {
        onDispose {
            if (serviceRunning) {
                scope.launch {
                    runCatching { viewModel.cameraController.detachPreview() }
                }
            } else {
                viewModel.cameraController.unbind()
            }
        }
    }
}

@Composable
private fun TelemetryCard(snapshot: TelemetrySnapshot) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("实时遥测", style = MaterialTheme.typography.titleMedium)
            MetricRow("状态", if (snapshot.serviceRunning) "上传中" else "已停止")
            MetricRow("定位诊断", snapshot.locationDiag)
            MetricRow("定位来源", snapshot.gpsProvider ?: "—")
            MetricRow("纬度", snapshot.lat?.let { fmt(it, 6) } ?: "—")
            MetricRow("经度", snapshot.lon?.let { fmt(it, 6) } ?: "—")
            MetricRow("海拔 (MSL)", snapshot.altMsl?.let { fmt(it, 1) + " m" } ?: "—")
            MetricRow("航向", snapshot.heading?.let { fmt(it, 1) + "°" } ?: "—")
            MetricRow("Roll / Pitch", "${fmt(snapshot.roll, 1)}° / ${fmt(snapshot.pitch, 1)}°")
            MetricRow("地速", fmt(snapshot.groundspeed, 2) + " m/s")
            MetricRow("爬升率", fmt(snapshot.climbRate, 2) + " m/s")
            MetricRow("电量", snapshot.battery?.let { fmt(it, 0) + "%" } ?: "—")
            MetricRow("GPS 精度", snapshot.gpsAccuracyM?.let { fmt(it.toDouble(), 1) + " m" } ?: "—")
            MetricRow("上传通道", snapshot.uploadChannel)
            MetricRow(
                "上传",
                "${snapshot.uploadsTotal} 成功 / ${snapshot.uploadsFailed} 失败" +
                    (snapshot.lastUploadError?.let { " · $it" } ?: ""),
            )
            if (snapshot.cameraUploadsTotal > 0 || snapshot.cameraLastFrameKb > 0) {
                MetricRow(
                    "摄像头",
                    "${snapshot.cameraUploadsTotal} 帧 · 最近 ${snapshot.cameraLastFrameKb} KB" +
                        (snapshot.cameraLastError?.let { " · $it" } ?: ""),
                )
            }
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
            label,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            modifier = Modifier.padding(end = 8.dp),
        )
        Text(
            value,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1f, fill = false),
        )
    }
}

private fun fmt(v: Double, digits: Int): String = String.format(Locale.US, "%.${digits}f", v)
