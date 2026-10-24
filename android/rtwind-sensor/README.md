# RTWind 机载传感器 Android 应用

安装在无人机上的 Android 手机端配套应用，采集 **GPS、IMU、电量** 并通过 HTTP 推送到 [RTWind](/rtwind) 实时仪表盘；支持 **CameraX** 机载摄像头预览与拍照。

## 技术栈

| 类别 | 选型 |
|------|------|
| 语言 | Kotlin 2.1 |
| UI | Jetpack Compose + Material 3 |
| 架构 | Hilt DI + ViewModel + 前台 Service |
| 网络 | Ktor Client + kotlinx.serialization |
| 定位 | Google Play Services Fused Location |
| 传感器 | SensorManager（Rotation Vector / 加速度计） |
| 相机 | CameraX |
| 配置 | DataStore Preferences |
| SDK | minSdk 26 · targetSdk 35 |

## 与 RTWind 对接

1. 启动 RTWind 服务（示例）：
   ```bash
   PYTHONPATH=src .venv-fig/bin/python -m windfarm.cli serve-rtwind --host 127.0.0.1 --port 8877
   ```
2. 在浏览器打开仪表盘，将数据源切换为 **Live**。
3. 在手机「设置」页填写服务器地址，默认：
   ```
   http://tetraploid.cn:11024/rtwind
   ```
4. 授予定位、相机、通知权限，点击 **开始上传**。点击后应用会先自动执行约 **0.8 秒的姿态零点校准**，请让手机保持静止。

应用以 1–20 Hz（默认 10 Hz）向 `POST {baseUrl}/api/ingest` 发送 JSON：

```json
{
  "lat", "lon", "alt_msl", "heading", "roll", "pitch", "yaw",
  "airspeed", "groundspeed", "climb_rate", "battery", "vehicle_id", "t"
}
```

字段与后端 `IngestBody`（`src/windfarm/rtwind/app.py`）一致。

## 构建

需要 **JDK 17**（本机已安装到 `~/jdks/jdk-17.0.20.1+1`）与 **Android SDK 35**。

```bash
export JAVA_HOME=~/jdks/jdk-17.0.20.1+1
export PATH="$JAVA_HOME/bin:$PATH"
cd android/rtwind-sensor
./gradlew assembleDebug
```

APK 输出：`app/build/outputs/apk/debug/app-debug.apk`

## 使用建议（机载）

- 固定手机时让摄像头朝向前方，开启「保持屏幕常亮」或接入 USB 供电。
- 户外首次定位可能需要 30–60 秒；请等待 GPS 精度 &lt; 10 m 再起飞。
- 弱网环境下可适当降低上传频率以省电。
- 生产环境建议使用 HTTPS 并在 `network_security_config` 中限制明文 HTTP。

### 姿态零点校准

每次点击 **开始上传** 时，应用都会先采集约 **0.8 秒** 的 IMU 姿态并取平均，把那一刻的 Roll / Pitch / Yaw 记为本次会话的零点（即「零姿态」）；校准完成前不会推送任何遥测帧，之后整个飞行过程中上传的是**相对该零点**的姿态变化量。

- 只对三个姿态角（Roll / Pitch / Yaw）做零点化；GPS、海拔、地速/空速、爬升率、电量、摄像头均不受影响，按原始值上传。
- 校准状态显示在「实时遥测 → 姿态零点」：校准中请保持手机静止；完成后显示本次零点与样本数，晃动较大时会提示“晃动较大，建议静止重试”。
- 若读不到 IMU 或样本不足，校准会显示失败原因；此时遥测仍会继续上传，但姿态是**未零点化**的绝对姿态，可「停止」后重新「开始上传」重试。
- 零点**不持久化**：每次开始上传都会重新校准，停止上传后清零。

## 项目结构

```
app/src/main/java/cn/tetraploid/rtwind/sensor/
├── camera/          CameraX 预览与拍照
├── data/            数据模型、DataStore、Ktor API
├── di/              Hilt 模块
├── sensor/          GPS / IMU 采集与聚合上传
├── service/         前台 TelemetryService
└── ui/              Compose 界面
```

## 后续可扩展

- WebRTC / RTSP 视频流回传
- 本地 Room 缓存断网续传
- MAVLink 直连飞控
- 与 RTWind 信念场 / 风场反演 pipeline 联动
