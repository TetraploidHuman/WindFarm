package cn.tetraploid.rtwind.sensor.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.Instant

@Serializable
data class TelemetryPayload(
    val lat: Double? = null,
    val lon: Double? = null,
    @SerialName("alt_msl") val altMsl: Double? = null,
    val heading: Double? = null,
    val roll: Double = 0.0,
    val pitch: Double = 0.0,
    val yaw: Double? = null,
    val airspeed: Double = 0.0,
    val groundspeed: Double? = null,
    @SerialName("climb_rate") val climbRate: Double = 0.0,
    val battery: Double? = null,
    @SerialName("vehicle_id") val vehicleId: String = "live-1",
    val t: String? = null,
    @SerialName("alt_agl") val altAgl: Double? = null,
    @SerialName("client_ts") val clientTs: Double? = null,
) {
    companion object {
        fun nowIso(): String = Instant.now().toString()
    }
}

enum class MountNoseAxis(val id: String, val label: String) {
    /** Screen up; phone top edge (+Y) points forward (short edge ahead). */
    DEVICE_Y("y", "手机顶部朝前"),
    /** Screen up; phone long edge (+X) points forward — common belly mount. */
    DEVICE_X("x", "手机长边朝前"),
    ;

    companion object {
        fun fromId(id: String?): MountNoseAxis =
            entries.firstOrNull { it.id == id } ?: DEVICE_X
    }
}

data class AppSettings(
    val serverBaseUrl: String = DEFAULT_SERVER,
    val vehicleId: String = "android-drone-1",
    val uploadHz: Int = 10,
    val enableCamera: Boolean = true,
    val enableImu: Boolean = true,
    /** Screen-up mount: which device axis points toward aircraft nose. */
    val mountNoseAxis: MountNoseAxis = MountNoseAxis.DEVICE_X,
    /** Output height: 360 / 480 / 720 */
    val cameraHeight: Int = 480,
    /** Camera JPEG upload rate (1–20 Hz, default 15). */
    val cameraUploadHz: Int = 15,
    /** JPEG quality 28–70; lower = smaller files. */
    val cameraJpegQuality: Int = 38,
) {
    companion object {
        const val DEFAULT_SERVER = "http://tetraploid.cn:11024/rtwind"
    }
}

data class TelemetrySnapshot(
    val lat: Double? = null,
    val lon: Double? = null,
    val altMsl: Double? = null,
    val heading: Double? = null,
    val roll: Double = 0.0,
    val pitch: Double = 0.0,
    val yaw: Double = 0.0,
    val groundspeed: Double = 0.0,
    val climbRate: Double = 0.0,
    val battery: Double? = null,
    val gpsAccuracyM: Float? = null,
    val gpsProvider: String? = null,
    val locationDiag: String = "未开始定位",
    val lastUploadOk: Boolean? = null,
    val lastUploadError: String? = null,
    val uploadChannel: String = "—",
    val uploadsTotal: Long = 0,
    val uploadsFailed: Long = 0,
    val serviceRunning: Boolean = false,
    val cameraUploadsTotal: Long = 0,
    val cameraUploadsFailed: Long = 0,
    val cameraLastFrameKb: Int = 0,
    val cameraLastError: String? = null,
)
