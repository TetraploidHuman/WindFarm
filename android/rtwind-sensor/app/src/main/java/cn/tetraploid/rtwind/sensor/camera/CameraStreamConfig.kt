package cn.tetraploid.rtwind.sensor.camera

import cn.tetraploid.rtwind.sensor.data.AppSettings

data class CameraStreamConfig(
    val targetHeight: Int,
    val analysisWidth: Int,
    val analysisHeight: Int,
    val uploadHz: Int,
    val jpegQuality: Int,
) {
    val minIntervalMs: Long get() = (1000L / uploadHz.coerceIn(1, 20)).coerceAtLeast(33L)

    companion object {
        fun fromSettings(settings: AppSettings): CameraStreamConfig {
            val height = settings.cameraHeight.coerceIn(360, 720)
            val (aw, ah) = when {
                height <= 360 -> 640 to 360
                height <= 480 -> 640 to 480
                else -> 1280 to 720
            }
            return CameraStreamConfig(
                targetHeight = height,
                analysisWidth = aw,
                analysisHeight = ah,
                uploadHz = settings.cameraUploadHz.coerceIn(1, 20),
                jpegQuality = settings.cameraJpegQuality.coerceIn(28, 70),
            )
        }
    }
}
