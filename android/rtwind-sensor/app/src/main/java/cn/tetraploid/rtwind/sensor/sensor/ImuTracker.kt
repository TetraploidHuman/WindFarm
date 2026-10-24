package cn.tetraploid.rtwind.sensor.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withTimeoutOrNull
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class ImuReading(
    val rollDeg: Double,
    val pitchDeg: Double,
    val yawDeg: Double,
)

/** 本次「开始上传」期间的平均姿态零点；spreadDeg 是样本相对均值的最大偏差（晃动程度）。 */
data class AttitudeZero(
    val rollDeg: Double,
    val pitchDeg: Double,
    val yawDeg: Double,
    val samples: Int,
    val spreadDeg: Double,
)

/**
 * Fixed mount: screen up, phone top (+Y device) = nose (+X body), right wing = +Y body.
 * Pitch/roll from gravity (accelerometer); yaw from remapped rotation vector.
 */
@Singleton
class ImuTracker @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnet = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val deviceRot = FloatArray(9)
    private val bodyRot = FloatArray(9)
    private val orient = FloatArray(3)
    private var lastAccel = floatArrayOf(0f, 0f, 9.81f)

    private companion object {
        /** 样本太少时零点不可信，按校准失败处理；正常 800 ms 在 SENSOR_DELAY_GAME 下约几十个样本。 */
        private const val MIN_ZERO_SAMPLES = 5
    }

    @Volatile
    private var zero: AttitudeZero? = null

    fun clearZero() {
        zero = null
    }

    fun readings(): Flow<ImuReading> = rawReadings().map { applyZero(it) }

    private fun rawReadings(): Flow<ImuReading> = callbackFlow {
        var last = ImuReading(0.0, 0.0, 0.0)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        lastAccel = event.values.copyOf()
                        if (rotation == null) {
                            last = bodyAnglesFromAccel(event.values)
                            trySend(last)
                        }
                    }
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        SensorManager.getRotationMatrixFromVector(deviceRot, event.values)
                        if (!SensorManager.remapCoordinateSystem(
                                deviceRot,
                                SensorManager.AXIS_Y,
                                SensorManager.AXIS_X,
                                bodyRot,
                            )
                        ) {
                            System.arraycopy(deviceRot, 0, bodyRot, 0, 9)
                        }
                        SensorManager.getOrientation(bodyRot, orient)
                        val tilt = bodyAnglesFromAccel(lastAccel)
                        last = ImuReading(
                            rollDeg = tilt.rollDeg,
                            pitchDeg = tilt.pitchDeg,
                            yawDeg = normalizeHeading(Math.toDegrees(orient[0].toDouble())),
                        )
                        trySend(last)
                    }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }

        val sensors = listOfNotNull(rotation, accel, magnet)
        if (sensors.isEmpty()) {
            close(IllegalStateException("设备无 IMU 传感器"))
            return@callbackFlow
        }

        sensors.forEach { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_GAME) }
        awaitClose { sensorManager.unregisterListener(listener) }
    }

    /**
     * 采集 [durationMs] 的原始 IMU 读数并取平均，作为本次会话的姿态零点。
     * 只有 roll/pitch/yaw 会被零点化，其它遥测量不受影响。
     */
    suspend fun calibrateZero(durationMs: Long = 800L): AttitudeZero {
        val samples = ArrayList<ImuReading>(128)
        withTimeoutOrNull(durationMs) {
            rawReadings().collect { samples.add(it) }
        }
        check(samples.isNotEmpty()) { "IMU 无读数，无法零点校准" }
        check(samples.size >= MIN_ZERO_SAMPLES) { "IMU 样本不足（${samples.size} 个），无法零点校准" }
        val n = samples.size
        val meanRoll = samples.sumOf { it.rollDeg } / n
        val meanPitch = samples.sumOf { it.pitchDeg } / n
        val meanSin = samples.sumOf { sin(Math.toRadians(it.yawDeg)) } / n
        val meanCos = samples.sumOf { cos(Math.toRadians(it.yawDeg)) } / n
        val meanYaw = normalizeHeading(Math.toDegrees(atan2(meanSin, meanCos)))
        var spread = 0.0
        for (s in samples) {
            val d = maxOf(
                abs(s.rollDeg - meanRoll),
                abs(s.pitchDeg - meanPitch),
                abs(normalizeSigned(s.yawDeg - meanYaw)),
            )
            if (d > spread) spread = d
        }
        val z = AttitudeZero(
            rollDeg = meanRoll,
            pitchDeg = meanPitch,
            yawDeg = meanYaw,
            samples = n,
            spreadDeg = spread,
        )
        zero = z
        return z
    }

    private fun applyZero(r: ImuReading): ImuReading {
        val z = zero ?: return r
        return r.copy(
            rollDeg = normalizeSigned(r.rollDeg - z.rollDeg),
            pitchDeg = normalizeSigned(r.pitchDeg - z.pitchDeg),
            yawDeg = normalizeHeading(r.yawDeg - z.yawDeg),
        )
    }

    /**
     * Map device accel to body tilt (screen up, nose = device +Y).
     * Nose up → pitch +; right wing down → roll +.
     */
    private fun bodyAnglesFromAccel(values: FloatArray): ImuReading {
        val ax = values[0].toDouble()
        val ay = values[1].toDouble()
        val az = values[2].toDouble()
        val gForward = -ay
        val gRight = -ax
        val gDown = az
        val pitch = -Math.toDegrees(atan2(gForward, sqrt(gRight * gRight + gDown * gDown)))
        val roll = Math.toDegrees(atan2(gRight, gDown))
        return ImuReading(
            rollDeg = normalizeSigned(roll),
            pitchDeg = normalizeSigned(pitch),
            yawDeg = 0.0,
        )
    }

    private fun normalizeSigned(deg: Double): Double {
        var d = deg
        while (d > 180.0) d -= 360.0
        while (d < -180.0) d += 360.0
        return d
    }

    private fun normalizeHeading(deg: Double): Double {
        var h = deg
        while (h < 0) h += 360.0
        while (h >= 360) h -= 360.0
        return h
    }
}
