package cn.tetraploid.rtwind.sensor.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import cn.tetraploid.rtwind.sensor.data.MountNoseAxis
import cn.tetraploid.rtwind.sensor.data.SettingsRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.atan2
import kotlin.math.sqrt

data class ImuReading(
    val rollDeg: Double,
    val pitchDeg: Double,
    val yawDeg: Double,
)

/**
 * Screen-up belly mount. Pitch/roll from gravity; yaw from rotation vector.
 * Nose axis configurable: phone long edge (+X) or top edge (+Y).
 */
@Singleton
class ImuTracker @Inject constructor(
    @ApplicationContext context: Context,
    private val settingsRepository: SettingsRepository,
) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnet = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val deviceRot = FloatArray(9)
    private val bodyRot = FloatArray(9)
    private val orient = FloatArray(3)
    private var lastAccel = floatArrayOf(0f, 0f, 9.81f)
    @Volatile private var mountNoseAxis: MountNoseAxis = MountNoseAxis.DEVICE_X

    fun readings(): Flow<ImuReading> = callbackFlow {
        mountNoseAxis = runBlocking { settingsRepository.settings.first().mountNoseAxis }

        var last = ImuReading(0.0, 0.0, 0.0)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        lastAccel = event.values.copyOf()
                        if (rotation == null) {
                            last = bodyAnglesFromAccel(event.values, mountNoseAxis)
                            trySend(last)
                        }
                    }
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        SensorManager.getRotationMatrixFromVector(deviceRot, event.values)
                        val remapped = remapForMount(mountNoseAxis)
                        if (!remapped) {
                            System.arraycopy(deviceRot, 0, bodyRot, 0, 9)
                        }
                        SensorManager.getOrientation(bodyRot, orient)
                        val tilt = bodyAnglesFromAccel(lastAccel, mountNoseAxis)
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

    /** Update mount axis when user changes settings (called from aggregator). */
    fun updateMountAxis(axis: MountNoseAxis) {
        mountNoseAxis = axis
    }

    private fun remapForMount(axis: MountNoseAxis): Boolean = when (axis) {
        MountNoseAxis.DEVICE_Y -> SensorManager.remapCoordinateSystem(
            deviceRot,
            SensorManager.AXIS_Y,
            SensorManager.AXIS_X,
            bodyRot,
        )
        MountNoseAxis.DEVICE_X -> SensorManager.remapCoordinateSystem(
            deviceRot,
            SensorManager.AXIS_X,
            SensorManager.AXIS_Y,
            bodyRot,
        )
    }

    /**
     * Gravity tilt; nose-up → pitch positive, right-wing-down → roll positive.
     * Uses accelerometer reaction vector (screen up, belly mount).
     */
    private fun bodyAnglesFromAccel(values: FloatArray, axis: MountNoseAxis): ImuReading {
        val ax = values[0].toDouble()
        val ay = values[1].toDouble()
        val az = values[2].toDouble()
        return when (axis) {
            MountNoseAxis.DEVICE_Y -> {
                // Nose = device +Y (portrait top forward)
                val pitch = Math.toDegrees(atan2(-ay, sqrt(ax * ax + az * az)))
                val roll = Math.toDegrees(atan2(-ax, az))
                ImuReading(normalizeSigned(roll), normalizeSigned(pitch), 0.0)
            }
            MountNoseAxis.DEVICE_X -> {
                // Nose = device +X (long edge forward) — typical flat belly mount
                val pitch = Math.toDegrees(atan2(ax, sqrt(ay * ay + az * az)))
                val roll = Math.toDegrees(atan2(-ay, az))
                ImuReading(normalizeSigned(roll), normalizeSigned(pitch), 0.0)
            }
        }
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
