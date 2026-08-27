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
 * Mount: screen up, phone top (+Y device) = nose (+X body), right wing = +Y body.
 * Aviation angles: roll = bank (right wing down +), pitch = nose up +, yaw = heading.
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

    fun readings(): Flow<ImuReading> = callbackFlow {
        var last = ImuReading(0.0, 0.0, 0.0)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
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
                        // orient[1] ≈ roll (about nose), orient[2] ≈ pitch (about wing).
                        // Screen-up level has orient[2]≈±π; offset so level screen-up → pitch 0.
                        last = ImuReading(
                            rollDeg = normalizeSigned(Math.toDegrees(orient[1].toDouble())),
                            pitchDeg = normalizeSigned(
                                -Math.toDegrees(orient[2].toDouble()) + 180.0,
                            ),
                            yawDeg = normalizeHeading(Math.toDegrees(orient[0].toDouble())),
                        )
                        trySend(last)
                    }
                    Sensor.TYPE_ACCELEROMETER -> if (rotation == null) {
                        last = bodyAnglesFromAccel(event.values)
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

    /** Gravity tilt in body frame (screen up, nose = device +Y). */
    private fun bodyAnglesFromAccel(values: FloatArray): ImuReading {
        val ax = values[0].toDouble()
        val ay = values[1].toDouble()
        val az = values[2].toDouble()
        val gForward = -ay
        val gRight = -ax
        val gDown = az
        val pitch = Math.toDegrees(atan2(gForward, sqrt(gRight * gRight + gDown * gDown)))
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
