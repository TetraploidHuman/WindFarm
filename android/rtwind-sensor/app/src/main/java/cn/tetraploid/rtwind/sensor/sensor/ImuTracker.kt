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

@Singleton
class ImuTracker @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnet = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    fun readings(): Flow<ImuReading> = callbackFlow {
        val rotMat = FloatArray(9)
        val orient = FloatArray(3)
        var last = ImuReading(0.0, 0.0, 0.0)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        SensorManager.getRotationMatrixFromVector(rotMat, event.values)
                        // Portrait, screen up, nose = top of phone (+Y), right wing = +X on screen.
                        val mapped = FloatArray(9)
                        val ok = SensorManager.remapCoordinateSystem(
                            rotMat,
                            SensorManager.AXIS_Y,
                            SensorManager.AXIS_MINUS_X,
                            mapped,
                        )
                        SensorManager.getOrientation(if (ok) mapped else rotMat, orient)
                        last = ImuReading(
                            rollDeg = Math.toDegrees(orient[2].toDouble()),
                            // Aviation: pitch + = nose up (after body-axis remap).
                            pitchDeg = Math.toDegrees(orient[1].toDouble()),
                            yawDeg = normalizeHeading(Math.toDegrees(orient[0].toDouble())),
                        )
                        trySend(last)
                    }
                    Sensor.TYPE_ACCELEROMETER -> if (rotation == null) {
                        // Portrait screen-up fallback (matches remap above).
                        val ax = event.values[0].toDouble()
                        val ay = event.values[1].toDouble()
                        val az = event.values[2].toDouble()
                        val pitch = Math.toDegrees(atan2(-ay, sqrt(ax * ax + az * az)))
                        val roll = Math.toDegrees(atan2(ax, az))
                        last = last.copy(rollDeg = roll, pitchDeg = pitch)
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

    private fun normalizeHeading(deg: Double): Double {
        var h = deg
        while (h < 0) h += 360.0
        while (h >= 360) h -= 360.0
        return h
    }
}
