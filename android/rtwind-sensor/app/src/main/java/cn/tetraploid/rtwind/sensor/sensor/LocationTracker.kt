package cn.tetraploid.rtwind.sensor.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.Looper
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.Priority
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import javax.inject.Inject
import javax.inject.Singleton

data class GpsReading(
    val lat: Double,
    val lon: Double,
    val altMsl: Double,
    val speedMps: Double,
    val bearing: Double,
    val accuracyM: Float,
    val verticalAccuracyM: Float?,
)

@Singleton
class LocationTracker @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val fused = com.google.android.gms.location.LocationServices
        .getFusedLocationProviderClient(context)
    @SuppressLint("MissingPermission")
    fun readings(): Flow<GpsReading> = callbackFlow {
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 200L)
            .setMinUpdateIntervalMillis(100L)
            .setWaitForAccurateLocation(false)
            .build()

        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                result.lastLocation?.let { trySend(it.toReading()) }
            }
        }

        fused.requestLocationUpdates(request, callback, Looper.getMainLooper())
        fused.lastLocation.addOnSuccessListener { loc ->
            loc?.let { trySend(it.toReading()) }
        }

        awaitClose { fused.removeLocationUpdates(callback) }
    }

    private fun Location.toReading() = GpsReading(
        lat = latitude,
        lon = longitude,
        altMsl = altitude,
        speedMps = speed.toDouble().coerceAtLeast(0.0),
        bearing = bearing.toDouble(),
        accuracyM = accuracy,
        verticalAccuracyM = if (android.os.Build.VERSION.SDK_INT >= 26) verticalAccuracyMeters else null,
    )
}
