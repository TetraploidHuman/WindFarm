package cn.tetraploid.rtwind.sensor.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.util.Log
import com.google.android.gms.common.ConnectionResult
import com.google.android.gms.common.GoogleApiAvailability
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
    val provider: String = "",
)

@Singleton
class LocationTracker @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val locationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    @SuppressLint("MissingPermission")
    fun readings(): Flow<GpsReading> = callbackFlow {
        fun emit(loc: Location) {
            trySend(loc.toReading())
        }

        // 1) Android 系统定位（国内手机主力：GPS + 网络）
        val sysListener = object : LocationListener {
            override fun onLocationChanged(location: Location) = emit(location)
            @Deprecated("Deprecated in Java")
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
            override fun onProviderEnabled(provider: String) = Unit
            override fun onProviderDisabled(provider: String) = Unit
        }

        val providers = buildList {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                add(LocationManager.GPS_PROVIDER)
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                add(LocationManager.NETWORK_PROVIDER)
            }
            if (android.os.Build.VERSION.SDK_INT >= 31 &&
                locationManager.isProviderEnabled(LocationManager.FUSED_PROVIDER)
            ) {
                add(LocationManager.FUSED_PROVIDER)
            }
        }

        if (providers.isEmpty()) {
            Log.w(TAG, "No location providers enabled")
            try {
                // 仍尝试注册 GPS，部分机型 isProviderEnabled 不准
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    200L,
                    0f,
                    sysListener,
                    Looper.getMainLooper(),
                )
            } catch (e: Exception) {
                Log.e(TAG, "GPS register failed", e)
            }
            try {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    500L,
                    0f,
                    sysListener,
                    Looper.getMainLooper(),
                )
            } catch (e: Exception) {
                Log.e(TAG, "NETWORK register failed", e)
            }
        } else {
            providers.forEach { provider ->
                runCatching {
                    locationManager.getLastKnownLocation(provider)?.let(::emit)
                    locationManager.requestLocationUpdates(
                        provider,
                        200L,
                        0f,
                        sysListener,
                        Looper.getMainLooper(),
                    )
                    Log.i(TAG, "Listening on $provider")
                }.onFailure { Log.e(TAG, "Failed $provider", it) }
            }
        }

        // 2) Google Fused（有 GMS 时作为补充）
        var fusedCallback: LocationCallback? = null
        val gmsOk = GoogleApiAvailability.getInstance()
            .isGooglePlayServicesAvailable(context) == ConnectionResult.SUCCESS
        if (gmsOk) {
            runCatching {
                val fused = com.google.android.gms.location.LocationServices
                    .getFusedLocationProviderClient(context)
                val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 200L)
                    .setMinUpdateIntervalMillis(100L)
                    .setWaitForAccurateLocation(false)
                    .build()
                val cb = object : LocationCallback() {
                    override fun onLocationResult(result: LocationResult) {
                        result.lastLocation?.let(::emit)
                    }
                }
                fusedCallback = cb
                fused.requestLocationUpdates(request, cb, Looper.getMainLooper())
                    .addOnFailureListener { Log.w(TAG, "Fused updates failed", it) }
                fused.lastLocation
                    .addOnSuccessListener { loc -> loc?.let(::emit) }
                    .addOnFailureListener { Log.w(TAG, "Fused lastLocation failed", it) }
            }.onFailure { Log.w(TAG, "GMS fused unavailable", it) }
        } else {
            Log.i(TAG, "Google Play Services unavailable, using system LocationManager only")
        }

        awaitClose {
            runCatching { locationManager.removeUpdates(sysListener) }
            fusedCallback?.let { cb ->
                runCatching {
                    com.google.android.gms.location.LocationServices
                        .getFusedLocationProviderClient(context)
                        .removeLocationUpdates(cb)
                }
            }
        }
    }

    private fun Location.toReading() = GpsReading(
        lat = latitude,
        lon = longitude,
        altMsl = altitude,
        speedMps = speed.toDouble().coerceAtLeast(0.0),
        bearing = bearing.toDouble(),
        accuracyM = accuracy,
        verticalAccuracyM = if (android.os.Build.VERSION.SDK_INT >= 26) verticalAccuracyMeters else null,
        provider = provider ?: "",
    )

    companion object {
        private const val TAG = "RtwindLocation"
    }
}
