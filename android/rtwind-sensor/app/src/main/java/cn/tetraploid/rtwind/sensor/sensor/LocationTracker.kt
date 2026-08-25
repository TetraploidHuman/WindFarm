package cn.tetraploid.rtwind.sensor.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Criteria
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.CancellationSignal
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
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
import java.util.concurrent.Executors
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
    private val executor = Executors.newSingleThreadExecutor()

    fun diagnose(): String {
        val fine = ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.ACCESS_FINE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.ACCESS_COARSE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        val enabled = runCatching { locationManager.isLocationEnabled }.getOrDefault(false)
        val gps = runCatching { locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) }
            .getOrDefault(false)
        val net = runCatching { locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) }
            .getOrDefault(false)
        val providers = locationManager.getProviders(true).joinToString(",")
        return "权限精确=$fine 粗略=$coarse | 系统定位=$enabled GPS=$gps 网络=$net | 可用:[$providers]"
    }

    @SuppressLint("MissingPermission")
    fun readings(): Flow<GpsReading> = callbackFlow {
        val diag = diagnose()
        Log.i(TAG, diag)
        if (!hasLocationPermission()) {
            close(SecurityException("缺少定位权限（请授予精确位置）"))
            return@callbackFlow
        }

        fun emit(loc: Location?) {
            if (loc == null) return
            // 忽略明显无效坐标
            if (loc.latitude == 0.0 && loc.longitude == 0.0) return
            trySend(loc.toReading())
        }

        val listener = object : LocationListener {
            override fun onLocationChanged(location: Location) = emit(location)
            @Deprecated("Deprecated in Java")
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
            override fun onProviderEnabled(provider: String) = Unit
            override fun onProviderDisabled(provider: String) = Unit
        }

        // 先吐出缓存位置（地图软件刚定位过时往往立刻有值）
        listOf(
            LocationManager.GPS_PROVIDER,
            LocationManager.NETWORK_PROVIDER,
            LocationManager.PASSIVE_PROVIDER,
        ).forEach { provider ->
            runCatching { locationManager.getLastKnownLocation(provider) }.getOrNull()?.let(::emit)
        }

        // 持续监听：网络（快）+ GPS（准）+ 被动（蹭其它 App 的定位）
        val watch = linkedSetOf(
            LocationManager.NETWORK_PROVIDER,
            LocationManager.GPS_PROVIDER,
            LocationManager.PASSIVE_PROVIDER,
        )
        runCatching {
            locationManager.getBestProvider(
                Criteria().apply {
                    accuracy = Criteria.ACCURACY_FINE
                    isCostAllowed = true
                },
                true,
            )
        }.getOrNull()?.let { watch.add(it) }

        if (android.os.Build.VERSION.SDK_INT >= 31) {
            runCatching {
                if (locationManager.isProviderEnabled(LocationManager.FUSED_PROVIDER)) {
                    watch.add(LocationManager.FUSED_PROVIDER)
                }
            }
        }

        watch.forEach { provider ->
            runCatching {
                locationManager.requestLocationUpdates(
                    provider,
                    100L,
                    0f,
                    listener,
                    Looper.getMainLooper(),
                )
                Log.i(TAG, "requestLocationUpdates: $provider")
            }.onFailure { Log.w(TAG, "listen $provider failed: ${it.message}") }

            // 主动拉一次当前定位
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                runCatching {
                    locationManager.getCurrentLocation(
                        provider,
                        CancellationSignal(),
                        executor,
                    ) { loc -> emit(loc) }
                }.onFailure { Log.w(TAG, "getCurrentLocation $provider: ${it.message}") }
            } else {
                @Suppress("DEPRECATION")
                runCatching {
                    locationManager.requestSingleUpdate(provider, listener, Looper.getMainLooper())
                }
            }
        }

        // GMS Fused 作补充（有 Play 服务时）
        var fusedCallback: LocationCallback? = null
        val gmsOk = GoogleApiAvailability.getInstance()
            .isGooglePlayServicesAvailable(context) == ConnectionResult.SUCCESS
        if (gmsOk) {
            runCatching {
                val fused = com.google.android.gms.location.LocationServices
                    .getFusedLocationProviderClient(context)
                val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 500L)
                    .setMinUpdateIntervalMillis(200L)
                    .setWaitForAccurateLocation(false)
                    .build()
                val cb = object : LocationCallback() {
                    override fun onLocationResult(result: LocationResult) {
                        emit(result.lastLocation)
                    }
                }
                fusedCallback = cb
                fused.requestLocationUpdates(request, cb, Looper.getMainLooper())
                fused.lastLocation.addOnSuccessListener { emit(it) }
                fused.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                    .addOnSuccessListener { emit(it) }
            }.onFailure { Log.w(TAG, "GMS fused failed", it) }
        }

        awaitClose {
            runCatching { locationManager.removeUpdates(listener) }
            fusedCallback?.let { cb ->
                runCatching {
                    com.google.android.gms.location.LocationServices
                        .getFusedLocationProviderClient(context)
                        .removeLocationUpdates(cb)
                }
            }
        }
    }

    private fun hasLocationPermission(): Boolean {
        val fine = ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.ACCESS_FINE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.ACCESS_COARSE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        return fine || coarse
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
