package cn.tetraploid.rtwind.sensor.service

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import cn.tetraploid.rtwind.sensor.MainActivity
import cn.tetraploid.rtwind.sensor.log.AppLog
import cn.tetraploid.rtwind.sensor.R
import cn.tetraploid.rtwind.sensor.sensor.TelemetryAggregator
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import javax.inject.Inject

@AndroidEntryPoint
class TelemetryService : LifecycleService() {

    @Inject lateinit var aggregator: TelemetryAggregator

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent): IBinder? = super.onBind(intent)

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        val enableCamera = intent?.getBooleanExtra(EXTRA_ENABLE_CAMERA, true) ?: true
        val notification = buildNotification("正在采集并上传传感器数据")
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, resolveForegroundServiceTypes(enableCamera))
            } else {
                @Suppress("DEPRECATION")
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: SecurityException) {
            AppLog.e(TAG, "startForeground failed", e)
            Log.e(TAG, "startForeground failed (missing permission for FGS type?)", e)
            stopSelf()
            return START_NOT_STICKY
        } catch (e: Exception) {
            AppLog.e(TAG, "onStartCommand failed", e)
            Log.e(TAG, "onStartCommand failed", e)
            stopSelf()
            return START_NOT_STICKY
        }
        acquireLocks()
        try {
            aggregator.start(scope)
            AppLog.i(TAG, "telemetry started enableCamera=$enableCamera")
        } catch (e: Exception) {
            AppLog.e(TAG, "aggregator.start failed", e)
            Log.e(TAG, "aggregator.start failed", e)
            releaseLocks()
            stopSelf()
            return START_NOT_STICKY
        }
        scope.launch {
            if (!enableCamera || !hasCameraPermission()) return@launch
            runCatching { aggregator.bindCamera(this@TelemetryService) }
                .onFailure { AppLog.e(TAG, "bindCamera failed", it) }
        }
        return START_REDELIVER_INTENT
    }

    /** 息屏后保持 CPU 和 Wi‑Fi，否则 IMU / 上传协程会被挂起。 */
    @SuppressLint("WakelockTimeout")
    private fun acquireLocks() {
        if (wakeLock?.isHeld != true) {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "rtwind:telemetry").apply {
                setReferenceCounted(false)
                acquire()
            }
        }
        if (wifiLock?.isHeld != true) {
            runCatching {
                val wm = applicationContext.getSystemService(WIFI_SERVICE) as WifiManager
                val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    WifiManager.WIFI_MODE_FULL_LOW_LATENCY
                } else {
                    @Suppress("DEPRECATION")
                    WifiManager.WIFI_MODE_FULL_HIGH_PERF
                }
                wifiLock = wm.createWifiLock(mode, "rtwind:telemetry-wifi").apply {
                    setReferenceCounted(false)
                    acquire()
                }
            }.onFailure { AppLog.w(TAG, "wifi lock failed", it) }
        }
        AppLog.i(TAG, "wake/wifi locks acquired")
    }

    private fun releaseLocks() {
        runCatching { if (wakeLock?.isHeld == true) wakeLock?.release() }
        runCatching { if (wifiLock?.isHeld == true) wifiLock?.release() }
        wakeLock = null
        wifiLock = null
    }

    /** Android 14+ 声明 CAMERA 类型时必须已授予相机权限，否则直接 SecurityException 闪退。 */
    private fun resolveForegroundServiceTypes(enableCamera: Boolean): Int {
        var types = ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION or
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE &&
            enableCamera &&
            hasCameraPermission()
        ) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
        }
        return types
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        AppLog.i(TAG, "service destroyed")
        aggregator.stop()
        releaseLocks()
        scope.cancel()
        super.onDestroy()
    }

    private fun createChannel() {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_DEFAULT,
        ).apply {
            description = getString(R.string.notification_channel_desc)
        }
        mgr.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        val pending = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setContentIntent(pending)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val TAG = "TelemetryService"
        const val CHANNEL_ID = "rtwind_telemetry_v2"
        const val NOTIFICATION_ID = 1001
        const val EXTRA_ENABLE_CAMERA = "enable_camera"

        fun start(context: Context, enableCamera: Boolean = true) {
            val intent = Intent(context, TelemetryService::class.java).apply {
                putExtra(EXTRA_ENABLE_CAMERA, enableCamera)
            }
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, TelemetryService::class.java))
        }
    }
}
