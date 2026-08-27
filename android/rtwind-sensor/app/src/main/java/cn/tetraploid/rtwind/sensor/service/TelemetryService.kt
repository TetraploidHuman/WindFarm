package cn.tetraploid.rtwind.sensor.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
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
import javax.inject.Inject

@AndroidEntryPoint
class TelemetryService : LifecycleService() {

    @Inject lateinit var aggregator: TelemetryAggregator

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

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
        try {
            aggregator.start(scope)
            AppLog.i(TAG, "telemetry started enableCamera=$enableCamera")
        } catch (e: Exception) {
            AppLog.e(TAG, "aggregator.start failed", e)
            Log.e(TAG, "aggregator.start failed", e)
            stopSelf()
            return START_NOT_STICKY
        }
        return START_STICKY
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
        scope.cancel()
        super.onDestroy()
    }

    private fun createChannel() {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW,
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
        const val CHANNEL_ID = "rtwind_telemetry"
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
