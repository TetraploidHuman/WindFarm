package cn.tetraploid.rtwind.sensor

import android.app.Application
import cn.tetraploid.rtwind.sensor.log.AppLog
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class RtwindSensorApp : Application() {
    override fun onCreate() {
        AppLog.init(this)
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            AppLog.logCrash(thread, error)
            defaultHandler?.uncaughtException(thread, error)
        }
        super.onCreate()
        AppLog.i("App", "started v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
    }
}
