package cn.tetraploid.rtwind.sensor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import cn.tetraploid.rtwind.sensor.ui.RtwindSensorApp
import cn.tetraploid.rtwind.sensor.ui.theme.RtwindSensorTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RtwindSensorTheme {
                RtwindSensorApp()
            }
        }
    }
}
