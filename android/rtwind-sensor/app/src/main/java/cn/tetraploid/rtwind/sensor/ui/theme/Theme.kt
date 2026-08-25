package cn.tetraploid.rtwind.sensor.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val SkyBlue = Color(0xFF38BDF8)
private val DeepNavy = Color(0xFF0F172A)
private val Panel = Color(0xFF1E293B)

private val DarkColors = darkColorScheme(
    primary = SkyBlue,
    onPrimary = DeepNavy,
    background = DeepNavy,
    surface = Panel,
    onBackground = Color(0xFFE2E8F0),
    onSurface = Color(0xFFE2E8F0),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF0284C7),
    onPrimary = Color.White,
    background = Color(0xFFF8FAFC),
    surface = Color.White,
)

@Composable
fun RtwindSensorTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content,
    )
}
