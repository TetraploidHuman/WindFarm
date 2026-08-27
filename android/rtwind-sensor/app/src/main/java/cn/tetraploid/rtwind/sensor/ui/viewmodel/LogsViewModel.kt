package cn.tetraploid.rtwind.sensor.ui.viewmodel

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.lifecycle.ViewModel
import cn.tetraploid.rtwind.sensor.BuildConfig
import cn.tetraploid.rtwind.sensor.log.AppLog
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

data class LogsUiState(
    val text: String = "",
    val lineCount: Int = 0,
    val byteSize: Long = 0L,
    val copyMessage: String? = null,
)

@HiltViewModel
class LogsViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
) : ViewModel() {

    private val _state = MutableStateFlow(LogsUiState())
    val state: StateFlow<LogsUiState> = _state.asStateFlow()

    fun refresh() {
        val body = AppLog.readText()
        val header = buildString {
            append("RTWind Sensor v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})\n")
            append("---\n")
        }
        _state.value = LogsUiState(
            text = header + body,
            lineCount = AppLog.lineCount(),
            byteSize = AppLog.byteSize(),
            copyMessage = null,
        )
    }

    fun copyAll() {
        val text = _state.value.text.ifBlank { AppLog.readText() }
        val clip = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clip.setPrimaryClip(ClipData.newPlainText("rtwind_logs", text))
        _state.value = _state.value.copy(copyMessage = "已复制到剪贴板（${text.length} 字符）")
    }

    fun clear() {
        AppLog.clear()
        AppLog.i("Logs", "log cleared by user")
        refresh()
        _state.value = _state.value.copy(copyMessage = "日志已清空")
    }

    fun dismissMessage() {
        _state.value = _state.value.copy(copyMessage = null)
    }
}
