package cn.tetraploid.rtwind.sensor.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import cn.tetraploid.rtwind.sensor.ui.viewmodel.LogsViewModel
import kotlin.math.roundToInt

@Composable
fun LogsScreen(
    viewModel: LogsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        viewModel.refresh()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("运行日志", style = MaterialTheme.typography.headlineSmall)
        Text(
            "闪退后重新打开应用，在此复制日志发给开发者排查。",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
        )
        Text(
            "${state.lineCount} 行 · ${formatKb(state.byteSize)} KB",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Button(
                onClick = { viewModel.copyAll() },
                modifier = Modifier.weight(1f),
            ) {
                Text("复制全部")
            }
            OutlinedButton(
                onClick = { viewModel.refresh() },
                modifier = Modifier.weight(1f),
            ) {
                Text("刷新")
            }
            OutlinedButton(
                onClick = { viewModel.clear() },
                modifier = Modifier.weight(1f),
            ) {
                Text("清空")
            }
        }

        state.copyMessage?.let { msg ->
            Text(
                msg,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
            )
        }

        SelectionContainer(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f, fill = true)
                .verticalScroll(rememberScrollState()),
        ) {
            Text(
                text = state.text.ifBlank { "（暂无日志）" },
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

private fun formatKb(bytes: Long): String {
    if (bytes <= 0) return "0"
    return (bytes / 1024.0).roundToInt().toString()
}
