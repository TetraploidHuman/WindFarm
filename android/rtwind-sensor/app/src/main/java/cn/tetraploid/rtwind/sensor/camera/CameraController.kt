package cn.tetraploid.rtwind.sensor.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

@Singleton
class CameraController @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val bindMutex = Mutex()
    private var imageCapture: ImageCapture? = null

    private var streamConfig: CameraStreamConfig? = null
    private var onJpegFrame: ((ByteArray) -> Unit)? = null
    private var boundOwner: LifecycleOwner? = null
    private var previewView: PreviewView? = null
    private val lastFrameMs = AtomicLong(0L)

    val isStreaming: Boolean get() = onJpegFrame != null

    suspend fun startStream(
        lifecycleOwner: LifecycleOwner,
        config: CameraStreamConfig,
        onJpeg: (ByteArray) -> Unit,
    ) = withMain {
        streamConfig = config
        onJpegFrame = onJpeg
        boundOwner = lifecycleOwner
        rebindInternal()
    }

    suspend fun attachPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView) = withMain {
        boundOwner = lifecycleOwner
        this.previewView = previewView
        rebindInternal()
    }

    suspend fun detachPreview() = withMain {
        previewView = null
        if (isStreaming) rebindInternal()
    }

    /** Preview only (no upload) when service is stopped. */
    suspend fun bindPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView) = withMain {
        boundOwner = lifecycleOwner
        this.previewView = previewView
        if (!isStreaming) {
            val provider = getProvider()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            imageCapture = capture
            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                capture,
            )
        } else {
            rebindInternal()
        }
    }

    suspend fun updateConfig(config: CameraStreamConfig) = withMain {
        streamConfig = config
        if (isStreaming) rebindInternal()
    }

    private suspend fun rebindInternal() {
        val owner = boundOwner ?: return
        val config = streamConfig
        val callback = onJpegFrame
        val provider = getProvider()
        provider.unbindAll()

        val useCases = mutableListOf<androidx.camera.core.UseCase>()

        previewView?.let { view ->
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = view.surfaceProvider
            }
            useCases.add(preview)
        }

        if (config != null && callback != null) {
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            imageCapture = capture
            useCases.add(capture)

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(config.analysisWidth, config.analysisHeight))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            analysis.setAnalyzer(cameraExecutor) { proxy ->
                val now = System.currentTimeMillis()
                val minGap = config.minIntervalMs
                if (now - lastFrameMs.get() < minGap) {
                    proxy.close()
                    return@setAnalyzer
                }
                lastFrameMs.set(now)
                try {
                    val jpeg = CameraFrameEncoder.encode(
                        proxy,
                        config.targetHeight,
                        config.jpegQuality,
                    )
                    if (jpeg != null) callback(jpeg)
                } catch (_: Exception) {
                } finally {
                    proxy.close()
                }
            }
            useCases.add(analysis)
        }

        if (useCases.isEmpty()) return

        provider.bindToLifecycle(
            owner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            *useCases.toTypedArray(),
        )
    }

    suspend fun stopStream() = withMain {
        onJpegFrame = null
        streamConfig = null
        previewView = null
        unbindInternal()
    }

    suspend fun unbind() = withMain {
        unbindInternal()
    }

    private fun unbindInternal() {
        runCatching { getProviderBlocking().unbindAll() }
        imageCapture = null
        if (!isStreaming) {
            boundOwner = null
        }
    }

    private suspend fun getProvider(): ProcessCameraProvider = suspendCoroutine { cont ->
        ProcessCameraProvider.getInstance(context).also { future ->
            future.addListener({ cont.resume(future.get()) }, context.mainExecutor)
        }
    }

    private fun getProviderBlocking(): ProcessCameraProvider =
        ProcessCameraProvider.getInstance(context).get()

    private suspend inline fun <T> withMain(crossinline block: suspend () -> T): T =
        withContext(Dispatchers.Main.immediate) {
            bindMutex.withLock {
                try {
                    block()
                } catch (e: Exception) {
                    Log.e(TAG, "camera op failed", e)
                    throw e
                }
            }
        }

    companion object {
        private const val TAG = "CameraController"
    }
}
