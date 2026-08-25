package cn.tetraploid.rtwind.sensor.camera

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

@Singleton
class CameraController @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val mainExecutor: Executor = ContextCompat.getMainExecutor(context)
    private var imageCapture: ImageCapture? = null

    suspend fun bindPreview(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        useFrontCamera: Boolean = false,
    ) {
        val provider = getProvider()
        val selector = if (useFrontCamera) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageCapture = capture

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, selector, preview, capture)
    }

    fun unbind() {
        runCatching {
            getProviderBlocking().unbindAll()
        }
        imageCapture = null
    }

    fun takePhoto(outputDir: File, onSaved: (File) -> Unit, onError: (Throwable) -> Unit) {
        val capture = imageCapture ?: run {
            onError(IllegalStateException("相机未就绪"))
            return
        }
        val file = File(
            outputDir,
            "rtwind_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg",
        )
        val options = ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(options, mainExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) = onSaved(file)
            override fun onError(exception: ImageCaptureException) = onError(exception)
        })
    }

    private suspend fun getProvider(): ProcessCameraProvider = suspendCoroutine { cont ->
        ProcessCameraProvider.getInstance(context).also { future ->
            future.addListener({ cont.resume(future.get()) }, mainExecutor)
        }
    }

    private fun getProviderBlocking(): ProcessCameraProvider =
        ProcessCameraProvider.getInstance(context).get()
}
