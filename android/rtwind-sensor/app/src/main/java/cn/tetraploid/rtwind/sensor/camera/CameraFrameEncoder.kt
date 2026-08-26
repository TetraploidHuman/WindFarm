package cn.tetraploid.rtwind.sensor.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

/** Downscale + JPEG compress for live upload (~15 fps). */
object CameraFrameEncoder {

    fun encode(image: ImageProxy, targetHeight: Int, jpegQuality: Int): ByteArray? {
        val quality = jpegQuality.coerceIn(28, 80)
        val height = targetHeight.coerceIn(240, 720)
        val rotation = image.imageInfo.rotationDegrees

        val jpeg = when (image.format) {
            ImageFormat.JPEG -> {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                bytes
            }
            ImageFormat.YUV_420_888 -> {
                val nv21 = yuv420888ToNv21(image) ?: return null
                val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                val out = ByteArrayOutputStream(128 * 1024)
                // First pass uses lower quality; final scale pass applies target quality.
                val passQ = (quality * 0.75f).roundToInt().coerceIn(28, 55)
                yuv.compressToJpeg(Rect(0, 0, image.width, image.height), passQ, out)
                out.toByteArray()
            }
            else -> return null
        }
        return scaleJpeg(jpeg, height, quality, rotation)
    }

    private fun scaleJpeg(jpeg: ByteArray, targetHeight: Int, quality: Int, rotation: Int): ByteArray {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
        if (bounds.outHeight <= 0 || bounds.outWidth <= 0) return jpeg

        var sample = 1
        while (bounds.outHeight / sample > targetHeight * 2) sample *= 2

        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        var bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, opts) ?: return jpeg

        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated !== bitmap) bitmap.recycle()
            bitmap = rotated
        }

        val scaled = if (bitmap.height > targetHeight) {
            val w = (bitmap.width * targetHeight.toFloat() / bitmap.height).roundToInt().coerceAtLeast(1)
            Bitmap.createScaledBitmap(bitmap, w, targetHeight, true).also {
                if (it !== bitmap) bitmap.recycle()
            }
        } else {
            bitmap
        }

        val out = ByteArrayOutputStream(96 * 1024)
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, out)
        scaled.recycle()
        return out.toByteArray()
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray? {
        val w = image.width
        val h = image.height
        val ySize = w * h
        val uvSize = w * h / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        copyPlane(yPlane.buffer, yPlane.rowStride, yPlane.pixelStride, w, h, nv21, 0, 1)

        val chromaHeight = h / 2
        val chromaWidth = w / 2
        var offset = ySize
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vuIndex = row * vPlane.rowStride + col * vPlane.pixelStride
                nv21[offset++] = vPlane.buffer.get(vuIndex)
                nv21[offset++] = uPlane.buffer.get(row * uPlane.rowStride + col * uPlane.pixelStride)
            }
        }
        return nv21
    }

    private fun copyPlane(
        buffer: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        out: ByteArray,
        offset: Int,
        outPixelStride: Int,
    ) {
        var output = offset
        buffer.rewind()
        if (pixelStride == 1 && outPixelStride == 1) {
            var row = 0
            while (row < height) {
                buffer.position(row * rowStride)
                buffer.get(out, output, width)
                output += width
                row++
            }
        } else {
            for (row in 0 until height) {
                var input = row * rowStride
                for (col in 0 until width) {
                    out[output] = buffer.get(input)
                    output += outPixelStride
                    input += pixelStride
                }
            }
        }
    }
}
