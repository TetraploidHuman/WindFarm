package cn.tetraploid.rtwind.sensor.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.MutablePreferences
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore("rtwind_settings")

@Singleton
class SettingsRepository @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private object Keys {
        val SERVER = stringPreferencesKey("server_base_url")
        val VEHICLE = stringPreferencesKey("vehicle_id")
        val HZ = intPreferencesKey("upload_hz")
        val CAMERA = booleanPreferencesKey("enable_camera")
        val IMU = booleanPreferencesKey("enable_imu")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            serverBaseUrl = prefs[Keys.SERVER] ?: AppSettings.DEFAULT_SERVER,
            vehicleId = prefs[Keys.VEHICLE] ?: "android-drone-1",
            uploadHz = (prefs[Keys.HZ] ?: 10).coerceIn(1, 20),
            enableCamera = prefs[Keys.CAMERA] ?: true,
            enableImu = prefs[Keys.IMU] ?: true,
        )
    }

    suspend fun updateServer(url: String) = edit { it[Keys.SERVER] = url.trim().trimEnd('/') }
    suspend fun updateVehicle(id: String) = edit { it[Keys.VEHICLE] = id.trim() }
    suspend fun updateHz(hz: Int) = edit { it[Keys.HZ] = hz.coerceIn(1, 20) }
    suspend fun updateCamera(enabled: Boolean) = edit { it[Keys.CAMERA] = enabled }
    suspend fun updateImu(enabled: Boolean) = edit { it[Keys.IMU] = enabled }

    private suspend fun edit(block: suspend (MutablePreferences) -> Unit) {
        context.dataStore.edit { block(it) }
    }
}
