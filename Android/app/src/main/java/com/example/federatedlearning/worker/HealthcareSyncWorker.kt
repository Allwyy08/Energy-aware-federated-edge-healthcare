package com.example.federatedlearning.worker

import android.content.Context
import android.util.Log
import androidx.room.Room
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.federatedlearning.data.AppDatabase
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class HealthcareSyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override async suspend fun doWork(): Result {
        val appContext = applicationContext
        val prefs = appContext.getSharedPreferences("fl_settings", Context.MODE_PRIVATE)
        val ip = prefs.getString("server_ip", "192.168.31.89") ?: "192.168.31.89"
        val port = prefs.getString("server_port", "5000") ?: "5000"
        val deviceId = prefs.getString("client_id", "android_client_01") ?: "android_client_01"

        val db = Room.databaseBuilder(appContext, AppDatabase::class.java, "patient_db")
            .fallbackToDestructiveMigration()
            .build()

        val pendingList = db.healthcareDao().getPendingScreenings()
        if (pendingList.isEmpty()) {
            Log.d("HealthcareSyncWorker", "No pending screening records to synchronize.")
            return Result.success()
        }

        Log.d("HealthcareSyncWorker", "Found ${pendingList.size} pending screenings to sync to http://$ip:$port")

        val baseUrl = "http://$ip:$port/"
        val retrofit = Retrofit.Builder()
            .baseUrl(baseUrl)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        val api = retrofit.create(FLNetworkService::class.java)

        val requestPayloads = pendingList.map { entity ->
            PatientScreeningRequest(
                screening_id = entity.screeningId,
                patient_id = entity.patientId,
                device_id = deviceId,
                timestamp = entity.timestamp,
                age = entity.age,
                sex = entity.sex,
                symptoms = entity.symptoms,
                medical_history = entity.medicalHistory,
                heart_rate = entity.heartRate,
                spo2 = entity.spo2,
                temperature = entity.temperature,
                systolic_bp = entity.systolicBp,
                diastolic_bp = entity.diastolicBp,
                respiratory_rate = entity.respiratoryRate,
                glucose = entity.glucose,
                bmi = entity.bmi,
                sync_status = "SYNCED"
            )
        }

        return try {
            val response = api.syncScreenings(requestPayloads).execute()
            if (response.isSuccessful && response.body()?.success == true) {
                pendingList.forEach { entity ->
                    db.healthcareDao().updateSyncStatus(entity.screeningId, "SYNCED")
                }
                Log.d("HealthcareSyncWorker", "Successfully synchronized ${pendingList.size} records!")
                Result.success()
            } else {
                Log.w("HealthcareSyncWorker", "Sync failed with status code ${response.code()}")
                Result.retry()
            }
        } catch (e: Exception) {
            Log.e("HealthcareSyncWorker", "Exception during healthcare sync: ${e.message}")
            Result.retry()
        }
    }
}
