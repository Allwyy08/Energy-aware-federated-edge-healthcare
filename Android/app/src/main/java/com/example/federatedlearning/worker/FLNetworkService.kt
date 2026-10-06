package com.example.federatedlearning.worker

import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

// Request and Response schemas
data class RegisterRequest(
    val cid: String,
    val name: String,
    val location: String,
    val battery: Float,
    val reliability: Float,
    val packet_drop: Float,
    val bandwidth: Float
)

data class RegisterResponse(
    val success: Boolean,
    val message: String
)

data class PingResponse(
    val status: String,
    val timestamp: Double,
    val server: String,
    val network_profile: String?
)

data class DownloadTestResponse(
    val size_kb: Int,
    val timestamp: Double,
    val payload: String
)

data class UploadTestResponse(
    val received_bytes: Long,
    val duration_sec: Float,
    val upload_mbps: Float
)

data class TelemetryRequest(
    val device_id: String,
    val battery: Float,
    val charging: Boolean,
    val latency_ms: Float,
    val download_mbps: Float,
    val upload_mbps: Float,
    val cpu_usage: Float,
    val memory_usage: Float,
    val network_type: String
)

data class GenericResponse(
    val success: Boolean,
    val message: String
)

data class DQNActionRequest(
    val cid: String,
    val battery: Float,
    val bandwidth: Float,
    val reliability: Float,
    val cpu: Float,
    val last_accuracy: Float
)

data class DQNActionResponse(
    val action: Int,
    val epochs: Int,
    val compress: Int,
    val delay_upload: Int
)

data class StatsUploadRequest(
    val cid: String,
    val battery: Float,
    val bandwidth: Float,
    val reliability: Float,
    val cpu: Float,
    val ram: Float,
    val epochs: Int,
    val compressed: Int,
    val uploaded_kb: Float,
    val energy_drained: Float,
    val accuracy_gain: Float,
    val state: List<Float>,
    val action: Int
)

data class StatsUploadResponse(
    val success: Boolean,
    val reward: Float
)

data class ModelWeightsResponse(
    val version: Long,
    val parameters: Map<String, List<List<Double>>>
)

// HEALTHCARE SCHEMAS
data class PatientScreeningRequest(
    val screening_id: String? = null,
    val patient_id: String = "PAT-001",
    val device_id: String = "android_client_01",
    val timestamp: String? = null,
    val age: Int = 35,
    val sex: String = "M",
    val symptoms: String = "",
    val medical_history: String = "",
    val heart_rate: Float? = null,
    val spo2: Float? = null,
    val temperature: Float? = null,
    val systolic_bp: Float? = null,
    val diastolic_bp: Float? = null,
    val respiratory_rate: Float? = null,
    val glucose: Float? = null,
    val bmi: Float? = null,
    val sync_status: String = "SYNCED"
)

data class RiskEvaluation(
    val risk_level: String,
    val risk_score: Float,
    val confidence: Float,
    val key_factors: List<String>,
    val recommended_action: String,
    val disclaimer: String
)

data class ScreeningResponse(
    val success: Boolean,
    val screening_id: String,
    val risk_evaluation: RiskEvaluation?,
    val message: String
)

data class SyncResponse(
    val success: Boolean,
    val synced_count: Int,
    val message: String
)

data class EmergencyAlertRequest(
    val alert_id: String? = null,
    val screening_id: String? = null,
    val device_id: String = "android_client_01",
    val timestamp: String? = null,
    val risk_level: String = "CRITICAL",
    val status: String = "AWAITING_ACTION"
)

data class HealthcareStatsResponse(
    val success: Boolean,
    val statistics: Map<String, Int>
)

interface FLNetworkService {

    @GET("/api/ping")
    fun ping(): Call<PingResponse>

    @GET("/api/network-test/download")
    fun downloadTest(@Query("size_kb") sizeKb: Int = 256): Call<DownloadTestResponse>

    @POST("/api/network-test/upload")
    fun uploadTest(@Body body: RequestBody): Call<UploadTestResponse>

    @POST("/api/device/telemetry")
    fun sendTelemetry(@Body request: TelemetryRequest): Call<GenericResponse>

    @POST("/api/register")
    fun registerClient(@Body request: RegisterRequest): Call<RegisterResponse>

    @POST("/api/dqn-action")
    fun queryDQNAction(@Body request: DQNActionRequest): Call<DQNActionResponse>

    @POST("/api/upload-stats")
    fun uploadStats(@Body request: StatsUploadRequest): Call<StatsUploadResponse>

    @GET("/api/model")
    fun getGlobalModel(): Call<ModelWeightsResponse>

    // HEALTHCARE REST ENDPOINTS
    @POST("/api/healthcare/screening")
    fun sendScreening(@Body request: PatientScreeningRequest): Call<ScreeningResponse>

    @POST("/api/healthcare/sync")
    fun syncScreenings(@Body requests: List<PatientScreeningRequest>): Call<SyncResponse>

    @GET("/api/healthcare/screening/{id}")
    fun getScreening(@Path("id") id: String): Call<ScreeningResponse>

    @GET("/api/healthcare/device/{device_id}/screenings")
    fun getDeviceScreenings(@Path("device_id") deviceId: String): Call<List<PatientScreeningRequest>>

    @POST("/api/healthcare/alert")
    fun sendAlert(@Body request: EmergencyAlertRequest): Call<GenericResponse>

    @GET("/api/healthcare/alerts")
    fun getAlerts(): Call<GenericResponse>

    @GET("/api/healthcare/statistics")
    fun getHealthcareStats(): Call<HealthcareStatsResponse>
}
