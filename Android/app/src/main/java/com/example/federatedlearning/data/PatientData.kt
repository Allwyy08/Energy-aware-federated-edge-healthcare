package com.example.federatedlearning.data

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "patient_diagnostics")
data class PatientData(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "glucose") val glucose: Float,
    @ColumnInfo(name = "blood_pressure") val bloodPressure: Float,
    @ColumnInfo(name = "insulin") val insulin: Float,
    @ColumnInfo(name = "bmi") val bmi: Float,
    @ColumnInfo(name = "age") val age: Float,
    @ColumnInfo(name = "outcome") val outcome: Int  // 0 = Healthy, 1 = Diabetic
)

@Dao
interface PatientDao {
    @Query("SELECT * FROM patient_diagnostics")
    fun getAllPatients(): List<PatientData>

    @Query("SELECT * FROM patient_diagnostics")
    fun getAllPatientsFlow(): Flow<List<PatientData>>

    @Query("SELECT COUNT(*) FROM patient_diagnostics")
    fun getCount(): Int

    @Insert
    fun insert(vararg patient: PatientData)

    @Delete
    fun delete(patient: PatientData)

    @Query("DELETE FROM patient_diagnostics")
    fun clearAll()
}

// HEALTHCARE EDGE SCREENING ROOM ENTITY & DAO
@Entity(tableName = "healthcare_screenings")
data class HealthcareScreeningEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "screening_id") val screeningId: String,
    @ColumnInfo(name = "patient_id") val patientId: String,
    @ColumnInfo(name = "timestamp") val timestamp: String,
    @ColumnInfo(name = "age") val age: Int,
    @ColumnInfo(name = "sex") val sex: String,
    @ColumnInfo(name = "symptoms") val symptoms: String,
    @ColumnInfo(name = "medical_history") val medicalHistory: String,
    @ColumnInfo(name = "heart_rate") val heartRate: Float?,
    @ColumnInfo(name = "spo2") val spo2: Float?,
    @ColumnInfo(name = "temperature") val temperature: Float?,
    @ColumnInfo(name = "systolic_bp") val systolicBp: Float?,
    @ColumnInfo(name = "diastolic_bp") val diastolicBp: Float?,
    @ColumnInfo(name = "respiratory_rate") val respiratoryRate: Float?,
    @ColumnInfo(name = "glucose") val glucose: Float?,
    @ColumnInfo(name = "bmi") val bmi: Float?,
    @ColumnInfo(name = "screening_result") val screeningResult: String,
    @ColumnInfo(name = "risk_level") val riskLevel: String, // LOW, MODERATE, HIGH, CRITICAL
    @ColumnInfo(name = "model_confidence") val modelConfidence: Float,
    @ColumnInfo(name = "recommended_action") val recommendedAction: String,
    @ColumnInfo(name = "sync_status") val syncStatus: String = "PENDING", // PENDING, SYNCING, SYNCED, FAILED
    @ColumnInfo(name = "emergency_status") val emergencyStatus: String = "NONE"
)

@Dao
interface HealthcareDao {
    @Query("SELECT * FROM healthcare_screenings ORDER BY id DESC")
    fun getAllScreeningsFlow(): Flow<List<HealthcareScreeningEntity>>

    @Query("SELECT * FROM healthcare_screenings ORDER BY id DESC")
    fun getAllScreenings(): List<HealthcareScreeningEntity>

    @Query("SELECT * FROM healthcare_screenings WHERE sync_status = 'PENDING'")
    fun getPendingScreenings(): List<HealthcareScreeningEntity>

    @Query("SELECT COUNT(*) FROM healthcare_screenings WHERE sync_status = 'PENDING'")
    fun getPendingCount(): Int

    @Query("SELECT COUNT(*) FROM healthcare_screenings")
    fun getTotalCount(): Int

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insert(screening: HealthcareScreeningEntity): Long

    @Query("UPDATE healthcare_screenings SET sync_status = :status WHERE screening_id = :screeningId")
    fun updateSyncStatus(screeningId: String, status: String)

    @Delete
    fun delete(screening: HealthcareScreeningEntity)
}

@Database(entities = [PatientData::class, HealthcareScreeningEntity::class], version = 2, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun patientDao(): PatientDao
    abstract fun healthcareDao(): HealthcareDao
}
