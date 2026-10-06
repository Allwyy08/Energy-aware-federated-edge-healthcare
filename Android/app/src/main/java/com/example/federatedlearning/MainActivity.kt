package com.example.federatedlearning

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.room.Room
import androidx.work.*
import com.example.federatedlearning.data.AppDatabase
import com.example.federatedlearning.data.HealthcareScreeningEntity
import com.example.federatedlearning.data.PatientData
import com.example.federatedlearning.monitoring.DeviceMonitor
import com.example.federatedlearning.monitoring.DeviceState
import com.example.federatedlearning.worker.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity() {

    private lateinit var db: AppDatabase

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        db = Room.databaseBuilder(applicationContext, AppDatabase::class.java, "patient_db")
            .fallbackToDestructiveMigration()
            .build()

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFF38BDF8),
                    secondary = Color(0xFF34D399),
                    background = Color(0xFF0F172A),
                    surface = Color(0xFF1E293B)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HealthcareAppMainLayout(
                        db = db,
                        onTriggerWorker = { triggerSyncWorker() }
                    )
                }
            }
        }
    }

    private fun triggerSyncWorker() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val syncWork = OneTimeWorkRequestBuilder<HealthcareSyncWorker>()
            .setConstraints(constraints)
            .build()

        WorkManager.getInstance(this).enqueueUniqueWork(
            "HealthcareSyncWorkOnDemand",
            ExistingWorkPolicy.REPLACE,
            syncWork
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HealthcareAppMainLayout(db: AppDatabase, onTriggerWorker: () -> Unit) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    var screeningsList by remember { mutableStateOf(listOf<HealthcareScreeningEntity>()) }
    var pendingCount by remember { mutableIntStateOf(0) }
    var selectedScreeningForDetail by remember { mutableStateOf<HealthcareScreeningEntity?>(null) }
    var criticalAlertDetail by remember { mutableStateOf<HealthcareScreeningEntity?>(null) }

    // Read stored settings
    val prefs = context.getSharedPreferences("fl_settings", Context.MODE_PRIVATE)
    var serverIp by remember { mutableStateOf(prefs.getString("server_ip", "192.168.31.89") ?: "192.168.31.89") }
    var serverPort by remember { mutableStateOf(prefs.getString("server_port", "5000") ?: "5000") }
    var clientId by remember { mutableStateOf(prefs.getString("client_id", "android_client_01") ?: "android_client_01") }
    var emergencyContact by remember { mutableStateOf(prefs.getString("emergency_contact", "102") ?: "102") }

    // Live Device State
    val monitor = remember { DeviceMonitor(context) }
    var deviceState by remember { mutableStateOf(monitor.getDeviceState()) }

    // Fetch Room screenings
    LaunchedEffect(Unit) {
        scope.launch(Dispatchers.IO) {
            db.healthcareDao().getAllScreeningsFlow().collect { list ->
                screeningsList = list
                pendingCount = list.count { it.syncStatus == "PENDING" }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("HEALTHCARE EDGE AI", fontWeight = FontWeight.Bold, fontSize = 17.sp, color = Color.White)
                        Text("Privacy-Preserving Screening Node", fontSize = 11.sp, color = Color(0xFF34D399))
                    }
                },
                actions = {
                    IconButton(onClick = {
                        deviceState = monitor.getDeviceState()
                        onTriggerWorker()
                        Toast.makeText(context, "Refreshed telemetry & triggered sync", Toast.LENGTH_SHORT).show()
                    }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh", tint = Color.White)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface)
            )
        },
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Default.Home, contentDescription = "Home") },
                    label = { Text("Home", fontSize = 10.sp) }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Default.AddCircle, contentDescription = "New") },
                    label = { Text("New Screening", fontSize = 10.sp) }
                )
                NavigationBarItem(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    icon = { Icon(Icons.Default.List, contentDescription = "History") },
                    label = { Text("History", fontSize = 10.sp) }
                )
                NavigationBarItem(
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 },
                    icon = { Icon(Icons.Default.Info, contentDescription = "Device") },
                    label = { Text("Device", fontSize = 10.sp) }
                )
                NavigationBarItem(
                    selected = selectedTab == 4,
                    onClick = { selectedTab = 4 },
                    icon = { Icon(Icons.Default.Settings, contentDescription = "Settings") },
                    label = { Text("Settings", fontSize = 10.sp) }
                )
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
        ) {
            when (selectedTab) {
                0 -> HealthcareHomeTab(
                    deviceState = deviceState,
                    serverIp = serverIp,
                    serverPort = serverPort,
                    clientId = clientId,
                    totalScreenings = screeningsList.size,
                    pendingSyncCount = pendingCount,
                    onStartNewScreening = { selectedTab = 1 }
                )
                1 -> NewPatientScreeningTab(
                    db = db,
                    serverIp = serverIp,
                    serverPort = serverPort,
                    deviceId = clientId,
                    onScreeningCompleted = { entity ->
                        if (entity.riskLevel == "CRITICAL") {
                            criticalAlertDetail = entity
                        } else {
                            selectedScreeningForDetail = entity
                        }
                        onTriggerWorker()
                    }
                )
                2 -> ScreeningHistoryTab(
                    screenings = screeningsList,
                    onSelectScreening = { selectedScreeningForDetail = it }
                )
                3 -> DeviceTelemetryTab(
                    deviceState = deviceState,
                    serverIp = serverIp,
                    serverPort = serverPort,
                    clientId = clientId
                )
                4 -> SettingsTab(
                    serverIp = serverIp,
                    serverPort = serverPort,
                    clientId = clientId,
                    emergencyContact = emergencyContact,
                    onSaveSettings = { ip, port, id, contact ->
                        serverIp = ip
                        serverPort = port
                        clientId = id
                        emergencyContact = contact
                        prefs.edit()
                            .putString("server_ip", ip)
                            .putString("server_port", port)
                            .putString("client_id", id)
                            .putString("emergency_contact", contact)
                            .apply()
                        Toast.makeText(context, "Settings saved!", Toast.LENGTH_SHORT).show()
                    }
                )
            }

            // Detail Dialog
            selectedScreeningForDetail?.let { entity ->
                ScreeningResultDialog(
                    entity = entity,
                    onDismiss = { selectedScreeningForDetail = null }
                )
            }

            // Critical Alert Dialog
            criticalAlertDetail?.let { entity ->
                EmergencyAlertDialog(
                    entity = entity,
                    emergencyContact = emergencyContact,
                    onDismiss = { criticalAlertDetail = null }
                )
            }
        }
    }
}

// 1. HOME TAB
@Composable
fun HealthcareHomeTab(
    deviceState: DeviceState,
    serverIp: String,
    serverPort: String,
    clientId: String,
    totalScreenings: Int,
    pendingSyncCount: Int,
    onStartNewScreening: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Status Banner
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("DEVICE STATUS", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF94A3B8))
                    Box(
                        modifier = Modifier
                            .clip(CircleShape)
                            .background(if (deviceState.isWifiConnected) Color(0xFF34D399) else Color(0xFFF87171))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = if (deviceState.isWifiConnected) "ONLINE" else "OFFLINE",
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.Black
                        )
                    }
                }

                Text("ID: $clientId", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color.White)
                Text("Server: http://$serverIp:$serverPort", fontSize = 12.sp, color = Color(0xFF38BDF8), fontFamily = FontFamily.Monospace)

                Divider(color = Color(0xFF334155), thickness = 1.dp)

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Battery: ${deviceState.batteryLevel}%", fontSize = 12.sp, color = Color.White)
                    Text("Latency: ${deviceState.latencyMs.toInt()} ms", fontSize = 12.sp, color = Color.White)
                    Text("Network: ${deviceState.networkType}", fontSize = 12.sp, color = Color.White)
                }
            }
        }

        // Action Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
            shape = RoundedCornerShape(16.dp),
            border = ButtonDefaults.outlinedButtonBorder
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("AI Healthcare Screening", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)
                Text("Perform offline-first patient risk assessment and triage.", fontSize = 12.sp, color = Color(0xFF94A3B8), textAlign = TextAlign.Center)

                Button(
                    onClick = onStartNewScreening,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF34D399))
                ) {
                    Icon(Icons.Default.AddCircle, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("START NEW PATIENT SCREENING", fontWeight = FontWeight.Bold, color = Color.Black)
                }
            }
        }

        // Summary Stats Grid
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Card(
                modifier = Modifier.weight(1f),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Total Screenings", fontSize = 11.sp, color = Color(0xFF94A3B8))
                    Text("$totalScreenings", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("Local Room DB", fontSize = 10.sp, color = Color(0xFF64748B))
                }
            }

            Card(
                modifier = Modifier.weight(1f),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Pending Sync", fontSize = 11.sp, color = Color(0xFF94A3B8))
                    Text("$pendingSyncCount", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = if (pendingSyncCount > 0) Color(0xFFFBBF24) else Color(0xFF34D399))
                    Text("WorkManager Queue", fontSize = 10.sp, color = Color(0xFF64748B))
                }
            }
        }

        // Disclaimer Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1B4B))
        ) {
            Row(modifier = Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFA5B4FC), modifier = Modifier.size(24.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "This is an AI-assisted screening tool and is not a substitute for a qualified healthcare professional.",
                    fontSize = 11.sp,
                    color = Color(0xFFC7D2FE),
                    lineHeight = 15.sp
                )
            }
        }
    }
}

// 2. NEW PATIENT SCREENING FORM TAB
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewPatientScreeningTab(
    db: AppDatabase,
    serverIp: String,
    serverPort: String,
    deviceId: String,
    onScreeningCompleted: (HealthcareScreeningEntity) -> Unit
) {
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    var patientId by remember { mutableStateOf("PAT-${(100..999).random()}") }
    var ageText by remember { mutableStateOf("45") }
    var sexText by remember { mutableStateOf("M") }
    var symptomsText by remember { mutableStateOf("") }
    var historyText by remember { mutableStateOf("") }

    var heartRateText by remember { mutableStateOf("76") }
    var spo2Text by remember { mutableStateOf("98") }
    var tempText by remember { mutableStateOf("37.0") }
    var sysBpText by remember { mutableStateOf("125") }
    var diaBpText by remember { mutableStateOf("82") }
    var respRateText by remember { mutableStateOf("16") }
    var glucoseText by remember { mutableStateOf("110") }
    var bmiText by remember { mutableStateOf("26.5") }

    var isAnalyzing by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("NEW PATIENT SCREENING", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)

        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Patient Demographics", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFF38BDF8))

                OutlinedTextField(
                    value = patientId,
                    onValueChange = { patientId = it },
                    label = { Text("Anonymized Patient ID") },
                    modifier = Modifier.fillMaxWidth()
                )

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = ageText,
                        onValueChange = { ageText = it },
                        label = { Text("Age (Years)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = sexText,
                        onValueChange = { sexText = it },
                        label = { Text("Sex (M/F)") },
                        modifier = Modifier.weight(1f)
                    )
                }

                OutlinedTextField(
                    value = symptomsText,
                    onValueChange = { symptomsText = it },
                    label = { Text("Symptoms (e.g., fatigue, chest pain, fever)") },
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = historyText,
                    onValueChange = { historyText = it },
                    label = { Text("Medical History Summary") },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }

        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Physiological Vitals & Laboratory", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFF34D399))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = heartRateText,
                        onValueChange = { heartRateText = it },
                        label = { Text("Heart Rate (bpm)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = spo2Text,
                        onValueChange = { spo2Text = it },
                        label = { Text("SpO2 (%)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = tempText,
                        onValueChange = { tempText = it },
                        label = { Text("Temp (°C)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = respRateText,
                        onValueChange = { respRateText = it },
                        label = { Text("Resp Rate (/min)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = sysBpText,
                        onValueChange = { sysBpText = it },
                        label = { Text("Systolic BP (mmHg)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = diaBpText,
                        onValueChange = { diaBpText = it },
                        label = { Text("Diastolic BP (mmHg)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = glucoseText,
                        onValueChange = { glucoseText = it },
                        label = { Text("Glucose (mg/dL)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = bmiText,
                        onValueChange = { bmiText = it },
                        label = { Text("BMI (kg/m²)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        Button(
            onClick = {
                val age = ageText.toIntOrNull() ?: 35
                val hr = heartRateText.toFloatOrNull()
                val spo2 = spo2Text.toFloatOrNull()
                val temp = tempText.toFloatOrNull()
                val sysBp = sysBpText.toFloatOrNull()
                val diaBp = diaBpText.toFloatOrNull()
                val resp = respRateText.toFloatOrNull()
                val gluc = glucoseText.toFloatOrNull()
                val bmi = bmiText.toFloatOrNull()

                isAnalyzing = true
                scope.launch {
                    val screeningId = "SCR-${System.currentTimeMillis()}"
                    val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

                    val req = PatientScreeningRequest(
                        screening_id = screeningId,
                        patient_id = patientId,
                        device_id = deviceId,
                        timestamp = timestamp,
                        age = age,
                        sex = sexText,
                        symptoms = symptomsText,
                        medical_history = historyText,
                        heart_rate = hr,
                        spo2 = spo2,
                        temperature = temp,
                        systolic_bp = sysBp,
                        diastolic_bp = diaBp,
                        respiratory_rate = resp,
                        glucose = gluc,
                        bmi = bmi,
                        sync_status = "PENDING"
                    )

                    var risk = "MODERATE"
                    var score = 2.0f
                    var confidence = 0.85f
                    var rec = "Schedule follow-up health screening and monitor vitals closely."
                    var syncState = "PENDING"

                    // Try backend API first
                    try {
                        withContext(Dispatchers.IO) {
                            val baseUrl = "http://$serverIp:$serverPort/"
                            val retrofit = Retrofit.Builder()
                                .baseUrl(baseUrl)
                                .addConverterFactory(GsonConverterFactory.create())
                                .build()
                            val api = retrofit.create(FLNetworkService::class.java)
                            val call = api.sendScreening(req).execute()
                            if (call.isSuccessful && call.body()?.success == true) {
                                val body = call.body()
                                body?.risk_evaluation?.let { ev ->
                                    risk = ev.risk_level
                                    score = ev.risk_score
                                    confidence = ev.confidence
                                    rec = ev.recommended_action
                                    syncState = "SYNCED"
                                }
                            }
                        }
                    } catch (e: Exception) {
                        // Offline local fallbacks rule evaluation
                        if ((spo2 != null && spo2 < 90.0) || (sysBp != null && sysBp >= 180.0) || symptomsText.lowercase().contains("chest pain")) {
                            risk = "CRITICAL"
                            score = 15.0f
                            confidence = 0.94f
                            rec = "Urgent medical evaluation and immediate emergency escalation recommended."
                        } else if ((sysBp != null && sysBp >= 140.0) || (gluc != null && gluc >= 180.0)) {
                            risk = "HIGH"
                            score = 4.0f
                            confidence = 0.89f
                            rec = "Prompt clinical evaluation by a qualified healthcare worker recommended."
                        }
                    }

                    val entity = HealthcareScreeningEntity(
                        screeningId = screeningId,
                        patientId = patientId,
                        timestamp = timestamp,
                        age = age,
                        sex = sexText,
                        symptoms = symptomsText,
                        medicalHistory = historyText,
                        heartRate = hr,
                        spo2 = spo2,
                        temperature = temp,
                        systolicBp = sysBp,
                        diastolicBp = diaBp,
                        respiratoryRate = resp,
                        glucose = gluc,
                        bmi = bmi,
                        screeningResult = "Risk: $risk | Score: $score",
                        riskLevel = risk,
                        modelConfidence = confidence,
                        recommendedAction = rec,
                        syncStatus = syncState
                    )

                    withContext(Dispatchers.IO) {
                        db.healthcareDao().insert(entity)
                    }

                    isAnalyzing = false
                    onScreeningCompleted(entity)
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isAnalyzing,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF38BDF8))
        ) {
            if (isAnalyzing) {
                CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.Black)
            } else {
                Text("ANALYZE PATIENT SCREENING", fontWeight = FontWeight.Bold, color = Color.Black)
            }
        }
    }
}

// 3. SCREENING HISTORY TAB
@Composable
fun ScreeningHistoryTab(
    screenings: List<HealthcareScreeningEntity>,
    onSelectScreening: (HealthcareScreeningEntity) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("SCREENING HISTORY", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)

        if (screenings.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No local screening records found.", color = Color(0xFF94A3B8), fontSize = 14.sp)
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(screenings) { item ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelectScreening(item) },
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text(item.patientId, fontWeight = FontWeight.Bold, fontSize = 15.sp, color = Color.White)
                                Text(item.timestamp, fontSize = 11.sp, color = Color(0xFF94A3B8))
                                Text("Glucose: ${item.glucose?.toInt() ?: '--'} | BP: ${item.systolicBp?.toInt() ?: '--'}/${item.diastolicBp?.toInt() ?: '--'}", fontSize = 12.sp, color = Color(0xFFCBD5E1))
                            }

                            Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                val riskColor = when (item.riskLevel) {
                                    "CRITICAL" -> Color(0xFFF87171)
                                    "HIGH" -> Color(0xFFFB923C)
                                    "MODERATE" -> Color(0xFFFACC15)
                                    else -> Color(0xFF4ADE80)
                                }
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(4.dp))
                                        .background(riskColor.copy(alpha = 0.2f))
                                        .padding(horizontal = 8.dp, vertical = 2.dp)
                                ) {
                                    Text(item.riskLevel, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = riskColor)
                                }

                                Text(
                                    text = item.syncStatus,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    color = if (item.syncStatus == "SYNCED") Color(0xFF34D399) else Color(0xFFFBBF24)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// 4. DEVICE TELEMETRY TAB
@Composable
fun DeviceTelemetryTab(
    deviceState: DeviceState,
    serverIp: String,
    serverPort: String,
    clientId: String
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("DEVICE TELEMETRY & DQN", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)

        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Hardware Metrics", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFF38BDF8))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Battery Level: ${deviceState.batteryLevel}%", fontSize = 13.sp, color = Color.White)
                    Text("Charging: ${if (deviceState.isCharging) "YES" else "NO"}", fontSize = 13.sp, color = Color.White)
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("RAM Usage: ${deviceState.memoryUsagePercent.toInt()}%", fontSize = 13.sp, color = Color.White)
                    Text("CPU: Unavailable", fontSize = 13.sp, color = Color(0xFF94A3B8))
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Latency: ${deviceState.latencyMs.toInt()} ms", fontSize = 13.sp, color = Color.White)
                    Text("Network: ${deviceState.networkType}", fontSize = 13.sp, color = Color.White)
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Down Speed: ${deviceState.downloadMbps.toInt()} Mbps", fontSize = 13.sp, color = Color.White)
                    Text("Up Speed: ${deviceState.uploadMbps.toInt()} Mbps", fontSize = 13.sp, color = Color.White)
                }
            }
        }

        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF311B92))) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Energy-Aware Training Scheduler (DQN)", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFFD8B4FE))
                Text("Inputs: Battery ${deviceState.batteryLevel}% | Latency ${deviceState.latencyMs.toInt()}ms | Down ${deviceState.downloadMbps.toInt()}Mbps", fontSize = 12.sp, color = Color(0xFFE9D5FF))
                Text("Policy Action Directive: 3 Local Training Epochs", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Color.White)
            }
        }
    }
}

// 5. SETTINGS TAB
@Composable
fun SettingsTab(
    serverIp: String,
    serverPort: String,
    clientId: String,
    emergencyContact: String,
    onSaveSettings: (String, String, String, String) -> Unit
) {
    var ip by remember { mutableStateOf(serverIp) }
    var port by remember { mutableStateOf(serverPort) }
    var id by remember { mutableStateOf(clientId) }
    var contact by remember { mutableStateOf(emergencyContact) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("SETTINGS & PAIRING", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)

        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B))) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = ip,
                    onValueChange = { ip = it },
                    label = { Text("Server Laptop IPv4 Address") },
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = port,
                    onValueChange = { port = it },
                    label = { Text("Server Port") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = id,
                    onValueChange = { id = it },
                    label = { Text("Android Edge Client ID") },
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = contact,
                    onValueChange = { contact = it },
                    label = { Text("Emergency Contact Number") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )

                Button(
                    onClick = { onSaveSettings(ip, port, id, contact) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF38BDF8))
                ) {
                    Text("SAVE CONFIGURATION", fontWeight = FontWeight.Bold, color = Color.Black)
                }
            }
        }
    }
}

// RESULT DIALOG
@Composable
fun ScreeningResultDialog(
    entity: HealthcareScreeningEntity,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("CLOSE", color = Color(0xFF38BDF8))
            }
        },
        title = {
            Text("SCREENING RESULT (${entity.riskLevel})", fontWeight = FontWeight.Bold, fontSize = 16.sp)
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Patient ID: ${entity.patientId}", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                Text("Timestamp: ${entity.timestamp}", fontSize = 11.sp, color = Color.Gray)
                Text("Risk Assessment: ${entity.screeningResult}", fontSize = 13.sp)
                Text("Recommended Action:\n${entity.recommendedAction}", fontSize = 12.sp, color = Color(0xFF34D399))
                Divider()
                Text("Disclaimer: This is an AI-assisted screening tool and is not a substitute for a qualified healthcare professional.", fontSize = 10.sp, color = Color.Gray)
            }
        },
        containerColor = Color(0xFF1E293B),
        titleContentColor = Color.White,
        textContentColor = Color.White
    )
}

// EMERGENCY ALERT DIALOG
@Composable
fun EmergencyAlertDialog(
    entity: HealthcareScreeningEntity,
    emergencyContact: String,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(
                onClick = {
                    val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:$emergencyContact"))
                    context.startActivity(intent)
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444))
            ) {
                Text("CALL EMERGENCY ($emergencyContact)", color = Color.White, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("DISMISS", color = Color.White)
            }
        },
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFEF4444))
                Spacer(modifier = Modifier.width(8.dp))
                Text("CRITICAL SCREENING RESULT", fontWeight = FontWeight.Bold, color = Color(0xFFEF4444), fontSize = 16.sp)
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Patient: ${entity.patientId}", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Text("Urgent medical evaluation is recommended immediately.", fontSize = 13.sp, color = Color.White)
                Text("Action: ${entity.recommendedAction}", fontSize = 12.sp, color = Color(0xFFFCA5A5))
            }
        },
        containerColor = Color(0xFF7F1D1D)
    )
}
