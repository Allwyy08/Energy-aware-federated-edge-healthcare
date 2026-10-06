import sqlite3
import os

DB_FILE = "federated_edge.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except:
        pass
    return conn


def initialize_database():
    """
    Initializes tables to track:
    - client_status: Client hardware & network metadata
    - devices: Normalized real/simulated edge device registry
    - telemetry: High-frequency real device telemetry entries
    - training_rounds: Aggregated server validation accuracy and loss
    - scheduler_actions: DQN actions & reward tracking
    - client_logs: Historic training step metrics uploaded by clients
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create client_status (backward compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_status (
            cid TEXT PRIMARY KEY,
            name TEXT,
            location TEXT,
            battery REAL,
            reliability REAL,
            packet_drop REAL,
            bandwidth REAL,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create normalized devices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            device_name TEXT,
            device_type TEXT DEFAULT 'SIMULATED', -- REAL or SIMULATED
            location TEXT,
            ip_address TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'ONLINE'
        )
    """)
    
    # Create high-frequency telemetry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            battery REAL,
            charging INTEGER,
            latency_ms REAL,
            download_mbps REAL,
            upload_mbps REAL,
            cpu_usage REAL,
            memory_usage REAL,
            network_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry(device_id, timestamp);")

    # Create training_rounds
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_rounds (
            round INTEGER PRIMARY KEY,
            accuracy REAL,
            loss REAL,
            active_clients INTEGER,
            packets_dropped INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create scheduler_actions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scheduler_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            battery REAL,
            bandwidth REAL,
            latency REAL,
            cpu REAL,
            accuracy REAL,
            selected_action INTEGER,
            selected_epochs INTEGER,
            reward REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scheduler_device ON scheduler_actions(device_id, timestamp);")

    # Create client_logs (history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cid TEXT,
            battery REAL,
            bandwidth REAL,
            latency REAL,
            cpu REAL,
            ram REAL,
            epochs INTEGER,
            compressed INTEGER,
            uploaded_kb REAL,
            energy_drained REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create patient_screenings table for healthcare edge
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_screenings (
            screening_id TEXT PRIMARY KEY,
            patient_id TEXT,
            device_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            sex TEXT,
            symptoms TEXT,
            medical_history TEXT,
            heart_rate REAL,
            spo2 REAL,
            temperature REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            respiratory_rate REAL,
            glucose REAL,
            bmi REAL,
            screening_result TEXT,
            risk_level TEXT,
            risk_score REAL,
            model_confidence REAL,
            recommended_action TEXT,
            sync_status TEXT DEFAULT 'SYNCED',
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        )
    """)
    try:
        cursor.execute("ALTER TABLE patient_screenings ADD COLUMN risk_score REAL;")
    except Exception:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_device ON patient_screenings(device_id, timestamp);")

    # Create emergency_alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_alerts (
            alert_id TEXT PRIMARY KEY,
            screening_id TEXT,
            device_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_level TEXT,
            status TEXT DEFAULT 'AWAITING_ACTION',
            FOREIGN KEY (screening_id) REFERENCES patient_screenings(screening_id)
        )
    """)

    # Insert default client mappings if not already populated
    defaults = [
        ("0", "Ooty Base Station", "Nilgiris (Open Hill)", 100.0, 0.95, 0.02, 100.0),
        ("1", "Yercaud Coffee Estate", "Shevaroys (Slope)", 100.0, 0.80, 0.05, 100.0),
        ("2", "Kodaikanal Valley", "Palani Hills (Forest Edge)", 100.0, 0.50, 0.20, 100.0),
        ("3", "Valparai Tea Range", "Anamalais (Deep Valley)", 100.0, 0.30, 0.40, 100.0),
        ("4", "Mudumalai Ranger Camp", "Reserve Forest (Dense Canopy)", 100.0, 0.20, 0.50, 100.0)
    ]

    
    for row in defaults:
        cursor.execute("""
            INSERT OR IGNORE INTO client_status (cid, name, location, battery, reliability, packet_drop, bandwidth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, row)
        cursor.execute("""
            INSERT OR IGNORE INTO devices (device_id, device_name, device_type, location, status)
            VALUES (?, ?, 'SIMULATED', ?, 'ONLINE')
        """, (row[0], row[1], row[2]))
        
    conn.commit()
    conn.close()
    print("Edge Database initialized.")

def update_client_status(cid, name, location, battery, reliability, packet_drop, bandwidth):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO client_status (cid, name, location, battery, reliability, packet_drop, bandwidth, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (cid, name, location, battery, reliability, packet_drop, bandwidth))
    
    dev_type = "REAL" if "android" in cid.lower() or "phone" in cid.lower() or "mobile" in cid.lower() else "SIMULATED"
    cursor.execute("""
        INSERT INTO devices (device_id, device_name, device_type, location, last_seen, status)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 'ONLINE')
        ON CONFLICT(device_id) DO UPDATE SET
            device_name=excluded.device_name,
            device_type=excluded.device_type,
            location=excluded.location,
            last_seen=CURRENT_TIMESTAMP,
            status='ONLINE'
    """, (cid, name, dev_type, location))
    conn.commit()
    conn.close()

def record_device_telemetry(device_id, battery, charging, latency_ms, download_mbps, upload_mbps, cpu_usage, memory_usage, network_type):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO telemetry (device_id, battery, charging, latency_ms, download_mbps, upload_mbps, cpu_usage, memory_usage, network_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (device_id, battery, 1 if charging else 0, latency_ms, download_mbps, upload_mbps, cpu_usage, memory_usage, network_type))
    
    dev_type = "REAL" if "android" in device_id.lower() or "phone" in device_id.lower() or "mobile" in device_id.lower() else "SIMULATED"
    cursor.execute("""
        INSERT INTO devices (device_id, device_name, device_type, location, last_seen, status)
        VALUES (?, 'My Android Phone', ?, 'Android Edge Device', CURRENT_TIMESTAMP, 'ONLINE')
        ON CONFLICT(device_id) DO UPDATE SET
            last_seen=CURRENT_TIMESTAMP,
            status='ONLINE'
    """, (device_id, dev_type))
    conn.commit()
    conn.close()

def record_scheduler_action(device_id, battery, bandwidth, latency, cpu, accuracy, selected_action, selected_epochs, reward):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scheduler_actions (device_id, battery, bandwidth, latency, cpu, accuracy, selected_action, selected_epochs, reward, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (device_id, battery, bandwidth, latency, cpu, accuracy, selected_action, selected_epochs, reward))
    conn.commit()
    conn.close()

def log_training_round(round_num, accuracy, loss, active_clients, packets_dropped):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO training_rounds (round, accuracy, loss, active_clients, packets_dropped, timestamp)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (round_num, accuracy, loss, active_clients, packets_dropped))
    conn.commit()
    conn.close()

def log_client_step(cid, battery, bandwidth, latency, cpu, ram, epochs, compressed, uploaded_kb, energy_drained):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO client_logs (cid, battery, bandwidth, latency, cpu, ram, epochs, compressed, uploaded_kb, energy_drained, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (cid, battery, bandwidth, latency, cpu, ram, epochs, compressed, uploaded_kb, energy_drained))
    conn.commit()
    conn.close()

def get_all_devices(threshold_seconds=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            d.device_id,
            d.device_name,
            d.device_type,
            d.location,
            d.registered_at,
            d.last_seen,
            CASE 
                WHEN (strftime('%s', 'now') - strftime('%s', d.last_seen)) <= ? THEN 'ONLINE'
                ELSE 'OFFLINE'
            END as computed_status,
            t.battery,
            t.charging,
            t.latency_ms,
            t.download_mbps,
            t.upload_mbps,
            t.cpu_usage,
            t.memory_usage,
            t.network_type,
            t.timestamp as telemetry_timestamp
        FROM devices d
        LEFT JOIN (
            SELECT t1.*
            FROM telemetry t1
            INNER JOIN (
                SELECT device_id, MAX(id) as max_id
                FROM telemetry
                GROUP BY device_id
            ) t2 ON t1.id = t2.max_id
        ) t ON d.device_id = t.device_id
    """, (threshold_seconds,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_all_clients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM client_status")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_rounds_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_rounds ORDER BY round ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_client_logs(cid, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM client_logs WHERE cid = ? ORDER BY timestamp DESC LIMIT ?", (cid, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_recent_telemetry(device_id, limit=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry WHERE device_id = ? ORDER BY id DESC LIMIT ?", (device_id, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_recent_scheduler_actions(limit=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scheduler_actions ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_latest_scheduler_action_for_device(device_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scheduler_actions WHERE device_id = ? ORDER BY id DESC LIMIT 1", (device_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# HEALTHCARE EDGE DATABASE HELPERS
def save_patient_screening(data):
    from AI.healthcare_model import evaluate_screening_risk
    if not data.get("risk_level"):
        eval_res = evaluate_screening_risk(data)
        data["risk_level"] = eval_res["risk_level"]
        data["risk_score"] = eval_res["risk_score"]
        data["model_confidence"] = eval_res["confidence"]
        data["recommended_action"] = eval_res["recommended_action"]
        data["screening_result"] = f"Risk: {eval_res['risk_level']} | Score: {eval_res['risk_score']}"
    elif not data.get("screening_result"):
        data["screening_result"] = f"Risk: {data['risk_level']} | Score: {data.get('risk_score', 0.0)}"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patient_screenings (
            screening_id, patient_id, device_id, timestamp, age, sex, symptoms, medical_history,
            heart_rate, spo2, temperature, systolic_bp, diastolic_bp, respiratory_rate, glucose, bmi,
            screening_result, risk_level, risk_score, model_confidence, recommended_action, sync_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("screening_id"),
        data.get("patient_id", "PAT-ANONYMOUS"),
        data.get("device_id", "android_client_01"),
        data.get("timestamp"),
        data.get("age"),
        data.get("sex"),
        data.get("symptoms"),
        data.get("medical_history"),
        data.get("heart_rate"),
        data.get("spo2"),
        data.get("temperature"),
        data.get("systolic_bp") or data.get("bp_sys"),
        data.get("diastolic_bp") or data.get("bp_dia"),
        data.get("respiratory_rate"),
        data.get("glucose"),
        data.get("bmi"),
        data.get("screening_result"),
        data.get("risk_level"),
        data.get("risk_score", 0.0),
        data.get("model_confidence", 0.90),
        data.get("recommended_action"),
        data.get("sync_status", "SYNCED")
    ))
    
    if data.get("risk_level") == "CRITICAL":
        alert_id = f"ALT-{data.get('screening_id') or 'UNK'}"
        cursor.execute("""
            INSERT OR IGNORE INTO emergency_alerts (alert_id, screening_id, device_id, timestamp, risk_level, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (alert_id, data.get("screening_id"), data.get("device_id", "android_client_01"), data.get("timestamp"), "CRITICAL", "AWAITING_ACTION"))
        
    conn.commit()
    conn.close()

def get_patient_screening(screening_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_screenings WHERE screening_id = ?", (screening_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_patient_screenings(limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_screenings ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_device_screenings(device_id, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_screenings WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?", (device_id, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def batch_sync_screenings(screenings_list):
    from AI.healthcare_model import evaluate_screening_risk
    conn = get_db_connection()
    cursor = conn.cursor()
    synced_count = 0
    for s in screenings_list:
        s["sync_status"] = "SYNCED"
        if not s.get("risk_level"):
            eval_res = evaluate_screening_risk(s)
            s["risk_level"] = eval_res["risk_level"]
            s["risk_score"] = eval_res["risk_score"]
            s["model_confidence"] = eval_res["confidence"]
            s["recommended_action"] = eval_res["recommended_action"]
            s["screening_result"] = f"Risk: {eval_res['risk_level']} | Score: {eval_res['risk_score']}"
        elif not s.get("screening_result"):
            s["screening_result"] = f"Risk: {s['risk_level']} | Score: {s.get('risk_score', 0.0)}"

        cursor.execute("""
            INSERT OR REPLACE INTO patient_screenings (
                screening_id, patient_id, device_id, timestamp, age, sex, symptoms, medical_history,
                heart_rate, spo2, temperature, systolic_bp, diastolic_bp, respiratory_rate, glucose, bmi,
                screening_result, risk_level, risk_score, model_confidence, recommended_action, sync_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s.get("screening_id"),
            s.get("patient_id", "PAT-ANONYMOUS"),
            s.get("device_id", "android_client_01"),
            s.get("timestamp"),
            s.get("age"),
            s.get("sex"),
            s.get("symptoms"),
            s.get("medical_history"),
            s.get("heart_rate"),
            s.get("spo2"),
            s.get("temperature"),
            s.get("systolic_bp") or s.get("bp_sys"),
            s.get("diastolic_bp") or s.get("bp_dia"),
            s.get("respiratory_rate"),
            s.get("glucose"),
            s.get("bmi"),
            s.get("screening_result"),
            s.get("risk_level"),
            s.get("risk_score", 0.0),
            s.get("model_confidence", 0.90),
            s.get("recommended_action"),
            "SYNCED"
        ))

        if s.get("risk_level") == "CRITICAL":
            alert_id = f"ALT-{s.get('screening_id') or 'UNK'}"
            cursor.execute("""
                INSERT OR IGNORE INTO emergency_alerts (alert_id, screening_id, device_id, timestamp, risk_level, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alert_id, s.get("screening_id"), s.get("device_id", "android_client_01"), s.get("timestamp"), "CRITICAL", "AWAITING_ACTION"))

        synced_count += 1
    conn.commit()
    conn.close()
    return synced_count

def create_emergency_alert(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO emergency_alerts (alert_id, screening_id, device_id, timestamp, risk_level, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.get("alert_id"),
        data.get("screening_id"),
        data.get("device_id", "android_client_01"),
        data.get("timestamp"),
        data.get("risk_level", "CRITICAL"),
        data.get("status", "AWAITING_ACTION")
    ))
    conn.commit()
    conn.close()

def get_active_alerts(limit=20):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emergency_alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_healthcare_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM patient_screenings")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patient_screenings WHERE risk_level = 'LOW'")
    low = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patient_screenings WHERE risk_level = 'MODERATE'")
    moderate = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patient_screenings WHERE risk_level = 'HIGH'")
    high = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patient_screenings WHERE risk_level = 'CRITICAL'")
    critical = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patient_screenings WHERE sync_status = 'PENDING'")
    pending_sync = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_screenings": total,
        "low_risk": low,
        "moderate_risk": moderate,
        "high_risk": high,
        "critical_risk": critical,
        "pending_sync": pending_sync
    }

if __name__ == "__main__":
    initialize_database()



