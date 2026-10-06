import sys
import os
server_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(server_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if server_dir not in sys.path:
    sys.path.append(server_dir)

import time
import sqlite3
import threading
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np


try:
    import server
except ModuleNotFoundError:
    from Server import server
from Server.database import (
    get_db_connection, initialize_database, update_client_status, log_client_step,
    record_device_telemetry, record_scheduler_action, get_all_devices, get_recent_telemetry,
    get_recent_scheduler_actions, get_latest_scheduler_action_for_device,
    save_patient_screening, get_patient_screening, get_all_patient_screenings, get_device_screenings, batch_sync_screenings,
    create_emergency_alert, get_active_alerts, get_healthcare_stats
)
from AI.dqn_agent import compute_drl_reward
from AI.healthcare_model import evaluate_screening_risk, validate_vitals, MEDICAL_DISCLAIMER


app = FastAPI(title="Edge Federated Learning Server")

# Allow CORS for Android clients connecting over local WiFi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulation & Network Profile state variables
simulation_thread = None
simulation_running = False
current_network_profile = "Good" # Options: Good, Medium, Poor, Intermittent

# Pydantic Schemas for REST endpoints
class ClientRegister(BaseModel):
    cid: str
    name: str
    location: str
    battery: float
    reliability: float
    packet_drop: float
    bandwidth: float

class DeviceRegister(BaseModel):
    device_id: str
    device_name: str
    device_type: str = "REAL"
    location: str = "Android Edge Device"

class TelemetryPayload(BaseModel):
    device_id: str
    battery: float
    charging: bool = False
    latency_ms: float = 0.0
    download_mbps: float = 0.0
    upload_mbps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_type: str = "WIFI"

class DQNActionRequest(BaseModel):
    cid: str
    battery: float
    bandwidth: float
    reliability: float
    cpu: float
    last_accuracy: float

class ClientStatsUpload(BaseModel):
    cid: str
    battery: float
    bandwidth: float
    reliability: float
    cpu: float
    ram: float
    epochs: int
    compressed: int
    uploaded_kb: float
    energy_drained: float
    accuracy_gain: float
    state: list  # [battery, bandwidth, reliability, cpu, last_accuracy]
    action: int  # action index

# Initial startup
@app.on_event("startup")
def startup_event():
    initialize_database()

# HTML Dashboard Route
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    dashboard_path = "Server/templates/dashboard.html"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h3>Dashboard Template Not Found</h3>", status_code=404)

# ----------------- NETWORK DIAGNOSTICS & PING -----------------

@app.get("/api/ping")
def ping():
    """Lightweight endpoint for RTT latency measurement."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "server": "FastAPI Edge FL Server",
        "network_profile": current_network_profile
    }

@app.get("/api/network-test/download")
def network_download_test(size_kb: int = 256):
    """
    Generates a controlled payload for mobile downstream bandwidth test.
    Default payload: 256 KB.
    """
    safe_size = max(64, min(size_kb, 1024))
    payload = "X" * (safe_size * 1024)
    return {
        "size_kb": safe_size,
        "timestamp": time.time(),
        "payload": payload
    }

@app.post("/api/network-test/upload")
async def network_upload_test(request: Request):
    """
    Measures upstream bandwidth by receiving a payload body.
    """
    start_time = time.time()
    body = await request.body()
    duration = time.time() - start_time
    size_bytes = len(body)
    size_mbits = (size_bytes * 8) / 1_000_000.0
    speed_mbps = size_mbits / max(0.001, duration)
    return {
        "received_bytes": size_bytes,
        "duration_sec": round(duration, 4),
        "upload_mbps": round(speed_mbps, 2)
    }

# ----------------- DEVICE MANAGEMENT & TELEMETRY -----------------

@app.post("/api/device/register")
def register_device(dev: DeviceRegister):
    """Register or update edge device metadata."""
    try:
        update_client_status(
            dev.device_id, dev.device_name, dev.location, 100.0, 0.95, 0.02, 50.0
        )
        print(f"[DEVICE] {dev.device_id} registered")
        return {"success": True, "message": f"Device {dev.device_id} registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/device/telemetry")
def submit_telemetry(t: TelemetryPayload):
    """Record high-frequency device telemetry."""
    try:
        record_device_telemetry(
            t.device_id, t.battery, t.charging, t.latency_ms,
            t.download_mbps, t.upload_mbps, t.cpu_usage, t.memory_usage, t.network_type
        )
        update_client_status(
            t.device_id, f"Device {t.device_id}", t.network_type,
            t.battery, round(1.0 - (t.latency_ms / 500.0), 2), 0.05, t.download_mbps
        )
        print(f"[TELEMETRY] {t.device_id} battery={t.battery:.1f}% latency={t.latency_ms:.1f}ms download={t.download_mbps:.1f}Mbps upload={t.upload_mbps:.1f}Mbps ram={t.memory_usage:.1f}%")
        return {"success": True, "message": "Telemetry recorded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices")
def list_devices():
    """List all registered edge devices (real and virtual)."""
    return get_all_devices(threshold_seconds=15)

@app.get("/api/device/{device_id}/history")
def get_device_history(device_id: str):
    """Get recent telemetry records for a specific device."""
    return get_recent_telemetry(device_id, limit=50)

@app.post("/api/network-profile")
def set_network_profile(data: dict):
    global current_network_profile
    profile = data.get("profile", "Good")
    if profile in ["Good", "Medium", "Poor", "Intermittent"]:
        current_network_profile = profile
        print(f"[NETWORK-PROFILE] Switched to {current_network_profile}")
        return {"success": True, "profile": current_network_profile}
    return {"success": False, "message": "Invalid network profile."}

# ----------------- ANDROID CLIENT ENDPOINTS -----------------

@app.post("/api/register")
def register_client(client: ClientRegister):
    """Called by Android clients to register location and battery status."""
    try:
        update_client_status(
            client.cid, client.name, client.location, client.battery,
            client.reliability, client.packet_drop, client.bandwidth
        )
        print(f"[DEVICE] {client.cid} registered")
        return {"success": True, "message": f"Client {client.cid} registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dqn-action")
def get_dqn_action(req: DQNActionRequest):
    """
    Called by Android clients to query the DQN Policy network for local training configuration.
    """
    try:
        state = [req.battery, req.bandwidth, req.reliability, req.cpu, req.last_accuracy]
        action = server.dqn_agent.act(state, epsilon=0.05)
        epochs = [0, 1, 3, 5][action]
        
        reward = compute_drl_reward(state, action, req.last_accuracy * 0.05)
        record_scheduler_action(req.cid, req.battery, req.bandwidth, req.reliability, req.cpu, req.last_accuracy, action, epochs, reward)

        print(f"[DQN] {req.cid} state received -> action={action} local_epochs={epochs}")
        return {
            "action": action,
            "epochs": epochs,
            "compress": 1 if req.bandwidth < 10.0 else 0,
            "delay_upload": 1 if req.reliability < 0.4 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-stats")
def upload_client_stats(stats: ClientStatsUpload):
    """
    Called by Android clients at the end of a round to upload local training metrics.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE client_status SET battery = ? WHERE cid = ?", (stats.battery, stats.cid))
        conn.commit()
        conn.close()
        
        log_client_step(
            stats.cid, stats.battery, stats.bandwidth, stats.reliability,
            stats.cpu, stats.ram, stats.epochs, stats.compressed,
            stats.uploaded_kb, stats.energy_drained
        )
        
        reward = compute_drl_reward(stats.state, stats.action, stats.accuracy_gain)
        next_state = [stats.battery, stats.bandwidth, stats.reliability, stats.cpu, stats.state[4] + stats.accuracy_gain]
        server.dqn_agent.remember(stats.state, stats.action, reward, next_state, False)
        server.dqn_agent.replay(32)
        
        print(f"[FL-UPLOAD] {stats.cid} uploaded stats: battery={stats.battery:.1f}% epochs={stats.epochs} reward={reward:.2f}")
        return {"success": True, "reward": reward}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model")
def get_global_model():
    global_model = server.models.TabularNet()
    if os.path.exists("AI/global_model.pth"):
        try:
            global_model.load_state_dict(torch.load("AI/global_model.pth"))
        except:
            pass
            
    serialized = {}
    for name, param in global_model.state_dict().items():
        serialized[name] = param.cpu().numpy().tolist()
        
    return {"version": int(time.time()), "parameters": serialized}

# ----------------- SIMULATION & DASHBOARD ENDPOINTS -----------------

def background_sim_runner(rounds):
    global simulation_running
    simulation_running = True
    try:
        server.run_edge_simulation(rounds=rounds)
    except Exception as e:
        print("Error inside simulation:", e)
    finally:
        simulation_running = False

@app.post("/api/start")
def start_simulation(data: dict):
    global simulation_thread, simulation_running
    if simulation_running:
        return {"success": False, "message": "Simulation already running."}
        
    rounds = int(data.get("rounds", 10))
    
    if os.path.exists("cancel.flag"):
        try:
            os.remove("cancel.flag")
        except:
            pass
            
    if os.path.exists("simulation.log"):
        try:
            os.remove("simulation.log")
        except:
            pass
            
    simulation_thread = threading.Thread(target=background_sim_runner, args=(rounds,), daemon=True)
    simulation_thread.start()
    
    return {"success": True, "message": "Simulation started successfully."}

@app.post("/api/stop")
def stop_simulation():
    global simulation_running
    if not simulation_running:
        return {"success": False, "message": "No active simulation."}
        
    with open("cancel.flag", "w", encoding="utf-8") as f:
        f.write("CANCELLED")
        
    return {"success": True, "message": "Cancellation flag set."}

@app.get("/api/dashboard-data")
@app.get("/api/status")
def get_dashboard_data():
    global simulation_running, current_network_profile
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM client_status")
    clients = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM training_rounds ORDER BY round ASC")
    rounds_data = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT COUNT(*) FROM client_logs WHERE epochs = 0")
    total_skips = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(uploaded_kb) FROM client_logs")
    bandwidth_saved_mb = round((cursor.fetchone()[0] or 0.0) / 1024.0, 2)
    
    cursor.execute("SELECT * FROM client_logs ORDER BY timestamp DESC LIMIT 20")
    recent_logs = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Fetch real edge devices with calculated online/offline status (15s threshold)
    devices = get_all_devices(threshold_seconds=15)
    
    # Target real device: android_client_01 or first REAL device
    real_device = None
    for d in devices:
        if d.get("device_id") == "android_client_01" or d.get("device_type") == "REAL":
            real_device = d
            break
            
    # Fetch history & latest DQN decision for real device
    target_id = real_device["device_id"] if real_device else "android_client_01"
    real_telemetry_history = get_recent_telemetry(target_id, limit=30)
    real_dqn_action = get_latest_scheduler_action_for_device(target_id)
    
    # Calculate real device summary metrics
    online_real_devices = [d for d in devices if d.get("computed_status") == "ONLINE"]
    real_connected_count = len(online_real_devices)
    
    if online_real_devices:
        real_bat_vals = [d["battery"] for d in online_real_devices if d.get("battery") is not None]
        avg_bat_str = f"{np.mean(real_bat_vals):.1f}%" if real_bat_vals else "N/A"
    elif real_device and real_device.get("battery") is not None:
        avg_bat_str = f"{real_device['battery']:.1f}%"
    else:
        avg_bat_str = "N/A"
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM scheduler_actions")
    total_dqn_actions = c.fetchone()[0]
    conn.close()
    
    best_acc = max([r["accuracy"] for r in rounds_data]) if rounds_data else 0.0
    total_drops = sum([r["packets_dropped"] for r in rounds_data]) if rounds_data else 0
    
    hc_stats = get_healthcare_stats()
    alerts_list = get_active_alerts(limit=10)
    
    return {
        "running": simulation_running,
        "clients": clients,
        "devices": devices,
        "real_device": real_device,
        "real_telemetry_history": real_telemetry_history,
        "real_dqn_action": real_dqn_action,
        "network_profile": current_network_profile,
        "rounds": rounds_data,
        "recent_logs": recent_logs,
        "healthcare_stats": hc_stats,
        "active_alerts": alerts_list,
        "patient_screenings": get_all_patient_screenings(limit=100),
        "metrics": {
            "connected_devices": real_connected_count if real_connected_count > 0 else (1 if real_device and real_device.get("computed_status") == "ONLINE" else 0),
            "current_round": len(rounds_data),
            "best_accuracy": f"{best_acc*100:.2f}%" if rounds_data else "N/A",
            "dropped_packets": total_drops,
            "avg_battery": avg_bat_str,
            "total_dqn_decisions": total_dqn_actions,
            "skips_count": total_skips,
            "bandwidth_saved": f"{bandwidth_saved_mb} MB",
            "total_telemetry_records": len(recent_logs),
            "total_screenings": hc_stats.get("total_screenings", 0),
            "pending_sync": hc_stats.get("pending_sync", 0)
        }
    }




@app.get("/api/stream-logs")
def stream_logs():
    def log_generator():
        log_path = "simulation.log"
        for _ in range(10):
            if os.path.exists(log_path):
                break
            time.sleep(0.5)
            
        if not os.path.exists(log_path):
            yield "data: [Edge Server] Standby. Waiting for logs...\n\n"
            return
            
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                yield f"data: {line.strip()}\n\n"
                
            while simulation_running:
                line = f.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                else:
                    time.sleep(0.2)
                    
            while True:
                line = f.readline()
                if not line:
                    break
                yield f"data: {line.strip()}\n\n"
                
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/dataset-preview")
def get_dataset_preview(client_id: str = "raw"):
    import pandas as pd
    try:
        if client_id == "raw":
            path = "Dataset/diabetes_raw.csv"
        else:
            path = f"Dataset/client_{client_id}.csv"
            
        if not os.path.exists(path):
            return {"success": False, "message": "Dataset not found."}
            
        df = pd.read_csv(path)
        total = len(df)
        healthy = int((df["Outcome"] == 0).sum())
        diabetic = int((df["Outcome"] == 1).sum())
        
        preview_rows = df.head(10).values.tolist()
        headers = df.columns.tolist()
        
        return {
            "success": True,
            "headers": headers,
            "rows": preview_rows,
            "summary": {
                "total": total,
                "healthy_pct": round((healthy / total) * 100, 1) if total > 0 else 0,
                "diabetic_pct": round((diabetic / total) * 100, 1) if total > 0 else 0
            }
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

# HEALTHCARE API MODELS
class PatientScreeningPayload(BaseModel):
    screening_id: str = None
    patient_id: str = "PAT-ANONYMOUS"
    device_id: str = "android_client_01"
    timestamp: str = None
    age: int = 35
    sex: str = "M"
    symptoms: str = ""
    medical_history: str = ""
    heart_rate: float = None
    spo2: float = None
    temperature: float = None
    systolic_bp: float = None
    diastolic_bp: float = None
    respiratory_rate: float = None
    glucose: float = None
    bmi: float = None
    risk_level: str = None
    risk_score: float = None
    model_confidence: float = None
    recommended_action: str = None
    screening_result: str = None
    sync_status: str = "SYNCED"

class EmergencyAlertPayload(BaseModel):
    alert_id: str = None
    screening_id: str = None
    device_id: str = "android_client_01"
    timestamp: str = None
    risk_level: str = "CRITICAL"
    status: str = "AWAITING_ACTION"


# HEALTHCARE API ENDPOINTS
@app.post("/api/healthcare/screening")
def post_healthcare_screening(payload: PatientScreeningPayload):
    try:
        data = payload.dict()
        if not data.get("screening_id"):
            data["screening_id"] = f"SCR-{int(time.time()*1000)}"
        if not data.get("timestamp"):
            data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # If client did not provide risk_level, calculate using canonical evaluate_screening_risk
        if not data.get("risk_level"):
            eval_res = evaluate_screening_risk(data)
            data["risk_level"] = eval_res["risk_level"]
            data["risk_score"] = eval_res["risk_score"]
            data["model_confidence"] = eval_res["confidence"]
            data["recommended_action"] = eval_res["recommended_action"]
            data["screening_result"] = f"Risk: {eval_res['risk_level']} | Score: {eval_res['risk_score']}"
        else:
            if not data.get("screening_result"):
                data["screening_result"] = f"Risk: {data['risk_level']} | Score: {data.get('risk_score', 0.0)}"
            eval_res = {
                "risk_level": data["risk_level"],
                "risk_score": data.get("risk_score", 0.0),
                "confidence": data.get("model_confidence", 0.90),
                "key_factors": [f"Ingested Risk Level: {data['risk_level']}"],
                "recommended_action": data.get("recommended_action", "Clinical follow-up"),
                "disclaimer": MEDICAL_DISCLAIMER
            }

        save_patient_screening(data)

        # Auto-create alert if CRITICAL
        if data.get("risk_level") == "CRITICAL":
            alert_id = f"ALT-{int(time.time()*1000)}"
            create_emergency_alert({
                "alert_id": alert_id,
                "screening_id": data["screening_id"],
                "device_id": data["device_id"],
                "timestamp": data["timestamp"],
                "risk_level": "CRITICAL",
                "status": "AWAITING_ACTION"
            })
            print(f"[ALERT] Critical screening recorded from {data['device_id']}! Alert ID: {alert_id}")

        print(f"[HEALTHCARE] Screening ingested: {data['screening_id']} Patient: {data['patient_id']} Risk: {data['risk_level']}")
        return {
            "success": True,
            "screening_id": data["screening_id"],
            "risk_evaluation": eval_res,
            "message": "Healthcare screening ingested successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/healthcare/screening/{screening_id}")
def get_healthcare_screening_by_id(screening_id: str):
    res = get_patient_screening(screening_id)
    if not res:
        raise HTTPException(status_code=404, detail="Screening record not found.")
    return {"success": True, "screening": res}

@app.get("/api/healthcare/device/{device_id}/screenings")
def get_device_healthcare_screenings(device_id: str, limit: int = 50):
    screenings = get_device_screenings(device_id, limit)
    return {"success": True, "device_id": device_id, "screenings": screenings}

@app.get("/api/healthcare/screenings")
def get_all_screenings_endpoint(limit: int = 100):
    screenings = get_all_patient_screenings(limit)
    return {"success": True, "screenings": screenings}

@app.post("/api/healthcare/sync")
def sync_offline_screenings(screenings: list[PatientScreeningPayload]):
    try:
        raw_list = [s.dict() for s in screenings]
        synced_count = batch_sync_screenings(raw_list)
        print(f"[HEALTHCARE] Batch sync completed for {synced_count} offline screening records.")
        return {"success": True, "synced_count": synced_count, "message": f"Successfully synchronized {synced_count} offline records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/healthcare/alert")
def post_emergency_alert(payload: EmergencyAlertPayload):
    try:
        data = payload.dict()
        if not data.get("alert_id"):
            data["alert_id"] = f"ALT-{int(time.time()*1000)}"
        if not data.get("timestamp"):
            data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        create_emergency_alert(data)
        print(f"[ALERT] Emergency alert triggered: {data['alert_id']} Device: {data['device_id']} Risk: {data['risk_level']}")
        return {"success": True, "alert_id": data["alert_id"], "message": "Emergency alert recorded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/healthcare/alerts")
def get_healthcare_alerts(limit: int = 20):
    alerts = get_active_alerts(limit)
    return {"success": True, "alerts": alerts}

@app.get("/api/healthcare/statistics")
def get_healthcare_statistics():
    stats = get_healthcare_stats()
    return {"success": True, "statistics": stats}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")



