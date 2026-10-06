# Real Device Setup & Demonstration Guide

**Project**: AI-Driven Energy-Aware Federated Edge Learning Framework  
**Document**: Step-by-Step Hardware Pairing & LAN Setup  

---

## 1. Prerequisites & Equipment

* **Laptop / Edge Control Server**: Running Windows OS with Python 3.9+ & PyTorch installed.
* **Physical Android Device**: Android 8.0+ (API 26+) physical smartphone or Android Emulator connected to local network.
* **Shared Wi-Fi Network**: Both laptop and Android device MUST be connected to the exact same Wi-Fi router / local network segment.

---

## 2. Step-by-Step Setup Guide

### Step 1: Discover Laptop Local IP Address
On your Windows laptop, open PowerShell or Command Prompt and run:
```cmd
ipconfig
```
Look for **IPv4 Address** under your active Wi-Fi adapter (e.g., `192.168.1.10` or `10.164.43.249`).

### Step 2: Configure Windows Firewall Rule for Port 5000
Allow inbound connections to FastAPI on port `5000`:
Open PowerShell as **Administrator** and run:
```powershell
New-NetFirewallRule -DisplayName "FastAPI FL Server (Port 5000)" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

### Step 3: Launch FastAPI Control Backend & FL Server
In your workspace terminal, start the server bound to `0.0.0.0:5000`:
```bash
python Server/app.py
```
Verify the server starts with message:
```text
INFO: Uvicorn running on http://0.0.0.0:5000
```
Open a browser on your laptop to verify dashboard loads:
`http://localhost:5000`

### Step 4: Install and Configure Android Mobile Application
1. Open the `/Android` directory in **Android Studio**.
2. Connect your physical Android phone via USB and enable **USB Debugging**.
3. Press **Run / Build APK** to deploy the application onto your phone.
4. Open **Federated Edge AI** app on your phone.
5. Navigate to the **Settings** tab.
6. Enter:
   * **Laptop Server IP**: `<Your-Laptop-IP>` (e.g., `192.168.1.10`)
   * **Server Port**: `5000`
   * **Device / Client ID**: `android_client_01`
7. Tap **SAVE CONFIGURATION**.

### Step 5: Verify Connectivity & Network Diagnostics
1. On the Android Settings tab, tap **TEST CONNECTION NOW**.
2. Verify green text: `SUCCESS: Connected to FastAPI Server!`.
3. Switch to the **Network** tab to view real-time RTT latency, download throughput, upload throughput, and Wi-Fi signal strength.

---

## 3. Demonstrating Real-Device Federated Learning

1. On the Android app **Dataset** tab, tap **+5 Records** to populate local patient diagnostic telemetry in Room DB.
2. Tap the **DQN** tab and tap **QUERY LIVE DQN POLICY** to view live Deep Q-Network state vector normalization and policy action selection (`0, 1, 3, 5` epochs).
3. Tap **TRIGGER FEDERATED UPDATE** on the Home tab.
4. On your laptop browser at `http://localhost:5000`:
   * Observe `android_client_01` appearing with a green **REAL EDGE NODE** badge.
   * Watch live battery percentage, latency, and CPU metrics update dynamically.
   * View global model accuracy and evaluation convergence graphs.

---

## 4. Connection Troubleshooting Checklist

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `FAILED to reach server` | Different Wi-Fi networks | Ensure phone and laptop are on the exact same Wi-Fi SSID. |
| `Connection Refused` | Firewall blocking | Run PowerShell firewall rule command above or temporarily disable Windows Private Firewall. |
| `Timeout` | Server IP typo | Double-check IPv4 address via `ipconfig`. Do not use `127.0.0.1` on physical phone. |
| `404 Not Found` | Wrong port | Ensure port is set to `5000`. |
