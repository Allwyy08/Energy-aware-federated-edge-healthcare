# Project Implementation Status & Audit

**Project**: AI-Driven Energy-Aware Federated Edge Learning Framework  
**Date**: September 21, 2026  

---

## 1. Executive Summary & Inventory

The project is an existing, research-grade, distributed edge learning framework for low-connectivity regions (e.g., patient telemetry classification across rural edge nodes).

### Component Audit Summary:
* **Python Backend (`/Server`)**:
  * [app.py](file:///d:/Final%20Year%20Project/Server/app.py): FastAPI web server with dashboard SSE streaming and REST API endpoints. Currently binds to `127.0.0.1`. Needs binding to `0.0.0.0` for LAN access.
  * [server.py](file:///d:/Final%20Year%20Project/Server/server.py): Flower (`flwr`) orchestrator with FedAvg and FedAsync strategies.
  * [database.py](file:///d:/Final%20Year%20Project/Server/database.py): SQLite database manager (`federated_edge.db`). Needs schema additions for `devices`, `telemetry`, `training_rounds`, and `scheduler_actions`.
* **AI & DRL Scheduler (`/AI`)**:
  * [dqn_agent.py](file:///d:/Final%20Year%20Project/AI/dqn_agent.py): PyTorch Deep Q-Network (`DQNSchedulerNetwork`) with 5D state input (`[battery, bandwidth, latency, cpu, accuracy]`) and 4 actions (`0, 1, 3, 5` epochs). Pre-trained checkpoint available at [dqn_scheduler.pth](file:///d:/Final%20Year%20Project/AI/dqn_scheduler.pth). Needs an enhanced multi-objective reward formulation and real telemetry normalization.
* **Android Client (`/Android`)**:
  * [MainActivity.kt](file:///d:/Final%20Year%20Project/Android/app/src/main/java/com/example/federatedlearning/MainActivity.kt): Jetpack Compose dark-theme UI. Needs settings screen for customizable Server IP/Port/Client ID, Network Diagnostics screen, and DQN decision inspector.
  * [DeviceMonitor.kt](file:///d:/Final%20Year%20Project/Android/app/src/main/java/com/example/federatedlearning/monitoring/DeviceMonitor.kt): Reads Android battery & network capabilities. Currently uses randomized CPU/RAM and hardcoded signal estimations. Needs real telemetry & speed tests.
  * [TFLiteTrainer.kt](file:///d:/Final%20Year%20Project/Android/app/src/main/java/com/example/federatedlearning/tflite/TFLiteTrainer.kt): Pure Kotlin multi-layer perceptron gradient descent over local Room patient records matching PyTorch `TabularNet`.
  * [FederatedWorker.kt](file:///d:/Final%20Year%20Project/Android/app/src/main/java/com/example/federatedlearning/worker/FederatedWorker.kt): WorkManager background worker querying FastAPI and sending stats. Uses fallback array of hardcoded server URLs (`10.0.2.2`, `10.164.43.249`). Needs dynamic host configuration.

---

## 2. Component Categorization Matrix

### ✅ Completed Components
- **PyTorch Model (`TabularNet`)**: 5 inputs -> 16 hidden -> 8 hidden -> 2 output classes ([models.py](file:///d:/Final%20Year%20Project/models.py)).
- **Flower FL Server**: 10-round evaluation loop executing both centralized evaluation and strategy aggregations.
- **Dataset Partitioning**: 1,200 patient records split into Non-IID partitions (`client_0.csv` to `client_4.csv`) and centralized validation set (`diabetes_raw.csv`).
- **Android Local Training Engine**: Native Kotlin forward & backpropagation implementation for tabular records matching PyTorch weights.
- **Android Room Database**: Local SQLite Room persistence for patient telemetry (`PatientData.kt`).
- **Initial Baseline Results**: Logged simulation achieving ~90.0% validation accuracy after 10 rounds.

### 🟡 Partially Completed Components
- **FastAPI Endpoints**: REST routes exist for client registration, DQN query, stat upload, and model fetching. Missing ping latency endpoint (`/api/ping`), network bandwidth test endpoints (`/api/network-test/download`, `/api/network-test/upload`), and device history query endpoints.
- **Android UI**: Jetpack Compose tabs exist (Dashboard, Patients, DRL Status, Logs), but missing dynamic IP/Port configuration settings screen, dedicated Network Diagnostics screen, and live DQN Decision breakdown view.
- **SQLite Database Schema**: Existing tables `client_status`, `training_rounds`, `client_logs` exist. Missing explicit normalized `devices`, `telemetry`, and `scheduler_actions` tables with indexes.

### 🔷 Simulated / Mocked Components
- **Device CPU & RAM Telemetry**: `DeviceMonitor.kt` randomly generates CPU (20-85%) and RAM (45-78%) metrics due to legacy Android kernel restrictions.
- **Network Bandwidth & Latency**: Link downstream speed is read from `NetworkCapabilities`, but no active RTT ping or socket speed test was being measured.
- **Server Address Configuration**: Currently hardcoded to localhost / emulator IPs in Android worker.

### ❌ Missing Components (To Be Implemented)
- **Dynamic LAN Server Configuration**: Settings UI on Android for configurable Server IP, Port, and Client ID.
- **Real Latency & Bandwidth Measurement**: Active ping RTT measurement (`GET /api/ping`) and controlled payload speed tests (`/api/network-test/...`).
- **Network Diagnostics Screen**: Dedicated Compose UI showing real-time connectivity status, IP addresses, RTT latency, upload/download throughput, and timestamped stats.
- **Enhanced DQN Multi-Objective Reward Function**: Comprehensive formulation incorporating accuracy gain, energy drain penalty, latency penalty, and communication cost.
- **Low-Network Condition Simulation Controller**: Server-side network profile throttle (Good, Medium, Poor, Intermittent) to test edge resilience under constrained environments.
- **Comprehensive Baseline & Energy Experiments Engine**: Automated logging and graph generation for `Centralized`, `FedAvg + Fixed`, `FedAsync + Fixed`, `FedAvg + DQN`, `FedAsync + DQN`.
- **Hybrid Real Device + Virtual Clients Orchestration**: Live identification of `REAL DEVICE` vs `SIMULATED DEVICE` on dashboard and Flower server loops.
- **Comprehensive Documentation Suite**: `REAL_DEVICE_SETUP.md`, `EXPERIMENT_GUIDE.md`, `DQN_REWARD_DESIGN.md`, `NETWORK_TESTING.md`.

---

## 3. Recommended Implementation Order

1. **Step 1: Network & Server Infrastructure (LAN & Telemetry APIs)**
   - Update `app.py` to listen on `0.0.0.0:5000`.
   - Add ping `/api/ping`, bandwidth test `/api/network-test/download` and `/upload`, and `/api/device/*` endpoints.
   - Upgrade SQLite database (`database.py`) with `devices`, `telemetry`, `training_rounds`, and `scheduler_actions` tables.

2. **Step 2: Android Networking & Real Telemetry Infrastructure**
   - Implement dynamic server IP/Port/Client ID configuration with `SharedPreferences`/DataStore.
   - Upgrade `DeviceMonitor.kt` with actual RTT latency probes, socket bandwidth testing, and real battery/CPU memory stats.
   - Add **Network Diagnostics** screen to Android Jetpack Compose UI.

3. **Step 3: DQN Scheduler & Multi-Objective Reward Upgrade**
   - Update `dqn_agent.py` reward formulation with documented multi-objective weights in `Documentation/DQN_REWARD_DESIGN.md`.
   - Normalize real Android telemetry inputs before feeding to DQN inference.

4. **Step 4: Hybrid Federated Learning (Real Android + 5 Virtual Clients)**
   - Integrate Android client into Flower server orchestrator alongside Virtual Clients `0` through `4`.
   - Clearly delineate `REAL DEVICE` vs `SIMULATED DEVICE` across backend and dashboard.

5. **Step 5: Network Controller, Experiments & Graphs**
   - Implement network condition profile controller (Good, Medium, Poor, Intermittent).
   - Implement automated experiment runner for baseline comparisons and energy analysis.
   - Generate all 13 required evaluation charts and save results to `Experiments/`.

6. **Step 6: Dashboard Upgrade & Documentation**
   - Enhance HTML5 glassmorphism dashboard with real device panels, DQN state breakdown, live network graphs, and experiment comparison charts.
   - Write `REAL_DEVICE_SETUP.md`, `EXPERIMENT_GUIDE.md`, `NETWORK_TESTING.md`, `DQN_REWARD_DESIGN.md`, and update `README.md` & `architecture.md`.
