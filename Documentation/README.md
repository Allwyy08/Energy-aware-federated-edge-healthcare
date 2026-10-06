# AI-Driven Energy-Aware Federated Edge Learning Framework

A distributed research-grade framework to optimize Federated Learning over Low-Power IoT Edge Networks in Low-Connectivity Regions (modeled after rural health camps and forest edge communities).

---

## 📁 Project Architecture & Components

* **`Android/`**: Native Kotlin Mobile Application built with Jetpack Compose UI, Room Database (local patient telemetry storage), WorkManager background workers, real hardware sensors (battery level, voltage, temperature, RAM, CPU), and native gradient descent execution matching PyTorch `TabularNet`.
* **`Server/`**: Python backend exposing FastAPI endpoints (`0.0.0.0:5000`), Flower (`flwr`) orchestrator supporting FedAvg and FedAsync strategies, SQLite database manager (`federated_edge.db`), and interactive HTML5 glassmorphism control dashboard.
* **`AI/`**: PyTorch implementation of the Deep Q-Network (DQN) scheduler agent ([dqn_agent.py](file:///d:/Final%20Year%20Project/AI/dqn_agent.py)) with normalized 5D state input, action selection (`0, 1, 3, 5` epochs), and multi-objective reward formulation.
* **`Dataset/`**: Non-IID diagnostic patient dataset split across 5 edge partitions (`client_0.csv` to `client_4.csv`) and centralized validation set (`diabetes_raw.csv`).
* **`Experiments/`**: Automated baseline benchmarking runner ([experiments.py](file:///d:/Final%20Year%20Project/Server/experiments.py)) and plot generator ([generate_plots.py](file:///d:/Final%20Year%20Project/Server/generate_plots.py)) producing all 13 research figures.
* **`Documentation/`**: Complete system documentation:
  * [IMPLEMENTATION_STATUS.md](file:///d:/Final%20Year%20Project/Documentation/IMPLEMENTATION_STATUS.md) - Project audit & status matrix.
  * [REAL_DEVICE_SETUP.md](file:///d:/Final%20Year%20Project/Documentation/REAL_DEVICE_SETUP.md) - Step-by-step physical Android phone pairing guide.
  * [EXPERIMENT_GUIDE.md](file:///d:/Final%20Year%20Project/Documentation/EXPERIMENT_GUIDE.md) - Baseline evaluation & chart generation guide.
  * [DQN_REWARD_DESIGN.md](file:///d:/Final%20Year%20Project/Documentation/DQN_REWARD_DESIGN.md) - Multi-objective DRL reward formulation.
  * [NETWORK_TESTING.md](file:///d:/Final%20Year%20Project/Documentation/NETWORK_TESTING.md) - RTT ping latency & speed test protocols.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Install required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset Partitions
Generate the 1,200 patient diagnostic dataset records and split them across 5 client files:
```bash
python Dataset/partition_data.py
```

### 3. Initialize SQLite Database
Initialize database tables (`devices`, `telemetry`, `client_status`, `training_rounds`, `scheduler_actions`):
```bash
python Server/database.py
```

### 4. Run Edge Server
Launch the FastAPI control server bound to LAN interface `0.0.0.0:5000`:
```bash
python Server/app.py
```
Open your laptop browser to:
[http://localhost:5000](http://localhost:5000)

---

## 📲 Physical Android Device Pairing

1. Find your laptop IP address (`ipconfig` on Windows).
2. Ensure phone and laptop are on the same Wi-Fi network.
3. Open the `/Android` folder in **Android Studio** and build APK to your phone.
4. Open app **Settings** tab $\rightarrow$ enter Laptop IP (e.g. `192.168.1.10`), Port `5000`, Client ID `android_client_01`.
5. Tap **TEST CONNECTION NOW** to verify connectivity.
6. Tap **TRIGGER FEDERATED UPDATE** to run real on-device gradient descent and stream telemetry to your laptop server.

---

## 📊 Running Baseline Benchmarks & Generating Figures

Run all 5 baseline comparative experiments (Centralized, FedAvg Fixed, FedAsync Fixed, FedAvg DQN, FedAsync DQN):
```bash
python Server/experiments.py
```

Generate all 13 research figures:
```bash
python Server/generate_plots.py
```
Figures will be saved in `Experiments/plots/`.
