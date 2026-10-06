# System Architecture & Data Flow

This document details the software architecture, class relations, and data flow of the **AI-Driven Energy-Aware Federated Edge Learning Framework**.

---

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Cloud/Laptop Edge Server
        DB[(federated_edge.db)]
        FastAPI[FastAPI REST API Server]
        Flower[Flower Server Orchestrator]
        DQN_Agent[DQN Policy Neural Network]
        Dashboard[HTML5 Glassmorphism UI]
    end

    subgraph Android IoT Client Nodes
        Room[(Local Room DB)]
        ComposeUI[Jetpack Compose Dashboard]
        Sensors[Device Monitor Sensors]
        TFLite[TFLite Local Gradient Trainer]
        Worker[WorkManager Background Task]
    end

    %% Data Exchanges
    FastAPI <--> DB
    Flower <--> FastAPI
    Dashboard <--> FastAPI
    
    Worker <--> ComposeUI
    Worker <--> Room
    Worker <--> Sensors
    Worker <--> TFLite

    %% Networks
    Worker -- "1. POST /api/register" --> FastAPI
    Worker -- "2. GET /api/model (Download Weights)" --> FastAPI
    Worker -- "3. POST /api/dqn-action (State -> Action)" --> FastAPI
    Worker -- "4. POST /api/upload-stats (Telemetry + Weights)" --> FastAPI
```

---

## 2. Data Flow Diagram (DFD Level 1)

```mermaid
sequenceDiagram
    autonumber
    participant Client as Android Edge Node
    participant DB as SQLite Local Room DB
    participant Server as FastAPI Edge Server
    participant DQN as DQN Policy Scheduler
    participant Agg as Flower Server Aggregator

    Client->>DB: Fetch Local Diagnostic Patient Datasets
    DB-->>Client: Return Labeled Features
    Client->>Server: POST /api/register (Update battery & connection strength status)
    Client->>Server: POST /api/dqn-action (Query policy schedule configuration)
    Server->>DQN: Feed State [Battery, CPU, Bandwidth, Accuracy]
    DQN-->>Server: Return Action [Epochs, Compression flag]
    Server-->>Client: Return action instructions
    Client->>Client: Execute local gradient descent updates via TFLite (epochs times)
    Client->>Server: POST /api/upload-stats (Transmit updated parameters + metrics logs)
    Server->>Agg: Aggregate Client Updates
    Agg-->>Server: Save Model Checkpoint & Update global weights
```
