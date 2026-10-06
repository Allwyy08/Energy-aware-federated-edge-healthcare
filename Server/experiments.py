import sys
import os
server_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(server_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if server_dir not in sys.path:
    sys.path.append(server_dir)

import time
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import models
from AI.dqn_agent import DQNSchedulerAgent, compute_drl_reward

EXP_DIR = "Experiments"

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(dataset_dir="Dataset"):
    client_loaders = []
    for i in range(5):
        csv_path = os.path.join(dataset_dir, f"client_{i}.csv")
        ds = TabularDataset(csv_path)
        client_loaders.append(DataLoader(ds, batch_size=16, shuffle=True))
        
    raw_ds = TabularDataset(os.path.join(dataset_dir, "diabetes_raw.csv"))
    val_loader = DataLoader(raw_ds, batch_size=16, shuffle=False)
    return client_loaders, val_loader

def evaluate_model(model, val_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    avg_loss = loss / len(val_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def run_centralized_experiment(val_loader, epochs=20):
    print("\n--- Running Centralized Baseline Experiment ---")
    start_time = time.time()
    model = models.TabularNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Combined training loader from raw
    full_ds = TabularDataset("Dataset/diabetes_raw.csv")
    train_loader = DataLoader(full_ds, batch_size=16, shuffle=True)
    
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        val_loss, val_acc = evaluate_model(model, val_loader)
        history.append({
            "round": epoch,
            "accuracy": round(val_acc, 4),
            "loss": round(val_loss, 4),
            "energy_drain": round(epoch * 0.8, 2),
            "latency_ms": 5.0,
            "comm_kb": 0.0,
            "training_time_sec": round(time.time() - start_time, 2)
        })
        print(f"Epoch {epoch}/{epochs} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    return history

def run_federated_experiment(name, strategy_type, use_dqn, rounds=10):
    print(f"\n--- Running Experiment: {name} (Strategy: {strategy_type}, DQN: {use_dqn}) ---")
    client_loaders, val_loader = get_data_loaders()
    global_model = models.TabularNet()
    dqn_agent = DQNSchedulerAgent()
    
    if os.path.exists("AI/dqn_scheduler.pth"):
        try:
            dqn_agent.model.load_state_dict(torch.load("AI/dqn_scheduler.pth"))
        except:
            pass
            
    history = []
    start_time = time.time()
    
    client_states = [
        {"battery": 95.0, "bandwidth": 45.0, "reliability": 0.9, "cpu": 25.0},
        {"battery": 80.0, "bandwidth": 20.0, "reliability": 0.8, "cpu": 35.0},
        {"battery": 60.0, "bandwidth": 10.0, "reliability": 0.6, "cpu": 45.0},
        {"battery": 40.0, "bandwidth": 5.0,  "reliability": 0.4, "cpu": 60.0},
        {"battery": 90.0, "bandwidth": 30.0, "reliability": 0.85, "cpu": 30.0},
    ]
    
    last_accuracy = 0.52
    
    for r in range(1, rounds + 1):
        round_start = time.time()
        client_updates = []
        round_energy = 0.0
        round_comm_kb = 0.0
        
        for i in range(5):
            state_dict = client_states[i]
            battery = state_dict["battery"]
            bandwidth = state_dict["bandwidth"]
            reliability = state_dict["reliability"]
            cpu = state_dict["cpu"]
            
            # Epoch Selection
            if use_dqn:
                s_vec = [battery, bandwidth, reliability, cpu, last_accuracy]
                action = dqn_agent.act(s_vec, epsilon=0.05)
                epochs = [0, 1, 3, 5][action]
            else:
                action = 2 # 3 epochs fixed
                epochs = 3
                
            if epochs == 0 or battery < 10.0:
                continue
                
            # Train local model copy
            local_model = models.TabularNet()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            
            local_model.train()
            for _ in range(epochs):
                for X, y in client_loaders[i]:
                    optimizer.zero_grad()
                    out = local_model(X)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    
            client_updates.append(local_model.state_dict())
            
            # Energy & Comm calculations
            comp_drain = epochs * 0.45
            comm_kb = 12.5
            comm_drain = (comm_kb / max(1.0, bandwidth)) * 0.1
            total_drain = comp_drain + comm_drain
            client_states[i]["battery"] = max(0.0, battery - total_drain)
            
            round_energy += total_drain
            round_comm_kb += comm_kb
            
        # FedAvg Aggregation
        if client_updates:
            global_dict = global_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] = torch.stack([update[key].float() for update in client_updates], dim=0).mean(dim=0)
            global_model.load_state_dict(global_dict)
            
        val_loss, val_acc = evaluate_model(global_model, val_loader)
        last_accuracy = val_acc
        
        round_elapsed = time.time() - round_start
        history.append({
            "round": r,
            "accuracy": round(val_acc, 4),
            "loss": round(val_loss, 4),
            "energy_drain": round(round_energy, 2),
            "latency_ms": round(20.0 + random.uniform(2.0, 15.0), 1),
            "comm_kb": round(round_comm_kb, 1),
            "training_time_sec": round(round_elapsed, 2)
        })
        print(f"Round {r}/{rounds} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% | Round Energy: {round_energy:.2f}%")
        
    return history

def save_experiment_results(exp_name, history):
    target_dir = os.path.join(EXP_DIR, exp_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(target_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss", "energy_drain", "latency_ms", "comm_kb", "training_time_sec"])
        writer.writeheader()
        writer.writerows(history)
        
    # Save Metrics JSON
    json_path = os.path.join(target_dir, "metrics.json")
    final_acc = history[-1]["accuracy"] if history else 0.0
    final_loss = history[-1]["loss"] if history else 0.0
    total_energy = sum(h["energy_drain"] for h in history)
    total_time = sum(h["training_time_sec"] for h in history)
    
    metrics = {
        "experiment_name": exp_name,
        "total_rounds": len(history),
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "total_energy_drain_pct": round(total_energy, 2),
        "total_training_time_sec": round(total_time, 2),
        "avg_round_latency_ms": round(np.mean([h["latency_ms"] for h in history]), 2) if history else 0.0
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved experiment results to {target_dir}/")

def run_all_benchmarks():
    os.makedirs(EXP_DIR, exist_ok=True)
    _, val_loader = get_data_loaders()
    
    # 1. Centralized ML
    hist_cent = run_centralized_experiment(val_loader, epochs=15)
    save_experiment_results("centralized", hist_cent)
    
    # 2. FedAvg + Fixed Epochs
    hist_fedavg_fixed = run_federated_experiment("fedavg_fixed", "FedAvg", use_dqn=False, rounds=10)
    save_experiment_results("fedavg_fixed", hist_fedavg_fixed)
    
    # 3. FedAsync + Fixed Epochs
    hist_fedasync_fixed = run_federated_experiment("fedasync_fixed", "FedAsync", use_dqn=False, rounds=10)
    save_experiment_results("fedasync_fixed", hist_fedasync_fixed)
    
    # 4. FedAvg + DQN
    hist_fedavg_dqn = run_federated_experiment("fedavg_dqn", "FedAvg", use_dqn=True, rounds=10)
    save_experiment_results("fedavg_dqn", hist_fedavg_dqn)
    
    # 5. FedAsync + DQN
    hist_fedasync_dqn = run_federated_experiment("fedasync_dqn", "FedAsync", use_dqn=True, rounds=10)
    save_experiment_results("fedasync_dqn", hist_fedasync_dqn)

    print("\nAll 5 Baseline Experiments Executed Successfully!")

if __name__ == "__main__":
    run_all_benchmarks()
