import sys
import os
server_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(server_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if server_dir not in sys.path:
    sys.path.append(server_dir)

import random
import logging
import sqlite3
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

import models
from AI.dqn_agent import DQNSchedulerAgent, compute_drl_reward
from Server.database import log_training_round, log_client_step, get_db_connection

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FederatedEdgeServer")

# Replay buffer path
DQN_MODEL_PATH = "AI/dqn_scheduler.pth"

class DiabetesDataset(Dataset):
    """Dataset wrapper for tabular diagnostic data."""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_diabetes_loaders(dataset_dir="Dataset"):
    """Load partitioned client datasets and centralized validation set."""
    client_loaders = []
    for i in range(5):
        csv_path = os.path.join(dataset_dir, f"client_{i}.csv")
        ds = DiabetesDataset(csv_path)
        client_loaders.append(DataLoader(ds, batch_size=16, shuffle=True))
        
    raw_path = os.path.join(dataset_dir, "diabetes_raw.csv")
    raw_ds = DiabetesDataset(raw_path)
    val_ds = torch.utils.data.Subset(raw_ds, range(min(200, len(raw_ds))))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    return client_loaders, val_loader

def get_evaluate_fn(model, val_loader, device="cpu"):
    """Centralized validation evaluation callback."""
    def evaluate(server_round, parameters, config):
        # Update parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        criterion = torch.nn.CrossEntropyLoss()
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
                
        loss /= len(val_loader.dataset)
        accuracy = correct / total
        return loss, {"accuracy": accuracy}
    return evaluate

# DQN Agent instance
dqn_agent = DQNSchedulerAgent(state_dim=5, action_dim=4)
if os.path.exists(DQN_MODEL_PATH):
    try:
        dqn_agent.model.load_state_dict(torch.load(DQN_MODEL_PATH))
        dqn_agent.update_target_network()
        logger.info("Loaded pre-trained DQN Scheduler model checkpoint.")
    except Exception as e:
        logger.warning(f"Could not load DQN model checkpoint: {e}")

class EdgeFederatedClient(fl.client.NumPyClient):
    """
    Simulated Android Edge device executing local tabular training scheduled by DQN.
    """
    def __init__(self, cid, trainloader, testloader, device="cpu"):
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.net = models.TabularNet().to(device)
        self.dqn_agent = dqn_agent
        
    def fit(self, parameters, config):
        if os.path.exists("cancel.flag"):
            raise Exception("Cancellation flag set by server.")
            
        # 1. Fetch current client status from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT battery, reliability, packet_drop, bandwidth FROM client_status WHERE cid = ?", (self.cid,))
        row = cursor.fetchone()
        conn.close()
        
        battery = row[0] if row else 100.0
        reliability = row[1] if row else 0.9
        packet_drop = row[2] if row else 0.1
        bandwidth = row[3] if row else 100.0
        
        # Simulate local CPU and RAM utilization
        cpu = random.uniform(10.0, 80.0)
        ram = random.uniform(20.0, 75.0)
        last_acc = float(config.get("last_accuracy", 0.5))
        
        # 2. Query DQN Agent for Action (Policy selection)
        state = [battery, bandwidth, reliability, cpu, last_acc]
        action = self.dqn_agent.act(state, epsilon=0.1)
        epochs = [0, 1, 3, 5][action]
        
        logger.info(f"Client {self.cid} DQN state: {state} -> Chosen epochs: {epochs}")
        
        if epochs == 0 or battery < 5.0:
            logger.warning(f"Client {self.cid} skipped training. Battery: {battery:.1f}%")
            # Log skipped log
            log_client_step(self.cid, battery, bandwidth, reliability, cpu, ram, 0, 0, 0.0, 0.0)
            return parameters, len(self.trainloader.dataset), {"accuracy": last_acc}

        # 3. Load global parameters
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        
        # 4. Perform Local Training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)
        self.net.to(self.device)
        self.net.train()
        
        for _ in range(epochs):
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
        # 5. Apply Energy Drains
        # Compute cost
        comp_drain = epochs * 0.5
        # Network transmission cost
        model_size_kb = sum(p.nbytes for p in self.net.parameters()) / 1024.0
        comm_drain = (model_size_kb / (bandwidth * reliability)) * 0.05
        
        # Simulating transmission drops with retries
        success = False
        attempts = 0
        total_comm_drain = 0.0
        
        while attempts < 5:
            attempts += 1
            total_comm_drain += comm_drain
            if random.random() >= packet_drop:
                success = True
                break
                
        if not success:
            logger.error(f"Client {self.cid} failed to upload parameters due to dropouts.")
            log_client_step(self.cid, max(0.0, battery - comp_drain - total_comm_drain), bandwidth, reliability, cpu, ram, epochs, 0, 0.0, comp_drain + total_comm_drain)
            raise Exception("Network transmission dropout timeout.")

        total_drain = comp_drain + total_comm_drain
        new_battery = max(0.0, battery - total_drain)
        
        # Update battery in client_status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE client_status SET battery = ? WHERE cid = ?", (new_battery, self.cid))
        conn.commit()
        conn.close()
        
        # 6. Log client steps
        log_client_step(self.cid, new_battery, bandwidth, reliability, cpu, ram, epochs, 0, model_size_kb, total_drain)
        
        # Compute reinforcement reward
        # Local evaluation
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.net(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy_gain = (correct / total) - last_acc
        
        reward = compute_drl_reward(state, action, accuracy_gain)
        next_state = [new_battery, bandwidth, reliability, cpu, correct / total]
        self.dqn_agent.remember(state, action, reward, next_state, False)
        
        # Perform target replay training iteration
        self.dqn_agent.replay(32)
        
        # Save model check weights
        updated_params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return updated_params, len(self.trainloader.dataset), {"accuracy": correct / total}

    def evaluate(self, parameters, config):
        # Local evaluation on client validation data
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.to(self.device)
        self.net.eval()
        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.net(X)
                loss += criterion(outputs, y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        loss /= len(self.testloader.dataset)
        accuracy = correct / total
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


class ResearchCustomFedAvg(fl.server.strategy.FedAvg):
    """
    Flower FedAvg implementation extending custom telemetry reporting.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_accuracy = 0.5

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        if fit_ins_list:
            for _, fit_ins in fit_ins_list:
                fit_ins.config["last_accuracy"] = self.last_accuracy
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Count total packets dropped in this round
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM client_logs WHERE epochs = 0") # failed drops
        drops = len(failures)
        conn.close()

        # Log training round status
        log_training_round(server_round, self.last_accuracy, 0.0, len(results), drops)
        
        # Train and save the DQN Agent at the end of the round
        torch.save(dqn_agent.model.state_dict(), DQN_MODEL_PATH)
        logger.info(f"DQN policy checkpoint saved to {DQN_MODEL_PATH}")
        
        return aggregated_parameters, metrics

    def evaluate(self, server_round, parameters):
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is not None:
            loss, metrics = eval_res
            self.last_accuracy = metrics.get("accuracy", 0.5)
            log_training_round(server_round, self.last_accuracy, loss, 5, 0)
        return eval_res


def run_edge_simulation(rounds=10):
    # Setup custom logger path
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
            
    file_handler = logging.FileHandler("simulation.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Initialize environment partitions
    client_loaders, val_loader = get_diabetes_loaders()
    
    # Global central validation model
    global_model = models.TabularNet()
    evaluate_fn = get_evaluate_fn(global_model, val_loader)
    
    # Initialize parameters
    init_params = ndarrays_to_parameters([val.cpu().numpy() for _, val in global_model.state_dict().items()])
    
    # Strategy setup
    strategy = ResearchCustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=5,
        min_available_clients=5,
        evaluate_fn=evaluate_fn,
        initial_parameters=init_params
    )
    
    def client_fn(cid: str):
        idx = int(cid)
        return EdgeFederatedClient(cid, client_loaders[idx], client_loaders[idx])
        
    logger.info("Initializing Flower Distributed Edge simulation...")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    logger.info("Flower Distributed Edge simulation complete.")

if __name__ == "__main__":
    run_edge_simulation(rounds=2)
