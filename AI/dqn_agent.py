import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNSchedulerNetwork(nn.Module):
    """
    Q-Network for scheduling federated learning training iterations on edge nodes.
    Input state: [battery_pct, bandwidth_mbps, latency_ms, cpu_pct, last_accuracy]
    Output action values: [Skip/0 epochs, 1 epoch, 3 epochs, 5 epochs]
    """
    def __init__(self, state_dim=5, action_dim=4):
        super(DQNSchedulerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNSchedulerAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNSchedulerNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DQNSchedulerNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def normalize_state(self, state):
        """
        Normalizes raw telemetry features to [0, 1] range for stability.
        State: [battery_pct, bandwidth_mbps, latency_ms/reliability, cpu_pct, last_accuracy]
        """
        battery = max(0.0, min(100.0, float(state[0]))) / 100.0
        bandwidth = max(0.1, min(100.0, float(state[1]))) / 100.0
        latency = max(0.0, min(1.0, float(state[2])))
        cpu = max(0.0, min(100.0, float(state[3]))) / 100.0
        accuracy = max(0.0, min(1.0, float(state[4])))
        return [battery, bandwidth, latency, cpu, accuracy]

    def act(self, state, epsilon=0.1):
        """
        Choose action based on state vector.
        Action: 0=Skip, 1=1 epoch, 2=3 epochs, 3=5 epochs
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        norm_s = self.normalize_state(state)
        state_t = torch.FloatTensor(norm_s).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())
        
    def remember(self, state, action, reward, next_state, done):
        norm_s = self.normalize_state(state)
        norm_ns = self.normalize_state(next_state)
        self.memory.append((norm_s, action, reward, norm_ns, done))
        
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        self.model.train()
        q_values = self.model(states_t).gather(1, actions_t).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_model(next_states_t).max(1)[0]
            targets = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

# Configurable DRL Multi-Objective Reward Weights
REWARD_WEIGHTS = {
    "w_accuracy": 15.0,    # Weight for accuracy gain
    "w_energy": 2.5,       # Weight for battery energy drain penalty
    "w_latency": 1.0,      # Weight for network latency/signal penalty
    "w_cpu": 0.5,          # Weight for CPU load penalty
    "low_battery_penalty": 5.0 # Extra penalty for heavy training when battery < 20%
}

def compute_drl_reward(state, action, accuracy_gain, weights=None):
    """
    Multi-Objective Reward Function Formulation:
    State: [battery, bandwidth, latency/reliability, cpu, accuracy]
    Action: 0=Skip (0 epochs), 1=1 epoch, 2=3 epochs, 3=5 epochs
    
    Formula:
    R = w_acc * Delta_Acc - w_energy * Energy_Drain - w_latency * Latency_Cost - w_cpu * CPU_Cost - Battery_Critical_Penalty
    """
    if weights is None:
        weights = REWARD_WEIGHTS
        
    battery, bandwidth, reliability, cpu, _ = state
    epochs = [0, 1, 3, 5][action]
    
    # 1. Performance Reward
    perf_reward = float(accuracy_gain) * weights["w_accuracy"]
    
    # 2. Skip Action Policy
    if action == 0:
        # If battery is low (< 25%), reward skipping to preserve battery; if battery high, minor opportunity cost penalty
        if battery < 25.0:
            return 2.0
        return -0.2
        
    # 3. Energy Drain Penalty
    energy_cost = (epochs * 0.4) + (0.1 if bandwidth < 5.0 else 0.02)
    energy_penalty = (energy_cost / 10.0) * weights["w_energy"]
    
    # 4. Latency / Bandwidth Penalty
    net_cost = (1.0 / max(0.1, bandwidth)) if bandwidth < 10.0 else 0.05
    network_penalty = net_cost * weights["w_latency"]
    
    # 5. CPU Thermal Penalty
    cpu_penalty = (cpu / 100.0) * weights["w_cpu"]
    
    # 6. Low Battery Critical Safety Constraint
    low_batt_penalty = 0.0
    if battery < 20.0 and epochs >= 3:
        low_batt_penalty = weights["low_battery_penalty"]
        
    reward = perf_reward - energy_penalty - network_penalty - cpu_penalty - low_batt_penalty
    return float(reward)

if __name__ == "__main__":
    print("Testing DQNSchedulerAgent initialization...")
    agent = DQNSchedulerAgent()
    
    test_state = [95.0, 50.0, 0.9, 20.0, 0.72]
    action = agent.act(test_state, epsilon=0.0)
    print(f"Decided action for test state {test_state}: {action} (epochs: {[0, 1, 3, 5][action]})")
    
    reward = compute_drl_reward(test_state, action, accuracy_gain=0.05)
    print(f"Computed DRL reward: {reward:.4f}")

