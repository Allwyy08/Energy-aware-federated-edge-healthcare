# Deep Q-Network (DQN) Scheduler: Multi-Objective Reward Design

**Project**: AI-Driven Energy-Aware Federated Edge Learning Framework  
**Document**: DRL Reward Specification  

---

## 1. Overview & Problem Formulation

In low-connectivity edge environments (such as forest borders and rural health camps), mobile IoT edge devices have dynamic hardware and networking constraints:
* Fluctuating battery levels
* Intermittent Wi-Fi / cellular bandwidth
* Varying network latency & signal drops
* Thermal CPU load limits

To prevent device battery exhaustion while optimizing global Federated Learning model accuracy, a **Deep Q-Network (DQN) Scheduler** dynamically selects the local training epoch configuration for each device prior to round execution.

---

## 2. State & Action Spaces

### State Vector $\mathbf{S}_t \in \mathbb{R}^5$
$$\mathbf{S}_t = \Big[ \text{BatteryPct},\; \text{BandwidthMbps},\; \text{ReliabilityScore},\; \text{CpuUsagePct},\; \text{LastAccuracy} \Big]$$

#### State Normalization:
$$\hat{\mathbf{S}}_t = \left[ \frac{\text{Battery}}{100},\; \min\left(1, \frac{\text{Bandwidth}}{100}\right),\; \text{Reliability},\; \frac{\text{CPU}}{100},\; \text{Accuracy} \right]$$

### Action Space $\mathcal{A} = \{0, 1, 2, 3\}$
| Action Index | Local Epochs | Description | Target Edge Condition |
| :---: | :---: | :--- | :--- |
| `0` | **0 Epochs (Skip)** | Skips local training completely | Battery $< 20\%$ or severe network outage |
| `1` | **1 Epoch** | Light training update | Low battery ($20\% - 45\%$) or poor bandwidth |
| `2` | **3 Epochs** | Moderate training update | Normal operating conditions |
| `3` | **5 Epochs** | Intensive local gradient descent | High battery ($>70\%$), fast Wi-Fi, low CPU load |

---

## 3. Multi-Objective Reward Function Formulation

The reward function $R(\mathbf{S}_t, a_t)$ balances model convergence against resource expenditure:

$$R(\mathbf{S}_t, a_t) = w_1 \cdot \Delta\text{Accuracy} - w_2 \cdot \text{EnergyDrain} - w_3 \cdot \text{LatencyCost} - w_4 \cdot \text{CpuCost} - \Omega_{\text{Critical}}$$

### Mathematical Components:

1. **Performance Reward ($\Delta\text{Accuracy}$)**:
   $$\text{PerfReward} = w_1 \cdot \Delta\text{Accuracy}$$
   Rewards parameter updates that improve global model validation accuracy.

2. **Energy Expenditure Penalty ($\text{EnergyDrain}$)**:
   $$\text{EnergyCost} = E(a_t) \cdot 0.4 + \mathbb{I}(\text{Bandwidth} < 5.0) \cdot 0.1$$
   $$\text{EnergyPenalty} = w_2 \cdot \left(\frac{\text{EnergyCost}}{10.0}\right)$$
   Penalizes high local compute loops that drain mobile battery power.

3. **Network Latency & Transmission Penalty ($\text{LatencyCost}$)**:
   $$\text{NetCost} = \frac{1.0}{\max(0.1, \text{BandwidthMbps})}$$
   $$\text{NetworkPenalty} = w_3 \cdot \text{NetCost}$$
   Penalizes parameter transmission over slow or congested channels.

4. **CPU Thermal Load Penalty ($\text{CpuCost}$)**:
   $$\text{CpuPenalty} = w_4 \cdot \left(\frac{\text{CpuUsagePct}}{100.0}\right)$$
   Discourages spawning intensive training on devices already suffering from high CPU load.

5. **Critical Battery Safety Constraint ($\Omega_{\text{Critical}}$)**:
   $$\Omega_{\text{Critical}} = \begin{cases} 5.0 & \text{if } \text{Battery} < 20\% \text{ and } E(a_t) \ge 3 \\ 0.0 & \text{otherwise} \end{cases}$$
   Imposes a severe penalty if the DQN attempts to assign intensive training (3 or 5 epochs) when device battery is critically low ($<20\%$).

---

## 4. Default Reward Coefficients

| Parameter | Symbol | Default Value | Rationale |
| :--- | :---: | :---: | :--- |
| **Accuracy Weight** | $w_1$ | `15.0` | Primary driver for global model convergence. |
| **Energy Weight** | $w_2$ | `2.5` | Strong penalty to conserve edge node battery life. |
| **Latency Weight** | $w_3$ | `1.0` | Penalty for slow uplink transmission times. |
| **CPU Load Weight** | $w_4$ | `0.5` | Moderate penalty to prevent thermal throttling. |
| **Low Battery Penalty** | $\Omega_{\text{Critical}}$ | `5.0` | Hard safety constraint against battery exhaustion. |

---

## 5. Experience Replay & Policy Execution

* **Exploitation Epsilon**: $\epsilon = 0.05$ during live deployment inference.
* **Replay Memory Buffer**: 2,000 transitions.
* **Batch Size**: 32 samples trained on MSE target $Q(\mathbf{S}, a) \approx r + \gamma \max_{a'} Q(\mathbf{S}', a')$.
* **Checkpoint Persistence**: Checkpoints saved automatically to `AI/dqn_scheduler.pth`.
