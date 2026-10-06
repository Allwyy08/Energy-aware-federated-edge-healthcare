# Network Testing & Latency Diagnostic Specification

**Project**: AI-Driven Energy-Aware Federated Edge Learning Framework  
**Document**: Active Latency RTT & Throughput Measurement Protocol  

---

## 1. Overview

In low-connectivity environments, edge devices face fluctuating network conditions (latency spikes, bandwidth throttling, packet dropouts). This framework implements active network testing protocols to collect real measured networking metrics between the Android client and the laptop server.

---

## 2. Latency Measurement Protocol (RTT)

### Endpoint: `GET /api/ping`
* **Method**: HTTP GET
* **Payload**: Lightweight JSON response
  ```json
  {
    "status": "ok",
    "timestamp": 1789984210.12,
    "server": "FastAPI Edge FL Server",
    "network_profile": "Good"
  }
  ```
* **Calculation**:
  $$\text{RTT Latency (ms)} = T_{\text{response\_received}} - T_{\text{request\_sent}}$$
* **Storage**: Stored locally on Android and sent to server database table `telemetry(latency_ms)`.

---

## 3. Bandwidth Throughput Protocol

### Downstream Test: `GET /api/network-test/download?size_kb=256`
* Server returns 256 KB buffer payload.
* Android calculates:
  $$\text{Download Mbps} = \frac{\text{BytesTransferred} \times 8}{1,000,000 \times \text{DurationSec}}$$

### Upstream Test: `POST /api/network-test/upload`
* Android sends payload buffer to server.
* Server calculates received duration and returns:
  ```json
  {
    "received_bytes": 262144,
    "duration_sec": 0.1245,
    "upload_mbps": 16.84
  }
  ```

---

## 4. Experimental Network Profile Profiles

The server includes a network profile throttle controller (`/api/network-profile`) to simulate constrained field conditions during testing:

| Profile | Bandwidth Limit | Simulated Latency | Packet Loss % | Target Field Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Good** | $>30.0$ Mbps | $<30$ ms | $2\%$ | Urban WiFi / Base Station |
| **Medium** | $10.0 - 30.0$ Mbps | $30 - 80$ ms | $5\%$ | Hill Station Slopes |
| **Poor** | $2.0 - 10.0$ Mbps | $80 - 250$ ms | $20\%$ | Forest Edge Clearing |
| **Intermittent** | $<2.0$ Mbps | $>250$ ms | $40\%$ | Deep Valley Forest Canopy |
