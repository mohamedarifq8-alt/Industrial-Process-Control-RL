# 🎛️ Autonomous Industrial Tank Level Control: From Q-Learning to Deep Q-Networks (DQN)

An advanced industrial process control application built using **Reinforcement Learning** and **Deep RL** inside the `Gymnasium` framework. This project simulates a chemical fluid tank in a manufacturing plant, demonstrating a complete engineering transition from **Tabular Q-Learning** to **Deep Q-Networks (DQN)** to achieve high-precision, automated control.

---

<p align="center">
  <video src="1.mp4" width="100%" controls autocomplete="off"></video>
</p>

## 📌 Project Overview & Evolution

In industrial automation, maintaining critical variables (like fluid levels, pressure, or temperature) at a precise setpoint is vital. Traditional controllers (like PID) rely heavily on manual tuning. This project replaces them with an AI agent that learns purely through environmental interaction.

The repository is structured into two evolutionary phases:
1.  **Phase 1: Tabular Q-Learning:** A discrete environment solved using a standard lookup table.
2.  **Phase 2: Deep Q-Network (DQN):** An upgraded continuous environment featuring multi-sensor data fusion, fluid inertia, and momentum, solved using Deep Neural Networks.

---

## 🏗️ Phase 1: Tabular Q-Learning (Discrete Domain)

The first phase models a simplified factory setup where sensors and actuators operate on discrete integer boundaries.

### 🔌 Environment Architecture
*   **State Space (Sensors):** Discrete levels from `0` (Critical Empty) to `10` (Critical Overflow).
*   **Action Space (Actuators):** 3 Discrete operations: `0` (Drain / Shifting), `1` (Hold / Stop), and `2` (Fill / Tapping).
*   **Disturbance Simulation:** A 20% random probability of a 1-meter fluid leak per second due to evaporation.

### 🧠 Algorithmic Framework
The agent updates a `11 × 3` matrix (Q-Table) based on the **Bellman Equation**:
\[\text{New } Q(s,a) = Q(s,a) + \alpha \left[ R(s,a) + \gamma \max Q(s',a') - Q(s,a) \right]\]

*   **Reward Function:** Implements shaped rewards. Reaching the target (`Level 5`) gives `+1.0`. Violating boundaries (`0` or `10`) gives `-1.0` (System Termination). Any deviation triggers a step-by-step linear penalty: \(-0.1 \times \vert{}Current Level - 5\vert{}\).

---

## ⚡ Phase 2: Deep Q-Network (DQN) (Continuous & Inertial Domain)

To reflect real-world manufacturing, Phase 2 breaks the constraints of tabular methods by introducing continuous physical dynamics that lead to an infinite number of possible states—shattering the capacity of a standard Q-Table.

### 🚀 The Architectural Upgrades
1.  **Continuous States (High-Precision Sensors):** Ultrasonic sensor readings are float-point metrics (e.g., `5.432`m).
2.  **Multi-Sensor Fusion:** The state observation expands to a 2D space:
    *   **Sensor₁:** Current Fluid Level \([0.0, 10.0]\)
    *   **Sensor₂:** Flow Velocity \([-2.0, 2.0]\) (Tracking fluid momentum)
3.  **Inertia & Momentum Physics:** Actuators generate *acceleration (force)* instead of immediate integer level changes. Shutting off a pump doesn't freeze the liquid instantly; the fluid continues rushing forward due to system momentum, forcing the AI to develop **anticipatory control logic**.

### 🧠 Deep Learning Architecture & Stability (PyTorch)
The controller approximates optimal actions using a Multi-Layer Perceptron (MLP):
*   **Neural Network Structure:** An input layer (2 dimensions) followed by two hidden dense layers with `64` neurons each utilizing **ReLU** activations, mapping out to 3 raw output Q-values.
*   **Experience Replay Memory:** A cyclic buffer (`capacity=2000`) that samples random mini-batches (`size=32`) to break chronological data correlations.
*   **Dual Network Optimization:** Decouples policy picking from value targets using a active **Policy Network** alongside a periodically synchronized **Target Network** (every 10 episodes) to maintain stable gradients.

---

## 📊 Hyperparameters Comparison

| Hyperparameter | Phase 1: Q-Learning | Phase 2: DQN |
| :--- | :--- | :--- |
| **State Dimension** | 11 Discrete States | Continuous Vector `[Level, Velocity]` |
| **Optimizer / Algorithm** | Tabular Bellman Update | Adam (Learning Rate = 0.001) |
| **Discount Factor (\(\gamma\))** | 0.95 | 0.95 |
| **Exploration Strategy** | \(\epsilon\)-Greedy (\(\epsilon_{decay} = 0.001\) per episode) | \(\epsilon\)-Greedy (\(\epsilon_{decay} = 0.995\) per step) |
| **Training Duration** | 10,000 Episodes | 300 Episodes (Fast Neural Generalization) |

---

## 🖥️ Human-Machine Interface (HMI) Visualizations

Both environments feature a live text-based HMI layout inside the terminal using carriage returns (`\r`) to show fluid adjustments in real time:

```text
--- Live Terminal Monitoring (DQN Mode) ---
Shift Turn: Cycle 24/50
Controller Decision: Fill 🔺
Fluid Level: 4.85m | Flow Velocity: +0.45m/s | [🟦🟦🟦🟦|  🎯  |          ] 
```

---

## 🛠️ Installation & Execution

1. **Clone the project repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPOSITORY_NAME
   ```

2. **Install requirements:**
   ```bash
   pip install gymnasium torch numpy pandas ipython
   ```

3. **Run the Notebooks / Scripts:**
   *   Execute the tabular agent script to see the Q-Table training pipeline.
   *   Execute the deep agent script to witness the PyTorch network mastering continuous inertial physics.

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*
