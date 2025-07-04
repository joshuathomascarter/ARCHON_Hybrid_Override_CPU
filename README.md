# ARCHON: Hybrid Analog–ML–Quantum Override CPU

ARCHON is a next-generation instruction pipeline that merges classical FSM hazard control with entropy-aware override systems sourced from analog signal conditioning, quantum entropy measurements, and ML-based classifiers.

## 🔍 Overview

This system includes:
- A 5-stage pipelined CPU (`pipeline_cpu.v`)
- A Finite State Machine (`fsm_entropy_overlay.v`) with entropy-triggered adaptive control
- Analog and quantum override inputs
- Entropy decoding logic
- Simulation testbench with waveform outputs

## 📂 Repository Structure

- `control_unit.v' — completed verilog module
- `fsm_entropy_overlay.v` — FSM with override detection
- `archon_top_testbench.v` — Full testbench for simulation
- `docs/` — FSM diagrams, writeups, architectural diagrams
- `demo/` — Live walkthrough and waveform logs

## 📸 Demo Preview
![video Banner](https://github.com/user-attachments/assets/13a63ee1-5871-4d97-89d0-d2ed6ff4bc94)

▶️ [Watch the demo video](link will go here)

## 🧠 Why It Matters

ARCHON explores adaptive fault tolerance in hardware, where entropy-triggered responses allow hazard control to operate under noise, delay, and non-determinism — making it ideal for hybrid quantum-classical systems.

## 🧪 Simulations + Artifacts

Waveform screenshots + simulation logs included in `/waveforms` and `/simulation_logs`.

📬 Want to collaborate?  
I'm looking for lab/startup partners for Fall 2025.  
Reach out via [LinkedIn](https://www.linkedin.com/in/joshua-carter-898868356/) or [joshtcarter0710@gmail.com]

