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

![Demo Thumbnail](demo/demo_thumbnail.png)

▶️ [Watch the demo video](link will go here)

## 🧠 Why It Matters

ARCHON explores adaptive fault tolerance in hardware, where entropy-triggered responses allow hazard control to operate under noise, delay, and non-determinism — making it ideal for hybrid quantum-classical systems.

## 🧪 Simulations + Artifacts

Waveform screenshots + simulation logs included in `/waveforms` and `/simulation_logs`.

## 📫 Contact

Feel free to reach out if this aligns with your lab, startup, or research interests.
