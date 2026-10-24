---
title: Autonomous Lunar Lander RL
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: DQN vs REINFORCE for safe, fuel-efficient lunar landing
---

# Reinforcement Learning Control Optimization for Autonomous Lander Systems

ML Lab (MACSE504) case study Q9, Assessments 8–10. This is an interactive Gradio demo that benchmarks a value-based agent (**DQN**) against a policy-gradient agent (**REINFORCE**) on Gymnasium `LunarLander-v3`. The goal is to land safely on the pad using as little fuel as possible.

## Run locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./run.sh
```
Then open http://localhost:7860

## What's in the app

| Tab | Covers | You can |
|---|---|---|
| Part A · Environment & Baseline | A.1–A.3 | Load the env, see the state, action and reward tables, run the random baseline (N episodes, seed), view trajectories and flight paths, and watch a random episode |
| Part B · Train Agents | B.1–B.3 | Train DQN or REINFORCE live with adjustable γ, learning-rate, ε and entropy schedules, plus batch, buffer and target-update settings. A progress card shows the timer and counters. Stop at any time, download the weights, or upload your own `.pt` |
| Part C · Watch & Evaluate | C.2, C.3 | Render a landing video for any agent, with optional sensor noise, gust or tilt kick, and measure landing success with a 95% CI on unseen seeds |
| Part C · Compare & Robustness | C.1–C.3 | Learning curves, a sample-efficiency and stability table, and a robustness sweep over perturbation levels |
| Inferences | C.3 | Findings and engineering inferences |

## Files

- `rl_core.py`: networks, schedules, agents, training and evaluation (same code as the notebook)
- `app.py`: Gradio interface
- `pretrain.py`: retrains the bundled agents with the notebook settings
- `models/`: trained weights (`dqn_lunarlander.pt`, `reinforce_lunarlander.pt`) and `part_b_training_logs.npz`, produced by running the notebook end to end
- `notebooks/`: the case-study notebook (Parts A, B and C) and the locally executed copy that produced `models/`

## Deployment

`.github/workflows/deploy-hf-space.yml` uploads the app to a Hugging Face Space on every push to `main`. It needs a repository secret `HF_TOKEN` (a Hugging Face token with write access). You can set the repository variable `HF_SPACE` to override the default Space id.
