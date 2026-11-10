# sg_ev_aggregator_rl

Reinforcement learning-based coordination of electric vehicle aggregators for grid service provision in Singapore.

An EV aggregator schedules fleet charging and discharging to balance electricity cost, battery degradation and travel needs while providing frequency regulation, peak shaving and demand response. RL agents are compared against rule-based and optimisation (MPC) baselines in the same simulator and with the same KPIs.

## Layout

The package follows the workflow of a virtual power plant: assets → forecasts → scheduling → dispatch → settlement.

| Module | Role |
| --- | --- |
| `sg_ev_rl.assets` | EV sessions, chargers, battery degradation |
| `sg_ev_rl.scenarios` | Session generators and price paths (currently a toy scenario) |
| `sg_ev_rl.forecast` | Arrival, price and load forecasts |
| `sg_ev_rl.market` | Singapore wholesale market data and rules (USEP, reserve, regulation) |
| `sg_ev_rl.scheduling` | Rule-based policies and MPC baselines |
| `sg_ev_rl.rl` | RL agents and the feasibility (safety) layer |
| `sg_ev_rl.dispatch` | Splitting aggregate setpoints into per-vehicle power |
| `sg_ev_rl.settlement` | Cost, revenue, degradation and service KPIs |
| `sg_ev_rl.sim` | Fleet simulator; policies propose power, the simulator enforces limits |
| `sg_ev_rl.api` | Service interface for dispatch decisions (planned) |

Experiment settings live in `configs/`.

## Setup

```bash
conda env create -f environment.yml   # creates env "sgev" and installs the package in editable mode
conda activate sgev
pre-commit install
pytest
python -m sg_ev_rl.demo --config configs/demo.yaml
```

## Data

Input data comes from the verified tables in the research workspace (`datasets/processed/`) and charging-occupancy snapshots collected by [SGPowerPage](https://github.com/zhichuanma/SGPowerPage). This repository reads those data and does not modify them. Set `SG_RESEARCH_ROOT` to point at the workspace; tests that need it skip when it is missing. All times are SGT (UTC+8).
