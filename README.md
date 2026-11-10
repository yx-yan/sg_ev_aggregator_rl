# sg_ev_aggregator_rl

[![CI](https://github.com/yx-yan/sg_ev_aggregator_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/yx-yan/sg_ev_aggregator_rl/actions/workflows/ci.yml)

Reinforcement learning-based coordination of electric vehicle aggregators for grid service provision in Singapore (NTU dissertation, work in progress).

An EV aggregator schedules fleet charging and discharging to balance electricity cost, battery degradation and travel needs while providing frequency regulation, peak shaving and demand response. RL agents are compared against rule-based and optimisation (MPC) baselines in the same simulator and with the same KPIs.

## Approach

- **Aggregate action, rule-based dispatch.** The agent sets one fleet-level power setpoint per 30-minute market period; a dispatch layer (least-laxity-first) splits it across vehicles. This keeps the action space independent of fleet size.
- **Safety layer.** Policies propose power and the simulator enforces charger and battery limits. A feasibility layer for the RL agents (planned) will also keep each vehicle's departure energy reachable.
- **Singapore market data.** Prices, reserve and regulation come from the half-hourly Uniform Singapore Energy Price (USEP) data published by EMC, with a chronological train / validation / test split.
- **Baselines from the literature.** Uncontrolled (charge on arrival), EDF and LLF rules, price-threshold + LLF, MPC and a perfect-foresight LP, against PPO / SAC agents.

## Status

| Part | State |
| --- | --- |
| Package layout, fleet simulator, rule baselines, settlement, CI | Done |
| Market data loader with raw-file cross-check (0 mismatches across 79,488 cells) | Done |
| USEP price diagnosis and data split | Done |
| V2G arbitrage and regulation revenue screen | Done |
| Literature survey of comparable EV aggregator RL papers | Done |
| Synthetic EV session generator calibrated to Singapore statistics | In progress |
| MPC / LP baselines | Planned |
| RL agents with safety layer | Planned |
| Dispatch service API | Planned |

## Findings so far

- **The price regime shifted mid-dataset.** USEP rose from a median of about S$100/MWh in the training period to about S$188/MWh in the test period. The March 2026 step matches EMC's fuel-price explanation; the cause of the spikes from July 2026 is not yet known. Agents therefore use relative price features, and results are reported separately for spike and normal weeks.
- **V2G arbitrage is thin at training-period prices.** With perfect foresight it is worth roughly S$100–300 per vehicle per year for home or workplace plug-in windows. On more than half of all days it earns nothing, and 10% of days provide most of the profit. Battery degradation cost matters more than round-trip efficiency.
- **Regulation capacity is the larger revenue stream** (scenario estimate about S$440–550 per vehicle per year), and regulation prices did not follow the energy price rise.

Details, figures and tables are in [`reports/c/`](reports/c/); the write-ups are in Chinese.

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

The demo runs one synthetic day for 50 vehicles and needs no external data:

```text
                  energy_kwh  energy_cost_sgd  avg_price_sgd_mwh  peak_kw  unmet_kwh
uncontrolled           979.8           184.83             188.64   256.21        0.0
cheapest_periods       979.8           183.55             187.34   294.52        0.0
```

## Data

Input data comes from the verified tables in the research workspace (`datasets/processed/`) and charging-occupancy snapshots collected by [SGPowerPage](https://github.com/zhichuanma/SGPowerPage). This repository reads those data and does not modify them. The raw data is not redistributed here. Set `SG_RESEARCH_ROOT` to point at the workspace; tests that need it skip when it is missing. All times are SGT (UTC+8).

## License

Code is released under the [MIT License](LICENSE). Market data remains subject to the terms of its original publishers (EMC, EMA).
