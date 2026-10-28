# sg_ev_aggregator_rl

Reinforcement learning-based coordination of electric vehicle aggregators for grid service provision in Singapore.

An EV aggregator schedules fleet charging and discharging to balance electricity cost, battery degradation and travel needs while providing frequency regulation, peak shaving and demand response. RL agents are compared against rule-based and optimisation baselines.

## Status

Project set up on 2026-10-28; no code yet.

## Data

Input data comes from the verified tables in the research workspace (`datasets/processed/`) and charging-occupancy snapshots collected by [SGPowerPage](https://github.com/zhichuanma/SGPowerPage). This repository reads those data and does not modify them. All times are SGT (UTC+8).
