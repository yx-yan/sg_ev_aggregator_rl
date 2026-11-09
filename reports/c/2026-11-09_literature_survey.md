# 文献调研：可对比的 EV 聚合商 RL 论文（2026-11-09）

## 一句话结论

我们的设计（RL 只定车队总功率，再用 least-laxity-first 分给单车）在文献里有先例，不是全新做法。所以创新点不能放在"聚合动作"上，而应该放在**有保证的安全层 + V2G + 调频 + 新加坡高波动电价**这个组合上。基准算法的选择在文献里很统一，可以直接照着搭。

## 1. 核实过的论文

"核实"指读过摘要或正文，出处、方法和数据来自原文。读不到的项写"未核实"。

| # | 论文 | 问题 | RL 和动作 | 对比的基准 | 数据和划分 | 代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Orfanoudakis et al., *EV2Gym*, IEEE T-ITS 2024, [10.1109/TITS.2024.3510945](https://doi.org/10.1109/TITS.2024.3510945)（[arXiv](https://arxiv.org/abs/2404.01849)） | V2G 仿真平台：功率跟踪、V2G 利润最大化 | 每个桩一个动作；用 SB3 跑了 PPO、SAC、TD3、DDPG 等 | AFAP（尽快充）、ALAP（尽晚充）、轮询、MPC、完全预知最优（Gurobi） | 荷兰 ElaadNL 车流分布、ENTSO-E 日前电价；划分未写明 | [开源](https://github.com/StavrosOrf/EV2Gym) |
| 2 | Yeh et al., *SustainGym*（EVChargingEnv），NeurIPS 2023 数据集与基准赛道（[OpenReview](https://openreview.net/forum?id=vZ9tA3o3hr)） | 单个充电网络，按碳排放调度充电，**无 V2G** | 每个桩一个动作；PPO、SAC（RLlib） | Greedy、Random、MPC、离线最优（[baselines.py](https://github.com/chrisyeh96/sustaingym/blob/main/sustaingym/algorithms/evcharging/baselines.py)） | ACN-Data（Caltech、JPL，2018–2021）；**按季节训练，换季节或换站点测试**，专门考察分布变化 | [开源](https://github.com/chrisyeh96/sustaingym)，需 MOSEK 许可 |
| 3 | Li, Sun, Chen, Ye, Low, Wierman, *Learning-Based Predictive Control via Real-Time Aggregate Flexibility*, IEEE TSG 2021, [10.1109/TSG.2021.3094719](https://doi.org/10.1109/TSG.2021.3094719)（[arXiv](https://arxiv.org/abs/2012.11261)） | 聚合商向系统运营商报告总灵活性 | SAC 学习灵活性信号；分配到单车用**固定规则**（LLF 等） | MPC、离线最优 | ACN-Data Caltech；**按时间顺序**：训练 2018-11-01 至 2019-12-01，测试 2019-12-02 至 2020-01-01；CAISO LMP | 未核实 |
| 4 | Kwon & Zhu, *Efficient Representation for EV Charging Station Operations using RL*, arXiv 2021/2022（[2108.03236](https://arxiv.org/abs/2108.03236)） | 单个充电站降低电费 | **只决定总充电功率，用 LLF 分给单车**；状态按 laxity 分组 | 另一种 LLF 降维方法（Wang et al.） | UC Davis 充电站 2019、ERCOT 电价；20 天训练 / 5 天测试 | 未提及 |
| 5 | Sales-Ortiz et al., *Efficient Trading of Aggregate Bidirectional EV Charging Flexibility with RL*, ACM e-Energy 2024, [10.1145/3632775.3661949](https://doi.org/10.1145/3632775.3661949) | 虚拟电厂在实时市场交易双向（V2G）灵活性 | **聚合 SAC**，动作被限制在保证单车需求可行的范围内；按比例公平分配 | 未核实（ACM 页面无法访问） | ElaadNL 充电记录、实时电价（据代码说明） | [开源](https://github.com/J27avier/EvCharge) |
| 6 | Xu, Luan, Zhao, Long, *Frequency Regulation Provision by EV Aggregator with Dynamic Balance between User Charging Anxiety and Aggregate Flexibility*, IEEE TSG 2025, [10.1109/TSG.2025.3570439](https://doi.org/10.1109/TSG.2025.3570439)（[作者录用稿](https://livrepository.liverpool.ac.uk/3192939/)） | 聚合商参与**调频**（PJM RegD），兼顾车主"电量焦虑" | 有界 PPO（BPPO），联合决定投标量和分配权重 | 固定权重、只顾焦虑、只顾灵活性、**LLF**、**EDF**（投标都用 BPPO） | ACN-Data Caltech 2019；PJM 电价和 RegD 信号 2023-12 至 2024-12；**随机抽 73 天（20%）做测试**，四季均衡 | 未提及 |
| 7 | Wan, Li, He, Prokhorov, *Model-Free Real-Time EV Charging Scheduling Based on DRL*, IEEE TSG 2019, [10.1109/TSG.2018.2879572](https://doi.org/10.1109/TSG.2018.2879572) | 单车充放电调度 | DQN + 电价特征提取网络 | 未核实 | 未核实 | 未核实 |
| 8 | Li, Wan, He, *Constrained EV Charging Scheduling Based on Safe DRL*, IEEE TSG 2020, [10.1109/TSG.2019.2955437](https://doi.org/10.1109/TSG.2019.2955437) | 同上，加约束 | 约束 MDP + 安全 DRL（是否为 CPO 未核实） | 未核实 | 未核实 | 未核实 |

## 2. 从文献里得到的四点

**(1) 基准算法很统一，可以照搬**

| 类别 | 文献常用 | 我们对应 |
| --- | --- | --- |
| 不控制 | AFAP（到了就满功率充） | B1 |
| 规则 | ALAP、轮询、LLF、EDF、Greedy | B2 电价阈值 + LLF；建议补 EDF |
| 优化 | MPC、完全预知最优 | B4 MPC、B3 LP |
| 标准 RL | PPO、SAC、TD3（多用 Stable-Baselines3） | D1 |

**(2) 我们的"聚合动作 + LLF"不是新东西**

#3、#4、#5、#6 都是聚合决策再分给单车。其中 #5 已经把动作限制在可行范围内，和我们的安全层思路接近。所以：
- 安全层要和 #5 的"有界动作"、#6 的 BPPO 明确比较；
- 我们的区别可以落在：同时处理 V2G 和调频的可行包络，并在新加坡的高波动电价下检验。

**(3) 数据划分有三种做法**

- 按时间顺序（#3）：最严格，最接近实际部署，和我们现在的划分一致。
- 随机抽天（#6）：训练和测试分布接近，结果更好看，但有时间泄漏。
- 按季节或站点做分布变化（#2）：专门检验泛化能力。

我们原计划的时间顺序划分有文献依据。#2 的做法正好支持"把高价期当作分布变化来报告"。

**(4) 调频信号在文献里来自 PJM**

#6 用的是 PJM 公开的 RegD 信号。新加坡没有公开的调频激活信号，我们只能用合成的激活曲线，这一点要在论文里写明（技术路线里已经这样计划）。

## 3. 建议

1. **仿真环境自己搭，基准照 EV2Gym 和 SustainGym 的配置配齐。** EV2Gym 是逐桩动作、荷兰数据、没有调频，直接改成新加坡的聚合环境，工作量不比自己写小。可以用它做一次交叉检查：同一组车流下，AFAP 和最优解的结果是否同一量级。
2. **基准名单**：AFAP、EDF、LLF、电价阈值 + LLF、MPC、完全预知 LP、PPO/SAC（惩罚项版），以及 #5 风格的"有界动作 SAC"作为最接近的已有方法。
3. **评价指标**跟文献对齐：成本或利润、未满足的离开电量（或车主满意率）、调频跟踪误差、约束违反次数、计算时间。

## 4. 还没做的

- #5 的基准和结果、#7 和 #8 的细节：需要原文。
- 没有覆盖到的方向：多智能体方法、配电网约束。和我们的保底版本关系不大，暂不展开。
- Qiu 2024 阅读指南、新加坡市场规则核查：第三个交付里的其他内容，还没做。

## 5. 需要原文的论文

2026-11-09 已整理到 `Research/papers/`，按 `作者_年份_主题.pdf` 命名，并建了 `papers/README.md` 索引和 `papers/notes/` 笔记。#2 SustainGym 因网站拦截没有下载到，需要手动下载。
