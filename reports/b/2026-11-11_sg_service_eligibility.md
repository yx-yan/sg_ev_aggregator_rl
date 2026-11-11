# 新加坡 EV 聚合商能参与哪些电网服务（2026-11-11 核查）

## 一句话结论

**今天 EV 聚合负荷不能参与调频。** 新加坡批发市场还没有允许分布式资源聚合后提供能量和辅助服务的框架。调频只能由注册为发电设施（GRF）的单元提供，EMA 2024 年提出的虚拟电厂（VPP）沙盒是唯一的试点通道。用电侧今天就能参与的是 **需求响应（DR）** 和 **可中断负荷（IL，按应急备用价付费）**，而且 EMA 已经明确把 EV 充电站纳入 DR。

## 关键事实（原文摘录）

| 主题 | 原文 | 出处 |
| --- | --- | --- |
| 聚合资源参与辅助服务 | "There is currently no regulatory framework allowing the aggregation of DERs via a VPP to provide energy and ancillary services in the Singapore Wholesale Electricity Market (SWEM)." | [1] 第 4 段 |
| VPP 沙盒的注册方式 | "each VPP shall aggregate DERs and register as a single Generation Registered Facility (GRF)"；可聚合 "any generation assets and load facilities" | [1] 第 5、6(b) 段 |
| VPP 必须提供的服务 | "The VPP shall provide energy and all ancillary services (frequency regulation, and primary and contingency reserves)." | [1] 第 6(c) 段 |
| 沙盒规模和期限 | 总容量 100 MW，期限一年，EMA 计划选 "two or three sizeable projects"，征集截止 2024-12-31 | [1] 第 6(a)(d)、13、14 段 |
| 调频的技术要求 | "must be able to increase or decrease its output in 60 seconds"；与调度 EMS 通信要用 "pilot wire or optical fibre"（可提议无线方案） | [1] 表 1、第 7–8 段 |
| 只有 FTP 通信时 | "the VPP shall participate only in energy and reserves market"（不能做调频） | [1] 脚注 6 |
| DR 纳入 EV 充电站 | "EMA expanded the DR programme to allow EV charging stations and BESS with nameplate rating below 10 megawatts (MW) to participate" | [2] 第 7 段 |
| EV 充电 DR 试点 | 与 ComfortDelGro 开展 "a 13-month regulatory sandbox to pilot the participation of EV charging stations … in the DR programme" | [2] 第 7(i) 段 |
| IL 激活时长 | "EMA plans to reduce each IL activation period to 30 minutes" | [2] 第 3 段 |
| DR 和 IL 的参与门槛 | 通过售电商或 "a DR Aggregator" 参与；直接参与 NEMS 需 "at least 0.1 megawatt (MW)" | [3] 第 1、3 页 |
| DR 的付费方式 | "one-third of the savings arising from the reduction in electricity prices"，上限 S$4,500/MWh | [3] 第 2 页 |
| IL 的付费方式 | "paid the clearing price for contingency reserves every period to be on standby"；"No additional payment is given out in the event of an activation." | [3] 第 3 页 |
| DR 合规要求 | 只有被调度且 100% 完成才付费；完成率低于 80% 要罚款 | [4] 第 3.1 段 |
| IL 规则（2025 起） | 恢复原框架：激活时长不设上限，不合规就不付该时段的备用费 | [4] 表 3 |
| 单辆 EV / V2G | "Individual EVs cannot meet 0.1MW minimum offer requirement"；"Individual EV unlikely to meet PSO's technical requirements for reserve/regulation provision"；V2G 需要一个中间储能注册为 GRF | [5] 第 16 页 |
| V2G 现状（2020） | "we yet to have V2G providers participating in the market today" | [5] 第 8 页 |

## 对论文的影响

1. **调频只能作为"未来情景"**：前提是 VPP 框架落地。而且按沙盒条件，VPP 要同时提供能量、调频和两类备用，60 秒内响应，还要专线通信。100 个公共端口平均只有约 0.09 MW 的调频容量，远小于"sizeable projects"的量级。
2. **今天可行的服务**：
   - **可中断负荷（IL）**：EV 充电负荷待命、需要时减少用电，按应急备用（contingency reserve）出清价付待命费。市场表里已经有 `reserve_price_CONRES*` 列，**有数据支撑**。只充电的车队也能做（减少充电），不需要 V2G。最小规模 0.1 MW，可以通过聚合商凑够。
   - **需求响应（DR）**：电价高时减少用电，拿系统节省的三分之一。EV 充电站已经明确可以参与，而且有 ComfortDelGro 的试点。但付费取决于对市场出清价格的影响，需要另外建模。
3. **V2G**：没有商业部署。2020 年的设想是通过中间储能注册为 GRF，仍然只能作为情景。

## 待核实

- `reserve_price_CONRESA`–`E`、`PRIRESA`–`E` 分别对应哪类备用、哪个价格组（EMC 原始文件说明）。
- VPP 沙盒的最终决定和入选项目（只找到 2024-10-21 的征询稿，没找到结果）。
- IL 激活缩短到 30 分钟的实施时间。

## 来源

1. EMA, *Harnessing Distributed Energy Resources via Virtual Power Plants to Provide Energy and Ancillary Services in the SWEM — Consultation Paper and Call for Regulatory Sandbox Proposals*, 21 Oct 2024. https://ema.gov.sg/content/dam/corporate/partnerships/consultations/harnessing-ders-via-vpps-to-provide-energy-and-ancillary-services/consultation/EMA-Consultations-Consultation-Harnessing-DERSs-via-VPPs-to-provide-energy-and-ancillary-services-20241021.pdf.coredownload.pdf
2. EMA, *Media Release: New Initiatives to Bolster Power Grid's Reliability Through Flexible Electricity Demand*, 29 Oct 2025. https://www.ema.gov.sg/content/dam/corporate/news/media-releases/2025/20251029-Media-Release-New-Initiatives-to-Bolster-Power-Grid-Reliability.pdf.coredownload.pdf
3. EMA, *Factsheet: Enhancing the DR and IL Programmes with a Demand Side Management Sandbox*, as at 3 Nov 2022. https://www.ema.gov.sg/content/dam/corporate/our-energy-story/energy-demand/factsheet-demand-response-interruptible-load_20221103.pdf
4. EMA, *Regulatory Enhancements to the DR and IL Programmes — Final Determination Paper*, 21 Oct 2024. https://www.ema.gov.sg/content/dam/corporate/partnerships/consultations/outcomes-of-dr-and-il-regulatory-sandbox-and-proposed-programme-enhancements/decision/EMA-Consultations-Final-Determination-Regulatory-Enhancements-to-DR-IL-Programmes-20241021.pdf
5. EMA, *Co-Creation with Industry: Facilitating Vehicle-to-Grid (V2G) Uptake in Singapore*, Version 1.1, 15 Sep 2020. https://www.ema.gov.sg/content/dam/corporate/partnerships/co-creations/facilitating-vehicle-to-grid-uptake-in-singapore/EMA-Partnerships-Co-Creation-Facilitating-V2G-Uptake-in-Singapore.pdf.coredownload.pdf
