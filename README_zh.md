# 面向 ConStellaration Problem 2 的离线代理引导搜索

**作者：Shengnian Liu**

[English](README.md) | [中文](README_zh.md)

本仓库给出一套离线代理（surrogate）引导工作流，用于在 ConStellaration
Problem 2 中发现准等动力（quasi-isodynamic）仿星器边界候选。公开发布内容包括：
方法学说明、可执行脚本与配置、中文技术报告、交互式 HTML 汇报，以及 3 张最终图。

## 项目概述

ConStellaration Problem 2 在约 80 维 Fourier 边界参数空间上优化等离子体边界，
并受五组物理与几何约束。本项目在离线训练子集上研究候选发现：该子集中**不存在**
同时满足全部严格 Problem 2 约束的样本。

工作流以边界 Fourier 系数为模型输入，预测 Problem 2 指标与归一化约束违背量，
在代理模型上搜索，再将固定规模候选批次送入官方 VMEC++ 正演模型做高保真审计。

主要研究产出包括：

- 无在线仿真的候选搜索内环；
- 约束感知的代理训练与排序；
- PCA、GMM、集成不确定性与信任域诊断；
- 来自附加指标与 wout 派生标签的辅助监督；
- 跨 Nfp 预训练与目标子集微调；
- 代理外推与约束地板的定量分析；
- 以官方得分对累积物理调用次数评价的稀疏 VMEC++ 主动学习路线图。

官方物理可行性由 ConStellaration 正演模型判定。代理值为**预测量**，VMEC++
值为**审计物理量**。公开排行榜记录确立最终审计边界与得分；各提交者的候选生成
历史与 VMEC++ 调用次数不在公开记录中。

## 报告与 HTML 汇报

- 中文技术报告：
  [`presentations/advisor_report/report_cn.md`](presentations/advisor_report/report_cn.md)
- 交互式 HTML 汇报：
  [`presentations/advisor_report/advisor_report_deck.html`](presentations/advisor_report/advisor_report_deck.html)

HTML 使用仓库内本地 CSS 与 JavaScript 资源。可直接用浏览器打开 HTML 文件，
并用方向键翻页。

### 最终图

![代理有效性边界](figures/final-negative-result/fig1_surrogate_validity_boundary.png)

![约束地板](figures/final-negative-result/fig2_constraint_floor_positive_violation.png)

![模型方案对比](figures/scheme-comparison/fig3_model_scheme_comparison.png)

## 仓库目录

实验目录按**科学角色**（赛道与阶段）命名，而非按运行机器命名：

```text
.
├── experiments/
│   ├── official_space/
│   │   ├── stage1_base/                # 基础 6/15 指标代理流水线
│   │   ├── stage2_latent/              # 潜空间可行性搜索
│   │   ├── stage3_trust_region/        # 代理套利与信任域诊断
│   │   └── stage4_alm_prescreen/       # 代理辅助 ALM/NGOpt 预筛选
│   ├── auxiliary_supervision/
│   │   └── wout24/                     # wout 派生辅助监督
│   ├── transfer/
│   │   └── cross_nfp/
│   │       ├── pretrain/               # 跨 Nfp 预训练与微调
│   │       └── downstream/             # 使用跨 Nfp 代理的 Stage 1–4
│   ├── hf_config_screening/            # 官方数据集配置探查
│   └── wout_download_estimate/         # wout 下载 / 流式工具
├── presentations/advisor_report/       # 公开报告与 HTML 汇报
├── figures/                            # 3 张最终图
├── docs/                               # 方法学、脚本目录与布局说明
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

生成的 `outputs*` 目录通过 `.gitignore` 仅保留在本地。脚本职责见
[`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md)。目录说明与后续包化重构计划见
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md)。

## 环境安装

主实验环境与已验证的 ConStellaration 栈对应 **Python 3.10**。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 将 ConStellaration 固定到实验所用的源码提交。GPU 机器可
先安装与 CUDA 匹配的 PyTorch 轮子，再安装其余依赖。

## 数据获取

脚本从官方 Hugging Face 数据集加载 ConStellaration：

```python
from datasets import load_dataset

dataset = load_dataset(
    "proxima-fusion/constellaration",
    "default",
    split="train",
)
```

数据集缓存、本地 Parquet 切分、VMEC++ wout 文件、训练权重、候选边界与逐行
审计记录均保留在本地实验存储中，不随本仓库发布。

wout24 赛道需显式提供本地 wout 分片目录：

```bash
python experiments/auxiliary_supervision/wout24/01_build_wout24_labels.py \
  --filtered-parts-dir /path/to/filtered/wout/parquet/parts
```

## 实验流程

```text
官方数据集
  -> 确定性过滤与划分构建
  -> 代理集成训练
  -> 松弛种子构造
  -> 代理引导候选搜索
  -> 固定预算 VMEC++ 审计
  -> 预测与物理量对照诊断
```

典型基础运行：

```bash
cd experiments/official_space/stage1_base
python 00_check_environment.py --config configs/quick.yaml
python 01_prepare_dataset.py --config configs/quick.yaml
python 02_train_surrogate.py --config configs/quick.yaml
python 03_generate_relaxed_seed_candidates.py --config configs/quick.yaml
python 04_run_conservative_cmaes.py --config configs/quick.yaml
python 05_vmec_audit_candidates.py --config configs/quick.yaml
python 06_analyze_results.py --config configs/quick.yaml
```

后续实验目录消费前序阶段的本地输出。输入约定与职责见
[`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md)。

## 方法学赛道

- **官方空间赛道（Official-space）：** 基准低阶 Fourier 参数化，并与对齐的
  VMEC++ 审计预算配套。
- **潜空间与信任域赛道：** PCA 支撑、邻域距离、集成不确定性与谱几何检查。
- **辅助监督赛道：** 附加默认指标与 wout 派生低维训练标签；推理输入仍为
  Fourier 系数。
- **跨 Nfp 赛道：** 在非目标场周期上做表示预训练，再在 Nfp=3 子集上微调。
- **稀疏反馈扩展：** 分批 VMEC++ 审计、标签获取与跨轮主动学习式代理重训。

## 复现规则

1. 记录上游 ConStellaration 提交号与环境版本。
2. 保持数据划分确定性，并将松弛种子与训练样本分开。
3. 每次离线运行中，VMEC++ 审计标签不得参与模型选择。
4. 统计每一个尝试的 VMEC++ 审计槽位，包括求解失败。
5. 代理预测与官方物理评价分字段、分表报告。
6. 在同一 Fourier 空间与同一审计预算下比较方法。
7. 扩展 Fourier 参数化作为独立赛道单独报告。

## 引用

引用元数据见 [`CITATION.cff`](CITATION.cff)。基准与 VMEC++ 的 BibTeX 见
[`docs/REFERENCES.md`](docs/REFERENCES.md)。上游归属见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 许可证

Copyright © 2026 Shengnian Liu。原创代码、文档、报告、HTML 汇报与最终图采用
MIT License。上游软件与数据集保留其原始条款。详见
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md)。
