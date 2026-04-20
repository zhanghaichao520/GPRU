# GPRU: Prompt-based Recommendation Unlearning

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Research%20Prototype-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyG-Graph%20Recommendation-3C2179?style=flat-square)
![Task](https://img.shields.io/badge/Task-Recommendation%20Unlearning-111827?style=flat-square)

GPRU 是论文 **Prompt-based Recommendation Unlearning** 的实验型实现。项目围绕推荐系统中的“被遗忘权”场景，提供一种轻量级的 **GPRU** 推荐遗忘流程：冻结已经训练好的推荐主干模型，仅训练少量 prompt 参数，在尽量保持保留数据推荐性能的同时，降低目标遗忘交互对模型输出的影响。

本仓库同时保留了 SISA、重训练等对照实验入口，覆盖 **LightGCN / MF / BPRMF** 三类推荐模型，以及 **Amazon-Book / Gowalla / Yelp2018** 三个常用推荐基准数据集。

## Highlights

- **Prompt-only unlearning**: 继承 teacher 模型参数，student 主干保持冻结，通过 `SimplePrompt` 或 `ComplexPrompt` 注入可学习扰动，避免全量重训练。
- **Teacher-student objective**: 在 retain set 上最小化 teacher/student 差异，在 forget set 上最大化二者输出差异，对齐论文中的保留与遗忘双目标。
- **Contrastive regularization**: 可选对比损失用于拉开遗忘表示与保留表示边界，提升遗忘过程的可控性。
- **Model-agnostic pipeline**: 同一套 prompt unlearning 思路适配 LightGCN、矩阵分解 MF 与 BPRMF。
- **Benchmark-ready evaluation**: 内置 HR@K、Recall@K、NDCG@K 指标，并提供 SISA 分片训练与分片重训对照。

## Method Overview

GPRU 的实验流程可以概括为四步：

1. **Full-data pretraining**: 使用完整交互图训练 teacher recommender。
2. **Retain / forget split**: 将训练交互拆成保留集 `D_r` 与遗忘集 `D_f`，默认实验采用随机 10% 作为 forget set。
3. **Prompt tuning**: 初始化 student 并加载 teacher 参数，冻结推荐主干，仅优化 prompt。
4. **Dual evaluation**: 在 retain set 上检查推荐质量，在 forget set 上检查遗忘效果。

训练目标在代码中对应：

```text
L_total = alpha * L_retain + beta * L_forget + gamma * L_contrastive
```

- `L_retain`: 让 student 在保留数据上的输出接近 teacher。
- `L_forget`: 让 student 在遗忘数据上的输出偏离 teacher。
- `L_contrastive`: 可选的对比约束，用于增强 retain/forget 表示边界。

## Repository Layout

```text
.
├── prompt.py                         # SimplePrompt / ComplexPrompt
├── model/
│   ├── LightGCN.py                   # LightGCN backbone
│   ├── MF.py                         # Matrix Factorization backbone
│   └── BPRMF.py                      # BPR-MF backbone
├── evaluation/
│   ├── LightGCN_evaluation.py        # LightGCN pretrain + GPRU evaluation
│   ├── MF_evaluation.py              # MF pretrain + GPRU evaluation
│   ├── BPRMF_evaluation.py           # BPRMF pretrain + GPRU evaluation
│   ├── LightGCN_SISA.py              # LightGCN SISA baseline
│   ├── MF_SISA.py                    # MF SISA baseline
│   └── BPRMF_SISA.py                 # BPRMF SISA baseline
├── preprocess/
│   ├── AmazonBook.py                 # Amazon-Book loader
│   ├── Gowalla.py                    # Gowalla loader
│   ├── Yelp.py                       # Yelp2018 loader
│   └── MovieLens1M.py                # MovieLens-1M loader
├── util/
│   ├── utils.py                      # BPR loss, retain/forget loss, NDCG, recommend
│   └── dataset_splitter.py           # retain/forget split persistence
├── dataset/                          # raw / processed benchmark data
├── pretrain/                         # pretrained checkpoints
├── test*.ipynb                       # reproducible experiment notebooks
├── plot.py / plot2.py                # runtime visualization scripts
└── _iconip__Prompt_based_Recommendation_Unlearning.pdf
```

## Installation

建议使用独立环境运行：

```bash
conda create -n gpru python=3.9
conda activate gpru
```

安装核心依赖：

```bash
pip install torch pandas numpy matplotlib seaborn jupyter
pip install torch-geometric
```

如果使用 CUDA，请根据本机 CUDA / PyTorch 版本安装匹配的 PyTorch Geometric wheel。CPU 环境可以直接运行小规模调试，完整 Amazon-Book、Gowalla、Yelp2018 实验建议使用 GPU。

## Quick Start

### 1. Load Dataset

```python
import torch
from preprocess.AmazonBook import AmazonBook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AmazonBook("./dataset/amazon-book")
data = dataset.get()
num_users, num_books = dataset.getNumber()
```

`AmazonBook`、`Gowalla`、`Yelp` 都会返回 PyG `Data` 对象：

- `edge_index`: 训练交互边。
- `edge_label_index`: 测试交互边。
- item 节点 id 会自动加上 `num_users` 偏移，形成 user-item 同构图。

### 2. Pretrain Teacher

```python
from model.LightGCN import LightGCN
from evaluation import lightgcn_evaluation

config = {
    "k": 20,
    "lr": 1e-3,
    "epochs": 10,
    "num_layers": 2,
    "batch_size": 8192,
    "embedding_dim": 64,
    "num_users": num_users,
    "num_books": num_books,
}

teacher = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=config["embedding_dim"],
    num_layers=config["num_layers"],
).to(device)

teacher = lightgcn_evaluation(teacher, config, data, device)
torch.save(teacher.state_dict(), "pretrain/LightGCN_Amazon_Book_10_Epochs_Top_20.pt")
```

### 3. Build Retain / Forget Split

```python
from torch_geometric.data import Data
from util.dataset_splitter import DatasetSplitter

num_edges = data.edge_index.size(1)
perm = torch.randperm(num_edges)
split = int(num_edges * 0.1)

forget_data = Data(
    num_nodes=data.num_nodes,
    edge_index=data.edge_index[:, perm[:split]],
    edge_label_index=data.edge_label_index,
)

retain_data = Data(
    num_nodes=data.num_nodes,
    edge_index=data.edge_index[:, perm[split:]],
    edge_label_index=data.edge_label_index,
)

DatasetSplitter("AmazonBook").save_split(retain_data, forget_data)
```

### 4. Run GPRU Unlearning

```python
from evaluation import prompt_lightgcn_evaluation, prompt_lightgcn_unlearning_evaluation

student = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=config["embedding_dim"],
    num_layers=config["num_layers"],
).to(device)

student.load_state_dict(teacher.state_dict())

config.update({
    "alpha": 0.3,
    "beta": 0.7,
    "gamma": 1e-6,
    "weight_decay": 1e-3,
    "tuning_type": "complexPrompt",   # "simplePrompt" or "complexPrompt"
    "number_p": 10,
    "Contrastive_loss": True,
})

student, prompt = prompt_lightgcn_evaluation(
    teacher,
    student,
    data,
    retain_data,
    forget_data,
    config,
    device,
)

prompt_lightgcn_unlearning_evaluation(
    student,
    prompt,
    data,
    forget_data,
    config,
    device,
)
```

## Experiment Entry Points

Notebook 命名遵循 “模型 × 数据集 × 任务” 的组织方式：

| Model | Pretrain | GPRU | SISA |
| --- | --- | --- | --- |
| LightGCN + Amazon-Book | `test01(AmazonBook).ipynb` | `test02.ipynb` | `test03.ipynb` |
| LightGCN + Gowalla | `test04(Gowalla).ipynb` | `test05.ipynb` | `test06.ipynb` |
| LightGCN + Yelp2018 | `test07(Yelp2018).ipynb` | `test08.ipynb` | `test09.ipynb` |
| MF + Amazon-Book | `test10(MF-AmazonBook).ipynb` | `test11.ipynb` | `test12.ipynb` |
| MF + Gowalla | `test13(MF-Gowalla).ipynb` | `test14.ipynb` | `test15.ipynb` |
| MF + Yelp2018 | `test16(MF-Yelp2018).ipynb` | `test17.ipynb` | `test18.ipynb` |
| BPRMF + Amazon-Book | `test19(BPRMF-AmazonBook).ipynb` | `test20.ipynb` | `test21.ipynb` |
| BPRMF + Gowalla | `test22(BPRMF-Gowalla).ipynb` | `test23.ipynb` | `test24.ipynb` |
| BPRMF + Yelp2018 | `test25(BPRMF-Yelp2018).ipynb` | `test26.ipynb` | `test27.ipynb` |

启动 Jupyter：

```bash
jupyter notebook
```

## Evaluation Protocol

代码默认输出：

- `HR@K`: Top-K 命中率。
- `Recall@K`: 用户级召回。
- `NDCG@K`: 排序质量。
- `Running time`: 用于比较 GPRU、SISA、Retrain 等方法的效率。

推荐同时观察两个方向：

- **Retain performance**: retain set 上 HR / Recall / NDCG 越接近 teacher，说明保留知识越稳定。
- **Forget effect**: forget set 上目标交互的 HR / Recall / NDCG 越低，说明目标数据影响被削弱得越充分。

## Supported Components

| Component | Implementation |
| --- | --- |
| Prompt modules | `SimplePrompt`, `ComplexPrompt` |
| Backbones | `LightGCN`, `MF`, `BPRMF` |
| GPRU training | `prompt_lightgcn_evaluation`, `prompt_MF_evaluation`, `prompt_BPRMF_unlearning_eva` |
| Forget evaluation | `prompt_lightgcn_unlearning_evaluation`, `MF_unlearning_evaluation`, `BPRMF_forget_data_eva` |
| SISA baselines | `sisa_lightgcn_*`, `sisa_MF_*`, `sisa_BPRMF_*` |
| Metrics | HR@K, Recall@K, NDCG@K |
| Split persistence | `DatasetSplitter.save_split`, `DatasetSplitter.load_split` |

## Notes

- 当前仓库是研究复现实验代码，不是打包好的命令行工具；推荐从 notebook 或 Python module 直接调用。
- `dataset/` 与 `pretrain/` 中已经包含部分处理后的数据与 checkpoint，便于快速复现实验。
- 不同系统对路径大小写敏感，使用 Gowalla / Yelp2018 时请保持 notebook 中的 dataset path 与本地目录一致。
- 大规模数据集上评估会构造 user-item score matrix，显存不足时请调小 `batch_size`。

## Reference

本实现主要参考仓库内论文：

```text
_iconip__Prompt_based_Recommendation_Unlearning.pdf
```
