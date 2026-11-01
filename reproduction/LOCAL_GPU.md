# 将 Cora 实验迁移到本地 RTX 4060

云端的数据和代码可以在本地使用。本项目目前只有公开 Cora 数据，没有企业私有数据。
RTX 4060 通常有 8GB 显存，适合尝试本实验的 Cora 规模，但显存是否足够和实际提速倍数必须实测。
CPU 预处理、数据加载等部分不会因 GPU 自动加速；不能把云端代码直接“连接”到家中显卡。

## 推荐系统与依赖

以下命令针对 **Linux 或 Windows + WSL2（Ubuntu，x86_64）**。
Windows 原生安装方式不同，不直接使用此 bash 安装脚本。
先确认 NVIDIA 驱动可用、`nvidia-smi` 能看到 RTX 4060；WSL2 需要 Windows 主机的 WSL GPU 驱动支持。
不要直接复制云端虚拟环境：云端是 CPU 版 PyTorch，且虚拟环境含绝对路径。

使用 Python 3.10，新建本地环境，并匹配 PyTorch 2.0.1 CUDA 11.8 和 DGL 1.1.2 CUDA 11.8。
官方预编译 PyTorch wheel 自带所需 CUDA 运行库，一般不需要另装完整 CUDA Toolkit；仍需要兼容的 NVIDIA 驱动。
GPU 安装脚本使用官方 PyTorch 索引和 DGL wheel 源，未关闭 TLS 或包校验。
该 GPU 安装路径未在本云环境执行验证（本机无 GPU），如对应旧 wheel 不可获取应读取真实报错处理，不能随意换最新版。

解压迁移包，进入包含 reproduction 的目录后：

```bash
bash reproduction/setup_local_cuda.sh
```

脚本最后要求 PyTorch 与 DGL 均能使用 CUDA，而不只是检测到显卡名称。

## 先做小实验

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DGLBACKEND=pytorch \
  .venv/bin/python -u reproduction/train_local.py --device cuda \
  --seed 42 --pretrain-epochs 2 --finetune-epochs 3 \
  --output reproduction/runs/local-gpu-smoke
```

结果文件会记录 GPU 名称、GPU 峰值张量显存和耗时。也可在另一终端用 `nvidia-smi` 观察总显存与利用率。
`peak_cuda_memory_bytes` 是 PyTorch 记录的峰值，不包含所有 DGL/驱动分配，不能当作总显存。
CUDA 数值结果不保证与 CPU 逐位一致；若严格确定性检查报错，先诊断具体算子，不要把它隐藏。

若成功，再运行较长的单次训练：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DGLBACKEND=pytorch \
  .venv/bin/python -u reproduction/train_local.py --device cuda \
  --seed 42 --pretrain-epochs 500 --finetune-epochs 200 \
  --output reproduction/runs/local-gpu-seed42
```

此入口与云端使用同一模型和协议，已在 CPU 分支做小实验回归检查；CUDA 分支待你的机器验证。
现有 `run_baseline.py` 仍是云端 CPU 批次入口，不会自动使用 GPU；不要把不同设备或不同配置的单次结果混入现有十次统计。
若要估计提速，应在本地分别用 `--device cpu`、`--device cuda` 跑相同轮数，均复用缓存，
比较日志和 result.json 中的耗时；不能仅根据显卡型号给固定倍数。

## 包中保留了什么

迁移包包含 reproduction 下的脚本、依赖版本、许可证、说明、公开 Cora 原始/处理数据和 5 聚类结构缓存。
不包含 .git、凭据、虚拟环境、云端训练中的输出目录或旧的 7 聚类缓存。
包中也附带 results/cora-baseline-10 下的本次十次基线报告及汇总，便于对照；不含大型训练检查点。
`TRANSFER_MANIFEST.json` 保存每个文件的 SHA-256，用于传输后完整性检查。

缓存含绝对数量相同的图张量，CPU 加载后再传入 GPU；在相同代码/图/特征/聚类参数下可复用。
本地修改预处理后应生成新缓存。新增 train_local.py 不改变正在运行的云端源码与结果。
完整复现的已知实现差异仍见 alignment.md，使用 GPU 不会修复增强模块的梯度路径。
