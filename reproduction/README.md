# DegFairGT / CGT：Cora 复现记录

现已增加 `train_cora.py`：对齐实际层数、聚类数、dropout、损失权重及增强系数，并输出 ΔSP。
新命令、验证结果及剩余限制见 [参数对齐说明](alignment.md)。旧 smoke 实验记录独立保留。

完整基线已按 [Cora 基线协议](baseline.md) 完成：10 个种子各 500 轮预训练及 200 轮微调，
保持现有增强实现，独立记录其与论文的差异；不混入新的边概率预测器。

已完成并独立验证全部 10 次：Accuracy **83.99 ± 1.86%**，ΔSP **6.03 ± 2.04**。
完整基线运行已结束，论文性能尚未复现；见 [最终报告](results/cora-baseline-10/REPORT.md)。
可用 [本地 GPU 说明](LOCAL_GPU.md) 将代码与公开 Cora 数据迁移到 RTX 4060；CUDA 分支需在本机验证。

本轮40次新增消融/p敏感性实验也已完成，连同复用基线的50条结果全部通过独立核验。
全开 p=4 为 **84.03 ± 1.66% / ΔSP 5.93 ± 2.03**，没有补齐论文目标差距。
重建损失单种子对照为83.73%/5.82，也没有补齐差距；详见 [消融报告](results/cora-ablation-10/REPORT.md)。

先前10对种子的限定范围核对已完成：候选边、结构注意力归约和增强梯度三处都继承官方代码。
同种子全开减全关：准确率差 **-0.94个百分点，95% CI [-2.64, +0.76]**；
ΔSP差 **+0.56，95% CI [-0.22, +1.34]**。逐种子差值及源码证据见
[官方源码与配对统计报告](results/official-three-checks/REPORT.md)。

按后续排查顺序，已记录原样README命令的启动阻塞，并核对隐藏参数与评估路径；
30对种子扩展已完成并独立核验：全开83.54 ± 2.40%，全关84.95 ± 2.17%；
准确率配对差-1.42个百分点，95% CI [-2.62, -0.21]；ΔSP差+0.56，95% CI [+0.17, +0.94]。
16组验证集系数网格也已完成并通过独立核验；seed100选中xi=0.6、zeta=0.2，
测试Accuracy83.73%、ΔSP3.97；同划分预设参照xi=0.8、zeta=0.2为80.04%/3.44。
该网格只覆盖一个预先指定的划分，不能作为十次均值复现或作者选择偏差的证据。
当前状态与最终结果见 [协议与后续实验报告](results/official-protocol-audit/REPORT.md)。

## 当前入口与结果

- [论文核对记录](paper-audit.md)：模型名称、公式、页码、参数及仍未说明的约定。
- [参数与公平指标实现](alignment.md)：四层、五社区、dropout、系数、ΔSP 和已知差异。
- [十次完整基线](results/cora-baseline-10/REPORT.md)：83.99 ± 1.86% / ΔSP 6.03 ± 2.04。
- [消融协议与只读审计](ablation-plan.md)：表8四种开关、同种子配对比较、仅SSL重建 p=4。
- [完整消融报告](results/cora-ablation-10/REPORT.md)：五组十次结果、配对差值、重建损失对照和核验记录。
- [本地RTX 4060说明](LOCAL_GPU.md)：CUDA分支需要在本机验证；云端当前仅CPU。

数据为公开 PyG Planetoid Cora：2708节点、1433维特征、7类、10556条双向存储边。
每个种子的训练/验证/测试为1626/541/541节点；按验证最佳轮次选模型。
ΔSP 已实现：仅测试节点按原图一跳度排序，低/高各108个，以全局节点编号处理同度边界。
这些取整和同度规则是明确记录的复现选择，论文没有逐项规定。

上游 [CGT仓库](https://github.com/NSLab-CUK/Community-aware-Graph-Transformer) 固定提交
`948f25477eb0253873ffdf6d38fcf53f258fdd83`，完整 checkout 保持原样；
`vendor/cgt` 保存本地可运行模型/预处理代码、许可证及来源。
上传的 arXiv:2504.15075v1 明确命名 DegFairGT 并链接该仓库。
旧版2312.16788v1已经报告相同分类目标，不能仅凭代码年份解释复现差距。

## 运行

工作目录：`/workspace/my-codex-project`。
Python3.10.22，PyTorch2.0.1+cpu、PyG2.3.1、DGL1.1.2；依赖见 `requirements-lock.txt`。

```bash
bash reproduction/setup.sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DGLBACKEND=pytorch \
  /workspace/venvs/cgt/bin/python -u reproduction/train_cora.py \
  --seed 42 --pretrain-epochs 500 --finetune-epochs 200 \
  --output reproduction/runs/my-cora
```

500轮预训练来自论文；200轮微调是沿用仓库的工作选择。
输出目录包含协议/损失/指标的JSON、模型检查点、预测/标签/划分和公平分组记录。
程序拒绝覆盖已有完整结果。用于检查流程时可缩短到2/3轮，但不能当作性能复现。
十种子批处理、消融恢复和独立验证命令分别见 `baseline.md`、`ablation-plan.md`。

`data/`、`cache/`、`runs/` 已被Git忽略；公开数据哈希记录在结果中。
缓存键包含预处理源码、图、特征、社区数和结构步数，防止误用旧的七社区缓存。
旧的 `smoke_cora.py` 及其21.63%少轮次结果仅作历史流程记录；新入口的ΔSP和正式结果单独保存。

## 结果解释的限制

增强概率目前由固定A/B及固定系数组成；GCN声明的卷积/MLP没有进入输出，
增强BCE没有参数梯度，采样后只用离散索引建图。
现有基线没有添加新的边概率网络。仅调整损失系数不能补出可学习增强器。

候选过滤、结构注意力归约及增强梯度已经逐行核对，均继承上游代码；源码一致不代表论文细节已充分说明。
原始/增强图的边特征处理与softmax后再cross-entropy等行为仍保留上游实现。
预训练损失的求和及未平方L2/N也是当前复现解释，未证实等同于作者约定；
单独的重建归一化诊断记录在 `runs/loss-convention-control/`。
完整十次训练已完成不等于论文87.10%/3.46目标已复现。
