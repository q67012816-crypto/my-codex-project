# 参数对齐与 ΔSP：实现及使用

## 已实现的修改

新入口 `train_cora.py` 使用 `paper_protocol.py` 的集中配置；
保留旧入口 `smoke_cora.py` 及其历史结果。

| 项目 | 旧实验 | 新入口的行为 | 原文依据 |
| --- | --- | --- | --- |
| 注意力层 | 传 4，实际 5 | 给上游构造器传 3，最终实际 4，并断言 | 第 8 页表 2 |
| KMeans 聚类 | 用 Cora 的 7 类 | 显式传 5，不与类别数绑定 | 第 8 页表 2 |
| Dropout | 写死 0.5 | 模型新增可选参数，新入口传 0.1；旧入口默认仍 0.5 | 第 8 页表 2 |
| 预训练 weight decay | 5e-5 | 1e-5 | 第 8 页表 2 |
| 结构/特征/增强损失 | 三项等权，混用均值和 logits BCE | β1=0.5、β2=0.5、α=1，显式求和与概率 BCE | 第 7 页式 (14)–(16) |
| 增强系数 | 0.1D + 0.9A | ζD + ξA = 0.2D + 0.8A | 第 4 页式 (1)、第 8 页表 2 |
| ΔSP | 未实现 | 测试节点分组，按类别预测频率宏平均绝对差 | 第 8 页式 (17) |

聚类改变后必须重新计算 B/sim/phi。新缓存键包含预处理源码、图、特征、聚类数及 transition steps，
不会误读旧的 7 聚类缓存；旧缓存也不被删除。

损失具体为：

```text
cosine = normalize(h) @ normalize(h).T
L_transition = sum_over_p(sum_over_i_j((cosine - logM[p])**2))
L_feature = vector_L2_norm(X_hat - X) / N
L_augmentation = probability_binary_cross_entropy(A_sampled, A, reduction="sum")
L_total = 0.5 * L_transition + 0.5 * L_feature + 1.0 * L_augmentation
```

原式的特征矩阵 `||.||_2` 在此明确解释为展平后的向量 L2 范数，不是平方 MSE；
硬采样得到 0/1 时，采用 PyTorch BCE 的有限端点处理（log 下限 -100）。
这两项属于公开记录的数值实现约定，不是论文明确给出的细节。
求和损失明显大于旧均值损失，不能直接比较两份日志中总损失的大小。

## ΔSP 的实现约定

- 在原始完整无向图上计算每个节点的一跳度，不加入额外自环。
- 只对测试节点按 `(度, 全局节点编号)` 升序排序。
- 每组人数为 `floor(0.2 * 测试节点数)`；低组取排序头部，高组取尾部。
  当前 541 个测试节点，每组 108 个，两个组不重叠。
- 类别集合取测试标签中的类别；对各类预测频率差取绝对值，再做宏平均。
- `delta_sp` 保存比例；`delta_sp_percent` 保存比例乘 100 的展示值。
- `fairness.json` 保存节点编号、类别频率、度范围、人数与约定，便于审计。

人数取整、同度处理与全图度的具体处理是显式复现约定；原文没有逐项规定。
本次只实现 r=1，不猜测 r=2 广义度定义。

## 如何运行

```bash
cd /workspace/my-codex-project
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DGLBACKEND=pytorch \
  /workspace/venvs/cgt/bin/python -u reproduction/train_cora.py \
  --seed 42 --pretrain-epochs 2 --finetune-epochs 3 \
  --output reproduction/runs/my-aligned-smoke
```

产物：`result.json`（配置、损失、准确率、ΔSP）、`fairness.json`（分组细节）、
`pretrained.pt` 和 `evaluation.pt`（检查点、预测、标签、划分）。
如果输出目录已有完整 `result.json`，脚本拒绝覆盖，请换新目录。

延长实验时可传 `--pretrain-epochs 500 --finetune-epochs 200`；
500 来自论文，微调 200 是沿用仓库的工作选择，论文未明确给出。
这仍是下述限制存在时的参数对齐实验，不代表完整论文复现。
后续已完成 10 次 500/200 轮基线并独立验证，见 results/cora-baseline-10/REPORT.md。

不必重新训练，也能给已有的本项目检查点补算 ΔSP：

```bash
/workspace/venvs/cgt/bin/python reproduction/evaluate_fairness.py \
  --evaluation reproduction/runs/smoke/evaluation.pt \
  --output reproduction/results/legacy-smoke-fairness.json
```

## 验证结果

- `python -m unittest discover -s reproduction -p test_protocol.py -v`：7 项测试通过。
  包含可手算的多分类 ΔSP、排除非测试节点、同度/取整、准确率差反例、空组、损失值与梯度、实际架构配置。
- 新入口 2 轮预训练 + 3 轮微调正常结束，损失和所检查梯度有限。
  测试准确率 **17.74%**，ΔSP 展示值 **1.85**；见 `results/aligned-smoke.json` 与运行日志。
- 同种子再次运行，损失、指标、预测、模型权重完全一致；测试分组成员范围也已验证。
  证据见 `results/aligned-repeatability.json`、`results/aligned-repeat.json`、`results/aligned-fairness.json`。
- 旧实验的独立后处理得到 ΔSP 展示值 **3.70**，见 `results/legacy-smoke-fairness.json`；
  没有修改旧实验的准确率、参数或原始结果 JSON。
- ΔSP 小不代表模型准确，预测塌缩为同一类别也可得到 0。必须同时报告准确率。

## 仍需解决的问题

运行证实上游增强损失 `requires_grad=False`：增强概率直接由固定的 A、B 和固定系数组成；
GCN 中声明的卷积/MLP 参数没有进入其 forward 输出，采样后只用离散边索引构建 DGL 图。
因此，仅调整 α 不能使增强模块成为可学习模块。系数已映射，但该项仍没有训练梯度；
没有偷偷引入论文未定义的参数化方式来掩盖它。

损失归约方式也不能视为已经验证的作者设定：本入口采用上述求和和未平方的 L2/N，
而上游训练函数使用转移误差均值和特征 MSE。单种子诊断已只改变这两个重建项，
结果为83.73%/ΔSP5.82，原损失同种子为83.55%/5.56，未补齐论文差距。
记录在 `runs/loss-convention-control/`；不能把全部性能差距都归因于上游模块或损失归约。

此外保留了上游分类头的 softmax 后再 cross-entropy 的行为、预处理过滤阈值和结构注意力实现；
它们尚未逐式重写。p=3 也沿用仓库工作值，表 2 没有规定它。
现有实现的长训练基线及本轮40次新增消融已全部完成并核验，见 baseline.md 和 ablation-plan.md；
未把新增边预测器混入这些实验。增强参数化与梯度路径的修改应作为后续独立实验。
