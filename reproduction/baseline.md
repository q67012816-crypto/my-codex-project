# Cora 第二步：完整基线协议

本实验运行当前已审阅的 CGT 实现和论文参数配置，不添加新的可学习边预测器。
它用于衡量现有实现与论文指标的差距，不预设能复现论文数值。

现已完成并验证全部 10 次，最终结果见 [报告](results/cora-baseline-10/REPORT.md)。
准确率 83.99 ± 1.86%，ΔSP 6.03 ± 2.04；不存在失败或替换种子。

固定协议：

- 数据：PyG Planetoid Cora，原始数据与结构缓存复用。
- 随机种子：42–51，共 10 次独立初始化与随机划分。
- 训练/验证/测试：1626/541/541，沿用仓库 60/20/20 的取整规则。
- 每次 500 轮预训练、200 轮微调；不提前结束训练。
- 微调 200 是仓库的工作设置，论文未明确给出；p=3 也是工作设置。
- 使用验证集选择微调轮次，最后对测试集评估准确率和 r=1 ΔSP。
- ΔSP：测试候选节点高低各 20%，人数 floor，同度按节点编号；见 alignment.md。
- 保存样本标准差（ddof=1）及总体标准差（ddof=0），主报告使用前者。
  论文未交代标准差的 ddof。
- 不调参、不挑选种子，也不因测试结果提前停止或替换运行。

启动或恢复已经完整完成的种子：

```bash
cd /workspace/my-codex-project
/workspace/venvs/cgt/bin/python -u reproduction/run_baseline.py --workers 2
```

运行结果位于 `reproduction/runs/cora-baseline-10/`：

- `protocol.json`：固定种子、训练轮数、实现文件 SHA-256。
- `seed-N/progress.json`：当前阶段、轮次、耗时；`training.log` 为完整日志。
- `seed-N/result.json`：只在该次训练、评估完整完成后生成。
- `seed-N/evaluation.pt`、`fairness.json`：最佳验证模型参数、预测、划分和公平分组。
- `summary.json`：累计进度、失败项及指标汇总。只有完整 10 次结束且无失败才标为 completed。
- `runs.csv`：每个种子的测试准确率、ΔSP、验证最佳轮次和耗时。

重启同一批次时，仅跳过协议与源码哈希匹配的完整结果。
尚未完成的种子会从头重跑，不支持从某个中间 epoch 恢复。
不要在运行过程中修改 train_cora.py、paper_protocol.py 或 vendor 模型/预处理文件。

已知限制见 alignment.md：增强概率固定且增强项没有训练梯度；
保留上游 softmax 后 cross-entropy、结构过滤与注意力实现；
特征 L2 范数和 BCE 端点按已公开的数值约定执行。
这些差异随每次结果保存，不能将“长训练跑完”等同于“已复现论文”。
