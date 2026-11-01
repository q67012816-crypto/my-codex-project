"""Publish a local reviewable report only after all ten runs verify."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
base = ROOT/'runs/cora-baseline-10'
summary = json.loads((base/'summary.json').read_text())
verification = json.loads((base/'verification.json').read_text())
assert summary['status']=='completed' and summary['completed_runs']==10 and not summary['failures']
assert verification['passed'] and verification['scope']=='full' and verification['distinct_splits']==10
dest = ROOT/'results/cora-baseline-10'
dest.mkdir(parents=True,exist_ok=True)
for name in ['summary.json','runs.csv','verification.json','protocol.json']:
    shutil.copyfile(base/name,dest/name)
rows=[]
for seed in summary['protocol']['seeds']:
    r=json.loads((base/f'seed-{seed}/result.json').read_text())
    rows.append(f"| {seed} | {100*r['test_accuracy']:.2f} | {r['delta_sp_percent']:.2f} | {r['best_validation_epoch']} |")
acc=summary['accuracy_percent'];sp=summary['delta_sp_percent']
report=f'''# Cora 完整基线结果

已完成 10 次独立随机划分（种子 42–51），每次 500 轮预训练和 200 轮微调。
所有运行和测试均正常结束；没有失败、跳过或替换种子。
这是现有 CGT 实现的参数对齐基线，尚不能称为完整论文复现。

## 与论文对照

| 指标 | 当前 10 次均值 ± 样本标准差 | 论文 | 当前均值减论文均值 |
| --- | --- | --- | --- |
| Accuracy (%) ↑ | {acc['mean']:.2f} ± {acc['std_sample']:.2f} | 87.10 ± 1.53 | {acc['mean_minus_paper']:+.2f} 个百分点 |
| ΔSP（乘100展示）↓ | {sp['mean']:.2f} ± {sp['std_sample']:.2f} | 3.46 ± 0.97 | {sp['mean_minus_paper']:+.2f} |

论文数值对应上传 PDF 第 10 页表 6（准确率）、第 9 页表 3（r=1，上下各20%公平分组）。
当前主报告标准差 ddof=1，论文没有声明 ddof；summary.json 也保留 ddof=0。
基线的准确率较低，ΔSP 较高；结果存在差距，不能仅以个别种子接近论文就称为复现成功。

## 每个种子的测试结果

| 种子 | Accuracy (%) | ΔSP（乘100） | 验证最佳轮次 |
| --- | --- | --- | --- |
{chr(10).join(rows)}

## 验证与可重复执行

- 逐次核对了 500/200 的完整轮次记录、有限损失与模型参数。
- 验证了十个不同的 1626/541/541 节点划分，掩码不重叠且覆盖全部节点。
- 从保存预测独立重算准确率与 ΔSP，并核对验证最佳轮次与源文件哈希。
- 实验批次墙钟耗时约 {summary['elapsed_seconds']/60:.1f} 分钟，运行于 CPU。
- 完整日志、最佳验证模型参数和预测位于 reproduction/runs/cora-baseline-10/seed-N/。
- 汇总、逐种子 CSV、协议和验证证据已复制到本报告同目录。

在项目根目录用既有环境运行：

```bash
/workspace/venvs/cgt/bin/python -u reproduction/run_baseline.py --workers 2
/workspace/venvs/cgt/bin/python reproduction/verify_baseline.py
/workspace/venvs/cgt/bin/python reproduction/report_baseline.py
```

批次会复用协议匹配的完整结果，不覆盖旧结果。

## 解释边界与后续工作

200 轮微调、p=3、20% 取整和同度规则是明确记录的工作约定，见 baseline.md、alignment.md、paper-audit.md。
保留了上游固定增强概率及无增强训练梯度、softmax 后 cross-entropy 和部分结构实现差异。
本实验没有新增边概率预测器；无法仅凭这组结果确定各个差异造成多少性能损失。

因此，第二步的“完整基线运行与指标核对”已经完成；论文性能的成功复现尚未达成。
下一阶段应逐项核查实现差异并做独立对照实验，不在测试结果上反复挑参数。
'''
(dest/'REPORT.md').write_text(report)
print(report)
