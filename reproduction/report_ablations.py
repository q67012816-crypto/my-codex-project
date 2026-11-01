import csv,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base=ROOT/'runs/cora-ablation-10'
summary=json.loads((base/'summary.json').read_text());verification=json.loads((base/'verification.json').read_text())
assert summary['status']=='completed' and verification['passed'] and verification['group_observations']==50
out=ROOT/'results/cora-ablation-10';out.mkdir(exist_ok=True)
for name in ['summary.json','runs.csv','protocol.json','verification.json','sampler-equivalence.json']:
    shutil.copyfile(base/name,out/name)
for name in ['community-audit.json','community-repeat-verification.json','attention-shape-audit.json','upstream-version-audit.json','edge-feature-scale-audit.json','candidate-hop-audit.json','loss-gradient-scale-audit.json']:
    if (ROOT/'results'/name).exists():shutil.copyfile(ROOT/'results'/name,out/name)
names={'none-p3':'全关','attention-p3':'只开结构注意力','augmentation-p3':'只开增强','full-p3':'全开（已有基线）','full-p4':'全开，仅重建 p=4'}
lines=['# Cora 消融与 p 敏感性结果','',
       '各组种子 42–51，500 轮预训练、200 轮微调；十次均值 ± 样本标准差（ddof=1）。',
       '全开 p=3 复用已有十次完整结果；其余四组为本轮新增40次正式实验。另做了一次完整采样等价性校验。','',
       '| 设置 | 本次准确率 (%) | 本次 Delta SP（乘100） | 论文准确率 / Delta SP |',
       '| --- | --- | --- | --- |']
for name,g in summary['groups'].items():
    a=g['accuracy_percent'];s=g['delta_sp_percent'];paper=f"{g['paper_accuracy_percent']:.2f} / {g['paper_delta_sp_percent']:.2f}" if 'paper_accuracy_percent' in g else '图7(b)，无精确表格目标'
    lines.append(f"| {names[name]} | {a['mean']:.2f} ± {a['std_sample']:.2f} | {s['mean']:.2f} ± {s['std_sample']:.2f} | {paper} |")
lines+=['','## 配对比较','', '差值 = 后者设置（after）减参照设置（before）；准确率为百分点，Delta SP 使用乘100后的尺度。',
        '这些是同种子、同数据划分的描述性比较，未做多重比较校正的显著性断言。','',
        '| 比较 | 准确率差均值 ± 配对差标准差 | Delta SP 差均值 ± 配对差标准差 |','| --- | --- | --- |']
for name,c in summary['comparisons'].items():
    a=c['test_accuracy'];s=c['delta_sp_percent']
    lines.append(f"| {names[c['after']]} - {names[c['before']]} | {a['mean_difference']:+.2f} ± {a['std_sample']:.2f} | {s['mean_difference']:+.2f} ± {s['std_sample']:.2f} |")
control_base=ROOT/'runs/loss-convention-control'
if (control_base/'verification.json').exists():
    control=json.loads((control_base/'verification.json').read_text())
    assert control['passed']
    control_out=out/'loss-convention-control';control_out.mkdir(exist_ok=True)
    for name in ['protocol.json','runner-diff.patch','verification.json']:
        shutil.copyfile(control_base/name,control_out/name)
    for name in ['result.json','fairness.json','training.log']:
        shutil.copyfile(control_base/'seed-42'/name,control_out/name)
    lines+=['','## 重建损失归约与范数：单种子诊断','',
            '此前基线采用论文公式的一个明确实现解释：转移误差求和、特征误差为未平方的展开 L2/N。',
            '这不是已经证实的作者训练约定，也可能影响复现差距。本诊断改用上游重建项的归约方式：',
            '转移误差除以 p*N*N、特征误差使用 MSE；仍保留本次固定增强的概率 BCE，以隔离重建项变化。',
            '种子42、同一划分/缓存/模型/系数、完整500/200轮；没有新增可学习增强器。','',
            '| seed42 设置 | 准确率 (%) | Delta SP（乘100） |','| --- | --- | --- |',
            f"| 原损失解释 | {control['reference_accuracy_percent']:.2f} | {control['reference_delta_sp_percent']:.2f} |",
            f"| 归一化重建项 | {control['control_accuracy_percent']:.2f} | {control['control_delta_sp_percent']:.2f} |",'',
            f"本种子准确率变化 {control['accuracy_difference_points']:+.2f} 个百分点，Delta SP变化 {control['delta_sp_difference_points']:+.2f}，未补齐论文目标差距。",
            '这只有一个种子，不能代替十次统计，也不能称为原作者完整入口重放。',
            '完整差异、哈希和验证见 loss-convention-control/。']
if (ROOT/'results/loss-gradient-scale-audit.json').exists():
    audit=json.loads((ROOT/'results/loss-gradient-scale-audit.json').read_text())
    profiles=audit['profiles']
    raw=profiles['raw_sum_flattened_L2_per_N']['transition_to_feature_gradient_norm_ratio']
    normalized=profiles['transition_mean_feature_MSE']['transition_to_feature_gradient_norm_ratio']
    lines+=['','### 同次初始前向的梯度尺度','',
            '种子42同一前向，分别对共享 embedding_h.weight 计算两个重建项的梯度，均计入0.5权重。',
            f'转移项/特征项的梯度范数比：原损失解释为 {raw:.3g}，均值与MSE约定为 {normalized:.2f}。',
            '这说明当前实现的重建目标存在明确的相对尺度差异；仅检查权重数字相同不足以证明目标已对齐。',
            '它只覆盖一个初始化、一个参数张量，不能单独估计十次训练的性能影响。',
            '可重跑 audit_loss_scale.py，数据见 loss-gradient-scale-audit.json。']
lines+=['','## 审计与解释边界','',
        '- 社区限制检查通过，B 的全部非零项在同社区；聚类为5类，但存在单节点簇。',
        '- 十个原基线检查点实际 xi=0.8、zeta=0.2，不是构造器默认值。',
        '- p=4 只改变 SSL 重建目标范围，结构搜索和 sim 维数仍固定 k=3。',
        '- 关闭结构注意力仍保留普通 Transformer 注意力。关闭增强后增强损失为0。',
        '- 逐次核对实际各层开关、相同划分、完整轮次、验证最佳轮次、预测指标和配对统计。',
        '- 固定概率采样加速与原实现完整 seed42 训练的损失、预测、权重一致；未新增可学习模块。',
        '- 预检发现并修正了无增强时的 DGL 图临时特征跨轮残留，新入口使用 local_var() 隔离。',
        '- 本轮仍保留原增强无参数梯度、分类头及损失约定等已知差异，不能将某组偏离论文唯一归因于该模块。',
        '- 另做只读注意力形状检查：phi 打开会使标量点积分数广播到每头16维后再求和；种子42的注意力检查点约16%–35%的边-头分数被截断。论文式(6)的归约维度也不够明确，尚未作为修复项改动。',
        '- 核心代码最后修改来自2023年CGT v1，README于2025年更新。已读旧版2312.16788v1，其中已有相同分类目标数值，故不能用年份直接解释差距；仍需作者确认准确提交。',
        '- 原始图与增强图的 sim 特征归一化路径不同，详细数值见 edge-feature-scale-audit.json；本轮没有变更。',
        '- 候选范围有具体差异：同社区3跳以内非原始边为68063对，代码经度差、恰好3步及概率阈值只保留11482对，75对相距2跳的度1节点全部漏选。计数不代表已识别其性能因果影响，见 candidate-hop-audit.json。',
        '- 详见 [ablation-plan.md](../../ablation-plan.md)。原基线四个源文件和结果未改写。','',
        f"本批墙钟耗时约 {summary['elapsed_seconds']/60:.1f} 分钟。逐次数据见 runs.csv，验证证据见 verification.json。",'']
(out/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines))
