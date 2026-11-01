"""Render the original-command, protocol, seed and bounded-grid evidence together."""
import csv,json,statistics
from datetime import datetime,timezone
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/official-protocol-audit'


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    attempt=ROOT/'runs/official-readme-20261101'
    for original,target in [('attempt.json','readme-attempt.json'),('command.log','readme-command.log'),('protocol.json','readme-source-protocol.json')]:
        shutil.copyfile(attempt/original,OUT/target)
    dependency_log=Path('/tmp/cgt-official-dependencies.log')
    if dependency_log.exists():shutil.copyfile(dependency_log,OUT/'dependency-build.log')
    command=json.loads((attempt/'attempt.json').read_text())
    base='https://github.com/NSLab-CUK/Community-aware-Graph-Transformer/blob/948f25477eb0253873ffdf6d38fcf53f258fdd83'
    lines=['# 原始命令、训练协议与后续有界实验','',
           '## 1. README原命令尝试','',
           '```bash','python exp.py --dataset cora','```','',
           f"官方所有已跟踪文件逐字复制到隔离目录，命令和源码均未更改；返回码 {command['return_code']}。",
           '本地Python3.10.22/PyTorch2.0.1+cpu，与作者environment.yml中的CUDA环境不同。',
           '实际在导入阶段因缺少torch_sparse失败，未产生准确率；补依赖遇到本地pkg_resources构建工具链冲突。',
           '独立CUDA分配检查也失败：当前PyTorch为CPU版本，CUDA设备数为0。',
           '因此不能把这次启动失败归类为“跑到84%”，也不能据此认定作者代码不能复现87%。',
           '日志见 readme-command.log；源码哈希、环境说明及返回码见 readme-source-protocol.json / readme-attempt.json。','',
           'README默认命令也不等于论文表2的配置：默认关闭预训练、2头/实际5层、7社区、lr=1e-3。',
           '所以即使命令能跑完，其数值也需要按实际配置解释。','',
           '## 2. 逐项参数与隐藏约定','',
           '| 项目 | 官方README命令的实际行为 | 当前固定CPU实验 |','| --- | --- | --- |',
           '| 预训练 | pretrain_check=0；epochs默认3未生效 | 500轮 |',
           '| 层数/头数 | num_layers=4后追加一层，实际5层；2头 | 实际4层；4头 |',
           '| Dropout | 模型写死0.5，CLI的drop未接入 | 0.1 |',
           '| 聚类 | 原始节点特征做KMeans，簇数=标签类别数，Cora为7 | 相同原始特征，显式5簇 |',
           '| k | 默认3，候选、sim和重建目标共用 | 结构k=3；本轮重建p=3 |',
           '| 增强系数 | xi=0.9、zeta=0.1，即0.9A+0.1B | xi=0.8、zeta=0.2 |',
           '| 候选过滤 | T^k>0.001、同社区、非已有边、绝对度差<=1 | 保留相同函数 |',
           '| sim过滤 | 每步概率>=0.001；另用第k步>=0.0001作门槛 | 保留相同函数 |',
           '| 预训练优化器 | 如启用：lr=1e-3、weight_decay=5e-5 | lr=1e-4、weight_decay=1e-5 |',
           '| 微调优化器 | lr共用args.lr=1e-3；weight_decay=1e-5 | lr=1e-4；weight_decay=1e-5 |',
           '| 微调轮数 | range(1,200)最多199轮；验证100轮不改善可提前停止 | 固定200轮 |',
           '| 编码器冻结 | 全部requires_grad=True，优化器包含编码器 | 同样端到端微调 |',
           '| 验证最佳模型 | best_model=model为别名，最终测试实际是最后更新的权重 | 深拷贝并恢复验证最佳权重 |',
           '| 随机种子 | 入口未设置固定训练种子；默认run_times=1 | 逐种子固定Python/NumPy/Torch |',
           '| 损失约定 | 结构误差均值、特征MSE、logits BCE均值；默认三项权重1 | 已记录的求和/L2解释；权重0.5/0.5/1 |','',
           f'参数见 [exp.py]({base}/exp.py#L229)，微调见 [script_classification.py]({base}/script_classification.py#L112)。',
           '分类入口传入num_heads=4，但构造器直接加载已有编码器，实际头数仍由已加载模型决定。',
           'README中的--layers和--alpha示例名与解析器不一致；实际选项为--num_layers和--alfa。裸命令不受这两个示例名影响。',
           f'候选见 [util.py]({base}/util.py#L46)，损失和设备分支见 [gnnutils.py]({base}/gnnutils.py#L195)。',
           '同一注意力归约代码的放大系数随每头维数变化：README默认2头为32，本地4头配置为16。',
           '额外复现了预训练入口的单卡分支错误：仅cuda.device_count()>1才前向；单卡情况下h未定义。',
           '该探针执行原函数AST、模拟设备数量为1，没有执行CUDA前向或训练更新。','',
           '## 3. 评估方式核对','',
           '- 官方代码是编码器与分类头共同微调，不是冻结嵌入后只训练分类器。',
           '- aug_check=1时，验证与测试都会再次执行采样，使用本次新增强图。model.eval()只关闭dropout等行为，不关闭该采样器。',
           '- 20节点eval探针连续两次前向，得到两个不同边集；没有执行优化器更新，官方源码只作临时CPU设备适配。',
           '- 当前固定CPU实验也保留测试时采样增强图的行为；公平分组用原始完整图度数。',
           '- 官方test仅返回Accuracy/micro-F1，没有公开Delta SP后处理。全关Delta SP均值接近论文不能证明所有评估约定相同。',
           '- 上传论文明确报告SSL预训练和下游任务，但未充分交代冻结策略及推断时图采样细节；上述确定结论来自代码，不冒充论文明确规定。','',
           f'代码证据：[编码器启用梯度]({base}/models.py#L332)、[测试调用]({base}/gnnutils.py#L242)、[增强调用]({base}/models.py#L57)。','',
           '## 4. 扩大配对种子','',
           '仅在现有固定CPU协议上增加种子；它不替代受阻的原样CUDA命令。每个新种子都训练全开、全关两次。',
           '按照既有配对标准差2.376估算：30对的95%区间半宽约0.887，50对约0.675；实际宽度以完成结果为准。',
           '已有10对，因此30对需新增40次训练，50对需新增80次训练；50对总共是100次训练，不是50次。']
    lr=ROOT/'runs/finetune-lr-probe-seed42'
    if (lr/'comparison.json').exists():
        comparison=json.loads((lr/'comparison.json').read_text())
        lines+=['','### 微调学习率单因子对照','',
                '同一seed42预训练权重及划分，新分类器初始化共同固定为42001；仅改变微调lr。',
                '两臂初始权重哈希及测试前Torch RNG哈希完全相同；各200轮微调，按验证最佳轮次选模。',
                '这是两条新训练臂，不能把其中一臂与旧seed42不同分类器初始化的成绩直接相减。','',
                '| 微调lr | 验证准确率(%) | 测试准确率(%) | Delta SP |','| --- | ---: | ---: | ---: |']
        for r in comparison['arms']:
            lines.append(f"| {r['lr']:g} | {100*r['best_validation_accuracy']:.2f} | {r['accuracy_percent']:.2f} | {r['delta_sp_percent']:.2f} |")
        lines+=['',f"高lr减低lr：准确率{comparison['accuracy_difference_points']:+.2f}个百分点，Delta SP{comparison['delta_sp_difference_points']:+.2f}。",'只有一个划分和一次初始化，不能当作平均性能或论文复现证据。']
        for name in ['comparison.json','protocol.json']:
            shutil.copyfile(lr/name,OUT/f'finetune-lr-{name}')
    paired=ROOT/'runs/cora-paired-30/summary.json'
    if paired.exists():
        s=json.loads(paired.read_text())
        lines+=['',f"当前状态：{s['status']}；已完成{s['completed_pairs']}/30对，新增{s['completed_new_jobs']}/40次。",'',
                '| 指标：全开减全关 | 配对均值 | 95%配对t区间 |','| --- | ---: | --- |']
        for label,key in [('准确率（百分点）','accuracy_points'),('Delta SP（乘100尺度）','delta_sp_x100_points')]:
            r=s['paired'][key];lo,hi=r['confidence_interval']
            lines.append(f"| {label} | {r['mean_difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
        accuracy=s['paired']['accuracy_points']
        lines+=['',f"实际准确率配对差值标准差为{accuracy['std_sample']:.3f}，95%区间半宽为{(accuracy['confidence_interval'][1]-accuracy['confidence_interval'][0])/2:.3f}个百分点。",
                '先前的种子数估算依赖旧的标准差；扩大样本后波动变大，30对并未达到预估的±0.9精度。']
        if s['pairs']:
            lines+=['','| 设置 | 测试准确率：均值 ± 样本标准差(%) | Delta SP：均值 ± 样本标准差 |',
                    '| --- | ---: | ---: |']
            for case,label in [('full','全开'),('none','全关')]:
                values=[[r[case][key] for r in s['pairs']] for key in ['accuracy_percent','delta_sp_percent']]
                cells=[f'{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}' if len(v)>1 else f'{v[0]:.2f}' for v in values]
                lines.append('| '+label+' | '+' | '.join(cells)+' |')
            lines+=['','配对区间针对同种子的差值；上表标准差反映各组在不同种子间的波动，两者含义不同。',
                    '准确率差为负、Delta SP差为正均表示全开较差；这些结论只适用于当前固定实现和协议。',
                    '全开与全关同时改变增强和结构注意力，不能据此将负效应单独归因其中一个模块；各指标区间也不是联合置信区间。']
        if (paired.parent/'attempts').exists():
            lines+=['','种子扩展曾因调度器预留启动时间而续跑；已完成结果保持原样复用，首轮摘要保存在 runs/cora-paired-30/attempts。',
                    '续跑摘要的elapsed_seconds只计该次调用；总预算仍以 ordered-followup-20261101/plan.json 的起始时间计算。']
        if s['status']!='running':
            for name in ['summary.json','protocol.json','paired-differences.csv']:
                shutil.copyfile(paired.parent/name,OUT/f'paired30-{name}')
    lines+=['','## 5. 有界验证集网格','',
            '搜索xi、zeta各取{0.2,0.4,0.6,0.8}，共16组；其他配置固定。',
            'xi、zeta进入边采样概率clamp(zeta*B+xi*A,max=1)，不是损失权重；本轮不额外限制xi+zeta=1。',
            '本轮预算内先做预先指定seed100的一次完整网格；它是单划分诊断，不是十次论文结果复现。',
            '各候选只计算验证指标；按最高验证准确率选参，平局按xi、zeta升序。先写selection.json，再计算赢家和预先指定(.8,.2)参照的测试指标。',
            '测试集不参与候选排名；不能用该实验单独证明或否定作者存在选择偏差。',
            '十个种子完整网格需160次完整训练；按每次5分钟估算，串行约13小时20分钟。']
    grid=ROOT/'runs/cora-validation-grid-seed-100'
    preflight=ROOT/'results/validation-grid-preflight.json'
    if preflight.exists():shutil.copyfile(preflight,OUT/preflight.name)
    if (grid/'summary.json').exists():
        g=json.loads((grid/'summary.json').read_text())
        lines+=['',f"网格状态：{g['status']}；完成{g['completed_candidates']}/16组。"]
        if g['candidates']:
            lines+=['','验证准确率矩阵（%，每项取其验证最佳轮次）：','',
                    '| xi / zeta | 0.2 | 0.4 | 0.6 | 0.8 |','| --- | ---: | ---: | ---: | ---: |']
            lookup={(r['xi'],r['zeta']):r for r in g['candidates']}
            for xi in [.2,.4,.6,.8]:
                cells=[f"{100*lookup[xi,z]['best_validation_accuracy']:.2f}" if (xi,z) in lookup else '—' for z in [.2,.4,.6,.8]]
                lines.append('| '+f'{xi:.1f}'+' | '+' | '.join(cells)+' |')
            with (OUT/'grid-validation.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['seed','xi','zeta','best_validation_accuracy_percent','best_validation_epoch'])
                for r in sorted(g['candidates'],key=lambda r:(r['xi'],r['zeta'])):
                    writer.writerow([r['seed'],r['xi'],r['zeta'],100*r['best_validation_accuracy'],r['best_validation_epoch']])
        if (grid/'test-after-selection.json').exists():
            evaluation=json.loads((grid/'test-after-selection.json').read_text())
            lines+=['','| seed100设置 | xi | zeta | 验证准确率(%) | 测试准确率(%) | Delta SP |','| --- | ---: | ---: | ---: | ---: | ---: |']
            for label,r in evaluation['results'].items():
                lines.append(f"| {label} | {r['xi']:.1f} | {r['zeta']:.1f} | {100*r['validation_accuracy']:.2f} | {r['accuracy_percent']:.2f} | {r['delta_sp_percent']:.2f} |")
            selected,reference=[evaluation['results'][k] for k in ['selected','reference']]
            lines+=['',f"验证集赢家减预设参照：测试准确率{selected['accuracy_percent']-reference['accuracy_percent']:+.2f}个百分点，Delta SP{selected['delta_sp_percent']-reference['delta_sp_percent']:+.2f}。",
                    '这是同一个划分上两组参数的一次测试比较，不能解释为跨种子的平均提升。']
            for name in ['protocol.json','summary.json','selection.json','test-after-selection.json']:
                shutil.copyfile(grid/name,OUT/f'grid-{name}')
    verification=ROOT/'runs/ordered-followup-20261101/verification.json'
    if verification.exists():
        v=json.loads(verification.read_text());shutil.copyfile(verification,OUT/'followup-verification.json')
        lines+=['',f"最终核验通过：{v['passed']}。"]
        if v['passed'] and s['status']=='completed' and g['status']=='completed':
            lines[2:2]=['30对固定协议对照及单划分16组验证集网格均已完成，并通过独立核验。',
                        '当前实现的全开组合准确率低于全关、Delta SP更高；验证选参后仍未达到论文87.10%的目标。',
                        '原样README命令未能在本环境启动训练，官方原始性能仍未获得，不能据此判定作者结果无法复现。','']
    ordered=ROOT/'runs/ordered-followup-20261101'
    for name in ['plan.json','runtime-control.json','status.json']:
        if (ordered/name).exists():shutil.copyfile(ordered/name,OUT/f'ordered-{name}')
    if (ordered/'status.json').exists():
        status=json.loads((ordered/'status.json').read_text())
        if status['status']=='finished':
            plan=json.loads((ordered/'plan.json').read_text())
            start=datetime.fromisoformat(plan['budget_start_utc'])
            finish=datetime.fromtimestamp((ordered/'status.json').stat().st_mtime,timezone.utc)
            minutes=(finish-start).total_seconds()/60
            lines+=['',f'从种子扩展启动至网格调度结束约{minutes:.1f}分钟；预设总训练时间上限为180分钟。',
                    '原命令/源码核查及结果整理时间不计入上述训练时段。']
    lines+=['','原样命令的性能仍未获得；后续CPU实验只能回答当前实现的噪声与系数敏感性问题。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(str(OUT/'REPORT.md'))


if __name__=='__main__':main()
