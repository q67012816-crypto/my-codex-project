"""Bounded official-source audit and paired full-minus-none analysis; no training."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('DGLBACKEND', 'pytorch')
sys.dont_write_bytecode = True
import dgl
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def definition(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def same_definition(a, b, name):
    return ast.dump(definition(a, name)) == ast.dump(definition(b, name))


class NormalizeDevice(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'edge_probs' and node.func.attr in ['cuda', 'to']):
            self.replacements += 1
            return ast.Name(id='edge_probs', ctx=ast.Load())
        return self.generic_visit(node)


def official_probe(path):
    """Execute the actual official module, adapting only its hard-coded CUDA call."""
    spec = importlib.util.spec_from_file_location('official_cgt_bounded_probe', path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    source = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0])
    target = torch.tensor([1, 0, 2, 1, 3, 2, 0, 3])
    graph = dgl.graph((source, target), num_nodes=4)
    x = torch.randn(4, 3)
    graph.ndata['x'] = x
    adjacency = torch.zeros(4, 4, dtype=torch.float64)
    adjacency[source, target] = 1
    sim, phi, b = torch.zeros(4, 4, 3), torch.zeros(4, 4, 1), torch.zeros(4, 4)
    graph.edata['sim'] = sim[source, target]
    graph.edata['phi'] = phi[source, target]
    model = official.Transformer(3, 64, 7, 64, 3, 4, 3, 1, 1, 1, .2, .8)
    device = torch.device('cpu')
    with patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
        h, reconstructed, sampled = model(graph.local_var(), adjacency, sim, phi, b, 3, 1, device, device)
    bce = F.binary_cross_entropy(sampled, adjacency, reduction='sum')
    (F.mse_loss(reconstructed, x) + bce).backward()
    unused = {name: parameter.grad is None for name, parameter in model.gcn.named_parameters()}
    assert unused and all(unused.values())
    assert not sampled.requires_grad and not bce.requires_grad
    assert model.embedding_h.weight.grad is not None
    assert torch.isfinite(model.embedding_h.weight.grad).all()
    assert torch.linalg.vector_norm(model.embedding_h.weight.grad) > 0

    scalar = torch.full((1, 4, 1), .125, dtype=torch.float64)
    projected = torch.arange(64, dtype=torch.float64).reshape(1, 4, 16) / 1000
    edges = SimpleNamespace(data={'score': scalar, 'proj_phi': projected})
    broadcast = official.imp_add_attn('score', 'proj_phi')(edges)['score']
    edges.data['score'] = broadcast
    actual = official.exp('score')(edges)['score']
    expected = torch.exp((16 * scalar + projected.sum(-1, keepdim=True)).clamp(-8, 8))
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    return dict(
        scope='One four-node forward/backward, no optimizer step; upstream default dropout retained.',
        device_adaptation='Temporarily replace Tensor.cuda with CPU identity; official source not edited.',
        sampled_requires_grad=sampled.requires_grad, augmentation_bce_requires_grad=bce.requires_grad,
        gcn_parameter_gradient_is_none=unused,
        main_embedding_gradient_norm=float(torch.linalg.vector_norm(model.embedding_h.weight.grad)),
        attention_scalar_shape=list(scalar.shape), attention_broadcast_shape=list(broadcast.shape),
        effective_implicit_score_multiplier=16, attention_formula_verified=True)


def metrics(saved):
    pred, truth, degree = [saved[k].numpy() for k in ['predictions', 'labels', 'degree']]
    test = saved['masks']['test'].numpy()
    ids = np.flatnonzero(test)
    assert len(ids) == 541
    correct = int((pred[test] == truth[test]).sum())
    ranked = ids[np.lexsort((ids, degree[ids]))]
    low, high = ranked[:108], ranked[-108:]
    classes = np.unique(truth[test])
    sp = float(np.mean([abs((pred[low] == c).mean() - (pred[high] == c).mean()) for c in classes]))
    return correct, len(ids), correct / len(ids) * 100, sp * 100


def paired_statistics(full, none, lower_is_better):
    difference = np.asarray(full) - np.asarray(none)
    n = len(difference)
    mean, sd = float(difference.mean()), float(difference.std(ddof=1))
    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(.975, n - 1))
    interval = [mean - critical * se, mean + critical * se]
    independent = stats.ttest_rel(full, none).confidence_interval(confidence_level=.95)
    assert np.allclose(interval, [independent.low, independent.high], rtol=0, atol=1e-12)
    direction = -1 if lower_is_better else 1
    return dict(n=n, mean_difference=mean, std_sample=sd, standard_error=se, df=n-1,
                t_critical=critical, confidence_level=.95, confidence_interval=interval,
                full_better=int((direction*difference > 1e-10).sum()),
                full_worse=int((direction*difference < -1e-10).sum()),
                ties=int((np.abs(difference) <= 1e-10).sum()),
                interpretation='Difference = full-p3 minus none-p3; conditional on this Cora graph and this training protocol.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path('/workspace/Community-aware-Graph-Transformer'))
    parser.add_argument('--remote-head', required=True, help='HEAD already observed using git ls-remote for this audit.')
    args = parser.parse_args()
    out = ROOT / 'results/official-three-checks'
    out.mkdir(exist_ok=True)
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(args.upstream), *arguments], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    assert commit == args.remote_head
    assert not git('status', '--porcelain')
    protocol = json.loads((ROOT / 'results/cora-ablation-10/protocol.json').read_text())
    for manifest in [protocol['source_sha256'], protocol['original_baseline_sources']]:
        for name, digest in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    original = {name: (args.upstream/name).read_text() for name in ['models.py', 'util.py']}
    local = {name: (ROOT/'vendor/cgt'/name).read_text() for name in original}
    assert (args.upstream/'util.py').read_bytes() == (ROOT/'vendor/cgt/util.py').read_bytes()
    functions = ['MultiHeadAttentionLayer', 'src_dot_dst', 'src_dot_dst_sim', 'scaling', 'imp_add_attn', 'exp', 'dgl_renew']
    comparisons = {name: same_definition(original['models.py'], local['models.py'], name) for name in functions}
    assert all(comparisons.values())
    a, b = NormalizeDevice(), NormalizeDevice()
    gcn_equal = ast.dump(a.visit(definition(original['models.py'], 'GCN'))) == ast.dump(b.visit(definition(local['models.py'], 'GCN')))
    assert gcn_equal and a.replacements == b.replacements == 1
    for name in original:
        difference = subprocess.run(['diff', '-u', '--label', f'official/{name}', '--label', f'local/{name}',
                                     str(args.upstream/name), str(ROOT/'vendor/cgt'/name)],
                                    capture_output=True, text=True, check=False)
        assert difference.returncode in [0, 1], difference.stderr
        (out/f'{name}.patch').write_text(difference.stdout)
    probe = official_probe(args.upstream/'models.py')
    audit = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), upstream_commit=commit,
                 audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 observed_remote_head=args.remote_head, upstream_worktree_clean=True,
                 official_source_hashes={n:hashlib.sha256((args.upstream/n).read_bytes()).hexdigest() for n in original},
                 local_source_hashes={n:hashlib.sha256((ROOT/'vendor/cgt'/n).read_bytes()).hexdigest() for n in original},
                 util_entire_file_identical=True, exact_ast_comparisons=comparisons,
                 GCN_identical_except_device_call=gcn_equal, official_runtime_probe=probe,
                 cached_B_note='Cached Cora B was generated by the byte-identical official utility, with paper-aligned KMeans=5 and structural k=3.',
                 fast_sampler_note='Ablations use fixed threshold sampling, previously verified against complete original seed42 training; not a newly removed gradient path.')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    rows = []
    for seed in range(42, 52):
        full_path = ROOT/f'runs/cora-baseline-10/seed-{seed}'
        none_path = ROOT/f'runs/cora-ablation-10/none-p3/seed-{seed}'
        full, none = [torch.load(p/'evaluation.pt', map_location='cpu') for p in [full_path, none_path]]
        for key in ['train', 'validation', 'test']:
            assert torch.equal(full['masks'][key], none['masks'][key])
        for key in ['labels', 'degree']:
            assert torch.equal(full[key], none[key])
        f, n = metrics(full), metrics(none)
        for directory, calculated in [(full_path, f), (none_path, n)]:
            result = json.loads((directory/'result.json').read_text())
            assert math.isclose(calculated[2], result['test_accuracy']*100, abs_tol=1e-5)
            assert math.isclose(calculated[3], result['delta_sp_percent'], abs_tol=1e-10)
        rows.append(dict(seed=seed, full_correct=f[0], none_correct=n[0], test_nodes=f[1],
                         full_accuracy_percent=f[2], none_accuracy_percent=n[2], accuracy_difference_points=f[2]-n[2],
                         full_delta_sp_x100=f[3], none_delta_sp_x100=n[3], delta_sp_difference_points=f[3]-n[3]))
    with (out/'paired-full-minus-none.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    statistics = dict(
        comparison='full-p3 minus none-p3', same_split_labels_original_degrees_verified=True,
        accuracy_source='Exact counts from saved predictions; legacy result JSON used float32 means, differing only below displayed precision.',
        interval_method='Two-sided paired Student t interval, mean(d) +/- t(.975,9)*sd(d)/sqrt(10); no multiplicity adjustment.',
        scope='Seed/split variation on this Cora graph; not uncertainty over new graphs. Equal seeds do not imply identical RNG use across enabled modules.',
        accuracy_points=paired_statistics([r['full_accuracy_percent'] for r in rows], [r['none_accuracy_percent'] for r in rows], False),
        delta_sp_x100_points=paired_statistics([r['full_delta_sp_x100'] for r in rows], [r['none_delta_sp_x100'] for r in rows], True))
    (out/'paired-full-minus-none.json').write_text(json.dumps(statistics, indent=2)+'\n')
    report(out, commit, rows, statistics)
    print(json.dumps(dict(output=str(out), source_audit_passed=True, probe=probe, paired=statistics), indent=2))


def report(out, commit, rows, statistics):
    base=f'https://github.com/NSLab-CUK/Community-aware-Graph-Transformer/blob/{commit}'
    lines=['# 三处官方源码核对与逐种子配对统计','',
           f'核对提交：`{commit}`。本次 git ls-remote 返回相同 HEAD，本地官方检出干净。',
           '本轮仅做代码核对、四节点梯度连通性检查和已有预测的后处理，没有启动新训练。','',
           '## 逐行结论','',
           '| 检查点 | 官方代码与本地实现 | 对照论文的结论 |','| --- | --- | --- |',
           f'| 候选边范围 | util.py 全文件逐字一致；[官方入口]({base}/util.py#L46) | 第54行只取最后一步转移矩阵，第75行要求概率>0.001，第80行加度差<=1，第83行限制同社区；未累计1到k步。 |',
           f'| 结构注意力归约 | 整个注意力类及相关函数AST一致；[相加]({base}/models.py#L452)、[求和]({base}/models.py#L409) | 标量分数广播到每头16维后求和，实际为16*s+sum(phi)，再截断/取指数；这来自官方，不是本地另选的归约。论文标量/向量约定仍不充分。 |',
           f'| 增强梯度 | GCN仅.cuda改为.to(device)；[官方forward]({base}/models.py#L240)、[图重建]({base}/models.py#L93) | 概率只依赖固定A/B和普通数值系数；声明的conv1/conv2/MLPA未被forward使用。采样后只用离散索引建图，无连续权重路径。 |','',
           'Cora候选审计已有数值：同社区、非已有边、3跳以内为68063对；公开代码最终B为11482对。',
           '其中75对同社区、相距2跳的度1节点全部不在B中。完整计数见 [候选审计](../cora-ablation-10/candidate-hop-audit.json)。','',
           '直接导入未修改的官方模型，仅将其硬编码.cuda调用暂时适配CPU，运行四节点前向/反向：',
           '增强采样与BCE的requires_grad均为False；GCN的全部参数grad=None；共享节点嵌入参数梯度非零。',
           '另直接调用官方注意力函数，验证16*s+sum(phi)公式。该检查不测性能，也不修改检查点。','',
           '本地消融另有固定概率采样加速，完整seed42校验的损失、预测、权重与原采样一致；',
           '它只支持无梯度概率，不能用于后续可学习增强器。完整差异见 models.py.patch；util.py.patch为空。','',
           '## 全开减全关：逐种子差值','',
           '两组都使用结构k=3、重建p=3；全开来自已核验基线，全关来自已完成消融。',
           '逐一验证训练/验证/测试划分、标签及原始图度相同；每次500轮预训练、200轮微调。',
           '准确率以百分点计；Delta SP使用比例乘100的尺度。准确率差为正更好，SP差为负更好。','',
           '| 种子 | 准确率差 | Delta SP差 |','| --- | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['accuracy_difference_points']:+.2f} | {row['delta_sp_difference_points']:+.2f} |")
    lines+=['','| 指标 | 配对均值 | 95%置信区间 | 全开更好/更差/相同 |','| --- | ---: | --- | --- |']
    for label, key in [('准确率（百分点）','accuracy_points'),('Delta SP（乘100尺度）','delta_sp_x100_points')]:
        item=statistics[key];lo,hi=item['confidence_interval']
        lines.append(f"| {label} | {item['mean_difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {item['full_better']}/{item['full_worse']}/{item['ties']} |")
    lines+=['','区间为双侧配对Student t区间：mean(d) ± t(0.975,9)*sd(d)/sqrt(10)。',
            '已与SciPy配对t接口的置信区间交叉核对；两个指标未做多重比较校正。',
            '该区间仅描述当前Cora图上划分/训练随机性的均值不确定性；不代表跨数据集表现。',
            '同种子不保证开启不同模块后每一步的随机扰动相同。','',
            '两个区间均包含0：当前实验没有充分证据支持全开的平均增益，也不能据此证明模块完全无效。',
            '该对照同时改变两个模块，不能拆分出增强或结构注意力各自的效果；已有其他消融可作条件比较。','',
            '## 结论与停止点','',
            '这三处行为都继承自当前公开官方代码；未发现本地独有的候选范围或注意力归约改写。',
            '但函数相同不等于完整训练流程相同：本地仍调整过层数、社区数、dropout、损失及训练入口等。',
            '因此不能认定差距已唯一归因于官方代码，更不能仅凭本轮核对断言论文数值有问题。',
            '本轮按限定范围结束，不追加调参训练。下一项可验证的候选应作为独立改动，先明确采用的定义再做配对对照。','',
            '机器可读证据见 audit.json、paired-full-minus-none.json 和 paired-full-minus-none.csv。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
