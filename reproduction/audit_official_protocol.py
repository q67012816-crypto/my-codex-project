"""Read the original CLI defaults and probe evaluation without training a model."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('DGLBACKEND', 'pytorch')
sys.dont_write_bytecode = True
import dgl
import numpy as np
from scipy.stats import t
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/workspace/Community-aware-Graph-Transformer')
OUT = ROOT / 'results/official-protocol-audit'


def cli_defaults():
    tree = ast.parse((UPSTREAM/'exp.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))
    setup = []
    for node in main.body:
        is_parser = isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'parser' for x in node.targets)
        is_argument = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'add_argument')
        if is_parser or is_argument:
            setup.append(node)
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(UPSTREAM/'exp.py'), 'exec'), namespace)
    return vars(namespace['parser'].parse_args(['--dataset', 'cora']))


def single_gpu_pretrain_probe():
    """Execute the original train function AST, stopping before any model forward."""
    tree = ast.parse((UPSTREAM/'gnnutils.py').read_text())
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['train', 'cosinSim']]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(UPSTREAM/'gnnutils.py'), 'exec'), namespace)
    class OptimizerProbe:
        def zero_grad(self):
            pass
    class ModelProbe:
        def train(self):
            pass
        def __call__(self, *args, **kwargs):
            raise AssertionError('Single-GPU branch should not reach model forward.')
    with patch.object(torch.cuda, 'device_count', return_value=1):
        try:
            namespace['train'](ModelProbe(), None, torch.ones(1, dtype=torch.bool), OptimizerProbe(),
                               torch.device('cpu'), torch.device('cpu'), None, None, None,
                               None, None, None, 3, 1, 1., 1., 1.)
        except UnboundLocalError as error:
            return dict(simulated_cuda_devices=1, error_type=type(error).__name__, message=str(error),
                        source='Unmodified train function AST, no CUDA operation or optimizer step executed.')
    raise AssertionError('Expected the original single-GPU pretraining branch to leave h undefined.')


def evaluation_probe():
    spec = importlib.util.spec_from_file_location('official_cgt_evaluation_probe', UPSTREAM/'models.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    n = 20
    source = torch.arange(n)
    target = (source+1) % n
    left, right = torch.cat([source, target]), torch.cat([target, source])
    graph = dgl.graph((left, right), num_nodes=n)
    x = torch.randn(n, 3)
    graph.ndata['x'] = x
    adjacency = torch.zeros(n, n, dtype=torch.float64)
    adjacency[left, right] = 1
    sim, phi, b = torch.zeros(n, n, 3), torch.zeros(n, n, 1), torch.zeros(n, n)
    graph.edata['sim'] = sim[left, right]
    graph.edata['phi'] = phi[left, right]
    model = official.Transformer(3, 64, 7, 64, 3, 4, 3, 1, 1, 1, .2, .8)
    with tempfile.TemporaryDirectory(prefix='cgt-evaluation-', dir='/tmp') as directory:
        checkpoint = Path(directory)/'model.pt'
        torch.save(model, checkpoint)
        classifier = official.Transformer_class(3, 64, 7, 64, 3, 4, 'cora', str(checkpoint), 1, 1, 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=.001, weight_decay=1e-5)
    encoder = list(classifier.model.parameters())
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(p.requires_grad and id(p) in optimized for p in encoder)
    classifier.eval()
    samples = []
    def capture(module, inputs, output):
        sampled = output[2]
        samples.append(dict(directed_edges=int(sampled.sum()),
                            sha256=hashlib.sha256(sampled.detach().numpy().tobytes()).hexdigest()))
    hook = classifier.model.gcn.register_forward_hook(capture)
    device = torch.device('cpu')
    with torch.no_grad(), patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
        for _ in range(2):
            scores = classifier(graph.local_var(), adjacency, sim, phi, b, 3, 1, device, device)
            assert torch.isfinite(scores).all()
    hook.remove()
    assert len(samples) == 2 and samples[0]['sha256'] != samples[1]['sha256']
    return dict(scope='Two eval forwards on a 20-node graph, original source, only CUDA calls adapted to CPU; no optimization.',
                encoder_parameter_tensors=len(encoder), all_encoder_parameters_enabled_and_in_optimizer=True,
                model_training_flag=classifier.training, sampled_graphs=samples,
                eval_resamples_graph=True, frozen_embedding_classifier=False)


def main():
    OUT.mkdir(exist_ok=True)
    defaults = cli_defaults()
    expected = dict(pretrain_check=0, epochs=3, lr=.001, num_layers=4, num_heads=2,
                    k_transition=3, run_times_fine=200, alfa=.1, beta=.9)
    assert all(defaults[key] == value for key,value in expected.items())
    try:
        torch.empty(1, device=defaults['device'])
        cuda_error = None
    except (AssertionError, RuntimeError) as error:
        cuda_error = dict(type=type(error).__name__, message=str(error))
    graph_probe = evaluation_probe()
    single_gpu = single_gpu_pretrain_probe()
    paired = json.loads((ROOT/'results/official-three-checks/paired-full-minus-none.json').read_text())
    sd = paired['accuracy_points']['std_sample']
    projections = [dict(total_paired_seeds=n, approximate_half_width_95=float(t.ppf(.975,n-1)*sd/np.sqrt(n)),
                        additional_training_jobs=2*(n-10),
                        serial_minutes_at_user_assumption_5min_each=2*(n-10)*5)
                   for n in [30,50]]
    sources = {name:hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest()
               for name in ['README.md','exp.py','script_classification.py','gnnutils.py','models.py','util.py']}
    result = dict(source_sha256=sources, defaults=defaults, cuda_devices=torch.cuda.device_count(),
                  cuda_allocation_probe=cuda_error, environment=dict(python=sys.version.split()[0],torch=torch.__version__,dgl=dgl.__version__),
                  evaluation_probe=graph_probe, single_gpu_pretraining_probe=single_gpu,
                  precision_projections=projections,
                  stage4_status='pending: keep one fixed, executable protocol before extending paired seeds',
                  stage5_status='pending: original CUDA entry has not completed; no claim of paper selection bias',
                  grid_values=[.2,.4,.6,.8], grid_pairs=16,
                  grid_cost=dict(one_seed_training_jobs=16,ten_seeds_training_jobs=160,
                                 ten_seeds_serial_minutes_at_5min=800),
                  original_command_accuracy=None)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'defaults.json').write_text(json.dumps(defaults,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
