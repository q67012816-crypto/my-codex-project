"""Export the research workspace and build tables from existing records only.

No training or model inference is performed. Original files are never modified.
Run using the reproduction venv; --workbook-only uses a Python with openpyxl.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
UPSTREAM = Path('/workspace/Community-aware-Graph-Transformer')
EXPORTS = Path('/workspace/exports')
TEXT = {'.py', '.md', '.json', '.jsonl', '.log', '.csv', '.txt', '.sh', '.yml', '.yaml', '.patch', '.cff', '.index'}


def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def workbook(directory):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    book = Workbook(write_only=True)
    instructions = book.create_sheet('README')
    for row in [
        ['用途', '汇总已有实验；不启动训练，不补算网格候选测试成绩。'],
        ['独立样本', '每行有原始路径；smoke/诊断/复用结果不可直接混为独立正式试验。'],
        ['概率/百分数', '字段名带_percent的值乘100；原始JSON保留原有单位。'],
        ['配对表单位', 'paired_30中的full_accuracy/none_accuracy也是百分数；Delta SP为乘100尺度。'],
        ['prediction来源', '只读取已有evaluation.pt；不对16个候选新增测试推断。'],
        ['网格', 'seed100单划分；只测试验证赢家和预设参照，不是十次均值复现。'],
        ['完整预测', 'node_predictions.csv含全部节点；本工作簿test_predictions只放测试节点。'],
        ['历史日志', '完整日志、JSON、源码、模型、数据和缓存见完整归档。'],
    ]: instructions.append(row)
    sources = [
        ('runs', 'run_index.csv', False), ('paired_30', 'paired_30.csv', False),
        ('validation_grid', 'validation_grid.csv', False),
        ('test_evaluations', 'evaluation_index.csv', False),
        ('epoch_history', 'epoch_history.csv', False),
        ('test_predictions', 'node_predictions.csv', True),
    ]
    for name, filename, test_only in sources:
        sheet = book.create_sheet(name)
        sheet.freeze_panes = 'A2'
        count = 0
        with (directory / filename).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            sheet.append(reader.fieldnames)
            for row in reader:
                if test_only and row['split'] != 'test': continue
                cells = []
                for key in reader.fieldnames:
                    value = row[key]
                    if value:
                        try: value = float(value)
                        except ValueError: pass
                    if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                        cell = WriteOnlyCell(sheet, value=value); cell.data_type = 's'; value = cell
                    cells.append(value)
                sheet.append(cells); count += 1
        emit(stage='workbook_sheet', name=name, rows=count)
    book.save(directory / 'all_experiments.xlsx')


def classify(path):
    name = path.parts[0]
    return {'cora-baseline-10': 'baseline', 'cora-ablation-10': 'ablation',
            'cora-paired-30': 'paired_extension', 'cora-validation-grid-seed-100': 'validation_grid_candidate',
            'finetune-lr-probe-seed42': 'finetune_lr_diagnostic',
            'loss-convention-control': 'loss_diagnostic'}.get(name, 'smoke_or_diagnostic')


def tables(out):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'vendor/cgt'))
    torch.set_num_threads(1)
    result_paths = sorted((ROOT / 'runs').rglob('*result.json'))
    indices, history, raw_records = [], [], []
    for path in result_paths:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or 'history' not in record: continue
        relative = path.relative_to(ROOT / 'runs')
        config = record.get('effective_config', {})
        row = dict(record_path=str(relative), record_kind=classify(relative), seed=record.get('seed'),
                   case=record.get('case', 'legacy_full' if relative.parts[0] == 'cora-baseline-10' else ''),
                   p=record.get('reconstruction_steps', config.get('transition_steps')),
                   xi=record.get('xi', config.get('xi')), zeta=record.get('zeta', config.get('zeta')),
                   lr=record.get('lr', config.get('lr', record.get('learning_rate'))),
                   pretrain_epochs=record.get('pretrain_epochs'), finetune_epochs=record.get('finetune_epochs'),
                   best_validation_epoch=record.get('best_validation_epoch'),
                   best_validation_accuracy_percent=100 * record['best_validation_accuracy'] if record.get('best_validation_accuracy') is not None else None,
                   original_test_accuracy_percent=100 * record['test_accuracy'] if record.get('test_accuracy') is not None else None,
                   original_delta_sp_percent=record.get('delta_sp_percent'),
                   test_evaluated=record.get('test_evaluated', record.get('test_accuracy') is not None),
                   effective_config_json=json.dumps(config, ensure_ascii=False, sort_keys=True))
        indices.append(row); raw_records.append(dict(record_path=str(relative), record=record))
        for step in record['history']:
            history.append(dict(record_path=str(relative), record_kind=row['record_kind'], seed=row['seed'], **step))
    csv_write(out / 'run_index.csv', list(indices[0]), indices)
    history_keys = ['record_path', 'record_kind', 'seed', 'phase', 'epoch'] + sorted(set().union(*(r.keys() for r in history)) - {'record_path', 'record_kind', 'seed', 'phase', 'epoch'})
    csv_write(out / 'epoch_history.csv', history_keys, history)
    with (out / 'all_results.jsonl').open('w') as stream:
        for row in raw_records: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copyfile(ROOT / 'results/official-protocol-audit/paired30-paired-differences.csv', out / 'paired_30.csv')
    shutil.copyfile(ROOT / 'results/official-protocol-audit/grid-validation.csv', out / 'validation_grid.csv')
    evaluation_rows, unsupported, nodes = [], [], 0
    fields = ['evaluation_path', 'node_id', 'split', 'label', 'prediction', 'degree', 'low_degree_group', 'high_degree_group']
    with (out / 'node_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for path in sorted((ROOT / 'runs').rglob('*evaluation.pt')):
            data = torch.load(path, map_location='cpu')
            required = {'predictions', 'labels', 'degree', 'masks'}
            if not isinstance(data, dict) or not required <= data.keys():
                unsupported.append(dict(path=str(path.relative_to(ROOT)), reason='unsupported evaluation schema; original retained')); continue
            relative = str(path.relative_to(ROOT / 'runs'))
            pred, labels, degree = [data[k].numpy() for k in ['predictions', 'labels', 'degree']]
            masks = {key: value.numpy() for key, value in data['masks'].items()}
            assert set(masks) == {'train', 'validation', 'test'}
            assert len(pred) == len(labels) == len(degree) == 2708
            assert np.all(sum(value.astype(int) for value in masks.values()) == 1)
            ids = np.flatnonzero(masks['test']); order = ids[np.lexsort((ids, degree[ids]))]
            size = int(.2 * len(ids)); low, high = set(order[:size].tolist()), set(order[-size:].tolist())
            low_rate = np.bincount(pred[list(low)], minlength=7) / size
            high_rate = np.bincount(pred[list(high)], minlength=7) / size
            correct = int((pred[ids] == labels[ids]).sum())
            evaluation_rows.append(dict(evaluation_path=relative, test_nodes=len(ids), correct=correct,
                                        accuracy_percent=100 * correct / len(ids),
                                        delta_sp_percent=100 * float(np.abs(low_rate - high_rate).mean()),
                                        metric_origin='export_recomputed_from_existing_predictions'))
            for node in range(len(pred)):
                split = next(key for key, mask in masks.items() if mask[node])
                writer.writerow(dict(evaluation_path=relative, node_id=node, split=split, label=int(labels[node]),
                                     prediction=int(pred[node]), degree=int(degree[node]),
                                     low_degree_group=int(node in low), high_degree_group=int(node in high)))
            nodes += len(pred)
    csv_write(out / 'evaluation_index.csv', list(evaluation_rows[0]), evaluation_rows)
    write_json(out / 'table_counts.json', dict(result_records=len(indices), epoch_records=len(history),
               evaluation_files=len(evaluation_rows), node_prediction_rows=nodes, unsupported_evaluations=unsupported,
               notes=['Records include smoke/diagnostic runs; do not pool every row as independent paper trials.',
                      'No new inference. Grid candidate test metrics remain absent except selected/reference existing files.']))
    assert not unsupported, unsupported
    emit(stage='tables_ready', results=len(indices), epochs=len(history), evaluations=len(evaluation_rows), nodes=nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook-only', type=Path)
    parser.add_argument('--xlsx-python', default=shutil.which('python'))
    args = parser.parse_args()
    if args.workbook_only: workbook(args.workbook_only); return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    base = EXPORTS / f'cgt-all-records-{stamp}'
    analysis = base / 'analysis'; analysis.mkdir(parents=True, exist_ok=False)
    emit(stage='started', output=str(base))
    tables(analysis)
    subprocess.run([args.xlsx_python, str(Path(__file__)), '--workbook-only', str(analysis)], check=True)
    packages = sorted((dict(name=d.metadata['Name'], version=d.version) for d in importlib.metadata.distributions()), key=lambda d:d['name'].lower())
    import torch
    write_json(analysis / 'environment.json', dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), torch=torch.__version__, cuda_available=torch.cuda.is_available(),
               packages=packages, upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()))
    shutil.copyfile(ROOT / 'results/official-protocol-audit/REPORT.md', analysis / 'LATEST_REPORT.md')
    entries, excluded = [], []
    def add_tree(root, prefix):
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            private_prefixes = ('chat-history', 'chat_history', 'conversation-history', 'conversation_history',
                                'chat-transcript', 'chat_transcript', '对话记录', '聊天记录')
            if any(part.lower() in {'.codex', 'conversations', 'chat_logs'} or part.lower().startswith(private_prefixes)
                   for part in relative.parts):
                continue
            if prefix == 'project' and relative.parts[0] == 'exported-records':
                continue
            if '.git' in relative.parts: continue
            if '__pycache__' in relative.parts or path.suffix in {'.pyc', '.pyo'}:
                if path.is_file(): excluded.append(dict(path=f'{prefix}/{relative}', reason='rebuildable Python bytecode'))
                continue
            if path.is_symlink(): excluded.append(dict(path=f'{prefix}/{relative}', reason='symlink not followed')); continue
            if path.is_file(): entries.append((path, f'{prefix}/{relative}'))
    add_tree(PROJECT, 'project')
    tracked = subprocess.check_output(['git','-C',str(UPSTREAM),'ls-files','-z']).decode().split('\0')
    entries += [(UPSTREAM / name, f'official-upstream/{name}') for name in tracked if name]
    for path in sorted(Path('/workspace/attachments').glob('**/2504.pdf')):
        entries.append((path, 'papers/2504.pdf'))
    for path in sorted(EXPORTS.iterdir()):
        if path.name.startswith('cgt-all-records-'): continue
        if path.is_file(): entries.append((path, f'historical-exports/{path.name}'))
        elif path.is_dir(): add_tree(path, f'historical-exports/{path.name}')
    add_tree(analysis, 'analysis')
    assert len({name for _,name in entries}) == len(entries)
    manifest = [dict(path=name, bytes=path.stat().st_size, sha256=digest(path)) for path,name in entries]
    write_json(base / 'FILE_MANIFEST.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               files=manifest, file_count=len(manifest), bytes=sum(r['bytes'] for r in manifest),
               scope='Project research files, tracked upstream snapshot, uploaded paper, prior exports and derived tables.',
               exclusions=['Git internal directories', 'Virtual environment/interpreter installations', 'Rebuildable Python bytecode',
                           'Non-experimental internal notes', 'Other full-export snapshots to avoid recursion'],
               explicitly_skipped=excluded,
               self_reference='Manifest, README and verification sidecars describe the payload; their own hashes are in SHA256SUMS.txt.'))
    csv_write(base / 'FILE_MANIFEST.csv', ['path','bytes','sha256'], manifest)
    readme = '''# CGT / DegFairGT 全量实验导出

完整归档包含 project/ 下全部研究代码、正式及历史试跑记录、失败日志、模型与预测检查点、公开 Cora 数据和预处理缓存；official-upstream/ 是固定提交的已跟踪文件；papers/ 是上传论文；historical-exports/ 保留此前的导出。

先看 analysis/all_experiments.xlsx 和 analysis/LATEST_REPORT.md。analysis/run_index.csv 汇总所有结果；epoch_history.csv 为逐轮训练记录；node_predictions.csv 为已有检查点中的全部节点预测、真实标签、度数和划分。工作簿仅嵌入测试节点预测，完整预测以 CSV 为准。

analysis/paired_30.csv 是正式30对的配对差值；validation_grid.csv 只含验证分数。analysis/evaluation_index.csv 的 Accuracy/ΔSP 由已有预测重新计算，标记为导出派生值；不会改写旧试跑中原本未实现的公平指标，也不会新增模型推断。

历史 smoke、少轮次诊断、复用结果和正式试验都有路径标记，不应将所有行混作独立样本。网格只覆盖 seed100；未新增查看其余14个候选的测试指标。原样作者命令仍受依赖/CUDA环境阻塞，官方原始性能尚未获得。

完整归档不包含 Python 虚拟环境、Git 内部目录或可重建字节码；环境包版本见 analysis/environment.json 与 project/reproduction/requirements-lock.txt。PyTorch整模型检查点使用项目中的类定义，读取时应使用相应版本和源码路径。

原始 JSON 保留运行时绝对路径；迁移后请将 /workspace/my-codex-project 映射到解压后的 project/，将官方源码路径映射到 official-upstream/。读取记录不需要维持旧绝对路径。

FILE_MANIFEST.json / CSV 列出每个负载文件及 SHA256。SHA256SUMS.txt 校验导出文件本身，VERIFY.json 记录对压缩包逐文件读取校验的结果。

解压：tar -xzf cgt-complete-records.tar.gz
校验：sha256sum -c SHA256SUMS.txt（在导出目录中运行）。

分析包 cgt-analysis-records.zip 含文本记录、日志、源码、论文OCR和上述分析表；完整归档额外含论文PDF、模型、数据和缓存。两者都不发布到外部服务。
'''
    (base / 'README.md').write_text(readme)
    sidecars = [(base / name, name) for name in ['README.md', 'FILE_MANIFEST.json', 'FILE_MANIFEST.csv']]
    payload = entries + sidecars
    lookup = {r['path']:r['sha256'] for r in manifest}
    lookup.update({name:digest(path) for path,name in sidecars})
    full = base / 'cgt-complete-records.tar.gz'
    light = base / 'cgt-analysis-records.zip'
    def make_full():
        emit(stage='archive_started', archive=full.name)
        with tarfile.open(full, 'w:gz', compresslevel=1) as archive:
            for path,name in payload: archive.add(path, arcname=name, recursive=False)
        emit(stage='archive_written', archive=full.name, bytes=full.stat().st_size)
        with tarfile.open(full, 'r:gz') as archive:
            members = archive.getmembers(); assert {m.name for m in members} == set(lookup)
            for member in members:
                stream = archive.extractfile(member); h=hashlib.sha256()
                for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
                assert h.hexdigest() == lookup[member.name], member.name
        return dict(archive=full.name, files=len(payload), all_payload_sha256_verified=True)
    light_payload = [(p,n) for p,n in payload if n.startswith('analysis/') or p.suffix.lower() in TEXT or p.name in {'LICENSE','.gitignore'}]
    def make_light():
        emit(stage='archive_started', archive=light.name)
        with zipfile.ZipFile(light, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path,name in light_payload: archive.write(path, arcname=name)
        with zipfile.ZipFile(light) as archive:
            assert set(archive.namelist()) == {n for _,n in light_payload}
            for name in archive.namelist():
                assert hashlib.sha256(archive.read(name)).hexdigest() == lookup[name], name
        emit(stage='archive_written', archive=light.name, bytes=light.stat().st_size)
        return dict(archive=light.name, files=len(light_payload), all_payload_sha256_verified=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(make_full); second=pool.submit(make_light)
        verification = [first.result(),second.result()]
    write_json(base / 'VERIFY.json', dict(passed=True, archives=verification,
               table_counts=json.loads((analysis / 'table_counts.json').read_text()), no_training_or_inference=True))
    checksum_paths = [full,light,base/'README.md',base/'FILE_MANIFEST.json',base/'FILE_MANIFEST.csv',base/'VERIFY.json'] + sorted(p for p in analysis.iterdir() if p.is_file())
    (base / 'SHA256SUMS.txt').write_text(''.join(f'{digest(path)}  {path.relative_to(base)}\n' for path in checksum_paths))
    links = ['# 导出文件索引', '', readme.split('\n')[2], '']
    for path in checksum_paths + [base/'SHA256SUMS.txt']:
        links.append(f'- [{path.relative_to(base)}]({path}) — {path.stat().st_size / 1024**2:.2f} MiB')
    (base / 'INDEX.md').write_text('\n'.join(links)+'\n')
    emit(stage='completed', output=str(base), full_archive_bytes=full.stat().st_size,
         analysis_archive_bytes=light.stat().st_size, files=len(manifest), verification_passed=True)


if __name__ == '__main__': main()
