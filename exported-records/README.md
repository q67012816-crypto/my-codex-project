# 实验记录导出与下载

本目录保存2026年11月1日导出的分析数据和完整归档分卷。原始导出在后续上传操作前生成，日期来自文件快照。

## 直接分析

- [Excel 分析表](analysis/all_experiments.xlsx)：结果索引、30对比较、网格验证分数、逐轮记录和测试节点预测。
- [轻量分析包](cgt-analysis-records.zip)：约20MB，包含源码、日志、报告、论文OCR和分析表；打开文件页面后用 Download raw file 下载。
- [结果索引 CSV](analysis/run_index.csv)。
- [76,080条逐轮记录 CSV](analysis/epoch_history.csv)。
- [297,880条已有节点预测 CSV](analysis/node_predictions.csv)。
- [30对逐种子差值 CSV](analysis/paired_30.csv)。
- [16组验证集网格 CSV](analysis/validation_grid.csv)。
- [最新分析报告](analysis/LATEST_REPORT.md)。

126份结果记录包括历史试跑和诊断，不能全部视为独立正式试验。节点预测只从已有检查点读取，未对网格其余候选新增测试推断。

## 完整记录：模型、数据、缓存与所有历史输出

完整压缩归档为1,060,960,887字节。GitHub单文件大小限制不适合直接存放该归档，因此本目录将它分为22卷，每卷最多48MiB；[分卷清单](PARTS_MANIFEST.json)记录各卷和完整文件的SHA256。

可以在GitHub本分支页面用 **Code → Download ZIP** 下载整个分支，也可以克隆：

```bash
git clone --single-branch --branch exports/cgt-records-20261101 https://github.com/q67012816-crypto/my-codex-project.git
cd my-codex-project
```

如果仓库是私有仓库，需要使用有访问权限的GitHub账号；不改变原来的仓库可见性。

用Python 3.10或更新版本合并、校验并解压；恢复操作不需要PyTorch：

```bash
python exported-records/restore_export.py --extract complete-records
```

Windows也可使用 `py` 代替 `python`。脚本按顺序验证22个分卷，再验证完整归档的大小和SHA256，最后解压到新目录。已有且正确的合并文件会复用；不会覆盖不匹配的已有文件或非空解压目录。

解压目录包含：

- `project/`：原始代码、所有runs/results、模型检查点、预测、公开Cora数据及预处理缓存。
- `official-upstream/`：固定提交的作者源码和许可证。
- `papers/`：上传的原始论文PDF。
- `historical-exports/`：此前导出文件。
- `analysis/`：本次汇总表与机器可读记录。

不包含Python虚拟环境、Git内部目录、可重建字节码或内部工作笔记。环境版本保存在 `analysis/environment.json`，项目依赖保存在 `reproduction/requirements-lock.txt`。

## 校验与来源

- [完整负载文件清单](FILE_MANIFEST.csv)与[JSON清单](FILE_MANIFEST.json)。
- [原始导出校验结果](VERIFY.json)。
- [GitHub目录校验和](GITHUB_SHA256SUMS.txt)：在本目录运行 `sha256sum -c GITHUB_SHA256SUMS.txt` 可校验当前文件布局，包括全部分卷；合并文件由恢复脚本独立校验。
- [原始导出校验和](SHA256SUMS.txt)：适用于原云端导出布局，README/INDEX对应下述保留文件；不是当前GitHub目录的校验单。
- 原云端导出说明保存在 [ORIGINAL_EXPORT_README.md](ORIGINAL_EXPORT_README.md)，原云端索引保存在 [ORIGINAL_EXPORT_INDEX.md](ORIGINAL_EXPORT_INDEX.md)。其中工作区绝对路径是历史位置，GitHub浏览请使用本说明中的相对链接。

文件中的 `/workspace/my-codex-project` 是原运行路径；本地读取记录时可映射到对应的解压目录，不需保留原绝对路径。
