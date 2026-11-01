# CGT / DegFairGT 全量实验导出

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
