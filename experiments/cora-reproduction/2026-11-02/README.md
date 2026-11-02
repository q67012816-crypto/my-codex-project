# Cora / DegFairGT 最近实验资料（2026-11-02）

这是一份实验归档，**尚未复现论文EO**。包含最近六轮对照和此前已锁定的p3/p5基线资料；没有把新增验证集结果当成论文测试结果。

## 下载和分析

- [下载分析包 recent-analysis.zip](recent-analysis.zip)：约24 MB，包含六轮配置、完整日志、源码、原始预测NPZ、输入特征与结构缓存、划分，以及统一CSV。
- [逐次指标 metrics-by-run.csv](metrics-by-run.csv)
- [逐节点预测 node-predictions.csv](node-predictions.csv)
- [逐类别/度数组召回 recall-by-class-and-group.csv](recall-by-class-and-group.csv)
- [分析口径与文件结构说明](ANALYSIS_GUIDE.md)

给分析工具使用时，可先下载分析包。验证集、测试集、短流程已在表格中区分；重复使用的基线不是独立重复实验。

## 模型权重

原模型和诊断张量压缩包约389 MB，因本次上传接口限制拆成17个分卷，位于[models](models)。无需手工按顺序合并：下载[download_models.py](download_models.py)后运行，例如在Windows把文件保存到D盘，然后：

```powershell
python download_models.py --output-dir D:\Codex\downloads\cora-20261102
```

脚本只使用Python标准库，自动下载分卷，并验证各分卷及合并文件的SHA256。若已下载分卷，可增加 `--parts-dir <分卷目录>`。完整包名为 `recent-models-and-tensors.zip`。

权重包按内容哈希去重；[checkpoint-index.json](checkpoint-index.json)将每个原实验路径映射到包内 `objects/<SHA256>.pt`。固定概率/残差模型恢复时需按配置显式设置 `gcn.learn_augmentation`，不能仅依赖state_dict。原实验各自的加载脚本和配置随分析包保存。

## 范围和指标

包括损失静态标定、增强PCGrad、冻结/联合分类的单种子及三种子确认、分类CE+BCE、固定概率/GCN残差匹配对照。最近六轮新增性能结果均为验证诊断，未新增测试评分；参考基线的test行来自此前已锁定预测的复算。

EO采用原始简单图r=1度数、对应划分内20/80百分位分组、逐类条件召回差绝对值的宏平均。不要将训练节点混入测试评价，不要取有符号平均抵消差距，不要挑选最好种子。

归档包含实验资料，不包含邮件、聊天记录或账号连接信息。大文件不包含Conda环境。代码和数据来源信息保留于原报告与源码，使用时请遵循原项目及Cora数据的许可。

## 完整性

- 分析包SHA256：`c056110df563b8d57b95ca71d4a4638893d224548f66fdcfe8d87758b36b94c0`
- 合并模型包SHA256：`74e9cd0c9b191ccad831d8cb566ada375215779d2b771fb31ea1429db495f263`
- [分卷清单与校验](model-parts.json)

原有仓库文件保留，本次资料集中放在本目录。
