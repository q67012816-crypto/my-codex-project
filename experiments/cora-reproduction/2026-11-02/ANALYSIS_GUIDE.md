# 最近六轮实验分析包

包含静态梯度标定、增强PCGrad、冻结/联合分类的种子0试验与三种子确认、分类CE+BCE、固定概率/GCN残差匹配对照。另附原p3/p5已锁定模型的预测和主基线报告。

优先读取 analysis/metrics-by-run.csv、analysis/node-predictions.csv、analysis/recall-by-class-and-group.csv。逐轮损失和验证指标保留在experiments各目录的history.jsonl/pretrain-history.jsonl/classification-history.jsonl，配置、协议、真实退出码、修改源码及原报告也在各实验目录。

注意：六轮新试验的split=validation是验证诊断，不能与论文测试EO4.26直接比较。stage=short/smoke是流程检查，低准确率时的小EO不是性能成果。reference-baseline中的测试预测是此前已经锁定并评估的结果，本次打包没有运行新训练或新测试。不同实验中的相同基线和种子可能是复制/回放结果，不能当成额外独立重复；同一seed的不同条件只在各实验预注册的配对范围内比较。

accuracy_percent、SP、EO_strict、较弱组召回均采用百分制。公平性按原简单图r=1度数，在各自划分内计算20/80百分位；low为degree<=q20，high为degree>q80，middle不计入两组差。保留所有同度并列。EO逐类条件召回差先取绝对值再宏平均，缺失任一组类别时EO_strict留空，missing_classes标记缺失类别；不能用有符号平均、全图含训练节点评价或填0代替。recall-by-class-and-group.csv中的recall为0到1，不是百分比。

node-predictions.csv中node_id是原Cora的0起始节点编号；degree_group针对该行split计算，不是全图统一分组。inputs提供三种子train/val/test划分、原边、节点特征及原结构缓存。features为2708x1433矩阵，标签为0到6。NPZ均为非object的数值数组，可用NumPy读取。

recent-models-and-tensors.zip另存原始.pt权重与诊断张量，按SHA256去重；checkpoint-index.json把原实验路径映射到objects/<SHA256>.pt。需要恢复目录时按索引复制，或者直接加载对应对象。固定概率与残差模型加载时须依据配置显式设置gcn.learn_augmentation，state_dict不自动保存该Python标志；优先使用实验自带验证脚本。

原source-inventory.json保留来源路径与哈希。原目录中的manifest.json描述原完整交付，分析ZIP排除了.pt而将它们移入模型包，故核验当前包请使用analysis/package-manifest.json。未打包邮件、聊天内容、账号连接记录或企业数据。本包只覆盖Cora实验。
