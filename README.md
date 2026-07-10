# 基于预训练词向量与深度神经网络的中文文本分类系统

本项目基于清华大学公开的 **THUCNews** 中文新闻数据集子集，设计并实现了一个端到端文本自动分类系统。系统融合了**分词过滤（Jieba）**、**分布式词表征（Word2Vec）**与**深度学习网络（TextCNN & BiLSTM-Attention）**，在测试集上达到了 **96.60%** 的分类准确率。

所有的核心源代码、数据集配置以及实验运行步骤均位于 [NN/](file:///c:/Users/asus/PycharmProjects/NNProject/NN/) 目录下。

## 📂 项目快速导航

- **完整开发及汇报文档**：请参阅 [NN/README.md](file:///c:/Users/asus/PycharmProjects/NNProject/NN/README.md)。该文档详尽记录了系统架构模块划分、核心算法数学公式、两款深度学习模型的详尽实验对比以及混淆矩阵分析。
- **核心源码目录**：
  - 数据处理工具集：[src/data_utils.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/src/data_utils.py)
  - 模型架构库：[src/models.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/src/models.py)
- **流水线运行脚本**：
  - 环境与数据自检：[check_env.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/check_env.py)
  - 数据清洗分词（步骤 1）：[run_preprocessing.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/run_preprocessing.py)
  - 词嵌入及降维（步骤 2）：[train_word2vec.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/train_word2vec.py)
  - 网络训练评估（步骤 3 & 4）：[train.py](file:///c:/Users/asus/PycharmProjects/NNProject/NN/train.py)

---

## 🚀 极速运行指南

```bash
# 切换至项目核心目录
cd NN

# 1. 运行自检，确认数据与环境就绪
python check_env.py

# 2. 执行全量文本分词与清洗
python run_preprocessing.py

# 3. 训练 Word2Vec 词向量并生成对齐矩阵与 t-SNE 可视化
python train_word2vec.py

# 4. 训练深度神经网络并评估分类指标
# 训练并评估 TextCNN 模型
python train.py --model textcnn
# 或者训练并评估 BiLSTM-Attention 模型
python train.py --model lstm
```

关于模型的具体性能分析、实验参数和混淆矩阵错判分析，请直接查阅核心文档：[NN/README.md](file:///c:/Users/asus/PycharmProjects/NNProject/NN/README.md)。
