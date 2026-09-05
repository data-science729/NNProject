# 基于预训练词向量与深度神经网络的中文文本分类系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-96.60%25-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Models-TextCNN%20%7C%20BiLSTM--Attention-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-THUCNews-success.svg)]()

本项目基于清华大学公开的 **THUCNews** 中文新闻数据集子集（涵盖体育、财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐共 10 个类别），设计并实现了一个工业级、模块化、高准确率的端到端中文文本自动分类系统。

系统全面贯穿了 **自然语言预处理（Jieba 分词 + 停用词过滤）**、**分布式词向量表征（Word2Vec Skip-Gram + t-SNE 2D 聚类降维）** 与 **深度神经网络分类器（TextCNN 卷积神经网络 & BiLSTM-Attention 双向循环长短期记忆注意力网络）**。在 10,000 条全量独立测试集上，模型达到了 **96.60%** 的顶级综合分类准确率。

---

## 📂 项目工程架构

项目遵循现代 Python 开源机器学习工程规范，采用模块化分层解耦设计：

```
NNProject/
├── dataset/                     # [数据中心] 词典、停用词表、标签映射与中间缓存
│   ├── stopwords.txt            # 中文停用词表
│   ├── cnews.vocab.txt          # 参考词表文件
│   ├── word_vocab.json          # 词级映射词典 (Word -> ID)
│   └── *.counts.json            # 词频统计过程文件
│
├── src/                         # [核心算法库] 模块化源码包
│   ├── __init__.py
│   ├── data_utils.py            # 数据清洗、Jieba高效分词、Dataset 与 DataLoader 封装
│   ├── models.py                # 深度学习模型库 (TextCNN, BiLSTM_Attention)
│   ├── count_vocab.py           # 词频分布分析工具
│   └── analyze_length_and_vocab.py # 语料长度分布与词表覆盖度统计
│
├── docs/                        # [文档与报告中心] 课程设计实习报告与演示文档
│   ├── 综合课程设计(Cnews文本分类).md # 飞书工程开发与实验详尽记录
│   ├── 第一小组综合课程设计实习报告.pdf # 正式结题实习报告 (排版成品)
│   ├── 第一小组课程设计报告.pdf
│   ├── 李笃光--综合课程设计一题目.pptx # 汇报答辩 PPT 演示文稿
│   ├── tfidf_stopwords_report.md  # TF-IDF 停用词效果量化分析报告
│   └── report.tex               # LaTeX 报告排版源码工程
│
├── check_env.py                 # [自检] 运行环境、GPU/CUDA 算力及数据集完整性自检
├── testgpu.py                   # [自检] 深度学习硬件加速单元自测
├── run_preprocessing.py         # [流水线 1] 单次扫描高效清洗、分词及词典生成
├── train_word2vec.py            # [流水线 2] Word2Vec 词向量训练、对齐及 t-SNE 降维
├── train.py                     # [流水线 3] 神经网络训练、早停验证与全类别测试评估
├── visualize.py                 # [分析] 绘制混淆矩阵、分类 F1 对比、注意力热图
├── visualize_vocab_zipf.py      # [分析] 齐夫定律 (Zipf's Law) 词频验证绘图
├── analyze_tfidf.py             # [分析] TF-IDF 停用词筛选与柱状图可视化
│
├── requirements.txt             # 项目第三方依赖清单
├── .gitignore                   # 大文件与运行缓存智能过滤规则
└── Git上传GitHub保姆级教程.md     # Git & GitHub 实战全流程操作手册
```

---

## 📊 模型性能对比

在 10 分类独立测试集（10,000 篇新闻）上的最终客观评测指标如下：

| 模型架构 | 核心特性 | 训练轮数 | 验证集最佳准确率 | 测试集准确率 | 优势与特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TextCNN** | 多尺度并行卷积核 (3, 4, 5) + 最大池化 | 5 Epochs | **96.84%** | **96.60%** | 局部特征捕捉极强，GPU 并行推理速度极快 |
| **BiLSTM-Attention** | 双向 2 层 LSTM + 自适应注意力加权机制 | 5 Epochs | **96.22%** | **95.82%** | 长程依赖建模能力优秀，注意力热图可解释性强 |

---

## 🚀 极速上手与复现指南

直接在项目根目录下，使用 PyCharm 内置终端或系统 PowerShell 运行以下命令：

### 1. 安装环境依赖
```bash
pip install -r requirements.txt
```

### 2. 环境与数据完整性自检
```bash
python check_env.py
```
*自动探测 GPU 显卡型号、CUDA 驱动版本以及各分词数据集就绪状态。*

### 3. 数据全量预处理与分词清洗（流水线 1）
```bash
python run_preprocessing.py
```
*采用单次扫描流式优化技术，1 分钟内完成 65,000 篇新闻文本清洗、Jieba 高频词过滤与 10,000 词级字典生成。*

### 4. 训练 Word2Vec 词向量（流水线 2）
```bash
python train_word2vec.py
```
*基于 Gensim 训练 100 维分布式词向量，生成 PyTorch 对齐嵌入矩阵 `embedding_matrix.npy`，并导出高维语义空间的 t-SNE 2D 聚类图。*

### 5. 训练深度神经网络并评估（流水线 3）
```bash
# 训练并测试 TextCNN 模型 (推荐，速度快且准确率最高)
python train.py --model textcnn --epochs 5 --batch_size 128

# 或者训练并测试 BiLSTM-Attention 模型
python train.py --model lstm --epochs 5 --batch_size 128
```
*训练过程包含验证集早停机制（Early Stopping），自动保存最佳模型权重 `best_model_*.pth`，并在测试集上生成涵盖 10 个类别的 Precision、Recall、F1-Score 全维分类报告。*

### 6. 模型效果综合可视化分析
```bash
python visualize.py
```
*一键批量生成混淆矩阵热力图、分门类 F1-Score 对比柱状图、训练收敛曲线及注意力热图。*

## 🛠️ 技术栈与依赖
- **Python 3.10+**
- **PyTorch 2.0+**（神经网络搭建、反向传播与 GPU 加速）
- **Jieba**（中文精准分词与停用词过滤）
- **Gensim**（Word2Vec 词向量训练）
- **Scikit-learn**（t-SNE 降维、分类评价指标及混淆矩阵）
- **Matplotlib / Seaborn**（训练曲线与可视化图表渲染）

---



---


