# -*- coding: utf-8 -*-
"""
THUCNews Word2Vec 词向量训练与可视化脚本 (同学 B 核心任务)
功能：
1. 读取清洗分词后的训练数据 (cnews.train.clean.txt)
2. 训练 Word2Vec 词向量模型并保存
3. 将词向量与同学 A 的词表 (word_vocab.json) 对齐，生成预训练 Embedding 矩阵 (embedding_matrix.npy)
4. 利用 t-SNE 算法对部分高频特定词汇进行 2D 降维
5. 绘制词向量在二维空间中的聚类分布图并保存为图片
"""

import os
import sys
import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 解决 Windows 下 matplotlib 中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

def load_clean_data(file_path):
    print(f"[B] 正在加载分词文本: {file_path}...")
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                word_list = parts[1].split()
                sentences.append(word_list)
    print(f"[B] 加载完毕，共 {len(sentences)} 条样本。")
    return sentences

def build_embedding_matrix(w2v_model, vocab_path, matrix_save_path):
    print(f"\n[B] 正在与词典对齐并构建预训练 Embedding 矩阵...")
    # 加载同学 A 的词表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    
    vocab_size = len(word2id)
    vector_size = w2v_model.vector_size
    print(f"  - 目标词表大小: {vocab_size}，词向量维度: {vector_size}")
    
    # 初始化权重矩阵（对特殊标记采用随机初始化，普通词如果在 Word2Vec 中有则加载，没有则随机初始化）
    embedding_matrix = np.random.normal(size=(vocab_size, vector_size))
    
    # 填充对齐
    # <PAD> 填充为全零向量，这对模型学习非特征信息有帮助
    embedding_matrix[0] = np.zeros(vector_size)

    #数字ID->向量
    found_count = 0
    for word, idx in word2id.items():
        if idx == 0:    #跳过填充词
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            found_count += 1
            
    print(f"  - 对齐完成！其中有 {found_count} 个词加载了 Word2Vec 词向量，其余为随机初始化值。")
    np.save(matrix_save_path, embedding_matrix)
    print(f"[B] 预训练 Embedding 矩阵已成功保存至: {matrix_save_path}")

def plot_tsne_visualization(w2v_model, img_save_path):
    print(f"\n[B] 正在执行 t-SNE 2D 词嵌入降维与可视化...")
    # 定义全部10个类别的代表测试词汇，精准对齐数据集分类
    word_categories = {
        "体育类": ["詹姆斯", "科比", "火箭", "季后赛", "夺冠"],
        "娱乐类": ["电影", "明星", "导演", "演唱会", "票房"],
        "家居类": ["装修", "家具", "客厅", "卧室", "建材"],
        "房产类": ["楼盘", "买房", "房价", "物业", "写字楼"],
        "教育类": ["高考", "高校", "招生", "录取", "研究生"],
        "时尚类": ["时尚", "搭配", "潮流", "彩妆", "服饰"],
        "时政类": ["选举", "总统", "反恐", "外交", "会谈"],
        "游戏类": ["网游", "玩家", "装备", "副本", "魔兽"],
        "科技类": ["手机", "电脑", "微软", "苹果", "互联网"],
        "金融类": ["股票", "基金", "投资", "股市", "银行"]
    }
    
    # 提取有向量表达的词及类别标签
    words = []
    vectors = []
    labels = []
    
    for category, word_list in word_categories.items():
        for w in word_list:
            if w in w2v_model.wv:
                words.append(w)
                vectors.append(w2v_model.wv[w])
                labels.append(category)
            else:
                print(f"  - 词汇 '{w}' 不在训练生成的 Word2Vec 词典中，已忽略。")
                
    vectors = np.array(vectors)
    
    # 调用 t-SNE 降维至 2 维空间
    # 将 perplexity 调整为 5 以适合 50 个点的降维分布
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, init='pca', n_iter=1000)
    vectors_2d = tsne.fit_transform(vectors)
    
    # 绘制散点图 (增加画幅大小至 12x10 以容纳 10 个类别)
    plt.figure(figsize=(12, 10), dpi=150)
    
    # 为 10 个类别指定 10 种高对比度的精美颜色
    colors = {
        "体育类": "#e74c3c",   # 红色
        "娱乐类": "#e67e22",   # 橙色
        "家居类": "#d35400",   # 深橘色
        "房产类": "#2c3e50",   # 深蓝灰色
        "教育类": "#f1c40f",   # 黄色
        "时尚类": "#9b59b6",   # 紫色
        "时政类": "#16a085",   # 墨绿色
        "游戏类": "#2ecc71",   # 绿色
        "科技类": "#1abc9c",   # 青色
        "金融类": "#3498db"    # 蓝色
    }
    
    # 画点
    for category in word_categories.keys():
        idx = [i for i, label in enumerate(labels) if label == category]
        if idx:
            plt.scatter(
                vectors_2d[idx, 0], 
                vectors_2d[idx, 1], 
                c=colors[category], 
                label=category, 
                s=100, 
                alpha=0.8,
                edgecolors='none'
            )
            
    # 标上词语文字
    for i, word in enumerate(words):
        plt.annotate(
            word, 
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
            xytext=(5, 2), 
            textcoords='offset points', 
            fontsize=10,
            alpha=0.9
        )
        
    plt.title("THUCNews Word2Vec 词嵌入高维空间 t-SNE 2D 聚类分布图", fontsize=14, pad=15)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 紧凑排列并保存
    plt.tight_layout()
    plt.savefig(img_save_path)
    plt.close()
    print(f"[B] t-SNE 可视化图片已成功保存至: {img_save_path}")

def main():
    dataset_dir = r"c:\Users\asus\PycharmProjects\NNProject\NN\dataset"
    train_clean_file = os.path.join(dataset_dir, "cnews.train.clean.txt")
    vocab_file = os.path.join(dataset_dir, "word_vocab.json")
    
    w2v_model_path = os.path.join(dataset_dir, "word2vec.model")
    matrix_save_path = os.path.join(dataset_dir, "embedding_matrix.npy")
    tsne_img_path = os.path.join(dataset_dir, "tsne_visualization.png")
    
    print("=" * 60)
    print("【第二部分】 Word2Vec 词向量模型训练与对齐")
    print("=" * 60)
    
    # 1. 载入分词文本
    sentences = load_clean_data(train_clean_file)
    
    # 2. 训练 Word2Vec 模型
    print("\n[B] 开始训练 Word2Vec 模型 (使用 CBOW 架构)...")
    # vector_size: 词向量维度 (例 100 维)
    # window: 上下文滑动窗口大小为 5
    # min_count: 忽略频数小于 5 的长尾词
    # workers: 开启 4 线程进行并行计算
    # epochs: 迭代次数 10 轮
    
    # 针对 Windows 下 NumPy 与 Cython 编译多线程冲突产生的 warnings 进行安全屏蔽
    original_stderr = sys.stderr
    devnull_f = open(os.devnull, 'w')
    sys.stderr = devnull_f
    
    try:
        model = Word2Vec(
            sentences, 
            vector_size=100, 
            window=5, 
            min_count=5, 
            workers=4, 
            epochs=10, 
            sg=0 # 0 表示 CBOW 架构，1 表示 Skip-gram 架构
        )
    finally:
        # 先恢复 sys.stderr 避免后续的异常输出被吞掉，再关闭 devnull_f 文件句柄
        sys.stderr = original_stderr
        devnull_f.close()
    
    # 保存模型
    model.save(w2v_model_path)
    print(f"[B] Word2Vec 模型已成功训练并存盘至: {w2v_model_path}")
    
    # 3. 进行相似词联想语义测试 (Console验证效果)
    print("\n[B] 进行控制台相似词语联想语义测试:")
    test_words = ["詹姆斯", "股票", "手机", "潮流", "游戏"]
    for w in test_words:
        if w in model.wv:
            sims = model.wv.most_similar(w, topn=5)
            sim_str = ", ".join([f"{word}({score:.3f})" for word, score in sims])
            print(f"  - 与 '{w}' 最接近的 5 个词: {sim_str}")
        else:
            print(f"  - 词汇 '{w}' 没在模型词表中。")
            
    # 4. 构建与同学 A 词典对齐的预训练 Embedding 矩阵
    build_embedding_matrix(model, vocab_file, matrix_save_path)
    
    # 5. 生成 t-SNE 2D 可视化图像
    plot_tsne_visualization(model, tsne_img_path)
    
    print("\n" + "=" * 60)
    print("[B] 恭喜！第二部分任务圆满完成。")
    print(f"  - 导出的对齐矩阵: {matrix_save_path} (同学 C 神经网络可以直接调用)")
    print(f"  - 降维聚类可视化图片: {tsne_img_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
