# -*- coding: utf-8 -*-
"""
TF-IDF Stopwords Analysis Script
功能：
1. 扫描分词后的训练集 (cnews.train.clean.txt)
2. 计算每个词的文档频数 (DF)、逆文档频率 (IDF) 以及在语料库中的平均 TF-IDF 值
3. 找出词频最高个前 100 个词，并按其平均 TF-IDF 值从小到大排序
4. 识别出 TF-IDF 接近 0 的高频词（这些是强力停用词候选）
5. 生成可视化柱状图并保存，输出 Markdown 格式的分析表
"""

import os
import math
import collections
import matplotlib.pyplot as plt
import numpy as np

# 解决 Windows 下 matplotlib 中文显示乱码及负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_CLEAN_PATH = os.path.join(DATASET_DIR, "cnews.train.clean.txt")
STOPWORDS_PATH = os.path.join(DATASET_DIR, "stopwords.txt")
SAVE_IMAGE_PATH = os.path.join(DATASET_DIR, "vis_tfidf_stopwords.png")

def load_stopwords(path):
    stopwords = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def main():
    if not os.path.exists(TRAIN_CLEAN_PATH):
        print(f"错误: 找不到清洗分词后的训练集文件: {TRAIN_CLEAN_PATH}")
        return
        
    print("[TF-IDF分析] 开始读取训练集并统计词频和文档频数...")
    
    # 统计项
    total_tf = collections.Counter()  # 单词 -> 在整个语料库中出现的总次数
    df = collections.Counter()        # 单词 -> 包含该单词的文档数
    total_normalized_tf = collections.defaultdict(float) # 单词 -> 归一化 TF 累加值
    
    N = 0  # 总文档数
    
    with open(TRAIN_CLEAN_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                words = parts[1].split()
                if not words:
                    continue
                N += 1
                
                # 统计当前文档中各词的频数
                doc_counter = collections.Counter(words)
                doc_len = len(words)
                
                # 更新全局计数
                total_tf.update(words)
                for word, count in doc_counter.items():
                    df[word] += 1
                    total_normalized_tf[word] += count / doc_len
                    
            if (idx + 1) % 15000 == 0:
                print(f"  - 已处理 {idx + 1} 行文本...")

    print(f"[TF-IDF分析] 读取完成！总有效文档数 N = {N}，独立词汇量 = {len(total_tf)}")
    
    # 加载现有停用词表
    existing_stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"[TF-IDF分析] 成功加载当前停用词表，共有 {len(existing_stopwords)} 个停用词。")
    
    # 计算每个词的 IDF 和平均 TF-IDF
    word_metrics = {}
    for word in total_tf:
        word_df = df[word]
        # IDF 计算：采用标准 ln(N / DF)
        # 如果 DF = N（在所有文档都出现），则 IDF 为 0
        idf = math.log(N / word_df)
        
        # 平均 Raw TF-IDF (基于总词频 / N)
        avg_raw_tf = total_tf[word] / N
        avg_raw_tfidf = avg_raw_tf * idf
        
        # 平均 Normalized TF-IDF (每个文档内归一化词频的平均)
        avg_norm_tf = total_normalized_tf[word] / N
        avg_norm_tfidf = avg_norm_tf * idf
        
        word_metrics[word] = {
            "total_tf": total_tf[word],
            "df": word_df,
            "df_ratio": word_df / N,
            "idf": idf,
            "avg_raw_tfidf": avg_raw_tfidf,
            "avg_norm_tfidf": avg_norm_tfidf,
            "in_stopwords": word in existing_stopwords
        }
        
    # 按照在语料库中的总词频 (total_tf) 从高到低排序，找出前 100 个高频词
    top_100_freq_words = sorted(word_metrics.items(), key=lambda x: x[1]["total_tf"], reverse=True)[:100]
    
    # 将这 100 个高频词，按照 avg_norm_tfidf 从小到大排序（TF-IDF 越接近 0 越排在前面）
    candidates = sorted(top_100_freq_words, key=lambda x: x[1]["avg_norm_tfidf"])
    
    print("\n" + "="*80)
    print("【高频词中 TF-IDF 值最低的前 30 个词（停用词候选）】")
    print(f"{'排名':<4}{'词语':<10}{'总频数':<10}{'文档数':<10}{'文档占比':<10}{'IDF值':<10}{'平均TF-IDF':<12}{'是否已在停用词表中'}")
    print("-" * 90)
    for rank, (word, metrics) in enumerate(candidates[:30], 1):
        status = "是" if metrics["in_stopwords"] else "【否，建议加入】"
        print(f"{rank:<4}{word:<10}{metrics['total_tf']:<12}{metrics['df']:<12}{metrics['df_ratio']:<12.4f}{metrics['idf']:<12.4f}{metrics['avg_norm_tfidf']:<14.6f}{status}")
    print("="*80)
    
    # 写入分析报告 markdown 表格
    report_path = os.path.join(BASE_DIR, "tfidf_stopwords_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 高频词 TF-IDF 分析报告\n\n")
        f.write("此报告展示了语料库中总频数最高的前 100 个词中，按照**平均归一化 TF-IDF 值从小到大**排序的前 50 个词。\n")
        f.write("TF-IDF 越接近 0 的词，说明它在几乎所有文档中都出现（DF% 极高），其信息量极低，是典型的停用词。\n\n")
        f.write("| 排名 | 词语 | 总频数 (TF) | 文档频数 (DF) | 文档占比 (DF%) | IDF 值 | 平均 TF-IDF (归一化) | 是否已在当前停用词表 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for rank, (word, metrics) in enumerate(candidates[:50], 1):
            status = "✅ 是" if metrics["in_stopwords"] else "❌ **否 (候选新停用词)**"
            f.write(f"| {rank} | `{word}` | {metrics['total_tf']} | {metrics['df']} | {metrics['df_ratio']*100:.2f}% | {metrics['idf']:.4f} | {metrics['avg_norm_tfidf']:.6f} | {status} |\n")
            
    print(f"\n[TF-IDF分析] 已将前 50 个候选词的详细表格保存至: {report_path}")
    
    # ------------------ 绘制可视化图表 ------------------
    # 提取前 30 个词进行可视化
    plot_words = [x[0] for x in candidates[:30]]
    plot_tfidfs = [x[1]["avg_norm_tfidf"] for x in candidates[:30]]
    plot_df_ratios = [x[1]["df_ratio"] * 100 for x in candidates[:30]]
    plot_in_stopwords = [x[1]["in_stopwords"] for x in candidates[:30]]
    
    x = np.arange(len(plot_words))
    
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    
    # 双 Y 轴：左轴画 TF-IDF，右轴画文档占比 DF%
    # 柱状图：TF-IDF（使用暖色调，已在停用词表的用灰色，不在的用橘红色突出）
    colors = ['#bdc3c7' if in_stop else '#e67e22' for in_stop in plot_in_stopwords]
    bars = ax1.bar(x, plot_tfidfs, color=colors, alpha=0.85, width=0.6, label='平均 TF-IDF (左轴)')
    
    ax1.set_xlabel('词语', fontsize=12, labelpad=10)
    ax1.set_ylabel('平均 TF-IDF 值', color='#2c3e50', fontsize=12)
    ax1.set_title('高频词中低 TF-IDF 候选词分析 (前30个)', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_words, rotation=45, fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # 右轴：折线图画文档占比 (DF%)
    ax2 = ax1.twinx()
    line = ax2.plot(x, plot_df_ratios, color='#3498db', marker='o', linewidth=1.8, markersize=5, label='文档覆盖率 DF% (右轴)')
    ax2.set_ylabel('文档覆盖率 (%)', color='#3498db', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#3498db')
    
    # 添加图例
    # 合并两个轴的图例
    # 手动创建图例句柄
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#bdc3c7', alpha=0.85, label='已在停用词表中的词 (TF-IDF)'),
        Patch(facecolor='#e67e22', alpha=0.85, label='未在停用词表的候选词 (TF-IDF)'),
        Line2D([0], [0], color='#3498db', marker='o', linewidth=1.8, label='文档覆盖率 DF%')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 为柱状图添加数值标注
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, alpha=0.8)
                    
    plt.tight_layout()
    plt.savefig(SAVE_IMAGE_PATH)
    plt.close()
    print(f"[TF-IDF分析] 可视化图表已保存至: {SAVE_IMAGE_PATH}")

if __name__ == "__main__":
    main()
