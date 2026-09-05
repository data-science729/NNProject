# -*- coding: utf-8 -*-
"""
THUCNews 词表大小合理性分析与 Zipf 定律可视化脚本
功能：
1. 扫描分词后的训练集 (cnews.train.clean.txt) 并统计所有词语的出现频数
2. 按频数从高到低排序，计算词语排名 (Rank) 和累积频数占比 (Cumulative Coverage)
3. 绘制 Zipf 定律双对数图与词表覆盖率曲线，直观佐证 10,000 词表大小的合理性
"""

import os
import collections
import numpy as np
import matplotlib.pyplot as plt

# 解决 Windows 下 matplotlib 中文显示乱码及负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_CLEAN_PATH = os.path.join(DATASET_DIR, "cnews.train.clean.txt")
SAVE_IMAGE_PATH = os.path.join(DATASET_DIR, "vis_vocab_zipf_coverage.png")

def main():
    if not os.path.exists(TRAIN_CLEAN_PATH):
        print(f"错误: 找不到清洗分词后的训练集文件: {TRAIN_CLEAN_PATH}")
        return
        
    print("[词表分析] 开始扫描分词后的训练集并统计全量词频...")
    word_counter = collections.Counter()
    total_words_count = 0
    
    with open(TRAIN_CLEAN_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                words = parts[1].split()
                word_counter.update(words)
                total_words_count += len(words)
                
            if (idx + 1) % 10000 == 0:
                print(f"  - 已扫描 {idx + 1} 行文本...")

    unique_words_count = len(word_counter)
    print(f"[词表分析] 扫描完成！总词数(含重复): {total_words_count}，独立词汇量: {unique_words_count}")
    
    # 按照频数降序排序
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    frequencies = np.array([x[1] for x in sorted_words])
    ranks = np.arange(1, len(frequencies) + 1)
    
    # 计算累积频数和百分比
    cumulative_sums = np.cumsum(frequencies)
    cumulative_coverage = (cumulative_sums / total_words_count) * 100
    
    # 查找 10,000 词表大小所对应的累积覆盖率
    vocab_target = 10000
    if len(cumulative_coverage) >= vocab_target:
        coverage_at_target = cumulative_coverage[vocab_target - 1]
    else:
        coverage_at_target = cumulative_coverage[-1]
        vocab_target = len(cumulative_coverage)
        
    print(f"[词表分析] 词表容量为 {vocab_target} 时，对应的累积词汇覆盖率为: {coverage_at_target:.2f}%")
    
    # 开始绘图 (1x2 双子图)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # ------------------ 左图：Zipf 定律验证 (Log-Log 图) ------------------
    ax1.loglog(ranks, frequencies, color='#2c3e50', linewidth=2.5, label='词频分布')
    
    # 拟合一条理想的 Zipf 定律直线 (f(r) = C / r^s) 作为对比
    # 在双对数坐标系下，Zipf 定律是一条斜率为 -1 左右的直线
    ideal_freqs = frequencies[0] / ranks
    ax1.loglog(ranks, ideal_freqs, color='#e74c3c', linestyle='--', linewidth=1.5, label='理想 Zipf 定律 (s=1)')
    
    ax1.set_title("词频与词汇排名分布 (Zipf定律双对数图)", fontsize=13, pad=10)
    ax1.set_xlabel("词频排名 (Rank, 对数尺度)", fontsize=11)
    ax1.set_ylabel("词汇频数 (Frequency, 对数尺度)", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # ------------------ 右图：词汇累积覆盖率曲线 ------------------
    # 为了绘图美观，横坐标只截取前 40,000 个词
    plot_limit = min(40000, len(cumulative_coverage))
    ax2.plot(ranks[:plot_limit], cumulative_coverage[:plot_limit], color='#3498db', linewidth=2.5, label='累积覆盖率曲线')
    
    # 画出 10,000 词表的交点虚线和标注点
    ax2.axvline(x=vocab_target, color='#e74c3c', linestyle='-.', linewidth=1.5)
    ax2.axhline(y=coverage_at_target, color='#e74c3c', linestyle='-.', linewidth=1.5)
    ax2.scatter([vocab_target], [coverage_at_target], color='#e74c3c', s=80, zorder=5)
    
    # 添加交点坐标说明文字
    ax2.annotate(f"词表容量: {vocab_target}\n语料覆盖率: {coverage_at_target:.2f}%", 
                 xy=(vocab_target, coverage_at_target), 
                 xytext=(vocab_target + 2000, coverage_at_target - 15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#fdfefe", ec="grey", lw=0.8))
                 
    ax2.set_title("词表大小与语料词汇累积覆盖率曲线", fontsize=13, pad=10)
    ax2.set_xlabel("词表容量 (Vocabulary Size)", fontsize=11)
    ax2.set_ylabel("累积覆盖率 (%)", fontsize=11)
    ax2.set_xlim(0, plot_limit)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("THUCNews 语料词频幂律分布与词表大小合理性科学论证", fontsize=15, y=0.98)
    plt.tight_layout()
    
    plt.savefig(SAVE_IMAGE_PATH)
    plt.close()
    print(f"[词表分析] 可视化分析完成！图表已成功保存至: {SAVE_IMAGE_PATH}")

if __name__ == "__main__":
    main()
