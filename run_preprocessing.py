# -*- coding: utf-8 -*-
"""
THUCNews 全量数据预处理执行脚本 (单次扫描优化版)
功能：
1. 一边清洗和分词训练集，一边统计词频，并直接导出分词后的 train 文件。
2. 避免了对 130MB 训练集进行两次分词扫描，处理速度提升 50%。
3. 基于词频统计构建并保存全局词典（vocab.json）。
4. 导出分词后的 val 和 test 数据集。
"""

import os
import json
import collections
from src.data_utils import TextPreprocessor

def preprocess_and_count(preprocessor, input_path, output_path, is_train=False):
    print(f"\n[A] 正在处理: {input_path} -> {output_path}")
    count = 0
    word_counter = collections.Counter() if is_train else None
    
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            label, content = parts[0], parts[1]
            
            # 清洗文本并分词
            cleaned = preprocessor.clean_text(content)
            words = preprocessor.segment_text(cleaned)
            
            # 统计词频 (仅对训练集进行)
            if is_train:
                word_counter.update(words)
            
            # 将分词结果连接成以空格分隔的字符串
            words_str = " ".join(words)
            fout.write(f"{label}\t{words_str}\n")
            
            count += 1
            if count % 10000 == 0:
                print(f"  - 已完成 {count} 条文本的清洗和分词...")
                
    print(f"[A] 处理完成，共保存 {count} 条样本。")
    return word_counter

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    train_raw = os.path.join(dataset_dir, "cnews.train.txt")
    val_raw = os.path.join(dataset_dir, "cnews.val.txt")
    test_raw = os.path.join(dataset_dir, "cnews.test.txt")
    
    train_clean = os.path.join(dataset_dir, "cnews.train.clean.txt")
    val_clean = os.path.join(dataset_dir, "cnews.val.clean.txt")
    test_clean = os.path.join(dataset_dir, "cnews.test.clean.txt")
    vocab_json = os.path.join(dataset_dir, "word_vocab.json")
    
    # 实例化预处理器（词表大小10000，句子最大长度888）
    stopwords_file = os.path.join(dataset_dir, "stopwords.txt")
    preprocessor = TextPreprocessor(vocab_size=10000, max_len=888, stopwords_path=stopwords_file)
    
    # 1. 对训练集进行单次扫描：清洗、分词、写入 clean 文件并统计词频
    print("=" * 60)
    print("步骤 1: 处理训练集并统计词频...")
    word_counter = preprocess_and_count(preprocessor, train_raw, train_clean, is_train=True)
    print("=" * 60)
    
    # 2. 构建并保存词表
    print("步骤 2: 构建词级别词典 (Vocab)...")
    min_freq = 5
    frequent_words = [word for word, count in word_counter.items() if count >= min_freq]
    print(f"[A] 训练集总词汇量: {len(word_counter)}，频数>= {min_freq} 的词汇量: {len(frequent_words)}")
    
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_words[:preprocessor.vocab_size - 2]]
    
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(top_words):
        word2id[word] = idx + 2
        
    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=4)
    print(f"[A] 词表成功保存至: {vocab_json}，最终词典大小为: {len(word2id)}")
    print("=" * 60)
    
    # 3. 处理验证集和测试集
    print("步骤 3: 预处理验证集与测试集...")
    preprocess_and_count(preprocessor, val_raw, val_clean, is_train=False)
    preprocess_and_count(preprocessor, test_raw, test_clean, is_train=False)
    print("=" * 60)
    
    print("\n[A] 恭喜！所有数据集清洗分词及词典构建全部完成。")
    print(f"  - 导出的词典: {vocab_json}")
    print(f"  - 导出的分词训练集: {train_clean}")
    print(f"  - 导出的分词验证集: {val_clean}")
    print(f"  - 导出的分词测试集: {test_clean}")
    print("=" * 60)

if __name__ == "__main__":
    main()
