# -*- coding: utf-8 -*-
"""
分析训练集文本长度分布并统计词汇表(Vocab)中词语的出现次数。
1. 计算能够覆盖训练集 90% 样本的句子最大长度 (max_len)。
2. 统计 vocab.txt 和 word_vocab.json 中各个词在训练集中的出现次数，并保存统计结果。
"""

import os
import json
import collections

def main():
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    train_clean_path = os.path.join(dataset_dir, "cnews.train.clean.txt")
    train_raw_path = os.path.join(dataset_dir, "cnews.train.txt")
    vocab_txt_path = os.path.join(dataset_dir, "cnews.vocab.txt")
    vocab_json_path = os.path.join(dataset_dir, "word_vocab.json")
    
    # 检查分词后的训练集是否存在
    if os.path.exists(train_clean_path):
        print(f"正在读取分词后的训练集: {train_clean_path}")
        use_clean = True
        train_path = train_clean_path
    elif os.path.exists(train_raw_path):
        print(f"分词训练集未找到，将读取原始训练集: {train_raw_path}")
        use_clean = False
        train_path = train_raw_path
    else:
        print("错误: 找不到训练集文件（cnews.train.clean.txt 或 cnews.train.txt）！")
        return

    lengths = []
    word_counter = collections.Counter()
    
    # 如果读取原始文件，需要使用 TextPreprocessor 进行分词，这里我们尝试导入
    preprocessor = None
    if not use_clean:
        try:
            from src.data_utils import TextPreprocessor
            stopwords_file = os.path.join(dataset_dir, "stopwords.txt")
            preprocessor = TextPreprocessor(stopwords_path=stopwords_file)
            print("已成功加载 TextPreprocessor 对原始文本进行清洗分词...")
        except ImportError as e:
            print(f"导入 TextPreprocessor 失败: {e}，将直接按空格/字符切分原始文本。")

    print("开始处理训练集数据...")
    total_samples = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            content = parts[1]
            
            if use_clean:
                # clean 文件已经是空格分隔的词
                words = content.split()
            else:
                if preprocessor:
                    cleaned = preprocessor.clean_text(content)
                    words = preprocessor.segment_text(cleaned)
                else:
                    # 简易 fallback
                    words = list(content)
            
            lengths.append(len(words))
            word_counter.update(words)
            total_samples += 1
            if total_samples % 10000 == 0:
                print(f"  - 已处理 {total_samples} 条样本...")

    if not lengths:
        print("未读取到任何有效样本！")
        return

    # 1. 计算 max_len (能覆盖 90% 样本的长度)
    sorted_lengths = sorted(lengths)
    percentile_idx = int(len(sorted_lengths) * 0.95)
    max_len_90 = sorted_lengths[percentile_idx]
    
    mean_len = sum(lengths) / len(lengths)
    print("\n" + "="*50)
    print("【训练集文本长度统计】")
    print(f"总样本数: {total_samples}")
    print(f"最小长度: {sorted_lengths[0]}")
    print(f"最大长度: {sorted_lengths[-1]}")
    print(f"平均长度: {mean_len:.2f}")
    print(f"能够覆盖 95% 样本的 max_len: {max_len_90}")
    print("="*50)

    # 2. 统计词表中各个词的出现次数
    # A. 针对 cnews.vocab.txt
    if os.path.exists(vocab_txt_path):
        print(f"\n正在统计 cnews.vocab.txt 中的词频...")
        vocab_txt_words = []
        with open(vocab_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    vocab_txt_words.append(word)
        
        vocab_txt_counts = {}
        for word in vocab_txt_words:
            # 特殊标记处理
            if word == "<PAD>":
                vocab_txt_counts[word] = 0
            else:
                vocab_txt_counts[word] = word_counter[word]
        
        # 保存到文件
        out_txt_counts = os.path.join(dataset_dir, "cnews.vocab.counts.json")
        with open(out_txt_counts, 'w', encoding='utf-8') as f:
            json.dump(vocab_txt_counts, f, ensure_ascii=False, indent=4)
        print(f"cnews.vocab.txt 词频统计完成，已保存至: {out_txt_counts}")
        
        # 打印部分高频词和未出现的词
        sorted_txt_counts = sorted(vocab_txt_counts.items(), key=lambda x: x[1], reverse=True)
        print("\ncnews.vocab.txt 出现频次最高的前 10 个词:")
        for w, c in sorted_txt_counts[:10]:
            print(f"  '{w}': {c} 次")
            
        zero_count_words = [w for w, c in vocab_txt_counts.items() if c == 0]
        print(f"cnews.vocab.txt 中未在训练集中出现的词数 (含特殊标记): {len(zero_count_words)} (例如: {zero_count_words[:5]})")

    # B. 针对 word_vocab.json
    if os.path.exists(vocab_json_path):
        print(f"\n正在统计 word_vocab.json 中的词频...")
        try:
            with open(vocab_json_path, 'r', encoding='utf-8') as f:
                word_vocab = json.load(f)
            
            vocab_json_counts = {}
            for word in word_vocab.keys():
                if word in ["<PAD>", "<UNK>"]:
                    vocab_json_counts[word] = 0
                else:
                    vocab_json_counts[word] = word_counter[word]
            
            out_json_counts = os.path.join(dataset_dir, "word_vocab.counts.json")
            with open(out_json_counts, 'w', encoding='utf-8') as f:
                json.dump(vocab_json_counts, f, ensure_ascii=False, indent=4)
            print(f"word_vocab.json 词频统计完成，已保存至: {out_json_counts}")
            
            sorted_json_counts = sorted(vocab_json_counts.items(), key=lambda x: x[1], reverse=True)
            print("\nword_vocab.json 出现频次最高的前 10 个词:")
            for w, c in sorted_json_counts[:10]:
                print(f"  '{w}': {c} 次")
                
            zero_count_json = [w for w, c in vocab_json_counts.items() if c == 0]
            print(f"word_vocab.json 中未在训练集中出现的词数 (含特殊标记): {len(zero_count_json)} (例如: {zero_count_json[:5]})")
        except Exception as e:
            print(f"读取或处理 word_vocab.json 失败: {e}")

if __name__ == "__main__":
    main()
