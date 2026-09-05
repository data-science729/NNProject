# -*- coding: utf-8 -*-
import os
import json
import re

def is_chinese_char(c):
    return '\u4e00' <= c <= '\u9fff'

def is_english_char(c):
    return 'a' <= c <= 'z' or 'A' <= c <= 'Z'

def is_number_char(c):
    return '0' <= c <= '9'

def analyze_word(word):
    has_zh = any(is_chinese_char(c) for c in word)
    has_en = any(is_english_char(c) for c in word)
    has_num = any(is_number_char(c) for c in word)
    
    # Classification logic:
    # 1. Pure categories
    if has_zh and not has_en and not has_num:
        return "pure_zh"
    if has_en and not has_zh and not has_num:
        return "pure_en"
    if has_num and not has_zh and not has_en:
        return "pure_num"
    
    # 2. Mixed categories
    if has_zh or has_en or has_num:
        parts = []
        if has_zh: parts.append("zh")
        if has_en: parts.append("en")
        if has_num: parts.append("num")
        return "mixed_" + "_".join(parts)
        
    return "other" # punctuation, symbols, empty, etc.

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, "dataset", "word_vocab.json")
    
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} does not exist!")
        return

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # Exclude <PAD> and <UNK> to get the 9998 words
    exclude_tokens = {"<PAD>", "<UNK>"}
    words = [w for w in vocab.keys() if w not in exclude_tokens]
    
    print(f"Total words in vocabulary: {len(vocab)}")
    print(f"Analyzing {len(words)} words (excluding {exclude_tokens})...\n")

    # 1. Character-level counts
    char_zh_count = 0
    char_en_count = 0
    char_num_count = 0
    char_other_count = 0
    total_chars = 0

    for word in words:
        for c in word:
            total_chars += 1
            if is_chinese_char(c):
                char_zh_count += 1
            elif is_english_char(c):
                char_en_count += 1
            elif is_number_char(c):
                char_num_count += 1
            else:
                char_other_count += 1

    print("=" * 40)
    print("【1. 字符级别统计 (Character-Level Counts)】")
    print(f"总字符数: {total_chars}")
    print(f"  - 汉字字符数: {char_zh_count}")
    print(f"  - 英文文字符数: {char_en_count}")
    print(f"  - 数字字符数: {char_num_count}")
    print(f"  - 其他字符数 (标点/符号等): {char_other_count}")
    print("=" * 40)

    # 2. Word-level counts (exclusive classification)
    word_categories = {
        "pure_zh": [],
        "pure_en": [],
        "pure_num": [],
        "mixed_zh_en": [],
        "mixed_zh_num": [],
        "mixed_en_num": [],
        "mixed_zh_en_num": [],
        "other": []
    }

    for word in words:
        cat = analyze_word(word)
        if cat in word_categories:
            word_categories[cat].append(word)
        else:
            word_categories["other"].append(word)

    print("\n" + "=" * 40)
    print("【2. 词级别统计 (Word-Level Classification)】")
    print(f"  - 纯汉字词 (e.g. '中国', '月'): {len(word_categories['pure_zh'])}")
    print(f"  - 纯英文词 (e.g. 'pad', 'hello'): {len(word_categories['pure_en'])}")
    print(f"  - 纯数字词 (e.g. '10', '2008'): {len(word_categories['pure_num'])}")
    print(f"  - 汉英混合词 (e.g. 'A股'): {len(word_categories['mixed_zh_en'])}")
    print(f"  - 汉数混合词 (e.g. '10月', '十一届'): {len(word_categories['mixed_zh_num'])}")
    print(f"  - 英数混合词 (e.g. '3D', 'iPhone6'): {len(word_categories['mixed_en_num'])}")
    print(f"  - 汉英数混合词: {len(word_categories['mixed_zh_en_num'])}")
    print(f"  - 其他词 (如纯标点/特殊符号): {len(word_categories['other'])}")
    print(f"总计: {sum(len(v) for v in word_categories.values())} 个词")
    print("=" * 40)

    # 3. Overlapping counts (How many words contain at least one...)
    contains_zh = sum(1 for w in words if any(is_chinese_char(c) for c in w))
    contains_en = sum(1 for w in words if any(is_english_char(c) for c in w))
    contains_num = sum(1 for w in words if any(is_number_char(c) for c in w))

    print("\n" + "=" * 40)
    print("【3. 包含特定字符类型的词数统计 (含有即可)】")
    print(f"  - 含有汉字的词数: {contains_zh}")
    print(f"  - 含有英文的词数: {contains_en}")
    print(f"  - 含有数字的词数: {contains_num}")
    print("=" * 40)

    # Print some examples of non-pure words to verify correctness
    print("\n【各分类示例】")
    for cat, wl in word_categories.items():
        print(f"  - {cat} (前 5 个): {wl[:5]}")

if __name__ == "__main__":
    main()
