# -*- coding: utf-8 -*-
"""
THUCNews 数据预处理与清洗模块
功能：
1. 读取文本并处理 UTF-8 编码
2. 保留上下文（保留句号、叹号、问号等标点符号，去除杂质字符）
3. 使用 Jieba 分词
4. 过滤中文停用词
5. 自动构建词级别词典（Vocab）并保存
6. 将文本转化为固定长度的 ID 序列（Padding/Truncating）
7. 封装为 PyTorch 的 Dataset 接口
"""

import os
import re
import json
import collections
import jieba
import torch
from torch.utils.data import Dataset, DataLoader

# 定义默认的10个类别
CLASSES = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
LABEL2ID = {label: idx for idx, label in enumerate(CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(CLASSES)}

# 预设的常用中文停用词（当外部停用词文件不存在时作为兜底使用）
DEFAULT_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '与', '为', '之', '而', '以', '于', '也', '而', '着', 
    '这', '那', '那', '就', '都', '而', '及', '同', '与', '及', '等', '被', '让', '把', '但', '但是', 
    '所以', '因为', '如果', '并且', '而且', '一个', '两个', '我们', '你们', '他们', '它们', '什么', 
    '怎么', '怎么', '这里', '那里', '这', '那', '它', '他', '她', '它', '们', '地', '得', '着',
}

class TextPreprocessor:
    def __init__(self, vocab_size=10000, max_len=888, stopwords_path=None):
        """
        初始化预处理器
        :param vocab_size: 词典最大词数（保留高频词）
        :param max_len: 文本固定长度
        :param stopwords_path: 停用词表路径
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2id = {}
        self.id2word = {}
        
        # 加载停用词表
        self.stopwords = set(DEFAULT_STOPWORDS)
        if stopwords_path and os.path.exists(stopwords_path):
            self.load_stopwords(stopwords_path)
            
    def load_stopwords(self, path):
        """从文件加载停用词"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:  
                    word = line.strip()
                    if word:
                        self.stopwords.add(word)
            print(f"[A] 成功从 {path} 加载了 {len(self.stopwords)} 个停用词。")
        except Exception as e:
            print(f"[A] 加载停用词失败: {e}，将使用内置的默认停用词表。")

    def clean_text(self, text):
        """
        文本清洗：保留基本编码和上下文。
        - 移除网页 HTML 标签
        - 移除奇怪的控制字符、无意义英文字符噪声等
        - 保留关键的中文标点符号（，。！？：），这些符号能表示句子的语气和自然断句，对于理解上下文非常重要。
        """
        if not text:
            return ""
        # 1. 移除网页标签
        text = re.sub(r'<[^>]+>', '', text)
        # 2. 仅保留汉字、英文字母、数字以及常用中文标点符号
        # 中文标点符号范围：\u3000-\u303f, \uFF00-\uFFEF
        pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？：、]')
        text = pattern.sub('', text)
        return text

    def segment_text(self, text):
        """使用 jieba 分词"""
        words = jieba.lcut(text)
        # 过滤停用词、单字中的空白字符
        words = [w.strip() for w in words if w.strip() and w.strip() not in self.stopwords]
        return words

    def build_vocab(self, train_data_path, min_freq=5):
        """
        基于训练集构建词典（同学 A 的高光展示点：统计分析与词表构建）
        """
        print(f"[A] 开始基于 {train_data_path} 构建词表...")
        word_counter = collections.Counter()  #格式(词语，词频)
        
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                content = parts[1]
                clean_content = self.clean_text(content)
                words = self.segment_text(clean_content)
                word_counter.update(words)
                
        # 过滤低频词
        frequent_words = [word for word, count in word_counter.items() if count >= min_freq]
        print(f"[A] 训练集总词汇量: {len(word_counter)}，频数大于等于{min_freq}的词汇量: {len(frequent_words)}")
        
        # 选取高频词构建词表
        sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)  #按照词频降序排序
        top_words = [word for word, count in sorted_words[:self.vocab_size - 2]] #  截取前N个高频词 －2留出给 PAD 和 UNK
        
        # 构建映射字典
        # 特殊标记：<PAD> 用于填充对齐，<UNK> 用于处理未登录词
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(top_words):
            self.word2id[word] = idx + 2
            
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        print(f"[A] 词表构建完成，最终词表大小为: {len(self.word2id)}")
        return self.word2id

    def save_vocab(self, vocab_path):
        """保存词表为 JSON 文件"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=4)
        print(f"[A] 词表成功保存至: {vocab_path}")

    def load_vocab(self, vocab_path):
        """加载已保存的词表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        print(f"[A] 词表成功从 {vocab_path} 加载，词表大小为: {len(self.word2id)}")

    def text_to_ids(self, text):
        """将文本转换为 ID 序列"""
        clean_content = self.clean_text(text)
        words = self.segment_text(clean_content)
        ids = [self.word2id.get(w, self.word2id['<UNK>']) for w in words]
        
        # 截断或填充到 max_len 长度
        if len(ids) < self.max_len:
            ids = ids + [self.word2id['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def clean_text_to_ids(self, words_str):
        """针对已经清洗分词好的空格分隔字符串转换为 ID 序列 (免Jieba分词，速度极快)"""
        words = words_str.split()
        ids = [self.word2id.get(w, self.word2id['<UNK>']) for w in words]
        
        # 截断或填充到 max_len 长度
        if len(ids) < self.max_len:
            ids = ids + [self.word2id['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids


class THUCNewsDataset(Dataset):
    def __init__(self, data_path, preprocessor, is_train=False, vocab_path=None):
        """
        THUCNews 数据集封装类 (PyTorch Dataset)
        """
        self.preprocessor = preprocessor
        self.data = []
        self.labels = []
        
        # 1. 词典准备：如果是训练集，并且没有指定已存词典，则需要构建词典；否则尝试加载
        if is_train:
            if vocab_path and os.path.exists(vocab_path):
                self.preprocessor.load_vocab(vocab_path)
            else:
                self.preprocessor.build_vocab(data_path)
                if vocab_path:
                    self.preprocessor.save_vocab(vocab_path)
        else:
            if vocab_path and os.path.exists(vocab_path):
                self.preprocessor.load_vocab(vocab_path)
            elif not self.preprocessor.word2id:
                raise ValueError("[A] 评估数据集必须指定并加载已有的词表（vocab_path）或者预处理器中已包含词表。")

        # 2. 读取并预处理数据
        print(f"[A] 开始读取并解析文件 {data_path}...")
        count = 0
        is_clean_file = data_path.endswith('.clean.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                label, content = parts[0], parts[1]
                if label not in LABEL2ID:
                    continue
                
                # 转换为 ID 列表
                if is_clean_file:
                    input_ids = self.preprocessor.clean_text_to_ids(content)
                else:
                    input_ids = self.preprocessor.text_to_ids(content)
                    
                self.data.append(input_ids)
                self.labels.append(LABEL2ID[label])
                count += 1
                if count % 10000 == 0:
                    print(f"  - 已处理并缓存 {count} 条文本数据...")
                    
        print(f"[A] 数据读取完毕，共 {len(self.data)} 条有效样本。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ==========================================
# 调试与验证接口
# ==========================================
if __name__ == "__main__":
    # 本地跑一个小型测试，验证代码的正确性
    print("[A] 开始运行 data_utils.py 的本地自检流程...")
    
    # 模拟一段数据清洗分词流程
    test_text = "<html>体育新闻：</html> 詹姆斯在今天的 NBA 季后赛中，砍下了 28分、18个篮板和5次助攻！这也太厉害了吧！"
    preprocessor = TextPreprocessor(vocab_size=100, max_len=15)
    
    print("\n[测试] 原始文本:")
    print(test_text)
    
    cleaned = preprocessor.clean_text(test_text)
    print("\n[测试] 清洗后保留上下文的文本:")
    print(cleaned)
    
    words = preprocessor.segment_text(cleaned)
    print("\n[测试] Jieba分词及停用词过滤后的词语列表:")
    print(words)
    
    # 测试词表的构建
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_train_file = os.path.join(base_dir, "dataset", "cnews.val.txt")
    if os.path.exists(sample_train_file):
        print("\n[测试] 尝试使用较小的验证集验证词表构建与 DataLoader 加载...")
        vocab = preprocessor.build_vocab(sample_train_file, min_freq=2)
        print("词表前 10 个词:", list(vocab.keys())[:10])

        # 封装为 Dataset 和 DataLoader
        test_dataset = THUCNewsDataset(sample_train_file, preprocessor, is_train=False, vocab_path=None)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True,pin_memory=True)
        
        print("\n[测试] 从 DataLoader 中读取一个 Batch 数据:")
        for x, y in test_loader:
            print("输入特征尺寸:", x.shape)
            print("对应标签编号:", y)
            print("样例输入第一条序列 (ID):", x[0])
            # 将 ID 还原回词语展示
            decoded_words = [preprocessor.id2word.get(idx.item(), '<UNK>') for idx in x[0]]
            print("样例输入第一条序列 (还原词语):", "/".join(decoded_words))
            break
    print("\n[A] 自检流程结束！代码一切正常。")
