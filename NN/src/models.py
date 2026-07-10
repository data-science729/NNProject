# -*- coding: utf-8 -*-
"""
THUCNews 文本分类深度学习模型定义 (同学 C 核心任务)
包含两个经典文本分类网络结构：
1. TextCNN (经典卷积结构，擅长局部 n-gram 特征捕捉，速度极快)
2. BiLSTM_Attention (双向长短期记忆网络 + 注意力机制，擅长序列及长短依赖学习)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, filter_sizes=(3, 4, 5), num_filters=256, pretrained_embeddings=None, freeze=False):
        """
        TextCNN 模型初始化
        :param vocab_size: 词表大小 (10000)
        :param embedding_dim: 词向量维度 (100)
        :param num_classes: 分类类别数 (10)
        :param filter_sizes: 卷积核大小列表 (常见为 3, 4, 5 对应短语、词组特征)
        :param num_filters: 卷积核通道数 (常见为 256)
        :param pretrained_embeddings: 预训练词嵌入权重矩阵 (同学 B 导出的 embedding_matrix)
        :param freeze: 是否冻结词向量权重 (True 表示不更新词向量，False 表示在反向传播中微调)
        """
        super(TextCNN, self).__init__()
        
        # 1. 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        # 2. 卷积层列表 (1D 卷积，分别提取不同大小滑动窗口的特征)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        # 3. Dropout 防过拟合
        self.dropout = nn.Dropout(0.5)
        
        # 4. 全连接分类器
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # 输入 x 维度: [batch_size, seq_len] -> 例如 [64, 888]
        
        # 1. 词嵌入查表 -> [batch_size, seq_len, embedding_dim] -> 例如 [64, 888, 100]
        out = self.embedding(x)
        
        # 2. 变换维度适应 1D 卷积的要求 -> [batch_size, embedding_dim, seq_len] -> 例如 [64, 100, 888]
        out = out.transpose(1, 2)
        
        # 3. 对各个尺寸的卷积核进行卷积操作、ReLU 激活以及 Max-Pooling (时间维度最大池化)
        pooled_outputs = []
        for conv in self.convs:
            # 卷积后形状: [batch_size, num_filters, seq_len - kernel_size + 1]
            c = F.relu(conv(out))
            # 在时间维度进行全局最大池化 -> 消除句子长度带来的影响，取最强特征
            # 池化后形状: [batch_size, num_filters, 1]
            p = F.max_pool1d(c, kernel_size=c.size(2))
            # 挤压多余维度变为二维: [batch_size, num_filters]
            pooled_outputs.append(p.squeeze(2))
            
        # 4. 拼接不同尺度的特征向量 -> [batch_size, num_filters * len(filter_sizes)] -> 例如 [64, 256 * 3]
        out = torch.cat(pooled_outputs, dim=1)
        
        # 5. Dropout & 全连接分类层 -> [batch_size, num_classes] -> 例如 [64, 10]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, pretrained_embeddings=None, freeze=False):
        """
        BiLSTM + Attention 序列模型
        :param vocab_size: 词表大小 (10000)
        :param embedding_dim: 词向量维度 (100)
        :param hidden_dim: LSTM 隐层神经元数 (如 128)
        :param num_classes: 分类类别数 (10)
        :param num_layers: 双向 LSTM 堆叠层数 (如 2 层)
        :param pretrained_embeddings: 预训练权重
        :param freeze: 是否冻结词向量
        """
        super(BiLSTM_Attention, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True, 
            dropout=0.5 if num_layers > 1 else 0.0
        )
        
        # 自注意力层机制 (Attention weights)
        # 输入为 lstm 的双向隐状态拼接长度: hidden_dim * 2，转换输出为单个注意力打分标量
        self.attention_w = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        nn.init.xavier_uniform_(self.attention_w) # 初始化权重
        
        self.dropout = nn.Dropout(0.5)
        # 全连接分类器 (双向隐藏状态拼接所以输入维度为 hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len]
        
        # 1. 词向量化 -> [batch_size, seq_len, embedding_dim]
        out = self.embedding(x)
        
        # 2. 喂入双向 LSTM -> output 的维度是 [batch_size, seq_len, hidden_dim * 2]
        output, (h_n, c_n) = self.lstm(out)
        
        # 3. 自注意力机制 (Self-Attention Layer)
        # 计算打分: [batch_size, seq_len, hidden_dim * 2] x [hidden_dim * 2, 1] -> [batch_size, seq_len, 1]
        score = torch.matmul(output, self.attention_w)
        # 在序列（时间）维度上进行 Softmax，计算每个词汇的注意力权重分配比例
        attn_weights = F.softmax(score, dim=1) # 形状: [batch_size, seq_len, 1]
        
        # 将权重应用于 LSTM 状态向量中进行加权求和，把整个长句子凝聚为一个特征向量
        # [batch_size, hidden_dim * 2, seq_len] x [batch_size, seq_len, 1] -> [batch_size, hidden_dim * 2]
        context_vector = torch.sum(output * attn_weights, dim=1) # 形状: [batch_size, hidden_dim * 2]
        
        # 4. 全连接层分类输出 -> [batch_size, num_classes]
        out = self.dropout(context_vector)
        out = self.fc(out)
        return out
