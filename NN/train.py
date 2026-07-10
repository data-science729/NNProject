# -*- coding: utf-8 -*-
"""
THUCNews 深度学习模型训练与评估脚本 (同学 C 与 同学 D 核心交集)
功能：
1. 从命令行读取模型选择 (--model: textcnn 或 lstm)
2. 加载已经分词清洗的数据集 (.clean.txt)
3. 加载同学 B 训练对齐的预训练词嵌入权重矩阵 (embedding_matrix.npy)
4. 实现训练与验证循环 (支持早停机制，保存验证集效果最佳的模型)
5. 在独立测试集上进行最终性能测试
6. 自动计算 10 个分类别各自的 Precision, Recall, F1-score 并输出分类报告
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# 导入自定义的数据包和模型包
from src.data_utils import TextPreprocessor, THUCNewsDataset, CLASSES
from src.models import TextCNN, BiLSTM_Attention

def get_args():
    parser = argparse.ArgumentParser(description="THUCNews Text Classification")
    parser.add_argument("--model", type=str, default="textcnn", choices=["textcnn", "lstm"], 
                        help="选择训练的模型结构: textcnn 或 lstm")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (Epochs)")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小 (Batch Size)")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率 (Learning Rate)")
    parser.add_argument("--freeze", action="store_true", help="是否冻结 Word2Vec 预训练权重 (默认不冻结，即在训练中微调)")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    eval_loss = total_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc, all_preds, all_targets

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"【第三部分】 神经网络模型训练与评估")
    print(f"运行设备: {device}")
    if torch.cuda.is_available():
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # 1. 定义相关路径
    dataset_dir = r"c:\Users\asus\PycharmProjects\NNProject\NN\dataset"
    train_clean = os.path.join(dataset_dir, "cnews.train.clean.txt")
    val_clean = os.path.join(dataset_dir, "cnews.val.clean.txt")
    test_clean = os.path.join(dataset_dir, "cnews.test.clean.txt")
    
    vocab_json = os.path.join(dataset_dir, "word_vocab.json")
    embedding_npy = os.path.join(dataset_dir, "embedding_matrix.npy")
    model_save_path = os.path.join(dataset_dir, f"best_model_{args.model}.pth")
    
    # 2. 载入数据集 (采用优化后的 DataLoader，秒级载入)
    print("\n[C] 正在从 clean 文件中快速载入数据集...")
    preprocessor = TextPreprocessor(vocab_size=10000, max_len=888)
    
    train_dataset = THUCNewsDataset(train_clean, preprocessor, is_train=False, vocab_path=vocab_json)
    val_dataset = THUCNewsDataset(val_clean, preprocessor, is_train=False, vocab_path=vocab_json)
    test_dataset = THUCNewsDataset(test_clean, preprocessor, is_train=False, vocab_path=vocab_json)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. 载入同学 B 导出的预训练词向量矩阵
    print(f"\n[C] 正在加载预训练 Word2Vec Embedding 矩阵: {embedding_npy}")
    embedding_weights = np.load(embedding_npy)
    pretrained_embeddings = torch.from_numpy(embedding_weights).float()
    
    # 4. 初始化模型
    print(f"\n[C] 正在构建模型: {args.model.upper()} ...")
    if args.model == "textcnn":
        # 卷积核尺寸 3,4,5，过滤器通道 256
        model = TextCNN(
            vocab_size=len(preprocessor.word2id), 
            embedding_dim=100, 
            num_classes=10, 
            pretrained_embeddings=pretrained_embeddings,
            freeze=args.freeze
        )
    else:
        # BiLSTM：隐层 128 双向，堆叠 2 层
        model = BiLSTM_Attention(
            vocab_size=len(preprocessor.word2id), 
            embedding_dim=100, 
            hidden_dim=128, 
            num_classes=10,
            num_layers=2,
            pretrained_embeddings=pretrained_embeddings,
            freeze=args.freeze
        )
        
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. 训练与验证循环
    print(f"\n[C] 开始训练，共 {args.epochs} 轮 (Epochs)...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练一轮
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证一轮
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | 耗时: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # 保存最佳模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> 发现更好的模型，已保存至: {model_save_path}")
            
    # 6. 【同学 D 核心评估环节】 加载最佳权重，在测试集上进行最终性能测试
    print("\n" + "=" * 60)
    print("【第四部分】 在独立测试集上进行性能评估与测试...")
    print("=" * 60)
    print(f"正在加载最佳模型权重: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)
    print(f"\n测试集总准确率 (Test Accuracy): {test_acc*100:.2f}%")
    print(f"测试集平均损失 (Test Loss): {test_loss:.4f}")
    
    # 输出分类别详细分类报告 (同学 D 写实习报告的核心图表！)
    print("\n[D] 测试集详细分类报告 (Classification Report):")
    report = classification_report(targets, preds, target_names=CLASSES, digits=4)
    print(report)
    
    # 打印混淆矩阵
    print("[D] 混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(targets, preds)
    print(cm)
    
    print("\n" + "=" * 60)
    print(f"[C/D] 神经网络分类阶段全部完成！最佳模型已生成。")
    print("=" * 60)

if __name__ == "__main__":
    main()
