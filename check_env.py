# -*- coding: utf-8 -*-
import os
import sys

def main():
    print("=" * 50)
    print("环境与数据集验证脚本")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    
    # 1. 检查数据集文件是否存在
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    files = ["cnews.train.txt", "cnews.val.txt", "cnews.test.txt", "cnews.vocab.txt"]
    
    print("\n--- 1. 数据集文件状态 ---")
    all_exist = True
    for f in files:
        path = os.path.join(dataset_dir, f)
        exists = os.path.exists(path)
        print(f"文件 {f}: {'存在' if exists else '不存在'}")
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - 大小: {size_mb:.2f} MB")
        else:
            all_exist = False
            
    if all_exist:
        print(">> 所有核心数据集文件均已就绪。")
    else:
        print(">> 警告: 缺少部分数据集文件，请检查 dataset 目录。")
        
    # 2. 检查依赖库是否安装
    print("\n--- 2. 依赖库安装状态 ---")
    libraries = ["jieba", "gensim", "torch"]
    for lib in libraries:
        try:
            mod = __import__(lib)
            version = getattr(mod, "__version__", "未知版本")
            print(f"库 {lib:7s}: 已安装 (版本: {version})")
            if lib == "torch":
                import torch
                print(f"  - CUDA (GPU) 是否可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  - 可用 GPU 数量: {torch.cuda.device_count()}")
                    print(f"  - 当前 GPU 名称: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print(f"库 {lib:7s}: 未安装 [X] (可通过 pip install {lib} 安装)")
            
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
