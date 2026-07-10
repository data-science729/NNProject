import torch
import time

# 1. 检测 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

if cuda_available:
    # 输出显卡设备名称
    device_name = torch.cuda.get_device_name(0)
    print(f"当前使用的显卡: {device_name}")

    # 2. 跑一个简单的 GPU 矩阵乘法测试
    device = torch.device("cuda")

    # 创建两个大矩阵并移动到 GPU
    size = 10000
    print(f"正在创建大小为 {size}x{size} 的矩阵...")
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    # 开始计算
    print("开始 GPU 矩阵乘法计算...")
    start_time = time.time()
    z = torch.matmul(x, y)

    # 确保 GPU 计算完成再计时
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"计算完成！耗时: {end_time - start_time:.4f} 秒")
    print(f"计算结果设备: {z.device}")
else:
    print("❌ 未检测到 CUDA，当前运行在 CPU 模式。请检查驱动或 PyTorch 安装版本。")