深度学习基础与卷积神经网络学习笔记
一、深度学习基础
（一）深度学习概述
定义：深度学习是机器学习的一个分支，通过构建多层神经网络结构学习数据中的复杂模式和特征。
应用领域：图像识别、语音识别、自然语言处理、推荐系统等。
（二）神经网络基础
神经元模型：模仿生物神经元，接收输入信号，经加权求和、激活函数处理后输出。
激活函数：为神经网络引入非线性，常用包括 Sigmoid、ReLU、Tanh 等。
损失函数：衡量预测值与真实值的差异，常见有均方误差（MSE）、交叉熵损失等。
优化算法：如梯度下降法及其变体（随机梯度下降、Adam 等），用于调整网络权重。
二、卷积神经网络（CNN）
（一）卷积操作
卷积核：提取输入数据的局部特征。
步长（Stride）：卷积核在输入数据上滑动的步长。
填充（Padding）：在输入边缘添加像素，保持输出尺寸。
import torch
import torch.nn.functional as F

# 定义输入矩阵和卷积核
input_matrix = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])
kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

# 调整输入维度以适配卷积运算
input_tensor = torch.reshape(input_matrix, (1, 1, 5, 5))
kernel_tensor = torch.reshape(kernel, (1, 1, 3, 3))

# 执行二维卷积操作
output = F.conv2d(input=input_tensor, weight=kernel_tensor, stride=1)
print("卷积输出结果：")
print(output)

（二）卷积神经网络的结构
卷积层：通过卷积操作提取特征。
池化层：降低特征图尺寸，减少计算量，常用最大池化和平均池化。
全连接层：将特征图展平后用于分类或回归。
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 卷积层1 + 池化层1
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            # 卷积层2 + 池化层2
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            # 卷积层3 + 池化层3
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            # 全连接层
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 测试模型
model = CNNModel()
input_tensor = torch.ones((64, 3, 32, 32))
output_tensor = model(input_tensor)
print(f"模型输出形状：{output_tensor.shape}")  # 预期输出：torch.Size([64, 10])
结构说明：
输入尺寸：32×32×3（RGB 图像）
经过 3 次卷积 + 池化后，特征图尺寸变为 4×4×64，展平后为 1024 维
全连接层将 1024 维特征映射到 64 维，最终输出 10 维（10 分类任务）

三、模型训练与测试
（一）数据集准备
数据集：CIFAR10（6 万张 32×32 彩色图像，10 个类别）。
数据加载：使用DataLoader批量加载数据，支持打乱顺序。
import torchvision
from torch.utils.data import DataLoader

# 加载训练集（自动下载，首次运行需联网）
train_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_chen",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 加载测试集
test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_chen",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集大小：{len(train_dataset)}")  # 输出：50000
print(f"测试集大小：{len(test_dataset)}")   # 输出：10000

（二）模型训练
损失函数：交叉熵损失（适用于多分类任务）。
优化器：随机梯度下降（SGD）。
训练流程：前向传播计算损失，反向传播更新权重。
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 初始化模型、损失函数和优化器
model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 修正：变量名改为optimizer

# 初始化TensorBoard
writer = SummaryWriter("logs_train")
total_train_step = 0
epochs = 10

for epoch in range(epochs):
    print(f"===== 第 {epoch+1}/{epochs} 轮训练开始 =====")
    for batch_idx, (images, targets) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练进度
        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"训练步数：{total_train_step}，Loss：{loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

writer.close()

(三）模型测试与保存
测试流程：在测试集上计算损失和准确率。
模型保存：将训练好的模型参数保存到文件。
# 测试模型
model.eval()  # 设置为评估模式（关闭Dropout等）
total_test_loss = 0.0
total_accuracy = 0

with torch.no_grad():
    for images, targets in test_loader:
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        total_test_loss += loss.item()
        
        # 计算准确率
        predictions = outputs.argmax(1)
        total_accuracy += (predictions == targets).sum().item()

# 计算平均损失和准确率
test_loss = total_test_loss / len(test_loader)
test_accuracy = total_accuracy / len(test_dataset)

print(f"测试集平均Loss：{test_loss:.4f}")
print(f"测试集准确率：{test_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), "model_save/cnn_model.pth")
print("模型已保存至 model_save/cnn_model.pth")

四、总结
深度学习基础：掌握了神经网络的核心组件（神经元、激活函数、损失函数、优化器）。
卷积神经网络：理解卷积操作的原理、CNN 的层次设计（卷积层 + 池化层 + 全连接层）。
模型训练流程：从数据加载、模型定义、训练到测试的完整流程，以及 TensorBoard 可视化和模型保存方法。