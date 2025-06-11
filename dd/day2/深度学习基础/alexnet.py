import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 卷积层部分
        self.features = nn.Sequential(
            # 卷积层 1：输入通道 3（RGB 图像），输出通道 48（可根据原结构对应调整，原论文中是 96 等，这里为适配示例可灵活改），卷积核 11x11，步长 4，填充 2
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 最大池化层 1：池化核 3x3，步长 2
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 卷积层 2：输入通道 48，输出通道 128，卷积核 5x5，填充 2
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 最大池化层 2：池化核 3x3，步长 2
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 卷积层 3：输入通道 128，输出通道 192，卷积核 3x3，填充 1
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积层 4：输入通道 192，输出通道 192，卷积核 3x3，填充 1
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积层 5：输入通道 192，输出通道 128，卷积核 3x3，填充 1
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 最大池化层 3：池化核 3x3，步长 2
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 全连接层部分
        self.classifier = nn.Sequential(
            #  Dropout 层，随机丢弃概率 0.5，用于防止过拟合
            nn.Dropout(),
            # 全连接层 1：输入维度需根据前面卷积层输出计算，这里经前面卷积池化后大致是 128 * 6 * 6（不同输入尺寸需留意，原 AlexNet 输入是 224x224 等情况），输出维度 2048
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            # 全连接层 2：输入 2048，输出 2048
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            # 全连接层 3：输入 2048，输出 num_classes（分类类别数，如 1000 类等）
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        # 先经过卷积层特征提取
        x = self.features(x)
        # 将卷积输出展平，为全连接层做准备
        x = x.view(x.size(0), 128 * 6 * 6)
        # 经过全连接层分类
        x = self.classifier(x)
        return x


# 测试网络结构
if __name__ == "__main__":
    # 模拟输入， batch_size=1，3 通道，224x224 尺寸图像（AlexNet 经典输入尺寸，可按需调整）
    input_tensor = torch.randn(1, 3, 224, 224)
    model = AlexNet()
    output = model(input_tensor)
    print(output.shape)