# 《深度学习代码学习笔记》  


## 一、代码1：`train_alex.py`（基于AlexNet的图像分类训练代码）  
### （一）代码功能  
本代码实现基于AlexNet架构的图像分类模型训练，使用自定义数据集。  


### （二）代码结构与关键点  
#### 1. 数据集加载  
- **自定义数据集**：使用`ImageTxtDataset`类通过`train.txt`文件加载图像路径和标签，图像存储于`D:\dataset\image2\train`。  
- **数据预处理**：  
  - `transforms.Resize(224)`：调整图像至224×224，适配AlexNet输入。  
  - `transforms.RandomHorizontalFlip()`：随机水平翻转，增强数据多样性。  
  - `transforms.ToTensor()`：转换图像为张量。  
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`：基于ImageNet数据集的归一化参数。  

#### 2. 模型定义  
- **简化AlexNet结构**：  
  - 5个卷积层+3个全连接层，卷积层使用`MaxPool2d`下采样。  
  - 输出层维度为10，适配10类分类任务。  
  - 输入为3通道RGB图像。  

#### 3. 训练流程  
- **数据加载**：`DataLoader`批量加载数据，batch_size=64。  
- **损失与优化**：  
  - 交叉熵损失函数（`CrossEntropyLoss`）。  
  - SGD优化器，lr=0.01，动量=0.9。  
- **日志与保存**：  
  - 每500步记录训练损失，用TensorBoard可视化。  
  - 每个epoch结束后评估测试集，计算损失和准确率，并保存模型为`.pth`文件。  

#### 4. 测试流程  
- `torch.no_grad()`禁用梯度计算，减少内存消耗。  
- 计算测试集总损失和准确率，记录至TensorBoard。  


### （三）学习重点  
1. **自定义数据集应用**  
   - 掌握通过文本文件加载图像路径和标签的方法。  
   - 理解数据预处理对模型输入的适配逻辑。  
2. **AlexNet架构解析**  
   - 明确卷积层、池化层、全连接层的功能及协作方式。  
   - 学习根据任务调整输出层维度的方法。  
3. **PyTorch训练框架**  
   - 掌握数据加载、损失计算、优化器更新、模型评估的全流程。  
   - 学习TensorBoard可视化训练过程的方法。  
4. **数据增强技术**  
   - 理解随机水平翻转对模型泛化能力的提升作用。  


## 二、代码2：`transformer.py`（基于Transformer的Vision Transformer模型代码）  
### （一）代码功能  
实现基于Transformer架构的Vision Transformer（ViT）模型，处理序列化图像数据。  


### （二）代码结构与关键点  
#### 1. 核心模块定义  
- **FeedForward模块**：  
  - 线性层→GELU激活→Dropout→线性层，搭配`LayerNorm`归一化。  
- **Attention模块**：  
  - 多头自注意力机制，通过`Softmax`计算注意力权重。  
  - 使用`rearrange`和`repeat`处理张量形状。  
- **Transformer模块**：  
  - 多层Transformer层，每层包含注意力模块和前馈模块，使用残差连接（`x = attn(x) + x`）。  
- **ViT模型**：  
  - 将图像序列化为patches，结合位置嵌入（`pos_embedding`）和类别嵌入（`cls_token`）。  
  - 通过Transformer处理后，经全连接层输出分类结果。  

#### 2. 模型输入输出  
- **输入**：序列化图像`time_series`，形状为`(batch_size, channels, seq_len)`。  
- **输出**：分类logits，形状为`(batch_size, num_classes)`。  

#### 3. 测试代码  
- 实例化ViT模型，输入随机张量`time_series`，验证输出形状正确性。  


### （三）学习重点  
1. **Transformer架构原理**  
   - 理解多头自注意力机制、前馈网络、残差连接的协同逻辑。  
   - 明确`LayerNorm`和`Dropout`在模型中的正则化作用。  
2. **Vision Transformer实现**  
   - 掌握图像序列化（patches）的处理方式。  
   - 理解位置嵌入和类别嵌入对序列建模的重要性。  
3. **einops库应用**  
   - 学习`rearrange`和`repeat`函数简化张量操作的技巧。  
4. **模型输入输出格式**  
   - 明确ViT模型对序列化图像的处理逻辑及分类结果的输出形式。  


## 三、总结  
今日学习了两类深度学习模型实现：  
1. **`train_alex.py`**：聚焦CNN架构（AlexNet），重点在自定义数据集处理、数据增强及PyTorch训练流程。  
2. **`transformer.py`**：聚焦Transformer架构（ViT），重点在序列建模、多头注意力机制及einops工具使用。  

通过实践，深化了对CNN与Transformer架构差异的理解，掌握了自定义数据集处理和模型可视化技巧。