import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights  # 导入 ResNet18_Weights

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理和增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),    # 随机裁剪并缩放到224x224
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.RandomRotation(20),        # 随机旋转
        transforms.ToTensor(),                # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),               # 先缩放至256
        transforms.CenterCrop(224),            # 然后中心裁剪为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# 加载数据
data_dir = 'train'  # 训练数据文件夹路径
test_dir = 'test'   # 测试数据文件夹路径

# 加载训练数据
full_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['train'])

# 按 80% 训练，20% 验证划分数据集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建数据加载器
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# 加载测试数据
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_size = len(test_dataset)

# 定义 ResNet18 模型（使用新的权重加载方法）
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用新的加载方法

# 修改最后的全连接层以适应我们的二分类任务
model.fc = nn.Linear(model.fc.in_features, 2)

# 移动模型到GPU（如果可用）
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练和评估模型的函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 加载数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向 + 反向 + 优化
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确度
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 只有在验证集上表现更好时才保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()

    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return model, history

# 训练模型
def main():
    model_trained, history = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

    # 保存训练好的模型
    torch.save(model_trained.state_dict(), 'pneumonia_model.pth')

    # 绘制训练过程中的损失和准确度曲线
    def plot_history(history):
        epochs = range(len(history['train_acc']))
        plt.figure()
        plt.plot(epochs, history['train_acc'], label='Training accuracy')
        plt.plot(epochs, history['val_acc'], label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.figure()
        plt.plot(epochs, history['train_loss'], label='Training loss')
        plt.plot(epochs, history['val_loss'], label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    plot_history(history)

    # 测试模型并计算评价指标
    def evaluate_model(model, test_loader):
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    evaluate_model(model_trained, test_loader)

# 防止多进程问题，只在主程序执行时启动训练
if __name__ == '__main__':
    main()
