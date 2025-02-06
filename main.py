import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from MyDataset import CustomDataset
from conv2former import Conv2Former

# ----------------------------------------
# 参数配置
# ----------------------------------------
config_1 = {
    # 数据集路径（需包含train/和valid/子文件夹，每个子文件夹按类别存放图片）
    # "data_dir": "/data/Projects/Python/Swim/traffic_dataset",
    "data_dir": "/data/Workspace/CIC-IoTDataset2023/bin-class",
    "num_classes": 2,  # 类别数
    "batch_size": 16384,
    "lr": 1e-4,  # 学习率
    "epochs": 500,
    "model_name": "Conv2Former",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

config_2 = {
    # 数据集路径（需包含train/和valid/子文件夹，每个子文件夹按类别存放图片）
    "data_dir": "/data/Projects/Python/Swim/traffic_dataset",
    # "data_dir": "/data/Workspace/dataset/png",
    "num_classes": 2,  # 类别数
    "batch_size": 128,
    "lr": 1e-3,  # 学习率
    "epochs": 100,
    "model_name": "Conv2Former",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
config = config_1

# 数据增强（适配单通道）
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 单通道归一化（假设像素范围0~1 → 归一化到[-1,1]）
    transforms.Normalize(mean=[0.5], std=[0.5])
])

valid_transform = transforms.Compose([
    # transforms.Resize(64),
    # transforms.CenterCrop(64),
    transforms.ToTensor(),
    # 单通道归一化（假设像素范围0~1 → 归一化到[-1,1]）
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = CustomDataset(os.path.join(config["data_dir"], "train"), transform=train_transform)
valid_dataset = CustomDataset(os.path.join(config["data_dir"], "valid"), transform=valid_transform)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                          num_workers=8,
                          pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# ----------------------------------------
# 模型定义（使用Conv2Former）
# ----------------------------------------
model = Conv2Former(dims=[32, 64, 128, 256], depths=[1, 1, 2, 1], num_classes=config["num_classes"])
model = model.to(config["device"])

# ----------------------------------------
# 训练与验证
# ----------------------------------------
class_weights = torch.tensor([1.0, 10.0])  # 正类权重设为 10
criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))
optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

best_val_acc = 0.0

scaler = GradScaler()  # 用于缩放梯度


def train():
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(config["device"])
        labels = labels.to(config["device"])
        optimizer.zero_grad()
        with autocast():  # 混合精度训练
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()  # 缩放梯度
        scaler.step(optimizer)  # 更新参数
        scaler.update()  # 更新缩放器

        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_dataset)


def validate():
    model.eval()
    correct = 0
    total = 0
    all_predicts = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])
            outputs = model(images)
            _, predict = torch.max(outputs, 1)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
            all_predicts.extend(predict.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    valid_acc = correct / total
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(classification_report(all_labels, all_predicts, zero_division=0))
    return valid_acc


# 训练循环
train_losses = []
valid_accs = []

for epoch in range(config["epochs"]):
    print(f"Epoch {epoch + 1}/{config['epochs']}")
    loss = train()
    train_losses.append(loss)
    valid_acc = validate()
    valid_accs.append(valid_acc)
    scheduler.step()

    # 保存最佳模型
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        torch.save(model.state_dict(), "intermediates/best_conv2former.pth")

    print(f"Train Loss: {loss:.4f}, Val Acc: {valid_acc:.4f}\n")

# ----------------------------------------
# 结果可视化
# ----------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(valid_accs, label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("intermediates/training_curve.png")
plt.show()
