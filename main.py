import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from MyDataset import CustomDataset


# -------------------------------
# 1. 设置随机种子，保证实验可复现
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# 创建保存中间结果的目录
os.makedirs("intermediates", exist_ok=True)

# -------------------------------
# 2. 参数配置
# -------------------------------
config = {
    "data_dir": "/data/Workspace/CIC-IoTDataset2023/bin-class",  # 数据集目录
    "num_classes": 2,  # 类别数
    "batch_size": 128,  # 批量大小（根据显存实际情况调整）
    "lr": 1e-4,  # 学习率
    "epochs": 100,  # 总训练轮数
    "model_name": "Conv2Former",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -------------------------------
# 3. 数据增强设置
# -------------------------------
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

valid_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------------
# 4. 数据集与 DataLoader
# -------------------------------
train_dataset = CustomDataset(os.path.join(config["data_dir"], "train"), transform=train_transform)
valid_dataset = CustomDataset(os.path.join(config["data_dir"], "valid"), transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# -------------------------------
# 5. 模型定义（使用 Conv2Former）
# -------------------------------
# 这里 dims 与 depths 可根据实验调节，本示例给出较轻量配置
from deepseek import ModFormer

model = ModFormer(dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], num_classes=config["num_classes"])
model = model.to(config["device"])

# -------------------------------
# 6. 损失函数、优化器与学习率调度器
# -------------------------------
# 这里采用了类别权重，适用于类别不平衡场景
class_weights = torch.tensor([7.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
# 如果需要可尝试 OneCycleLR 或 Warmup 策略

best_val_acc = 0.0
scaler = GradScaler()  # 用于混合精度训练


# -------------------------------
# 7. 训练与验证函数
# -------------------------------
def train(epoch_num: int):
    model.train()
    total_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Training: {epoch_num:>3}/{config['epochs']}", leave=False)
    for images, labels in train_bar:
        images = images.to(config["device"])
        labels = labels.to(config["device"])
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        # 梯度裁剪，防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        train_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_dataset)


def validate(epoch_num: int):
    model.eval()
    correct = 0
    total = 0
    all_predicts = []
    all_labels = []
    valid_bar = tqdm(valid_loader, desc=f"Validating: {epoch_num:>3}/{config['epochs']}", leave=False)
    with torch.no_grad():
        for images, labels in valid_bar:
            images = images.to(config["device"])
            labels = labels.to(config["device"])
            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            correct += (predicts == labels).sum().item()
            total += labels.size(0)
            all_predicts.extend(predicts.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            valid_bar.set_postfix(batch_acc=(predicts == labels).float().mean().item())
    valid_acc = correct / total
    print(classification_report(all_labels, all_predicts, zero_division=0, digits=4))
    return valid_acc


# -------------------------------
# 8. 训练循环与早停机制
# -------------------------------
train_losses = []
valid_accs = []

for epoch in range(config["epochs"]):
    loss = train(epoch + 1)
    train_losses.append(loss)
    valid_acc = validate(epoch_num=epoch + 1)
    valid_accs.append(valid_acc)
    scheduler.step()

    # 保存最佳模型
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        torch.save(model.state_dict(), "intermediates/best_conv2former.pth")

    print(f"\nTrain Loss: {loss:.4f}, Val Acc: {valid_acc:.4f}\n")

# -------------------------------
# 9. 结果可视化
# -------------------------------
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
