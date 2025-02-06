import os

from PIL import Image
from torch.utils.data import Dataset


# ----------------------------------------
# 数据集加载（适配单通道图像）
# ----------------------------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        for label, clazz in enumerate(self.classes):
            label_dir = os.path.join(root_dir, clazz)
            if not os.path.isdir(label_dir): continue
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 关键修改：转为灰度图（"L"模式）
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
