from datasets import load_dataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TinyImagenet(Dataset):
    def __init__(self, path, split, transform=None):
        self.path = path
        self.split = split
        self.transform = transform

        if split == 'train':
            self.data, self.labels = self._load_train_data()
        elif split in ['valid', 'validation']:
            self.data, self.labels = self._load_val_data()
        else:
            raise ValueError(f"Invalid split: {split}. Choose 'train' or 'valid'.")


        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.target = [self.class_to_idx[label] for label in self.labels]

    def _load_train_data(self):
        train_dir = os.path.join(self.path, "train")
        data, labels = [], []

        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name, "images")
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                data.append(img_path)
                labels.append(class_name)

        return data, labels

    def _load_val_data(self):
        val_dir = os.path.join(self.path, "val")
        annotations_file = os.path.join(val_dir, "val_annotations.txt")
        data, labels = [], []

        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name, class_name = parts[0], parts[1]
                img_path = os.path.join(val_dir, "images", img_name)
                data.append(img_path)
                labels.append(class_name)

        return data, labels

    def __getitem__(self, index):
        img_path, label = self.data[index], self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label_idx = self.class_to_idx[label]
        return {"img": img, "label": label_idx}

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    path = "/data/tinyimagenet"
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = TinyImagenet(path, 'train', transform=transform)
    val_dataset = TinyImagenet(path, 'valid', transform=transform)


    sample = train_dataset[0]
    print(f"shape: {sample['img'].shape}, label: {sample['label']}")
