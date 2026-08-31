import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class Brisc2025Dataset(Dataset):
    def __init__(self, csv_path, task="train", image_size=(512, 512), augment=None):

        self.df = pd.read_csv(csv_path, sep=";")
        self.df = self.df[self.df["task"] == task].reset_index(drop=True)
        self.image_size = image_size
        # self.task = task

        self.augment = augment if augment is not None else (task == "train")

        self.image_trans = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

        self.mask_trans = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

        self.tumor_type_map = {
            "no_tumor": 0,
            "glioma": 1,
            "meningioma": 2,
            "pituitary": 3
        }

        self.direction_map = {
            "axial": 0,
            "coronal": 1,
            "sagittal": 2,
            "unknown": 3
        }

    def __len__(self):
        return len(self.df)

    def _augment(self, img, mask):
        
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))

        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        

        has_mask = bool(row["has_mask"])
        if has_mask:
            mask = Image.open(row["segmentation_mask_path"]).convert("L")            
        else:
            mask = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = self.image_trans(img)
        mask = self.mask_trans(mask)
        mask = (mask > 0.5).float()
        direction_idx = self.direction_map[row["direction"]]
        direction = torch.tensor(direction_idx, dtype=torch.long)

        tumor_type_idx = self.tumor_type_map[row["tumor_type"]]
        tumor_type = torch.tensor(tumor_type_idx, dtype=torch.long)

        return {
            "image": img,
            "has_mask": has_mask,
            "mask": mask,
            "direction": direction,
            "tumor_type": tumor_type,
            "filename": row["filename"]
        }