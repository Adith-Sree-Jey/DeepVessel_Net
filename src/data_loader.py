import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class RetinaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, img_size=512, patch_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.augment = augment
        self.img_size = img_size
        self.patch_size = patch_size

        # Augmentation
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=50),
                A.RandomBrightnessContrast(p=0.3),
                A.CLAHE(p=0.5),  # Increased probability
                A.Normalize(mean=(0.0, 0.0, 0.0),
                            std=(1.0, 1.0, 1.0),
                            max_pixel_value=255.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.0, 0.0, 0.0),
                            std=(1.0, 1.0, 1.0),
                            max_pixel_value=255.0),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('float32')

        # Patch extraction (optional)
        if self.patch_size:
            h, w = image.shape[:2]
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            image = image[y:y+self.patch_size, x:x+self.patch_size]
            mask = mask[y:y+self.patch_size, x:x+self.patch_size]

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)

        return image, mask


def get_dataloaders(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir,
                    batch_size=4, img_size=512, patch_size=None, num_workers=0):

    train_dataset = RetinaDataset(train_img_dir, train_mask_dir, augment=True, img_size=img_size, patch_size=patch_size)
    test_dataset = RetinaDataset(test_img_dir, test_mask_dir, augment=False, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
