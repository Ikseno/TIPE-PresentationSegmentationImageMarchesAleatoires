"""Sous-ensemble « chats » du dataset Oxford-IIIT Pet, avec cartes de seeds cibles."""

import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import InterpolationMode

CAT_CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
    "Russian_Blue", "Siamese", "Sphynx",
]


class OxfordPetCatsOnly(Dataset):
    """
    Retourne (image, seed_map, binary_mask) :
    - image       : tenseur [3, H, W]
    - seed_map    : 1 = objet sûr, 0 = fond sûr, -1 = zone ignorée (bord de l'objet)
    - binary_mask : vérité terrain (1 = chat)
    La seed_map est obtenue par érosion du masque, pour ne garder que les zones
    éloignées de la frontière.
    """

    def __init__(self, root="data/pets", image_size=(256, 256), radius=5):
        self.full_dataset = OxfordIIITPet(root=root, download=True, target_types="segmentation")
        self.radius = radius
        self.img_transform = T.Compose([T.Resize(image_size), T.ToTensor()])
        self.mask_transform = T.Resize(image_size, interpolation=InterpolationMode.NEAREST)

        self.indices = [
            i for i, path in enumerate(self.full_dataset._images)
            if os.path.basename(path).split("_")[0] in CAT_CLASSES
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.full_dataset[self.indices[idx]]
        image = self.img_transform(image)
        np_mask = np.array(self.mask_transform(mask))        # trimap : 1 objet, 2 fond, 3 bord
        binary_mask = (np_mask == 1).astype(np.uint8)

        kernel = np.ones((self.radius, self.radius), np.uint8)
        fg = cv2.erode(binary_mask, kernel, iterations=2)
        bg = cv2.erode(1 - binary_mask, kernel, iterations=2)

        seed_map = np.full(binary_mask.shape, -1, dtype=np.int64)
        seed_map[fg == 1] = 1
        seed_map[bg == 1] = 0
        return image, torch.from_numpy(seed_map), torch.from_numpy(binary_mask)


def train_val_split(dataset, train_ratio=0.8, seed=42):
    """Découpage 80/20 reproductible (même graine que pendant le TIPE)."""
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(train_ratio * len(dataset))
    return Subset(dataset, indices[:n_train]), Subset(dataset, indices[n_train:])
