"""MiniSeedNet : petit CNN encodeur-décodeur qui prédit où placer les seeds."""

import random

import numpy as np
import torch
import torch.nn as nn

FG_COLOR = "#ff0000"  # objet (chat)
BG_COLOR = "#00ff00"  # fond


class MiniSeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(          # extrait les motifs locaux
            nn.Conv2d(3, 16, 3, padding=1),    # -> [B, 16, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> [B, 16, 128, 128]
            nn.Conv2d(16, 32, 3, padding=1),   # -> [B, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> [B, 32, 64, 64]
        )
        self.decoder = nn.Sequential(          # restitue une carte à la taille de l'image
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # -> [B, 16, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # -> [B, 16, 256, 256]
            nn.ReLU(),
            nn.Conv2d(16, 2, 1),                      # -> [B, 2, 256, 256] (fond / objet)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))  # logits


def load_model(path, device="cpu"):
    model = MiniSeedNet()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, image_tensor):
    """image_tensor [3, H, W] -> (fg_probs, bg_probs) en numpy [H, W]."""
    device = next(model.parameters()).device
    out = model(image_tensor.unsqueeze(0).to(device))
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs[1], probs[0]


def draw_disk(center_y, center_x, radius, shape):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return np.where((Y - center_y) ** 2 + (X - center_x) ** 2 <= radius ** 2)


def get_limited_seeds(fg_probs, bg_probs, n_fg=12, n_bg=12, radius=3,
                      fg_thresh=0.9, bg_thresh=0.9, rng=random):
    """
    Tire n_fg / n_bg centres parmi les pixels où le CNN est confiant, puis marque un
    disque de rayon `radius` autour de chacun. Retourne {(y, x): '#RRGGBB'}.
    """
    H, W = fg_probs.shape
    seeds = {}
    for probs, thresh, n, color in ((fg_probs, fg_thresh, n_fg, FG_COLOR),
                                    (bg_probs, bg_thresh, n_bg, BG_COLOR)):
        coords = list(zip(*np.where(probs > thresh)))
        sample = rng.sample(coords, n) if len(coords) >= n else coords
        for y, x in sample:
            for yy, xx in zip(*draw_disk(y, x, radius, (H, W))):
                seeds[(int(yy), int(xx))] = color
    return seeds
