"""Entraînement de MiniSeedNet sur les chats d'Oxford-IIIT Pet.

Exemple :
    python train.py --epochs 50                      # nouvelle session
    python train.py --epochs 50 --resume models/seednet_final.pth
"""

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset import OxfordPetCatsOnly, train_val_split
from segmentation.seednet import MiniSeedNet

LOSSES_CSV = "results/logs/losses.csv"


def last_session_number(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        return max((int(r["Session"]) for r in csv.DictReader(f)), default=0)


def save_losses(train_losses, val_losses, csv_path):
    new_file = not os.path.exists(csv_path)
    session = last_session_number(csv_path) + 1
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Session", "Epoch", "Train Loss", "Validation Loss"])
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([session, i, tl, vl])
    return session


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for images, seed_maps, _ in tqdm(loader, leave=False):
            images, seed_maps = images.to(device), seed_maps.to(device)
            loss = criterion(model(images), seed_maps)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--resume", help="checkpoint .pth à partir duquel reprendre")
    p.add_argument("--data-root", default="data/pets")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set, val_set = train_val_split(OxfordPetCatsOnly(root=args.data_root))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    print(f"{len(train_set)} images d'entraînement / {len(val_set)} de validation")

    model = MiniSeedNet().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Poids 5 sur la classe « objet » pour compenser le déséquilibre fond/objet ;
    # ignore_index=-1 : les pixels proches du bord ne comptent pas dans la loss
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1.0, 5.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        train_losses.append(run_epoch(model, train_loader, criterion, device, optimizer))
        val_losses.append(run_epoch(model, val_loader, criterion, device))
        print(f"Epoch {epoch}/{args.epochs} | train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")

    session = save_losses(train_losses, val_losses, LOSSES_CSV)
    os.makedirs("models", exist_ok=True)
    path = f"models/seednet_session{session}_{datetime.now():%Y%m%d_%H%M%S}.pth"
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé : {path}")


if __name__ == "__main__":
    main()
