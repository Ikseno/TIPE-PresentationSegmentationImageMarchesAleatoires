"""Évaluation quantitative (IoU / Dice) sur les images de validation d'Oxford-IIIT Pet.

Compare deux chaînes sur les mêmes images, avec la vérité terrain du dataset :
  1. CNN seul          : masque = argmax des probabilités de MiniSeedNet
  2. CNN + marches     : seeds tirées des zones où le CNN est confiant, puis
                         segmentation par marches aléatoires (chaîne du TIPE)

La marche aléatoire est lente (pure Python, ~20 s à plusieurs minutes par image) :
commencer avec --n 10 pour vérifier que tout fonctionne.

Exemple :
    python evaluate.py --n 10 --max-steps 20000
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from segmentation.dataset import OxfordPetCatsOnly, train_val_split
from segmentation.random_walker import RandomWalkerSegmentation
from segmentation.seednet import FG_COLOR, get_limited_seeds, load_model, predict_probs


def iou(pred, gt):
    union = np.logical_or(pred, gt).sum()
    return np.logical_and(pred, gt).sum() / union if union else 1.0


def dice(pred, gt):
    total = pred.sum() + gt.sum()
    return 2 * np.logical_and(pred, gt).sum() / total if total else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/seednet_final.pth")
    p.add_argument("--n", type=int, default=None, help="nombre d'images (défaut : toutes)")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--sigma", type=float, default=7)
    p.add_argument("--weight-method", choices=["rgb", "patch"], default="rgb")
    p.add_argument("--n-seeds", type=int, default=12, help="disques de seeds par classe")
    p.add_argument("--skip-rw", action="store_true", help="n'évaluer que le CNN (rapide)")
    p.add_argument("--data-root", default="data/pets")
    p.add_argument("--out", default="results/evaluation.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    _, val_set = train_val_split(OxfordPetCatsOnly(root=args.data_root))
    n = len(val_set) if args.n is None else min(args.n, len(val_set))
    rng = random.Random(0)

    rows = []
    for k in range(n):
        image, _, gt = val_set[k]
        gt = gt.numpy().astype(bool)
        fg, bg = predict_probs(model, image)
        row = {"index": k, "iou_cnn": iou(fg > bg, gt), "dice_cnn": dice(fg > bg, gt)}

        if not args.skip_rw:
            seeds = get_limited_seeds(fg, bg, n_fg=args.n_seeds, n_bg=args.n_seeds, rng=rng)
            if seeds:
                pil = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                rw = RandomWalkerSegmentation(seeds, pil, *gt.shape, max_steps=args.max_steps,
                                              sigma=args.sigma, weight_method=args.weight_method)
                t0 = time.perf_counter()
                rw.color_all_pixels(verbose=False)
                pred = rw.label_array() == FG_COLOR
                row.update(iou_rw=iou(pred, gt), dice_rw=dice(pred, gt),
                           time_rw=time.perf_counter() - t0, n_seeds=len(seeds))
        rows.append(row)
        msg = f"[{k + 1}/{n}] IoU CNN {row['iou_cnn']:.3f}"
        if "iou_rw" in row:
            msg += f" | IoU CNN+marches {row['iou_rw']:.3f} ({row['time_rw']:.0f} s)"
        print(msg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = sorted({key for r in rows for key in r}, key=lambda s: (s != "index", s))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Moyennes ===")
    for key in ("iou_cnn", "dice_cnn", "iou_rw", "dice_rw", "time_rw"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"{key:>9} : {np.mean(vals):.3f}  (médiane {np.median(vals):.3f}, n={len(vals)})")
    print(f"Détails : {args.out}")


if __name__ == "__main__":
    main()
