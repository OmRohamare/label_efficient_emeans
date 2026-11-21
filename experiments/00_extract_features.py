"""Extract DINOv2 ViT-S/14 CLS features for Tiny-ImageNet and cache to .npy.

Usage:
    python experiments/00_extract_features.py --cache-dir cache --batch-size 256
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import load_tiny_imagenet
from src.features import extract_and_cache, load_dinov2

logger = logging.getLogger("extract")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="recompute even if cache exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    cache = Path(args.cache_dir)
    train_loader, valid_loader, n_classes = load_tiny_imagenet(
        cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_dinov2(device)

    for name, loader in (("train", train_loader), ("valid", valid_loader)):
        feats, labels = extract_and_cache(
            loader,
            cache / f"{name}_feats.npy",
            cache / f"{name}_labels.npy",
            model=model,
            device=device,
            force=args.force,
            desc=f"{name} features",
        )
        logger.info("%s: feats %s labels %s", name, feats.shape, labels.shape)

    logger.info("Done. classes=%d", n_classes)


if __name__ == "__main__":
    main()
