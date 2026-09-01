"""Run a pretrained BigEarthNet v2.0 model on a local Sentinel-1 / Sentinel-2 patch."""

import argparse
from pathlib import Path

import rasterio
import torch
from configilm.extra.BENv2_utils import STANDARD_BANDS, stack_and_interpolate, NEW_LABELS

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier


def find_band_file(folder: Path, required_band: str, extensions):
    for ext in extensions:
        for file in folder.rglob(f"*{required_band}*.{ext}"):
            if file.is_file():
                return file
    return None


def load_patch(s1_dir: Path, s2_dir: Path, model):
    channels = model.config.channels
    image_size = model.config.image_size

    if channels == 10:
        bands = STANDARD_BANDS[10]
        data = {}
        for band in bands:
            if band in STANDARD_BANDS["S1"]:
                file = find_band_file(s1_dir, band, ["tif", "tiff"])
            else:
                file = find_band_file(s2_dir, band, ["tif", "tiff"])
            if file is None:
                raise FileNotFoundError(f"Missing band {band} in {s1_dir} or {s2_dir}")
            with rasterio.open(file) as src:
                data[band] = src.read(1)
        img = stack_and_interpolate(data, order=bands, img_size=image_size, upsample_mode="nearest")
    elif channels == 12:
        # Model expects S2 + S1. Use S2 first 10 bands and S1 VV/VH if both are available.
        s2_bands = STANDARD_BANDS[10]
        s1_bands = STANDARD_BANDS[2]
        data = {}
        for band in s2_bands:
            file = find_band_file(s2_dir, band, ["tif", "tiff"])
            if file is None:
                raise FileNotFoundError(f"Missing S2 band {band} in {s2_dir}")
            with rasterio.open(file) as src:
                data[band] = src.read(1)
        for band in s1_bands:
            file = find_band_file(s1_dir, band, ["tif", "tiff"])
            if file is None:
                raise FileNotFoundError(f"Missing S1 band {band} in {s1_dir}")
            with rasterio.open(file) as src:
                data[band] = src.read(1)
        img = stack_and_interpolate(data, order=s2_bands + s1_bands, img_size=image_size, upsample_mode="nearest")
    else:
        raise ValueError(f"Unsupported model channel count: {channels}")

    return img.unsqueeze(0)


def interpret_outputs(logits):
    probs = torch.sigmoid(logits).squeeze(0)
    ranked = torch.argsort(probs, descending=True)
    print("Top predicted classes:")
    for idx in ranked[:5]:
        label = NEW_LABELS[int(idx)]
        score = float(probs[int(idx)])
        print(f"  {label}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run the pretrained BigEarthNet v2.0 model on a local Sentinel patch.")
    parser.add_argument("--model-name", type=str, default="hackelle/resnet18-all-v0.1.1")
    parser.add_argument("--s1-dir", type=str, default="scripts/data/S1")
    parser.add_argument("--s2-dir", type=str, default="scripts/data/S2")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    s1_dir = Path(args.s1_dir)
    s2_dir = Path(args.s2_dir)
    x = load_patch(s1_dir, s2_dir, model)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    interpret_outputs(logits)


if __name__ == "__main__":
    main()
