"""
Phase 1: Feature Caching
Run all patches through a frozen ConvNeXt-Base backbone and cache the 1024D feature
vectors to disk. After this step, no images are touched during training.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def build_feature_extractor(device):
    """Load ConvNeXt-Base, strip classifier head, freeze everything."""
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights)

    # Strip the classification head — keep everything up to avgpool
    # ConvNeXt structure: features -> avgpool -> classifier
    # We want features -> avgpool -> flatten -> 1024D
    # ConvNeXt classifier = Sequential(Flatten, LayerNorm, Linear).
    # We keep the Flatten to go from (B, 1024, 1, 1) → (B, 1024).
    model.classifier = nn.Flatten(1)

    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_image_transform():
    """ImageNet normalization as specified in prompt2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def cache_features(
    patches_dir="data/patches",
    output_dir="data/cached_features",
    batch_size=64,
):
    """Extract and cache ConvNeXt-Base features for all patches."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the frozen feature extractor
    model = build_feature_extractor(device)
    transform = get_image_transform()

    os.makedirs(output_dir, exist_ok=True)

    patches_path = Path(patches_dir)
    donor_dirs = sorted([d for d in patches_path.iterdir() if d.is_dir() and not d.name.startswith('.')])

    total_vectors = 0

    for donor_dir in donor_dirs:
        donor_id = donor_dir.name
        print(f"\nProcessing {donor_id}...")

        # Collect all patch files for this donor
        patch_files = sorted(donor_dir.glob("*.png"))
        if not patch_files:
            print(f"  No patches found, skipping.")
            continue

        barcodes = []
        features_list = []

        # Process in batches to avoid OOM
        for i in tqdm(range(0, len(patch_files), batch_size), desc=f"  {donor_id}"):
            batch_paths = patch_files[i:i + batch_size]
            batch_tensors = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)
                barcodes.append(p.stem)  # barcode = filename without extension

            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                feats = model(batch)  # (B, 1024)

            features_list.append(feats.cpu())

        # Concatenate all features for this donor
        all_features = torch.cat(features_list, dim=0)  # (N, 1024)

        # Save as a dictionary: {barcode: 1024D tensor}
        donor_cache = {
            "barcodes": barcodes,
            "features": all_features,  # (N, 1024) tensor
        }

        out_path = os.path.join(output_dir, f"{donor_id}.pt")
        torch.save(donor_cache, out_path)
        print(f"  Saved {len(barcodes)} vectors to {out_path}")
        total_vectors += len(barcodes)

    print(f"\nFeature caching complete! Total vectors: {total_vectors}")
    print(f"Cache size: {sum(f.stat().st_size for f in Path(output_dir).glob('*.pt')) / 1e6:.1f} MB")


if __name__ == "__main__":
    cache_features()
