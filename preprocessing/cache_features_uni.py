"""
Feature extraction script using the UNI Vision Foundation Model.
Processes patches from data/raw_data/ and saves 1024D embeddings to data/cached_features_uni/.

UNI Model: ViT-Large/16 (307M params)
Output: 1024D feature vectors (CLS token)
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# Donor IDs (same as in ConvNeXt extraction)
DONOR_IDS = [
    "Br2743_ant", "Br3942_ant", "Br6423_ant", "Br8492_ant",
    "Br6471_ant", "Br6522_ant", "Br8325_ant", "Br8667_ant",
]

def load_uni_model(checkpoint_path="uni.bin", device="mps"):
    print(f"Loading UNI model from {checkpoint_path}...")
    
    # Initialize ViT-L/16 architecture specific to UNI
    # UNI uses registar tokens (if available in timm version) and specific init
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()
    
    # Use timm's recommended transforms for this model
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    print("Model loaded successfully.")
    return model, transform

def cache_features_for_donor(donor_id, model, transform, raw_dir, output_dir, device, batch_size=64):
    donor_patch_dir = os.path.join(raw_dir, donor_id)
    if not os.path.exists(donor_patch_dir):
        print(f"Directory {donor_patch_dir} not found. Skipping.")
        return

    patch_files = [f for f in os.listdir(donor_patch_dir) if f.endswith(".png")]
    patch_files.sort()  # Ensure deterministic order
    
    print(f"Processing {len(patch_files)} patches for donor {donor_id}...")
    
    features_list = []
    barcodes = []
    
    # Process in batches
    for i in tqdm(range(0, len(patch_files), batch_size)):
        batch_files = patch_files[i : i + batch_size]
        batch_tensors = []
        
        for f in batch_files:
            img_path = os.path.join(donor_patch_dir, f)
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(transform(img))
            barcodes.append(f.replace(".png", ""))
            
        input_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            # Get CLS token output (n_images, 1024)
            # For ViT-L in timm with num_classes=0, forward() returns the features
            features = model(input_tensor)
            features_list.append(features.cpu())
            
    # Concatenate and save
    all_features = torch.cat(features_list, dim=0)
    output_path = os.path.join(output_dir, f"{donor_id}.pt")
    
    torch.save({
        "barcodes": barcodes,
        "features": all_features,
        "model": "UNI (vit_large_patch16_224)"
    }, output_path)
    
    print(f"Saved {all_features.shape[0]} features to {output_path}")

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw_dir = "data/raw_data"
    output_dir = "data/cached_features_uni"
    os.makedirs(output_dir, exist_ok=True)
    
    model, transform = load_uni_model("uni.bin", device=device)
    
    for donor_id in DONOR_IDS:
        cache_features_for_donor(donor_id, model, transform, raw_dir, output_dir, device)
        
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()
