import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import gc
import cv2
import numpy as np
import anndata as ad
from PIL import Image

def extract_patches(
    h5ad_path="data/spatialDLPFC.h5ad", 
    raw_dir="data/raw_data", 
    out_dir="data/patches", 
    patch_size=224
):
    print(f"Loading generic matrix: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    # Create master patch directory
    os.makedirs(out_dir, exist_ok=True)
    
    # We iterate by unique sample ID so we only load ONE massive TIF into memory at a time
    unique_samples = adata.obs['sample_id'].unique()
    
    half_patch = patch_size // 2
    total_patches = 0
    
    for sample in unique_samples:
        tif_path = os.path.join(raw_dir, f"{sample}.tif")
        
        if not os.path.exists(tif_path):
            print(f"WARNING: No TIF found for {sample} at {tif_path}! Skipping.")
            continue
            
        print(f"\\nProcessing {sample}...")
        sample_dir = os.path.join(out_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. Load the Massive TIF Image
        print(f"Loading {tif_path} into memory...")
        # Since these are big biomedical TIFs, flags like cv2.IMREAD_UNCHANGED are optimal.
        img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Failed to load image for {sample} using OpenCV. Image might be too large or corrupted.")
            continue
            
        # Visium histology images naturally originate in RGB, but cv2 loads as BGR. Let's flip it for sanity.
        # This prevents our Deep Learning model from training on "blue" tissues.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_h, img_w, _ = img.shape
        print(f"Image shape: {img.shape}")
        
        # 2. Subset to only the spots corresponding to THIS specific physical slide
        sample_adata = adata[adata.obs['sample_id'] == sample]
        
        print(f"Extracting {sample_adata.n_obs} patches...")
        
        sample_patches_count = 0
        for i in range(sample_adata.n_obs):
            barcode = sample_adata.obs.index[i]
            
            # The spatial coords are inside obsm['spatial']
            # We use integer position i to ensure we grab the exact row even with duplicate barcodes
            spatial_row = sample_adata.obsm['spatial'].iloc[i]
            
            cx = int(spatial_row['pxl_col_in_fullres'])
            cy = int(spatial_row['pxl_row_in_fullres'])
            
            
            # Calculate bounding box
            x_min = cx - half_patch
            y_min = cy - half_patch
            x_max = cx + half_patch
            y_max = cy + half_patch
            
            # Handle out-of-bounds crops by safely padding the boundary with zeros (black padding)
            pad_left = max(0, -x_min)
            pad_top = max(0, -y_min)
            pad_right = max(0, x_max - img_w)
            pad_bottom = max(0, y_max - img_h)
            
            # Safe crop ranges
            safe_x_min = max(0, x_min)
            safe_y_min = max(0, y_min)
            safe_x_max = min(img_w, x_max)
            safe_y_max = min(img_h, y_max)
            
            # Crop real pixels
            cropped = img[safe_y_min:safe_y_max, safe_x_min:safe_x_max]
            
            # Pad if we went out of bounds (so neural network always gets exact 224x224)
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                cropped = cv2.copyMakeBorder(
                    cropped, 
                    pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            
            # Save Lossless PNG
            out_file = os.path.join(sample_dir, f"{barcode}.png")
            
            # Using basic cv2 imwrite writes BGR natively so we have to swap back, so we use PIL:
            pil_img = Image.fromarray(cropped)
            pil_img.save(out_file)
            
            sample_patches_count += 1
            
        print(f"Successfully saved {sample_patches_count} patches to {sample_dir}")
        total_patches += sample_patches_count
        
        # Wipe memory immediately before the next multi-GB TIF
        del img
        gc.collect()

    print(f"\\nExtraction Complete! Generated {total_patches} physical 224x224 PNG patches across all regions.")

if __name__ == "__main__":
    extract_patches()
