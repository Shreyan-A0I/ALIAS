import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_spatial_data, load_image_rgb

def extract_patch(img, row, col, patch_size=224):
    """
    Extract a patch_size x patch_size square from img centered at (row, col).
    Pad with zeros if out of bounds.
    """
    half_size = patch_size // 2
    
    # Calculate crop boundaries
    r_start = row - half_size
    r_end = row + half_size + (patch_size % 2)
    c_start = col - half_size
    c_end = col + half_size + (patch_size % 2)
    
    # Pad handling
    pad_top = max(0, -r_start)
    pad_bottom = max(0, r_end - img.shape[0])
    pad_left = max(0, -c_start)
    pad_right = max(0, c_end - img.shape[1])
    
    img_padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # Adjust crop boundaries after padding
    r_start_pad = max(0, r_start) + pad_top
    r_end_pad = r_end + pad_top
    c_start_pad = max(0, c_start) + pad_left
    c_end_pad = c_end + pad_left
    
    patch = img_padded[r_start_pad:r_end_pad, c_start_pad:c_end_pad]
    return patch

def verify_spatial_alignment(spatial_dir, image_path, output_path, patch_size=224):
    """
    Mission 1 bridge logic: Verifies spatial alignment by cropping a patch
    and overlaying the spot.
    """
    df, scalefactors = load_spatial_data(spatial_dir)
    img_rgb = load_image_rgb(image_path)
    
    # Filter for spots in tissue
    in_tissue_df = df[df['in_tissue'] == 1]
    
    # Select a "high-confidence" barcode (we'll just take a random one from inside the tissue)
    # Using a fixed seed for reproducibility
    spot = in_tissue_df.sample(n=1, random_state=42).iloc[0]
    
    row = int(spot['pxl_row_in_fullres'])
    col = int(spot['pxl_col_in_fullres'])
    
    # Crop the 224x224 patch
    patch = extract_patch(img_rgb, row, col, patch_size=patch_size)
    
    # Overlay the spot (55um diameter -> spot_diameter_fullres)
    # Radius in fullres:
    spot_radius = int(scalefactors.get('spot_diameter_fullres', 100) / 2.0)
    
    # In the patch coordinate system, the spot center is exactly at the patch center
    center_y, center_x = patch_size // 2, patch_size // 2
    
    # Draw circle using OpenCV
    # We will blend it slightly
    overlay = patch.copy()
    cv2.circle(overlay, (center_x, center_y), spot_radius, (0, 255, 0), 2)
    
    alpha = 0.5
    visualized_patch = cv2.addWeighted(overlay, alpha, patch, 1 - alpha, 0)
    
    # Save the visual
    plt.figure(figsize=(6, 6))
    plt.imshow(visualized_patch)
    plt.title(f"Spot Alignment verification\nBarcode: {spot['barcode']}")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

if __name__ == "__main__":
    spatial_data = "data/spatial"
    hires_image = "data/spatial/tissue_hires_image.png"  # In practice use the big TIFF
    # For mission 1 demo let's assume we can use the 1.7GB TIFF if we have RAM,
    # or the hires png for quick verification. Let's try the TIFF.
    tiff_image = "data/Targeted_Visium_Human_BreastCancer_Immunology_image.tif"
    
    out = "patch_alignment_demo.png"
    print(f"Running Mission 1 verification on {tiff_image}...")
    if os.path.exists(tiff_image):
       verify_spatial_alignment(spatial_data, tiff_image, out)
       print(f"Saved alignment verification to {out}")
    else:
       print("TIFF not found, using hires PNG instead...")
       verify_spatial_alignment(spatial_data, hires_image, out)
       print(f"Saved alignment verification to {out}")
