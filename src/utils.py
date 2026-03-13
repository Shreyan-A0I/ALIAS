import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_spatial_data(spatial_dir):
    """
    Loads spatial mapping data from Space Ranger outputs.
    Handles the "Headerless" CSV trap by explicitly assigning columns.
    """
    positions_path = os.path.join(spatial_dir, "tissue_positions_list.csv")
    if not os.path.exists(positions_path):
        # newer version of space ranger might use tissue_positions.csv
        positions_path = os.path.join(spatial_dir, "tissue_positions.csv")

    # The "Headerless" CSV Trap fix:
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    
    try:
        # Try reading with header, if first row barcode is weird, read without head
        df = pd.read_csv(positions_path, header=None, names=columns)
    except Exception as e:
        df = pd.read_csv(positions_path)
        if len(df.columns) == 6:
            df.columns = columns
    
    # Load scalefactors
    scalefactors_path = os.path.join(spatial_dir, "scalefactors_json.json")
    with open(scalefactors_path, "r") as f:
        scalefactors = json.load(f)
        
    return df, scalefactors

def load_image_rgb(image_path):
    """
    Loads a High-Resolution image via OpenCV and converts BGR to RGB 
    to fix The "Channel Order" Check.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # The "Channel Order" Check fix:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
