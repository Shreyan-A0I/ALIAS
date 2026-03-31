import os
import shutil
import pandas as pd

def rename_raw_images(raw_dir="data/raw_data", xlsx_path="data/Visium_dlpfc_mastersheet.xlsx"):
    print("Reading Master Sheet...")
    df = pd.read_excel(xlsx_path)
    
    # We only care about rows that actually have an image file path
    df_valid = df.dropna(subset=['image file path'])
    
    # Create mapping of expected filename to actual sample name
    # e.g., '1000140_dlpfc_ant_round3_A1.tif' -> 'Br6471_ant'
    filename_to_sample = {}
    for _, row in df_valid.iterrows():
        expected_filename = row['image file path'].split('/')[-1]
        sample_name = row['sample name']
        filename_to_sample[expected_filename] = sample_name
        
    print(f"Created mapping dictionary with {len(filename_to_sample)} valid entries from mastersheet.")

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.tif') or f.endswith('.tiff')]
    print(f"Found {len(raw_files)} raw TIFs in {raw_dir}.")
    
    renamed_count = 0
    for f in raw_files:
        if f in filename_to_sample:
            old_path = os.path.join(raw_dir, f)
            new_name = filename_to_sample[f] + ".tif"
            new_path = os.path.join(raw_dir, new_name)
            
            # Rename in place
            os.rename(old_path, new_path)
            print(f"Renamed: {f} -> {new_name}")
            renamed_count += 1
        elif "_ant" in f or "_mid" in f or "_post" in f:
            # Maybe already renamed? Check
            print(f"Skipping {f}: Might already be renamed or unknown file.")
        else:
            print(f"Warning: {f} found in directory but not matched in the Master Sheet mapping!")
            
    print(f"\\nTotal files successfully renamed: {renamed_count}")

if __name__ == "__main__":
    rename_raw_images()
