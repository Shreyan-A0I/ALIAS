import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

def robust_hvg_selection(h5_paths, output_plot="hvg_spatial_plot.png"):
    """
    Mission 2: Robust HVG Selection Strategy
    Implements normalization, dropout removal, and intersection logic.
    """
    if not h5_paths:
        raise ValueError("No H5 paths provided.")
        
    adatas = []
    
    # 1. Load data and basic pre-processing
    for path in h5_paths:
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        adatas.append(adata)
    
    # 2. Inter-sample Logic (Intersection of genes)
    print("Extracting master_gene_list across all samples...")
    master_gene_list = set(adatas[0].var_names)
    for adata in adatas[1:]:
        master_gene_list = master_gene_list.intersection(set(adata.var_names))
    
    master_gene_list = list(master_gene_list)
    print(f"Total overlapping genes: {len(master_gene_list)}")
    
    # 3. Filter subset
    for i in range(len(adatas)):
        adatas[i] = adatas[i][:, master_gene_list].copy()
    
    # We will process the first sample for the Oracle demonstration
    # (In the framework, you apply this consistently)
    target_adata = adatas[0]
    
    # Dropout Removal: Filter genes present in < 5% of spots
    min_spots = int(0.05 * target_adata.n_obs)
    sc.pp.filter_genes(target_adata, min_cells=min_spots)
    print(f"Genes after dropout removal (min {min_spots} spots): {target_adata.n_vars}")
    
    # 4. The "Library Size" vs. "HVG" Conflict
    print("Performing Library Size Normalization...")
    sc.pp.normalize_total(target_adata, target_sum=1e4)
    sc.pp.log1p(target_adata)
    
    # 5. Finding Visually Descriptive Genes (HVGs)
    print("Calculating Highly Variable Genes...")
    sc.pp.highly_variable_genes(target_adata, flavor='seurat', n_top_genes=20)
    
    # Extract top visually descriptive genes
    top_hvgs = target_adata.var[target_adata.var['highly_variable']].sort_values(by="dispersions_norm", ascending=False).index[:3].tolist()
    print(f"Top spatially relevant HVGs identified: {top_hvgs}")
    
    # (Optional) Since this is a spatial plot verify, we normally add spatial coords if we had the directory
    # For now we just return the top HVGs and the processed object
    return top_hvgs, target_adata

if __name__ == "__main__":
    sample_h5 = "data/Targeted_Visium_Human_BreastCancer_Immunology_filtered_feature_bc_matrix.h5"
    if h5_paths := [sample_h5]: # Simulate 5 files if we had them
        top_genes, _ = robust_hvg_selection(h5_paths)
        print("Mission 2 Execution Complete.")
