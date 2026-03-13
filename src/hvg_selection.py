import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import os
from src.utils import load_spatial_data

def robust_hvg_selection(h5_paths, spatial_dir=None, output_plot="hvg_spatial_plot.png"):
    """
    Mission 2: Robust HVG Selection Strategy
    Implements normalization, dropout removal, and intersection logic, and plots.
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
    
    # Generate spatial plots if spatial_dir is provided
    if spatial_dir and os.path.exists(spatial_dir):
        print("Generating spatial plot for top HVGs...")
        df, _ = load_spatial_data(spatial_dir)
        df.set_index("barcode", inplace=True)
        # join spatial coordinates
        target_adata.obs = target_adata.obs.join(df)
        
        # Need to flip y-axis because image coordinates have y pointing down
        y_coords = -target_adata.obs["pxl_row_in_fullres"]
        x_coords = target_adata.obs["pxl_col_in_fullres"]
        
        fig, axes = plt.subplots(1, len(top_hvgs), figsize=(5 * len(top_hvgs), 5))
        if len(top_hvgs) == 1:
            axes = [axes]
            
        for ax, gene in zip(axes, top_hvgs):
            expr = target_adata[:, gene].X.toarray().flatten() if hasattr(target_adata.X, "toarray") else target_adata[:, gene].X.flatten()
            sc_plot = ax.scatter(x_coords, y_coords, c=expr, cmap="viridis", s=10, alpha=0.8)
            ax.set_title(f"Spatial Expression: {gene}")
            ax.axis("off")
            plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches="tight")
        print(f"Saved spatial plot to {output_plot}")
        plt.close()

    return top_hvgs, target_adata

if __name__ == "__main__":
    sample_h5 = "data/Targeted_Visium_Human_BreastCancer_Immunology_filtered_feature_bc_matrix.h5"
    s_dir = "data/spatial"
    if os.path.exists(sample_h5):
        top_genes, _ = robust_hvg_selection([sample_h5], spatial_dir=s_dir)


    # (Optional) Since this is a spatial plot verify, we normally add spatial coords if we had the directory
    # For now we just return the top HVGs and the processed object