import scanpy as sc
import squidpy as sq
import gc
import os

def preprocess_and_shrink_h5ad(
    input_h5ad="data/spatialDLPFC_data.h5ad",
    output_h5ad="data/spatialDLPFC_anterior_processed.h5ad",
    top_n_genes=2000
):
    print(f"Loading {input_h5ad} ...")
    adata = sc.read_h5ad(input_h5ad)
    
    # 1. Strip images from memory
    # The image arrays inside adata.uns['spatial'] take up huge amounts of space.
    # Since we have the raw TIFs, we don't need these low/hires png copies hanging in memory.
    if 'spatial' in adata.uns:
        print("Stripping heavy image arrays from .uns['spatial']...")
        for library_id in list(adata.uns['spatial'].keys()):
            if 'images' in adata.uns['spatial'][library_id]:
                # Clear references to pixel arrays
                adata.uns['spatial'][library_id]['images'] = {}
                
    # Free memory
    gc.collect()

    # 2. Subset to Anterior sections only
    # Our master sheet renaming mapped 10 _ant samples, we should subset adata to match.
    print("Subsetting object specifically for anterior samples ('_ant')")
    ant_mask = adata.obs['sample_id'].str.endswith('_ant', na=False)
    adata = adata[ant_mask].copy()
    print(f"Spots after filtering: {adata.n_obs}")
    
    gc.collect()

    # 3. Moran's I Selection
    # To reduce the 28K+ genes, we select the top spatially correlated genes.
    print("Normalizing counts before calculating Moran's I...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Optional: we can filter completely useless genes first to speed up spatial neighbor compute
    sc.pp.filter_genes(adata, min_cells=10)
    
    print("Computing Spatial Neighbors Graph...")
    # Calculate neighbors based on coordinate proximity
    sq.gr.spatial_neighbors(adata)
    
    print("Calculating Moran's I spatial autocorrelation...")
    # Moran's I calculation (can take a minute)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=5)
    
    # Sort and slice top expected spatially significant genes
    print(f"Subsetting to top {top_n_genes} spatially variable genes...")
    top_spatial_genes = adata.uns['moranI'].sort_values("I", ascending=False).index[:top_n_genes]
    
    adata = adata[:, top_spatial_genes].copy()
    
    # 4. Save
    print(f"Saving preprocessed lightweight generic .h5ad to {output_h5ad}")
    adata.write(output_h5ad)
    
    print(f"Finished! New object size: {adata.n_obs} spots x {adata.n_vars} genes.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    preprocess_and_shrink_h5ad()
