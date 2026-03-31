# 1. Install necessary packages
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("spatialLIBD", "zellkonverter"))

# 2. Fetch the spatial DLPFC data
install.packages("rlang")
library(rlang)
packageVersion("rlang") 

# Now try loading the main package again
library(spatialLIBD)
library(zellkonverter)

spe <- fetch_data(type = "spatialDLPFC_Visium")

# 3. Convert and save as H5AD
# This writes the R object directly to a format Python's Scanpy can read
writeH5AD(spe, file = "spatialDLPFC_data.h5ad")

test_metadata <- zellkonverter::readH5AD("spatialDLPFC_data.h5ad", reader = "python", skip_data = TRUE)
print(test_metadata)
