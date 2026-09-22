import numpy as np
import scanpy as sc
import pandas as pd


# get pert set
pert_set = pd.read_csv('predictions/map/predictions_map.csv')
pert_list = pert_set['perturbation'].values

# load adata
adata_list = []
for pert in pert_list:
    adata = sc.read_h5ad(f'data/rep_hepg2_pretrain_{pert}.h5ad')
    adata_list.append(adata)

# concat adata
adata = sc.concat(adata_list)

