import scanpy as sc

adata_real_test = sc.read_h5ad('data/test_mask/rep_hepg2_test_mask.h5ad')
# adata_real_test.var_names = adata_real_test.var['gene_name']
# # 去除test_adata中可能存在的重复基因
# adata_real_test = adata_real_test[:, ~adata_real_test.var_names.duplicated()]

true_adata = adata_real_test[~(adata_real_test.obs.condition == 'ctrl')]
control_adata = adata_real_test[adata_real_test.obs.condition == 'ctrl']

# true_adata = sc.read_h5ad('F:\\pythonProject\\stack_mix\\predictions\\aligned_hepg2_test.h5ad')
# control_adata = sc.read_h5ad('F:\\pythonProject\\stack_mix\\predictions\\aligned_hepg2_test_ctrl.h5ad')

adata_list = []
for gene in true_adata.obs.gene.unique():
    # adata = sc.read_h5ad(f'F:\\pythonProject\\stack_mix\\predictions\\mask_v0\\{gene}_ctrl.h5ad')
    adata = sc.read_h5ad(f'predictions/mask_v0_mask_input/{gene}.h5ad')
    adata.obs['gene'] = gene
    adata_list.append(adata)

pred_adata = sc.concat(adata_list)

pred_adata.write_h5ad('merged_hepg2_prediction_mask_v0_mask_input.h5ad')
