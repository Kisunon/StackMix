import pandas as pd
import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import jaccard_score
from gears import PertData, GEARS
import scanpy as sc
import pickle
# from src.stack.cli.generation import _align_genes_to_target_list


def calculate_correlation(real_expr, pred_expr):
    """Calculate Pearson and Spearman correlations between real and predicted expression"""
    print(real_expr.shape, pred_expr.shape)

    mean_real = np.mean(real_expr, axis=0)
    mean_pred = np.mean(pred_expr, axis=0)
    # mean_pred = pred_expr

    pearson_r, _ = pearsonr(mean_real, mean_pred)
    spearman_r, _ = spearmanr(mean_real, mean_pred)

    # Return average correlations
    return pearson_r, spearman_r



if __name__ == '__main__':
    # load data
    # adata = sc.read_h5ad('./data/hepg2_train.h5ad')

    test_adata = sc.read_h5ad("data/rep_hepg2_test.h5ad")

    # 加载预训练基因列表
    genelist_path = "weight/Stack-Large/basecount_1000per_15000max.pkl"
    with open(genelist_path, 'rb') as f:
        target_genes = pickle.load(f)

    # 对齐基因
    # aligned_adata = _align_genes_to_target_list(test_adata, target_genes, gene_name_col=None)

    # predict
    test_pert_gene = ['TOP2A', 'BIRC5', 'CDC20', 'UTP20', 'TAF3', 'WARS', 'ACTR3', 'AURKB']

    test_adata.var_names = test_adata.var['gene_name']
    # 去除test_adata中可能存在的重复基因
    test_adata = test_adata[:, ~test_adata.var_names.duplicated()]

    results = []
    for perturbation in test_pert_gene:
        test_pred = sc.read_h5ad(f"predictions/test_pretrain/{perturbation}_ctrl.h5ad")

        # print(test_adata)
        # print(test_adata.var_names[:5])
        # print(test_pred)
        # print(test_pred.var_names[:5])

        common_genes = list(set(test_adata.var_names) & set(test_pred.var_names))
        # print(len(common_genes))

        test_pred_common = test_pred[:, common_genes]
        test_adata_common = test_adata[:, common_genes]

        result = calculate_correlation(test_adata_common.X[test_adata_common.obs['gene'] == perturbation], test_pred_common.X)
        results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results, index=test_pert_gene, columns=['Pearson', 'Spearman'])

    print(results_df)
    results_df.to_csv('./data/hepg2_pretrain.csv')
