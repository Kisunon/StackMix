import scanpy as sc
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_squared_error
import warnings
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

warnings.filterwarnings('ignore')


def preprocess_perturbation_data(adata,
                                 min_cells_per_pert=30,
                                 max_cells_per_pert=2000,
                                 n_hvgs=5000,
                                 mito_threshold=0.1,
                                 min_genes=200):
    """
    按照文章标准预处理单个 AnnData 对象（对照、预测或真实数据）
    """
    # 1. 细胞过滤
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 人类线粒体基因前缀
    # sc.pp.filter_cells(adata, max_counts=None)  # 线粒体比例过滤需额外计算
    if 'mt' in adata.var.columns:
        adata.obs['percent_mito'] = np.sum(adata[:, adata.var['mt']].X, axis=1) / np.sum(adata.X, axis=1)
        adata = adata[adata.obs['percent_mito'] < mito_threshold, :].copy()

    # 2. 按扰动条件过滤细胞数
    pert_counts = adata.obs['condition'].value_counts()
    valid_perts = pert_counts[(pert_counts >= min_cells_per_pert) & (pert_counts.index != 'ctrl')].index
    adata = adata[adata.obs['condition'].isin(valid_perts) | (adata.obs['condition'] == 'ctrl')].copy()

    # 3. 下采样：每个扰动最多保留 max_cells_per_pert 个细胞
    if max_cells_per_pert:
        new_obs = []
        for cond, group in adata.obs.groupby('condition'):
            if len(group) > max_cells_per_pert:
                keep = random.sample(list(group.index), max_cells_per_pert)
            else:
                keep = list(group.index)
            new_obs.extend(keep)
        adata = adata[new_obs].copy()

    # 4. 基因过滤（仅基于训练集，这里我们使用真实数据作为参照，但若预测数据需与真实一致，则后续取交集）
    # 此步暂不删除基因，统一在后续进行高变基因选择
    return adata


def align_genes(adata_list):
    """
    多个 AnnData 对象取基因交集
    """
    common_genes = set(adata_list[0].var_names)
    for ad in adata_list[1:]:
        common_genes &= set(ad.var_names)
    common_genes = list(common_genes)
    for i, ad in enumerate(adata_list):
        adata_list[i] = ad[:, common_genes].copy()
    return adata_list


def select_hvgs(adata_true, n_hvgs=5000, batch_key=None):
    """
    在真实数据上选择高变基因，并过滤其他数据至相同基因集
    """
    sc.pp.highly_variable_genes(adata_true, n_top_genes=n_hvgs, batch_key=batch_key, flavor='seurat_v3')
    hvgs = adata_true.var_names[adata_true.var['highly_variable']]
    # 强制保留扰动靶标基因（如果已知，这里需要传入；若无，则忽略）
    # 若需要，可扩展参数
    return hvgs


def calculate_deg(adata_control, adata_true, condition_column='condition', control_tag='ctrl'):
    """
    计算差异表达基因，参考 scPerturBench 的 calDEG 函数
    """
    # 合并对照组和真实扰动数据
    adata_combined = adata_control.concatenate(adata_true)

    # 确保 log1p 配置正确
    adata_combined.uns['log1p'] = {}
    adata_combined.uns['log1p']['base'] = None

    # 获取所有扰动条件
    perturbations = adata_combined.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i != control_tag]

    # 计算差异表达基因
    sc.tl.rank_genes_groups(adata_combined, condition_column, groups=perturbations,
                            reference=control_tag, method='t-test')

    result = adata_combined.uns['rank_genes_groups']
    deg_dict = defaultdict(dict)

    for perturbation in perturbations:
        final_result = pd.DataFrame({
            key: result[key][perturbation]
            for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']
        })
        final_result['foldchanges'] = 2 ** final_result['logfoldchanges']
        final_result.drop(labels=['logfoldchanges'], inplace=True, axis=1)
        final_result.set_index('names', inplace=True)
        final_result['abs_scores'] = np.abs(final_result['scores'])
        final_result.sort_values('abs_scores', ascending=False, inplace=True)
        deg_dict[perturbation] = final_result

    return deg_dict


def get_average_expression(adata, condition_col='condition', control_name='ctrl'):
    """
    返回每个条件（扰动）的平均表达矩阵，行=条件，列=基因。
    跳过 control_name 指定的对照条件。
    """
    avg = {}
    for cond in set(adata.obs[condition_col]):
        if cond == control_name:
            continue
        sub = adata[adata.obs[condition_col] == cond]
        # 计算均值，结果可能是 np.matrix (稀疏) 或 np.ndarray (稠密)
        mean_vals = sub.X.mean(axis=0)
        # 转换为平坦的一维数组
        if hasattr(mean_vals, 'A1'):
            mean_vals = mean_vals.A1  # 稀疏矩阵的 .A1 方法
        else:
            mean_vals = np.asarray(mean_vals).flatten()  # 稠密数组处理
        avg[cond] = mean_vals
    return pd.DataFrame(avg).T  # 返回 DataFrame，行=条件，列=基因

# def get_average_expression(adata, condition_col='condition'):
#     """返回每个条件的平均表达矩阵（条件 × 基因）"""
#     avg = {}
#     for cond in set(adata.obs[condition_col]):
#         if cond == 'ctrl':
#             continue
#         avg[cond] = adata[adata.obs[condition_col] == cond].X.mean(axis=0).A1
#     return pd.DataFrame(avg).T  # 行=条件，列=基因

def get_distributions(adata, condition_col='condition'):
    """返回每个条件的表达矩阵列表（用于分布比较）"""
    dists = {}
    for cond in set(adata.obs[condition_col]):
        if cond == 'ctrl':
            continue
        dists[cond] = adata[adata.obs[condition_col] == cond].X
    return dists


def mse_score(pred_avg, true_avg):
    """均方误差，基于平均表达向量"""
    return mean_squared_error(true_avg, pred_avg)


def pcc_delta(pred_avg, true_avg, ctrl_avg):
    """PCC-delta：计算扰动前后变化量的Pearson相关系数"""
    delta_true = true_avg - ctrl_avg
    delta_pred = pred_avg - ctrl_avg
    # 避免常数向量
    if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
        return np.nan
    # corr = np.corrcoef(delta_true, delta_pred)[0, 1]
    corr, _ = pearsonr(delta_true, delta_pred)
    return corr


def calculate_correlation(real_expr, pred_expr):
    """Calculate Pearson and Spearman correlations between real and predicted expression"""
    print(real_expr.shape, pred_expr.shape)

    # mean_real = np.mean(real_expr, axis=0)
    # mean_pred = np.mean(pred_expr, axis=0)
    mean_real = real_expr
    mean_pred = pred_expr
    # mean_pred = pred_expr

    pearson_r, _ = pearsonr(mean_real, mean_pred)
    spearman_r, _ = spearmanr(mean_real, mean_pred)

    # Return average correlations
    return pearson_r, spearman_r


def e_distance(pred_avg, true_avg, ctrl_avg):
    """
    E-distance：基于平均表达，计算两组距离的期望差异。
    此处简化为欧氏距离（原文E-distance定义见公式(2)，需计算组间与组内距离）。
    为简便，我们使用欧氏距离作为代理，若需精确实现请参考原文。
    """
    return np.linalg.norm(pred_avg - true_avg)


def wasserstein_per_gene(pred_X, true_X):
    """对每个基因计算Wasserstein距离，返回平均值"""
    n_genes = pred_X.shape[1]
    ws = []
    for i in range(n_genes):
        w = wasserstein_distance(pred_X[:, i], true_X[:, i])
        ws.append(w)
    return np.mean(ws)


def kl_divergence_per_gene(pred_X, true_X, bins=50):
    """
    对每个基因计算KL散度，使用直方图估计概率密度。
    返回平均值（需确保概率分布归一化）。
    """
    kl_vals = []
    n_genes = pred_X.shape[1]
    for i in range(n_genes):
        p_hist, bin_edges = np.histogram(pred_X[:, i], bins=bins, density=True)
        q_hist, _ = np.histogram(true_X[:, i], bins=bin_edges, density=True)
        # 添加小偏移避免零
        p_hist += 1e-10
        q_hist += 1e-10
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        kl = entropy(p_hist, q_hist)
        kl_vals.append(kl)
    return np.mean(kl_vals)


def common_deg_metric(adata_pred, adata_true, condition_col='condition', top_n=100):
    """
    计算每个扰动下预测与真实DEGs的重叠率，返回平均值。
    需要先运行差异表达分析（以对照为参照）。
    """
    # 假设对照条件在 obs['condition'] 中标识为 'control'
    # 我们为每个扰动计算相对于对照的差异基因
    common_deg_ratios = []
    for pert in set(adata_pred.obs[condition_col]):
        if pert == 'ctrl':
            continue
        # 真实数据：该扰动 vs 对照
        true_mask = (adata_true.obs[condition_col] == pert)
        ctrl_mask = (adata_true.obs[condition_col] == 'ctrl')
        if true_mask.sum() == 0 or ctrl_mask.sum() == 0:
            continue
        # 使用t检验（或scanpy的rank_genes_groups）得到差异基因
        # 简便：用均值差排序
        true_avg_pert = adata_true[true_mask].X.mean(axis=0).A1
        true_avg_ctrl = adata_true[ctrl_mask].X.mean(axis=0).A1
        true_fold = true_avg_pert - true_avg_ctrl
        true_top_genes = np.argsort(-np.abs(true_fold))[:top_n]  # 按变化幅度排序
        true_top_genes = adata_true.var_names[true_top_genes]

        # 预测数据
        pred_mask = (adata_pred.obs[condition_col] == pert)
        pred_avg_pert = adata_pred[pred_mask].X.mean(axis=0).A1
        pred_avg_ctrl = adata_pred[adata_pred.obs[condition_col] == 'ctrl'].X.mean(axis=0).A1
        pred_fold = pred_avg_pert - pred_avg_ctrl
        pred_top_genes = np.argsort(-np.abs(pred_fold))[:top_n]
        pred_top_genes = adata_pred.var_names[pred_top_genes]

        overlap = len(set(true_top_genes) & set(pred_top_genes))
        ratio = overlap / top_n
        common_deg_ratios.append(ratio)
    return np.mean(common_deg_ratios)


def evaluate_perturbation_generalization(ctrl, pred, true,
                                         min_cells=30, max_cells=2000, n_hvgs=5000,
                                         mito_threshold=0.1, min_genes=200):
    """
    ctrl, pred, true: AnnData 对象，包含 .obs['condition']
    返回所有扰动条件下各项指标的平均值及详细结果
    """
    # ---------- 1. 预处理 ----------
    print("预处理：细胞过滤、下采样...")
    ctrl = preprocess_perturbation_data(ctrl, min_cells, max_cells, n_hvgs, mito_threshold, min_genes)
    pred = preprocess_perturbation_data(pred, min_cells, max_cells, n_hvgs, mito_threshold, min_genes)
    true = preprocess_perturbation_data(true, min_cells, max_cells, n_hvgs, mito_threshold, min_genes)

    # 对齐基因
    print("对齐基因...")
    ctrl, pred, true = align_genes([ctrl, pred, true])

    # 在真实数据上选择高变基因，并过滤其他数据
    print("选择高变基因...")
    hvgs = select_hvgs(true, n_hvgs=n_hvgs)

    print(len(hvgs))
    print(hvgs[:10])

    # ctrl = ctrl[:, hvgs].copy()
    # pred = pred[:, hvgs].copy()
    # true = true[:, hvgs].copy()

    print(ctrl.shape)
    print(pred.shape)
    print(true.shape)

    # 归一化：每个细胞标准化到 10000 总计数，然后 log1p
    for ad in [ctrl, pred, true]:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    # ---------- 2. 计算指标 ----------
    # 获得平均表达矩阵（条件 × 基因）
    # ctrl_avg = get_average_expression(ctrl)

    ctrl_avg = ctrl.X.mean(axis=0)

    pred_avg = get_average_expression(pred)
    true_avg = get_average_expression(true)

    # 获取每个条件的细胞表达矩阵（用于分布指标）
    pred_dist = get_distributions(pred)
    true_dist = get_distributions(true)

    # 保证条件顺序一致
    common_conds = set(pred_avg.index) & set(true_avg.index)
    print(pred_avg.index)
    print(true_avg.index)
    print(ctrl_avg.shape)

    print(len(common_conds))

    common_conds = sorted(common_conds)

    metrics = {
        'MSE': [],
        'PCC_delta': [],
        'Pearson': [],
        'Spearman': [],
        'E_distance': [],
        'Wasserstein': [],
        'KL_divergence': [],
        'Common_DEGs': []
    }

    for cond in common_conds:
        # 平均向量
        pred_vec = pred_avg.loc[cond].values
        true_vec = true_avg.loc[cond].values
        ctrl_vec = ctrl_avg

        metrics['MSE'].append(mean_squared_error(true_vec, pred_vec))
        metrics['PCC_delta'].append(pcc_delta(pred_vec, true_vec, ctrl_vec))
        pearson_r, spearman_r = calculate_correlation(true_vec, pred_vec)
        metrics['Pearson'].append(pearson_r)
        metrics['Spearman'].append(spearman_r)
        metrics['E_distance'].append(e_distance(pred_vec, true_vec, ctrl_vec))

        # 分布指标
        pred_X = pred_dist[cond]
        true_X = true_dist[cond]
        metrics['Wasserstein'].append(wasserstein_per_gene(pred_X, true_X))
        metrics['KL_divergence'].append(kl_divergence_per_gene(pred_X, true_X))

    # Common-DEGs 需要对照信息，单独计算
    # 为计算准确，将预测和真实数据合并为一个对象？这里我们直接使用原有数据
    # 注意：Common-DEGs 计算需所有扰动条件，上面已循环得到 deg_ratios，但为了与其它指标保持相同条件，也可单独计算。
    deg_ratio = common_deg_metric(pred, true, top_n=100)
    # 注意：上面函数对所有扰动计算平均，但 common_conds 可能不完全一致，这里我们假设一致。
    # 为简化，我们直接使用 common_deg_metric 结果，并复制到每个条件（展示时用平均值）

    # 汇总平均
    summary = {k: np.nanmean(v) for k, v in metrics.items()}
    summary['Common_DEGs'] = deg_ratio

    return summary, metrics


import scanpy as sc

adata_real_test = sc.read_h5ad('data/rep_hepg2_test.h5ad')
adata_real_test.var_names = adata_real_test.var['gene_name']
# 去除test_adata中可能存在的重复基因
adata_real_test = adata_real_test[:, ~adata_real_test.var_names.duplicated()]

true_adata = adata_real_test[~(adata_real_test.obs.condition == 'ctrl')]
control_adata = adata_real_test[adata_real_test.obs.condition == 'ctrl']

adata_list = []
for gene in true_adata.obs.gene.unique():
    adata = sc.read_h5ad(f'predictions/{gene}_ctrl.h5ad')
    adata.obs['gene'] = gene
    adata_list.append(adata)

pred_adata = sc.concat(adata_list)

common_genes = list(set(pred_adata.var_names) & set(true_adata.var_names))

pred_adata = pred_adata[:, common_genes]
true_adata = true_adata[:, common_genes]
control_adata = control_adata[:, common_genes]

pred_adata.obs['original_index'] = pred_adata.obs_names
pred_adata.obs_names = pd.RangeIndex(len(pred_adata))

# 假设已加载数据
# import scanpy as sc
# ctrl = sc.read_h5ad('control.h5ad')
# pred = sc.read_h5ad('predicted.h5ad')
# true = sc.read_h5ad('true.h5ad')

# 确保 .obs['condition'] 存在
# ctrl.obs['condition'] = 'control'   # 如果还没有，则添加
# pred.obs['condition'] = ...  # 每个细胞对应的扰动标签
# true.obs['condition'] = ...

summary, per_cond = evaluate_perturbation_generalization(control_adata, pred_adata, true_adata, n_hvgs=5000)
print("Average metrics over conditions:")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")