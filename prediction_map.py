import torch
import logging
import scanpy as sc
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader

from src.stack.models.finetune.model_mix import predict_perturbation_effect, PertMix
from src.stack.data.finetuning.perturbation_dataset import PerturbationMappingDataset, \
    create_perturbation_mapping_dataloader
from src.stack.finetune.utils import override_model_config_n_cells

if __name__ == '__main__':
    # 准备输入数据
    logging.basicConfig(level=logging.INFO)

    test_adata = sc.read_h5ad("data/rep_hepg2_test.h5ad")
    print(test_adata)
    print(test_adata.obs.condition.value_counts())

    control_adata = test_adata[test_adata.obs.condition == 'ctrl']

    # 准备扰动基因集
    # sara_set = pd.read_csv("data/feature_set/sara_geneset.csv")
    # cosmic_set = pd.read_csv("data/feature_set/cosmic_geneset.csv")
    # pert_set = list(set(sara_set['Gene Symbol'].tolist() + cosmic_set['Gene Symbol'].tolist()))

    pert_set = ['TOP2A', 'BIRC5', 'CDC20', 'UTP20', 'TAF3', 'WARS', 'ACTR3', 'AURKB']

    # 将pert_set中所有基因转换为大写
    pert_set = [gene.upper() for gene in pert_set]

    print(len(pert_set))

    sample_size = 20
    batch_size = 64

    # 创建预测数据集和数据加载器
    test_dataloader, dataset = create_perturbation_mapping_dataloader(
        adata=test_adata,
        perturbation_genes=pert_set,
        batch_size=batch_size,
        num_control_cells=sample_size,
        max_control_cells_per_perturbation=300,
        random_state=42,
        num_workers=0
    )

    pert_list = dataset.pert_names
    node_map_pert = dataset.node_map_pert

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_pert_emb = {
        "num_perts": len(pert_list),
        "hidden_size": 16,  # 与stack的hidden_size保持一致
        "num_go_gnn_layers": 1,
        "pert_list": pert_list,
        "node_map_pert": node_map_pert,
        "device": device,
    }

    pre_path = "./weight/Stack-Large/bc_large.ckpt"
    model_config = override_model_config_n_cells(pre_path, sample_size)

    # 创建并加载模型
    model = PertMix(args_pert_emb, model_config)

    model_path = './checkpoints/last-v1.ckpt'

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 创建predictions目录
    os.makedirs('predictions', exist_ok=True)

    # 初始化存储
    all_pred_expr = []
    all_emb = []
    all_obs = []
    all_cell_idxs = []

    # 遍历所有batch进行预测
    for batch_idx, (control_cells, pert_idx, perturbation, cell_idxs) in enumerate(test_dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")

        # 预测 - 使用已创建的模型
        predicted_expression, predicted_embedding = predict_perturbation_effect(
            control_cells,
            pert_idx,
            model=model,
            return_embeddings=True,
        )

        # 处理当前批次的结果
        cell_idxs = np.array(cell_idxs).transpose()
        for i, p in enumerate(perturbation):
            # 获取当前样本的预测结果
            pred_expr = predicted_expression[i].detach().cpu().numpy()
            emb = predicted_embedding[i].detach().cpu().numpy()
            cell_idx = cell_idxs[i]

            # 存储结果
            all_pred_expr.append(pred_expr)
            all_emb.append(emb)
            all_cell_idxs.append(cell_idx)

            # 存储obs信息
            gene = p.split('+')[0]
            all_obs.append({
                'gene': gene,
                'condition': p,
                'cell_type': 'predicted'
            })

    # 释放模型相关内存
    del model
    del dataset
    del test_dataloader
    del checkpoint

    # 合并所有结果
    print("Merging results...")
    pred_expr = np.concatenate(all_pred_expr, axis=0)
    emb = np.concatenate(all_emb, axis=0)
    cell_idxs = np.concatenate(all_cell_idxs, axis=0)
    obs_df = pd.DataFrame(all_obs)

    # 释放临时内存
    del all_pred_expr
    del all_emb
    del all_obs
    del all_cell_idxs

    # 创建单个adata
    print("Creating AnnData object...")
    pred_adata = sc.AnnData(
        X=pred_expr,
        obs=obs_df,
        var=pd.DataFrame(index=dataset.data_target_genes)
    )
    pred_adata.obsm['latent'] = emb
    pred_adata.obs.index = cell_idxs

    print(pred_adata)

    pred_adata.write("predictions/test_v1/predictions_map.h5ad")

    print("Prediction completed successfully!")