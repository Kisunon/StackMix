import torch
import logging
import scanpy as sc
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader

from src.stack.models.finetune.model_mix import predict_perturbation_effect, PertMix
from src.stack.data.finetuning.perturbation_dataset import PredictionDataset, create_prediction_dataloader
from src.stack.finetune.utils import override_model_config_n_cells

if __name__ == '__main__':
    # 准备输入数据
    logging.basicConfig(level=logging.INFO)

    test_adata = sc.read_h5ad("data/rep_hepg2_test.h5ad")
    print(test_adata)
    print(test_adata.obs.condition.value_counts())

    sample_size = 20
    batch_size = 64

    # 创建预测数据集和数据加载器
    test_dataloader, dataset = create_prediction_dataloader(
        adata=test_adata,
        batch_size=batch_size,
        num_control_cells=sample_size,
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

    model_path = './checkpoints/pretrain/train/last.ckpt'

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 创建predictions目录
    os.makedirs('predictions', exist_ok=True)

    # 收集预测结果
    predictions = {}
    predictions_id = {}

    # 遍历所有batch进行预测
    # todo 在dataset getitem 中添加id
    for batch_idx, (control_cells, pert_idx, perturbation, cell_idxs) in enumerate(test_dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")

        # 预测 - 使用已创建的模型
        predicted_expression = predict_perturbation_effect(
            control_cells,
            pert_idx,
            model=model
        )

        # 按perturbation分组保存结果
        cell_idxs = np.array(cell_idxs)
        cell_idxs = cell_idxs.transpose()
        for i, p in enumerate(perturbation):
            if p not in predictions:
                predictions[p] = []

            # 获取当前样本的预测结果
            pred_expr = predicted_expression[i].detach().cpu().numpy()
            predictions[p].append(pred_expr)

            if p not in predictions_id:
                predictions_id[p] = []
            predictions_id[p].append(cell_idxs[i])

    # 创建基因对齐后的真实数据
    aligned_adata = dataset.create_aligned_adata()

    # 为每个perturbation创建单独的adata
    for perturbation, pred_list in predictions.items():
        # 合并预测结果
        pred_expr = np.concatenate(pred_list, axis=0)
        # 合并cell_idxs
        cell_idxs = np.concatenate(predictions_id[perturbation], axis=0)

        # 创建adata
        pred_adata = sc.AnnData(
            X=pred_expr,
            obs=pd.DataFrame({
                'condition': [perturbation] * pred_expr.shape[0],
                'cell_type': ['predicted'] * pred_expr.shape[0]
            }),
            var=aligned_adata.var.copy()
        )
        pred_adata.obs.index = cell_idxs

        # 保存到文件
        output_path = f"predictions/test_pretrain/{perturbation.replace('+', '_')}.h5ad"
        # pred_adata.write(output_path)
        # print(f"Saved predictions for {perturbation} to {output_path}")
        break


    print(pred_adata.obs_names[:10])

    print("Prediction completed successfully!")