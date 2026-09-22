import scanpy as sc
import logging
import torch
import pytorch_lightning as pl
from pathlib import Path

from src.stack.data.finetuning.perturbation_dataset import create_perturbation_dataloader
from src.stack.models.finetune.model_mix import PertMix
from src.stack.finetune.utils import override_model_config_n_cells, build_scheduler_config


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)


    adata = sc.read_h5ad("data/perturb_processed.h5ad")
    print(adata)
    print(adata.obs.condition.value_counts())

    sample_size = 20

    # 创建DataLoader
    dataloader, num_genes, pert_list, node_map_pert = create_perturbation_dataloader(
        adata=adata,
        batch_size=32,  # 每个batch包含32组细胞
        num_control_cells=sample_size,  # 每组包含10个对照细胞
        num_perturb_cells=sample_size,  # 每组包含10个扰动细胞
        random_state=42,
        mode='train',
        num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_pert_emb = {
        "num_genes": num_genes,
        "num_perts": len(pert_list),
        "hidden_size": 128,  # todo 与stack的hidden_size保持一致
        "num_go_gnn_layers": 1,
        "pert_list": pert_list,
        "node_map_pert": node_map_pert,
        "device": device,
    }

    print("num_genes:", num_genes)
    print("num_perts:", len(pert_list))
    print("num_pert_graph:", len(node_map_pert.keys()))

    checkpoint_path = "./weight/Stack-Large/bc_large.ckpt"
    model_config = override_model_config_n_cells(checkpoint_path, sample_size)
    # model_config = {'n_genes': 15012,
    #                 'n_hidden': 100,
    #                 'token_dim': 16,
    #                 'n_cells': sample_size,
    #                 'n_layers': 9,
    #                 'n_heads': 8,
    #                 'dropout': 0.0,
    #                 'mask_rate_min': 0.1,
    #                 'mask_rate_max': 0.8,
    #                 'sw_weight': 0.01,
    #                 'n_proj': 64}

    print(model_config)

    args_scheduler = {"scheduler": "cosine", "scheduler_T_max": 20, "scheduler_warmup_epochs": 0, "scheduler_eta_min": 1e-6}

    scheduler_config = build_scheduler_config(args_scheduler)

    model = PertMix(args_pert_emb, model_config, checkpoint_path=checkpoint_path, scheduler_config=scheduler_config)
    model.to(device)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        deterministic=False,
        benchmark=True,
    )
    trainer.fit(model, dataloader)



    # 在训练循环中使用
    for batch in dataloader:
        observed_features, ground_truth_features, pert_idx = batch
        print(observed_features.shape)
        print(ground_truth_features.shape)
        print(pert_idx.shape)
        print(pert_idx)
        break
