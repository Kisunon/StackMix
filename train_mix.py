import scanpy as sc
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import sparse

from src.stack.data.finetuning.perturbation_dataset import create_perturbation_dataloader
from src.stack.models.finetune.model_mix import PertMix
from src.stack.finetune.utils import override_model_config_n_cells, build_scheduler_config


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)


    # train_adata = sc.read_h5ad("data/perturb_processed.h5ad")
    train_adata = sc.read_h5ad("data/rpe1/rep_rpe1_train.h5ad")
    print(train_adata)
    print(train_adata.obs.condition.value_counts())

    test_adata = sc.read_h5ad("data/rpe1/rep_rpe1_test.h5ad")
    print(test_adata)
    print(test_adata.obs.condition.value_counts())

    # 判断训练数据是否为稀疏矩阵
    if sparse.issparse(train_adata.X):
        print("训练数据为稀疏矩阵")
    else:
        print("训练数据为稠密矩阵")

    sample_size = 20
    max_epochs = 100

    # 创建DataLoader
    dataloader, num_genes, pert_list, node_map_pert = create_perturbation_dataloader(
        adata=train_adata,
        batch_size=64,
        num_control_cells=sample_size,
        num_perturb_cells=sample_size,
        random_state=42,
        mode='train',
        num_workers=5
    )

    print(len(dataloader))

    test_dataloader = create_perturbation_dataloader(
        adata=test_adata,
        batch_size=64,
        num_control_cells=sample_size,
        num_perturb_cells=sample_size,
        random_state=42,
        mode='test',
        num_workers=5
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_pert_emb = {
        "num_genes": num_genes,
        "num_perts": len(pert_list),
        "hidden_size": 16,  # todo 与stack的hidden_size保持一致
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

    args_scheduler = {"scheduler": "cosine", "scheduler_T_max": max_epochs, "scheduler_warmup_epochs": 0, "scheduler_eta_min": 1e-6}

    scheduler_config = build_scheduler_config(args_scheduler)

    model = PertMix(args_pert_emb, model_config, checkpoint_path=checkpoint_path, scheduler_config=scheduler_config)
    model.to(device)

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/rpe1",  # 保存目录
        filename="model-{epoch:02d}-{val_loss:.4f}",  # 文件名模板
        save_top_k=1,  # 保存前3个最佳模型
        monitor="val_loss",  # 监控验证损失
        mode="min",  # 损失越小越好
        save_last=True,  # 保存最后一个检查点
        every_n_epochs=5,  # 每个epoch保存一次
    )

    # 配置 TensorBoard 日志器
    logger = TensorBoardLogger(
        save_dir="./logs",  # 日志保存目录
        name="pertmix_rpe1"  # 实验名称
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],  # 添加检查点回调
        logger=logger,  # 添加 TensorBoard 日志器
    )
    trainer.fit(model, dataloader, val_dataloaders=test_dataloader)



    # 在训练循环中使用
    # for batch in dataloader:
    #     observed_features, ground_truth_features, pert_idx = batch
    #     print(observed_features.shape)
    #     print(ground_truth_features.shape)
    #     print(pert_idx.shape)
    #     print(pert_idx)
    #     break
