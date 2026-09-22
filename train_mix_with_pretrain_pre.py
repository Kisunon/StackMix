import scanpy as sc
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import sparse
import gc

from src.stack.data.finetuning.perturbation_dataset import create_perturbation_dataloader
from src.stack.models.finetune.model_mix import PertMix
from src.stack.finetune.utils import override_model_config_n_cells, build_scheduler_config


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    sample_size = 20
    pretrain_epochs = 50
    dataset_index = 2

    pretrain_datasets = [
        ("K562", "data/rep_K562_pretrain.h5ad"),
        ("rpe1", "data/rep_rpe1_pretrain.h5ad"),
        ("jurkat", "data/rep_jurkat_pretrain.h5ad"),
    ]

    dataset_name, pretrain_adata_path = pretrain_datasets[dataset_index]

    # match_teacher = True if dataset_index == 0 else False
    match_teacher = True

    # 加载所有数据集
    print("=== 加载数据集 ===")
    train_adata = sc.read_h5ad(pretrain_adata_path)
    print(f"训练数据: {train_adata.shape}")
    
    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/pretrain_match/{dataset_name}",  # 保存目录
        filename="model-{epoch:02d}-{val_loss:.4f}",  # 文件名模板
        save_top_k=1,  # 保存最佳模型
        monitor="val_loss",  # 监控验证损失
        mode="min",  # 损失越小越好
        save_last=True,  # 保存最后一个检查点
        every_n_epochs=5,  # 每个epoch保存一次
    )

    # 配置 TensorBoard 日志器
    logger = TensorBoardLogger(
        save_dir="./logs",  # 日志保存目录
        name=f"pertmix_hepg2_pretrain_match_{dataset_name}"  # 实验名称
    )

    # 预训练阶段
    print("\n=== 开始预训练 ===")
    checkpoint_path = "./weight/Stack-Large/bc_large.ckpt"
    model_config = override_model_config_n_cells(checkpoint_path, sample_size)
    print(f"模型配置: {model_config}")

    # 首先使用第一个预训练数据集初始化模型
    dataloader, num_genes, pert_list, node_map_pert = create_perturbation_dataloader(
        adata=train_adata,
        batch_size=64,
        num_control_cells=sample_size,
        num_perturb_cells=sample_size,
        random_state=42,
        mode='train',
        num_workers=0
    )

    # 释放内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    args_pert_emb = {
        "num_genes": num_genes,
        "num_perts": len(pert_list),
        "hidden_size": 16,  # todo 与stack的hidden_size保持一致
        "num_go_gnn_layers": 1,
        "pert_list": pert_list,
        "node_map_pert": node_map_pert,
        "device": device,
    }

    print(f"num_genes: {num_genes}")
    print(f"num_perts: {len(pert_list)}")
    print(f"num_pert_graph: {len(node_map_pert.keys())}")

    args_scheduler = {"scheduler": "cosine", "scheduler_T_max": pretrain_epochs, "scheduler_warmup_epochs": 0,
                      "scheduler_eta_min": 1e-6}

    scheduler_config = build_scheduler_config(args_scheduler)

    model = PertMix(args_pert_emb, model_config, checkpoint_path=checkpoint_path, scheduler_config=scheduler_config, match_teacher=match_teacher)
    if dataset_index > 0:
        # 加载训练好的参数
        pretrain_checkpoint_path = f"./checkpoints/pretrain/{pretrain_datasets[dataset_index-1][0]}/last.ckpt"
        print(f"加载预训练模型参数: {pretrain_checkpoint_path}")
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)

        # 加载教师模型参数
        checkpoint_teacher = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_teacher['state_dict']
        model.teacher_model.load_state_dict(state_dict, strict=False)

    model.to(device)

    print(f"\n=== 在 {dataset_name} 数据集上预训练 ===")


    # 释放内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 创建临时验证集DataLoader
    # val_dataloader = create_perturbation_dataloader(
    #     adata=pretrain_adata,
    #     batch_size=64,
    #     num_control_cells=sample_size,
    #     num_perturb_cells=sample_size,
    #     random_state=42,
    #     mode='test',
    #     num_workers=0
    # )

    # 配置预训练的Trainer
    pretrain_trainer = pl.Trainer(
        max_epochs=pretrain_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # 执行预训练
    pretrain_trainer.fit(model, dataloader, val_dataloaders=None)

    print(f"=== {dataset_name} 数据集预训练完成 ===")
