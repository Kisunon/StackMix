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
    pretrain_epochs = 1
    max_epochs = 2
    
    # 加载所有数据集
    print("=== 加载数据集 ===")
    train_adata = sc.read_h5ad("data/rep_hepg2_train.h5ad")
    print(f"训练数据: {train_adata.shape}")
    
    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/pretrain",  # 保存目录
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
        name="pertmix_hepg2"  # 实验名称
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

    model = PertMix(args_pert_emb, model_config, checkpoint_path=checkpoint_path, scheduler_config=scheduler_config)
    model.to(device)

    # 预训练循环：遍历三个预训练数据集
    pretrain_datasets = [
        ("jurkat", "data/rep_jurkat_pretrain.h5ad"),
        ("K562", "data/rep_K562_pretrain.h5ad"),
        ("rpe1", "data/rep_rpe1_pretrain.h5ad")
    ]

    for dataset_name, pretrain_adata in pretrain_datasets:
        print(f"\n=== 在 {dataset_name} 数据集上预训练 ===")
        
        # 创建当前预训练数据集的DataLoader
        pretrain_adata = sc.read_h5ad(pretrain_adata)
        pretrain_dataloader, _, _, _ = create_perturbation_dataloader(
            adata=pretrain_adata,
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
        pretrain_trainer.fit(model, pretrain_dataloader, val_dataloaders=None)
        
        print(f"=== {dataset_name} 数据集预训练完成 ===")

        # 释放内存
        del pretrain_adata
        del pretrain_dataloader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 正式训练阶段
    train_adata = sc.read_h5ad("data/rep_hepg2_train.h5ad")
    print(f"训练数据: {train_adata.shape}")

    test_adata = sc.read_h5ad("data/rep_hepg2_test.h5ad")
    print(f"测试数据: {test_adata.shape}")

    print("\n=== 开始正式训练 ===")
    print("训练数据条件分布:")
    print(train_adata.obs.condition.value_counts())
    print("测试数据条件分布:")
    print(test_adata.obs.condition.value_counts())

    # 释放内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 判断训练数据是否为稀疏矩阵
    if sparse.issparse(train_adata.X):
        print("训练数据为稀疏矩阵")
    else:
        print("训练数据为稠密矩阵")

    # 创建训练和测试DataLoader
    train_dataloader, num_genes, pert_list, node_map_pert = create_perturbation_dataloader(
        adata=train_adata,
        batch_size=64,
        num_control_cells=sample_size,
        num_perturb_cells=sample_size,
        random_state=42,
        mode='train',
        num_workers=0
    )

    print(f"训练数据加载器长度: {len(train_dataloader)}")

    test_dataloader = create_perturbation_dataloader(
        adata=test_adata,
        batch_size=64,
        num_control_cells=sample_size,
        num_perturb_cells=sample_size,
        random_state=42,
        mode='test',
        num_workers=0
    )

    # 释放内存
    del train_adata
    del test_adata
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 更新模型的扰动嵌入参数以适应新的训练数据
    args_pert_emb.update({
        "num_perts": len(pert_list),
        "pert_list": pert_list,
        "node_map_pert": node_map_pert
    })

    # 更新调度器配置以适应正式训练的轮数
    args_scheduler = {"scheduler": "cosine", "scheduler_T_max": max_epochs, "scheduler_warmup_epochs": 0,
                      "scheduler_eta_min": 1e-6}
    scheduler_config = build_scheduler_config(args_scheduler)

    # 重新初始化模型以适应新的扰动嵌入参数
    model = PertMix(args_pert_emb, model_config, checkpoint_path=None, scheduler_config=scheduler_config)
    # 加载预训练后的权重
    try:
        model.load_state_dict(torch.load("./checkpoints/last.ckpt")['state_dict'])
        print("成功加载预训练权重")
    except Exception as e:
        print(f"加载预训练权重失败: {e}")
    model.to(device)

    # 配置正式训练的Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # 执行正式训练
    trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader)
    print("=== 正式训练完成 ===")



    # 在训练循环中使用
    # for batch in dataloader:
    #     observed_features, ground_truth_features, pert_idx = batch
    #     print(observed_features.shape)
    #     print(ground_truth_features.shape)
    #     print(pert_idx.shape)
    #     print(pert_idx)
    #     break
