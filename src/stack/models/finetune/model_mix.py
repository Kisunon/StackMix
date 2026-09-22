"""Fine-tuning model built on top of the core StateICL architecture."""

from __future__ import annotations

import os
import logging
from typing import  Any, Dict, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import SGConv
import pytorch_lightning as pl

from ..core import StateICLModel
from .mixins import FinetuneInitializationMixin, FinetuneLossMixin
from ...model_loading import load_model_from_checkpoint

log = logging.getLogger(__name__)

class PertMix(pl.LightningModule):
    def __init__(self,
                 args_pert_emb,
                 arg_icl,
                 checkpoint_path: Optional[str] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 match_teacher: bool = True,
                 ):
        super().__init__()

        self.pert_emb = PertEmbedding(args_pert_emb)
        self.model = ICL_FinetunedModelMix(**arg_icl)
        self.teacher_model = StateICLModel(**arg_icl)

        self.match_teacher = match_teacher

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.teacher_ema_decay = 0.95
        self.ema_every_n_steps = 500

        self.train_metrics = []
        self.val_metrics = []

        # 添加预训练参数加载逻辑
        if checkpoint_path is not None:
            self.load_pretrained_parameters(checkpoint_path)

    def load_pretrained_parameters(self, checkpoint_path: str) -> None:
        """加载预训练参数到model和teacher_model"""
        log.info(f"Loading pretrained parameters from checkpoint: {checkpoint_path}")

        # 首先加载预训练的StateICLModel
        pretrained_model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_class="StateICLModel",
            strict=False  # 设置为False以允许部分参数加载
        )

        # 将预训练参数加载到student model
        self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
        log.info("Pretrained parameters loaded into student model")

        # 将相同的预训练参数加载到teacher model
        self.teacher_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        log.info("Pretrained parameters loaded into teacher model")

        # 确保teacher_model保持冻结状态
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:  # type: ignore[override]
        if self.match_teacher:
            log.info("`on_fit_start` hook called. Syncing teacher model weights from student...")
            self.teacher_model.load_state_dict(self.model.state_dict())

            log.info("VERIFYING teacher weights against student weights...")
            student_param = self.model.layers[0].cell_attn.qkv.weight
            teacher_param = self.teacher_model.layers[0].cell_attn.qkv.weight

            if torch.equal(student_param, teacher_param):
                log.info("SUCCESS: Teacher weights match student weights.")
                log.info("Sample tensor sum: %s", student_param.sum().item())
            else:
                diff = torch.sum(torch.abs(student_param - teacher_param))
                log.error("FAILURE: Teacher and student weights diverged; diff=%s", diff.item())
                raise RuntimeError("Teacher model failed to sync with student model.")

            log.info("Teacher model weights synced successfully.")
        # try:
        #     model_config = self.hparams.model_config
        #     n_kept_cell = self.hparams.n_kept_cell
        #
        #     dummy_obs = torch.rand(1, model_config['n_cells'], model_config['n_genes'], device=self.device)
        #     dummy_gt = torch.rand(1, model_config['n_cells'], model_config['n_genes'], device=self.device)
        #     dummy_ct = torch.zeros(1, model_config['n_cells'], dtype=torch.long, device=self.device)
        #     dummy_mask = torch.ones(1, model_config['n_cells'], dtype=torch.bool, device=self.device)
        #
        #     with torch.no_grad():
        #         _ = self.model(
        #             observed_features=dummy_obs,
        #             ground_truth_features=dummy_gt,
        #             cell_type_ids=dummy_ct,
        #             position_mask=dummy_mask,
        #             n_kept_cell=n_kept_cell,
        #             return_loss=False,
        #         )
        #         _ = self.teacher_model(
        #             observed_features=dummy_gt,
        #             ground_truth_features=dummy_gt,
        #             cell_type_ids=dummy_ct,
        #             position_mask=dummy_mask,
        #             mask_genes=False,
        #             n_kept_cell=model_config['n_cells'],
        #             return_loss=False,
        #         )
        #     log.info("Model architecture verification passed for both student and teacher.")
        # except Exception as exc:  # pragma: no cover - diagnostic logging path
        #     log.error("Model architecture mismatch detected during verification: %s", exc)
        #     raise

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        # 只保存pert_emb和student model，不保存teacher_model等其他组件
        # 首先获取完整的状态字典
        full_state_dict = checkpoint['state_dict']

        # 创建一个新的状态字典，只包含pert_emb和model的参数
        filtered_state_dict = {}
        for key, value in full_state_dict.items():
            # 只保留pert_emb和model的参数
            if key.startswith('pert_emb.') or key.startswith('model.'):
                filtered_state_dict[key] = value

        # 替换原来的state_dict为过滤后的版本
        checkpoint['state_dict'] = filtered_state_dict

        # 可选：添加一些元数据，说明我们只保存了部分参数
        checkpoint['save_info'] = {
            'saved_components': ['pert_emb', 'model'],
            'excluded_components': ['teacher_model']
        }

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------
    def _forward_pass_with_teacher(self, batch: Any) -> Dict[str, Any]:
        ground_truth_features, observed_features, pert_id = batch
        with torch.no_grad():
            teacher_output = self.teacher_model(
                ground_truth_features,
                return_loss=False,
            )
            target_embeddings = teacher_output['cell_embeddings'].detach()

        pert_track, emb_total = self.pert_emb.get_pert_emb(pert_id)

        # for idx, j in enumerate(pert_track.keys()):
        #     pert_emb = emb_total[idx]
        #     # 确保索引是正确的整数类型
        #     indices = pert_track[j]
        #     if isinstance(indices, torch.Tensor):
        #         indices = indices.long()
        #     target_embeddings[indices] = pert_emb

        result = self.model(
            batch,
            pert_track,
            emb_total,
            t_cell_embeddings=target_embeddings,
            return_loss=True,
        )
        return result

    def forward(self, *inputs: Any, **kwargs: Any):  # type: ignore[override]
        return self.model(*inputs, **kwargs)

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        result = self._forward_pass_with_teacher(batch)

        total_loss = result['loss']
        recon_loss = result['recon_loss']
        mmd_loss = result['mmd_loss']
        sw_loss = result['sw_loss']

        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/mmd_loss', mmd_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/sw_loss', sw_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        result = self._forward_pass_with_teacher(batch)

        total_loss = result['loss']
        recon_loss = result['recon_loss']
        mmd_loss = result['mmd_loss']
        sw_loss = result['sw_loss']

        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', recon_loss, on_epoch=True, sync_dist=True)
        self.log('val/mmd_loss', mmd_loss, on_epoch=True, sync_dist=True)
        self.log('val/sw_loss', sw_loss, on_epoch=True, sync_dist=True)

        if 'masked_mae' in result:
            self.log('val/masked_mae', result['masked_mae'], on_epoch=True, sync_dist=True)
        if 'masked_corr' in result:
            self.log('val/masked_corr', result['masked_corr'], on_epoch=True, sync_dist=True)
        if 'sw_predict' in result:
            self.log('val/sw_predict', result['sw_predict'], on_epoch=True, sync_dist=True)
        if 'mask_rate' in result:
            self.log('val/mask_rate', result['mask_rate'], on_epoch=True, sync_dist=True)

        return {'val_loss': total_loss}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        config: Dict[str, Any] = {'optimizer': optimizer}

        if not self.scheduler_config:
            return config

        scheduler_type = self.scheduler_config.get('type', 'cosine')
        if scheduler_type == 'cosine':
            warmup_epochs = self.scheduler_config.get('warmup_epochs', 0)
            T_max = self.scheduler_config.get('T_max', 100)
            eta_min = self.scheduler_config.get('eta_min', 1e-6)

            if warmup_epochs > 0:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, T_max - warmup_epochs),
                    eta_min=eta_min,
                )
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                         milestones=[warmup_epochs])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            config['lr_scheduler'] = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
            return config

        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 10),
                verbose=True,
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
            return config

        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        datamodule = self.trainer.datamodule
        if getattr(datamodule, "resample_each_epoch", False):
            datamodule.train_dataset.resample_training_data()
            if self.global_rank == 0:
                log.info(
                    "[Epoch %s] Resampled training data -> %s samples",
                    self.current_epoch,
                    len(datamodule.train_dataset),
                )

    @torch.no_grad()
    def _ema_update_teacher(self):
        ema = float(self.teacher_ema_decay)

        for t, s in zip(self.teacher_model.parameters(), self.model.parameters()):
            t.data.mul_(ema).add_(s.data, alpha=1.0 - ema)

        for t_buf, s_buf in zip(self.teacher_model.buffers(), self.model.buffers()):
            t_buf.copy_(s_buf)

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer):
        self.teacher_model.eval()
        step = int(self.global_step)
        if step > 0 and (step % int(self.ema_every_n_steps) == 0):
            self._ema_update_teacher()
            if getattr(self, "global_rank", 0) == 0:
                logging.info(
                    f"EMA teacher updated @ global_step={step}, ema={self.teacher_ema_decay}"
                )



class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class PertEmbedding(torch.nn.Module):
    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(PertEmbedding, self).__init__()
        self.args = args
        # self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.num_layers = args['num_go_gnn_layers']
        self.pert_emb_lambda = 0.2

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # globel perturbation embedding dictionary lookup
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        ### perturbation gene ontology GNN
        sim_network = GeneSimNetwork(args['pert_list'], node_map=args['node_map_pert'])
        self.G_sim = sim_network.edge_index.to(args['device'])
        self.G_sim_weight = sim_network.edge_weight.to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

    def get_pert_emb(self, pert_idx):

        ## get perturbation index and embeddings

        pert_index = []
        for idx, i in enumerate(pert_idx):
            for j in i:
                if j != -1:
                    pert_index.append([idx, j])
        pert_index = torch.tensor(pert_index).T

        pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))

        ## augment global perturbation embedding with GNN
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        if pert_index.shape[0] != 0:
            ### in case all samples in the batch are controls, then there is no indexing for pert_index.
            pert_track = {}
            for i, j in enumerate(pert_index[0]):
                if j.item() in pert_track:
                    pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                else:
                    pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

            if len(list(pert_track.values())) > 0:
                if len(list(pert_track.values())) == 1:
                    # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                else:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                # for idx, j in enumerate(pert_track.keys()):
                #     base_emb[j] = base_emb[j] + emb_total[idx]

                return pert_track, emb_total


class ICL_FinetunedModelMix(
    FinetuneInitializationMixin,
    FinetuneLossMixin,
    StateICLModel,
):
    """Fine-tuning head on top of :class:`StateICLModel`."""

    def forward(
        self,
        batch,
        pert_track,
        emb_total,
        t_cell_embeddings: Optional[torch.Tensor] = None,
        mask_genes: bool = True,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}

        observed_features, ground_truth_features, pert_idx = batch

        batch_size, n_cells, n_genes = observed_features.shape

        features_log = torch.log1p(observed_features)
        # 新的 n_context_cell 定义：等于 n_cells，表示所有细胞都是需要预测的 Query

        mask = torch.zeros(
            batch_size,
            n_cells,
            n_genes,
            dtype=torch.bool,
            device=observed_features.device,
        )
        if mask_genes:
            masked_features, mask = self.apply_finetune_mask(features_log)
            tokens = self._reduce_and_tokenize(masked_features)
        else:
            tokens = self._reduce_and_tokenize(features_log)

        x = tokens

        # mask_expanded = torch.ones_like(x)
        # if n_kept_cell < n_cells:
        #     mask_expanded[:, n_context_cell:, :, :] = 1.0
        # query_emb = self.query_pos_embedding.unsqueeze(0).unsqueeze(0)
        # x = x + query_emb * mask_expanded  # todo check 此处是否是将query的mask设为1，即query的位置设为1

        attn_mask = torch.zeros(
            n_cells,
            n_cells,
            dtype=torch.bool,
            device=observed_features.device,
        )
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        for idx, j in enumerate(pert_track.keys()):
            pert_emb = emb_total[idx]
            # 配置扰动嵌入：将pert_emb加到对应batch索引的细胞组的每个基因嵌入上
            # pert_emb形状: [hidden_size]
            # x形状: [batch_size, n_cells, n_genes, hidden_size]
            # 扩展pert_emb到[1, 1, 1, hidden_size]，然后广播到对应batch的所有细胞和基因
            pert_emb_expanded = pert_emb.view(1, 1, -1)
            x[j] = x[j] + pert_emb_expanded

        x = self._run_attention_layers(x, gene_attn_mask=attn_mask)

        final_cell_embeddings = x.reshape(batch_size, n_cells, -1)
        lib_size = ground_truth_features.sum(dim=-1, keepdim=True).clamp(min=1.0)
        nb_mean, nb_dispersion, px_scale = self._compute_nb_parameters(
            final_cell_embeddings,
            lib_size,
        )

        result.update(
            {
                "px_scale": px_scale,
                "nb_mean": nb_mean,
                "nb_dispersion": nb_dispersion,
                "final_cell_embeddings": final_cell_embeddings,
            }
        )

        if not return_loss:
            return result

        # mean_kept_cell = final_cell_embeddings[:, :n_kept_cell].mean(dim=1).detach()
        # mid_feat = final_cell_embeddings[:, n_kept_cell:n_context_cell, :].detach()
        # tail_feat = final_cell_embeddings[:, n_context_cell:, :].detach()
        # cls_loss, cls_acc = self._compute_cls_loss(mean_kept_cell, mid_feat, tail_feat)

        recon_loss, _ = self._compute_reconstruction_loss(
            nb_mean,
            nb_dispersion,
            ground_truth_features,
            mask,
        )

        time = torch.rand(1, device=observed_features.device).item() * 0.9375
        mmd_loss, pred_dist_all, true_dist_all = self._compute_mmd_loss_in_mix(
            nb_mean,
            nb_dispersion,
            ground_truth_features,
            lib_size,
            final_cell_embeddings,
            t_cell_embeddings,
            n_cells,
            n_genes,
            time,
            observed_features.device,
        )
        n_proj = getattr(self, "n_proj", None)
        subsample_size = max(32, min(128, n_cells))
        sw_loss = self._compute_sw_loss(
            final_cell_embeddings,
            subsample_size=subsample_size,
            min_size=subsample_size,
            max_size=subsample_size,
                n_proj=n_proj,
        )

        total_loss = recon_loss + mmd_loss + self.sw_weight * sw_loss
        result.update(
            {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "mmd_loss": mmd_loss,
                "sw_loss": sw_loss,
            }
        )

        # todo 重新定义评估部分
        if not self.training:
            metrics = self._compute_eval_metrics(
                nb_mean,
                ground_truth_features,
                mask,
            )

            if pred_dist_all is not None and true_dist_all is not None:
                sw_predict = self.sw_distance(pred_dist_all, true_dist_all, n_proj=256)
            else:
                sw_predict = torch.tensor(0.0, device=observed_features.device)

            result.update({**metrics, "sw_predict": sw_predict})
        return result

    def _compute_mmd_loss_in_mix(
            self,
            nb_mean: torch.Tensor,
            nb_dispersion: torch.Tensor,
            ground_truth_features: torch.Tensor,
            lib_size: torch.Tensor,
            final_cell_embeddings: torch.Tensor,
            t_cell_embeddings: Optional[torch.Tensor],
            n_cells: int,
            n_genes: int,
            time: float,
            device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute MMD loss for mix model, handling both feature distributions and embeddings."""
        # 处理基因掩码，选择方差最大的1000个基因
        mask_indices = torch.ones(n_genes, dtype=torch.bool, device=device)
        rep_lib_size = lib_size
        if mask_indices.sum().item() > 1000:
            theta = 100
            expected_counts = (
                    ground_truth_features.sum(dim=1, keepdim=True)
                    * ground_truth_features.sum(dim=2, keepdim=True)
                    / rep_lib_size.sum(dim=1, keepdim=True)
            )
            variance = (expected_counts + (expected_counts ** 2) / theta).clamp_min(1e-4)
            residuals = (
                                ground_truth_features
                                - expected_counts
                        ) / torch.sqrt(variance)
            residuals_var = residuals.var(dim=1).mean(dim=0)
            mask_indices &= torch.zeros_like(mask_indices).index_fill_(
                0,
                torch.topk(residuals_var, min(1000, residuals_var.numel())).indices,
                True,
            )
        # 计算预测分布和真实分布
        pred_mean = nb_mean
        # 使用所有细胞的dispersion的中位数
        pred_dispersion = nb_dispersion.median(dim=1, keepdim=True).values.detach()
        pred_dist_all = self.nblog_sampler(
            mu=pred_mean[:, :, mask_indices],
            theta=pred_dispersion[:, :, mask_indices],
            N=rep_lib_size / 1e4,
        )
        true_dist_all = torch.log1p(
            1e4 * ground_truth_features[:, :, mask_indices] / rep_lib_size
        )
        # 计算预测嵌入和真实嵌入
        pred_embed_all = final_cell_embeddings
        true_embed_all = None if t_cell_embeddings is None else t_cell_embeddings
        # 计算特征分布的MMD损失
        mmd_loss = self.mmd_loss(pred_dist_all, true_dist_all).mean()

        # 计算嵌入的MMD损失
        if true_embed_all is None:
            true_embed_all = pred_embed_all
        mmd_embed_loss = self.mmd_loss(pred_embed_all, true_embed_all).mean()
        # 综合两种损失
        total_mmd_loss = (0.5 * mmd_loss + 0.5 * mmd_embed_loss) / (1 - time)
        return total_mmd_loss, pred_dist_all, true_dist_all

class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, gene_list, node_map, num_similar_genes_go_graph=20):
        """
        Initialize GeneSimNetwork class
        """
        self.num_similar_genes_go_graph = num_similar_genes_go_graph
        self.edge_list = self.get_go_network()
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                                         target='target', edge_attr=['importance'],
                                         create_using=nx.DiGraph())
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                       self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        # self.edge_weight = torch.Tensor(self.edge_list['importance'].values)

        edge_attr = nx.get_edge_attributes(self.G, 'importance')
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

    def get_go_network(self,):
        """
        Get GO network
        """
        data_path = "./data"
        df_jaccard = pd.read_csv(os.path.join(data_path, 'go_essential_all.csv'))
        k = self.num_similar_genes_go_graph
        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                                                         ['importance'])).reset_index(drop=True)
        return df_out


def predict_perturbation_effect(
        control_cells: torch.Tensor,
        pert_id: torch.Tensor,
        model: Optional[PertMix] = None,
        checkpoint_path: Optional[str] = None,
        args_pert_emb: Optional[dict] = None,
        arg_icl: Optional[dict] = None,
        return_embeddings: bool = False,
        device: Optional[torch.device] = None
):
    """
    预测对照细胞在扰动条件下的基因表达

    Args:
        control_cells: 对照细胞的基因表达，形状为 (batch_size, n_cells, n_genes)
        pert_id: 扰动条件列表，每个元素是扰动的索引
        model: 已经创建好的模型实例，如果提供则使用
        checkpoint_path: 训练好的模型 checkpoint 路径，当model为None时需要
        args_pert_emb: PertEmbedding 的参数，当model为None时需要
        arg_icl: ICL_FinetunedModelMix 的参数，当model为None时需要
        device: 运行设备，默认为自动选择

    Returns:
        预测的扰动后基因表达，形状为 (batch_size, n_cells, n_genes)
    """
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果没有提供模型，则创建并加载
    if model is None:
        if checkpoint_path is None or args_pert_emb is None or arg_icl is None:
            raise ValueError("If model is not provided, checkpoint_path, args_pert_emb, and arg_icl must be provided")

        # 创建模型实例
        model = PertMix(args_pert_emb, arg_icl)

        # 加载训练好的参数
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)

        # 移至设备并设置为评估模式
        model.to(device)
        model.eval()

    # 准备输入数据
    control_cells = control_cells.to(device)

    # 构建 batch
    # 注意：这里 ground_truth_features 可以是任意值，因为我们只需要预测结果
    ground_truth_features = torch.zeros_like(control_cells)
    batch = (control_cells, ground_truth_features, pert_id)

    # 获取扰动嵌入
    pert_track, emb_total = model.pert_emb.get_pert_emb(pert_id)

    # 预测
    with torch.no_grad():
        result = model.model(
            batch,
            pert_track,
            emb_total,
            return_loss=False
        )

    # 返回预测的基因表达（nb_mean）
    if return_embeddings:
        return result['nb_mean'], result['final_cell_embeddings']
    else:
        return result['nb_mean']
