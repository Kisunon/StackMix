# StackMix

**StackMix** is a deep-learning model for predicting single-cell transcriptional responses to **unseen genetic perturbations**. It combines the pretrained [**Stack**](https://huggingface.co/arcinstitute/Stack-Large) tabular-transformer foundation model as a context-aware cell encoder with a **Gene Ontology (GO) graph neural network** that produces generalizable perturbation embeddings.

Given the expression profiles of unperturbed (control) cells and a set of target genes to perturb, StackMix predicts the genome-wide expression profile after the perturbation, including for genes or gene combinations that never appear in training.

- Encoder: pretrained **Stack-Large** (217M parameters, pretrained on ~150 million human cells)
- Prior knowledge: GO-based gene pathway-similarity graph + graph neural network
- Prediction: genome-wide post-perturbation expression with a negative-binomial decoder
- Strong quantitative accuracy: **39.8% / 45.4% lower MSE than GEARS** on the perturbation-generalization benchmark (top 100 / 5000 DEGs)

## Overview

High-throughput Perturb-seq screens can profile thousands of genetic perturbations, but the combinatorial space of single- and multi-gene perturbations makes exhaustive experimental measurement impossible. Computational models must therefore extrapolate from a limited set of observed perturbations to unseen genes and unseen combinations.

GEARS addresses this by coupling learned perturbation embeddings with a Gene Ontology graph, but it encodes every cell independently and cannot exploit the rich context carried by a cell population. StackMix keeps the GO-prior design that makes zero-shot perturbation possible, while replacing the cell encoder with **Stack**, a tabular transformer that treats a *set of cells* as the input unit and performs attention along both the gene-module axis and the cell axis. Information therefore flows in two directions — between genes within each cell and between cells within the population.

### Key features

- **Tabular attention over cell sets.** Each cell is compressed into 100 trainable gene-module tokens; alternating intra-cellular and inter-cellular attention layers capture both intracellular gene regulation and population-level phenotypic structure.
- **Generalization to unseen genes via GO priors.** A gene–pathway bipartite graph is built from Gene Ontology annotations; pairwise Jaccard similarity defines a sparse gene-similarity graph through which a GNN propagates perturbation information. A gene that was never perturbed can still obtain a meaningful embedding from its functional neighbors.
- **Combinatorial perturbation modeling.** Perturbation embeddings are summed over the perturbed gene set and passed through an MLP, so the model accepts arbitrary perturbation sizes (including multi-gene combinations) without enumerating combinations at training time.
- **Distribution-aware training.** Optimization combines a negative-binomial reconstruction loss, an MMD loss between predicted and true expression/embedding distributions, and a sliced-Wasserstein prior on cell embeddings.

## Model architecture

StackMix consists of three components:

1. **Stack cell encoder.** Raw UMI counts are log-normalized (`ln(1 + counts)`) and projected by a single-layer MLP into `n = 100` gene-module tokens of dimension `d = 16`. Nine stacked tabular-transformer layers then apply, in order: intra-cellular multi-head attention (8 heads, over gene modules within each cell), inter-cellular multi-head attention (8 heads, over flattened cell representations), and a per-token feed-forward network with residual connections and LayerNorm. The Stack-Large weights are initialized from the publicly released pretrained checkpoint.

2. **GO-based perturbation embedding generator.** A gene–GO-term bipartite graph is constructed from Gene Ontology annotations. For every gene pair `(u, v)` the Jaccard index of their annotated pathway sets is computed:

   ```
   J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
   ```

   For each gene the `H_pert` most similar genes are retained as neighbors, producing a graph that is derived purely from prior knowledge and is independent of the training data. Each gene receives a trainable embedding that is refined by a graph neural network over this similarity graph. For a perturbation set `P`, gene embeddings are summed and transformed by an MLP into a single combination perturbation representation.

3. **Combination decoder.** The perturbation representation is fused with the encoded cell tokens; the tabular attention layers then decode the post-perturbation cell state, and a cell-level two-layer MLP outputs negative-binomial mean and dispersion parameters for every gene. The predicted perturbation effect is added to sampled control expression, so the model focuses its capacity on the perturbation-induced change rather than on reconstructing background expression.

The total training objective is

```
L = L_NB + L_MMD + λ · L_SW
```

where `L_NB` is the masked negative-binomial negative log-likelihood, `L_MMD` aligns predicted and true distributions in both expression and embedding spaces, and `L_SW` is a sliced-Wasserstein penalty that pushes cell embeddings toward an isotropic Gaussian prior for improved identifiability and cross-dataset generalization. An EMA teacher model supplies target embeddings during training.

## Benchmark results

We evaluate perturbation generalization on held-out perturbations in HepG2 Perturb-seq data (unseen single-gene perturbations: TOP2A, BIRC5, CDC20, UTP20, TAF3, WARS, ACTR3, AURKB) following the scPerturBench evaluation protocol: for every perturbation, distances between predicted and true cell distributions are computed on the top differentially expressed genes (`numDEG = 100` and `5000`). Raw per-metric summaries are provided in [`perturbation_generalization_summary_pretrain.csv`](perturbation_generalization_summary_pretrain.csv) (StackMix) and [`gears_perturbation_generalization_summary.csv`](gears_perturbation_generalization_summary.csv) (GEARS).

**For every metric below, lower is better.** Values are mean ± standard deviation across perturbations.

| Metric | numDEG | StackMix (ours) | GEARS |
|---|---:|---:|---:|
| **MSE** | 100 | **0.2598 ± 0.3456** | 0.4319 ± 0.4752 |
| **MSE** | 5000 | **0.0549 ± 0.0588** | 0.1006 ± 0.0599 |
| Pearson distance | 100 | **1.0857 ± 0.4908** | 1.1719 ± 0.2803 |
| Pearson distance | 5000 | **0.9228 ± 0.1390** | 0.9759 ± 0.0384 |
| Euclidean distance | 100 | **4.4046 ± 2.7426** | 5.9298 ± 3.0284 |
| Euclidean distance | 5000 | **15.0916 ± 7.2902** | 21.7795 ± 5.6968 |
| MMD | 100 | **25.9824 ± 34.5663** | 43.1871 ± 47.5170 |
| MMD | 5000 | **274.2584 ± 293.9056** | 502.7436 ± 299.6498 |
| Symmetric KL divergence | 100 | **35.5689 ± 20.1736** | 37.2365 ± 21.8306 |
| Symmetric KL divergence | 5000 | **32.3033 ± 17.1649** | 33.3799 ± 20.5005 |
| Wasserstein distance | 100 | **37.7143 ± 37.0589** | 93.1118 ± 55.6978 |
| Wasserstein distance | 5000 | **1188.7213 ± 149.3067** | 3678.4236 ± 190.3443 |

> **Note on Pearson distance.** The scPerturBench/pertpy `pearson_distance` is defined as `1 − Pearson r` between the predicted and true mean perturbation effects (deltas) on the selected DEGs, so a smaller value corresponds to a higher Pearson correlation.

Key takeaways:

- StackMix reduces **MSE by 39.8%** (top 100 DEGs) and **45.4%** (top 5000 DEGs) relative to GEARS — the prediction errors of absolute expression levels are nearly halved.
- StackMix is also better on all distribution-level distances, with especially large gains on Wasserstein distance (59.5% / 67.7% lower) and MMD (39.8% / 45.4% lower).
- Pearson distance is lower for StackMix at both DEG cutoffs (1.086 vs 1.172 and 0.923 vs 0.976), although the implicit correlation values are small in absolute terms for both methods, indicating that predicting the *shape* of the DEG response vector from a few genes remains challenging. The dominant advantage of StackMix is its substantially lower MSE and distribution-level error.

## Pretrained model

The Stack-Large encoder weights and the aligned 15,000-gene list are hosted on Hugging Face:

- Model repository: https://huggingface.co/arcinstitute/Stack-Large
- Files: `bc_large.ckpt` (checkpoint) and `basecount_1000per_15000max.pkl` (gene list)

Download with the Hugging Face Hub CLI:

```bash
pip install huggingface_hub
huggingface-cli download arcinstitute/Stack-Large \
    bc_large.ckpt basecount_1000per_15000max.pkl \
    --local-dir weight/Stack-Large
```

Or from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcinstitute/Stack-Large",
    repo_type="model",
    local_dir="weight/Stack-Large",
)
```

The Stack-Large weights are released by Arc Institute under a **non-commercial** license; see `MODEL_LICENSE.md` and `MODEL_ACCEPTABLE_USE_POLICY.md` in the model repository.

## Installation

Requirements: Python >= 3.9, a CUDA-capable GPU is recommended.

```bash
pip install arc-stack
pip install torch pytorch-lightning torch-geometric
pip install scanpy anndata networkx pandas numpy scipy
```

The GEARS baseline code used for comparison is included under [`gears/`](gears/). All commands below assume the repository root as the working directory.

## Data preparation

### AnnData format

Training and test data are stored as `h5ad` files (scanpy AnnData) with raw UMI counts in `adata.X` and the following annotations:

- `adata.obs['condition']`: perturbation label. Use `ctrl` for non-targeting controls and `GENE+ctrl` for a single-gene perturbation (e.g. `TOP2A+ctrl`); multi-gene perturbations use the `G1+G2+ctrl` convention.
- `adata.obs['cell_type']`: cell-type label.
- `adata.var['gene_name']`: gene symbols. Duplicated gene names should be removed.

The notebooks [`prepare_pretrain_test_data.ipynb`](prepare_pretrain_test_data.ipynb) and [`prepare_train_test_data.ipynb`](prepare_train_test_data.ipynb) show how raw Perturb-seq `h5ad` files are converted into this format and how held-out perturbations are split.

### Required files

Place the downloaded Stack-Large checkpoint and gene list under `weight/Stack-Large/`, and the Gene Ontology annotation files under `data/`:

```
weight/
└── Stack-Large/
    ├── bc_large.ckpt
    └── basecount_1000per_15000max.pkl
data/
├── gene2go_all.pkl                     # gene -> GO term mapping
├── essential_all_data_pert_genes.pkl   # gene universe for the perturbation graph
├── go_essential_all.tar.gz             # archive containing go_essential_all.csv (Jaccard edges)
└── *.h5ad                              # Perturb-seq datasets
```

`go_essential_all.csv` contains the precomputed GO Jaccard edge list (columns `source`, `target`, `importance`); extract it from `go_essential_all.tar.gz` before training.

## Training

### Single-dataset training

Fine-tune StackMix on one Perturb-seq dataset (a train/test split of RPE1 is shown in [`train_mix.py`](train_mix.py)):

```bash
python train_mix.py
```

### Cross-cell-line warm-up followed by HepG2 fine-tuning

[`train_mix_with_pretrain.py`](train_mix_with_pretrain.py) implements a two-stage schedule used for the reported HepG2 results:

1. **Cross-dataset warm-up:** the perturbation embedding module and decoder are adapted on genome-wide Perturb-seq screens from three cell lines (Jurkat, K562, RPE1) using the pretrained Stack-Large encoder.
2. **Target fine-tuning:** the model is re-initialized on the HepG2 training split, the warm-up checkpoint is loaded, and training proceeds with a cosine learning-rate schedule.

```bash
python train_mix_with_pretrain.py
```

Key default settings: 20 control + 20 perturbed cells per cell set (`sample_size`), batch size 64, perturbation embedding dimension 16, one GO GNN layer, AdamW optimizer, gradient clipping at 1.0, and an EMA teacher updated every 500 steps. Checkpoints are saved under `checkpoints/` and TensorBoard logs under `logs/`.

## Inference

[`predicton.py`](predicton.py) predicts held-out perturbation responses. It builds control cell sets with `PredictionDataset`, instantiates `PertMix` with the same configuration used for training, loads the fine-tuned checkpoint, and calls `predict_perturbation_effect`:

```python
from src.stack.models.finetune.model_mix import (
    PertMix,
    predict_perturbation_effect,
)

# control_cells: (batch_size, n_cells, n_genes) tensor of control expression
# pert_id:       list of perturbation gene index lists
predicted_expression = predict_perturbation_effect(
    control_cells,
    pert_id,
    model=model,
)
```

Run the script:

```bash
python predicton.py
```

Predictions are written per perturbation as AnnData objects under `predictions/`. For building a genome-wide in silico perturbation map, see [`prediction_map.py`](prediction_map.py), [`merge_prediction_map.py`](merge_prediction_map.py) and [`draw_pert_map.py`](draw_pert_map.py).

## Evaluation

[`evaluaton.py`](evaluaton.py) aligns genes across predicted and true AnnData objects, normalizes to 10,000 counts per cell with `log1p`, and computes MSE, Pearson/Spearman correlation, E-distance, Wasserstein distance, KL divergence and common-DEG overlap, averaged over perturbation conditions:

```bash
python evaluaton.py
```

The benchmark summaries reported above (`perturbation_generalization_summary_pretrain.csv`, `gears_perturbation_generalization_summary.csv`) follow the scPerturBench protocol with MSE, Pearson distance, Euclidean distance, MMD, symmetric KL divergence and Wasserstein distance at `numDEG = 100 / 5000`.

## Repository structure

```
├── configs/                           # YAML configs for Stack pretraining / fine-tuning
├── data/                              # GO annotation files and Perturb-seq h5ad data
├── gears/                             # GEARS baseline code used for comparison
├── src/stack/
│   ├── data/finetuning/
│   │   └── perturbation_dataset.py    # control/perturbed cell-set datasets & dataloaders
│   ├── models/
│   │   ├── core/                      # Stack tabular-transformer core model & losses
│   │   └── finetune/model_mix.py      # PertMix: GO GNN embeddings + Stack fine-tuning
│   └── finetune/                      # Lightning modules and utilities
├── train_mix.py                       # single-dataset fine-tuning
├── train_mix_with_pretrain.py         # cross-cell-line warm-up + HepG2 fine-tuning
├── predicton.py                       # held-out perturbation prediction
├── evaluaton.py                       # perturbation-generalization metrics
├── prepare_pretrain_test_data.ipynb   # data formatting example
├── prepare_train_test_data.ipynb      # train/test split example
├── perturbation_generalization_summary_pretrain.csv
└── gears_perturbation_generalization_summary.csv
```

## Citation

If you use StackMix in your work, please cite the Stack foundation model:

- Dong et al. (2026). *Stack: In-context modeling of single-cell biology.* bioRxiv. https://www.biorxiv.org/content/10.64898/2026.01.09.698608v1
- Stack source code: https://github.com/ArcInstitute/stack

This work also builds upon:

- **GEARS**: Roohani et al., *Predicting transcriptional outcomes of novel multigene perturbations with GNNs*.
- **Gene Ontology**: The Gene Ontology Consortium.
- **scPerturBench**: the single-cell perturbation-effects prediction benchmark used for the evaluation protocol.

## License

Code in this repository is provided for research use. The pretrained Stack-Large weights downloaded from Hugging Face are governed by Arc Institute's non-commercial model license and acceptable use policy; consult the model repository for details before any commercial use.
