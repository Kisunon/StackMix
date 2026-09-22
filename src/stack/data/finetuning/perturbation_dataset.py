import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List, Union
import anndata
import logging
import pickle
import os
import random

log = logging.getLogger(__name__)


def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))


class PerturbationDataset(Dataset):
    """
    Dataset for ICL_FinetunedModelMix training, inspired by gears' PertData.
    
    This dataset handles perturbation data stored in AnnData format, where:
    1. adata.obs has 'condition' and 'cell_type' columns
    2. adata.var has 'gene_name' column
    3. adata.X stores post-perturbed gene expression
    
    Each batch contains batch_size groups of cells, where each group contains:
    -若干对照细胞和同样数量的扰动细胞
    -对应的扰动条件
    """
    
    def __init__(
        self,
        adata: anndata.AnnData,
        batch_size: int,
        num_control_cells: int,
        num_perturb_cells: int,
        random_state: Optional[int] = 42,
        mode: str = 'train'
    ):
        """
        Initialize the dataset.
        
        Args:
            adata: AnnData object containing the perturbation data
            batch_size: Number of groups per batch
            num_control_cells: Number of control cells per group
            num_perturb_cells: Number of perturbation cells per group
            random_state: Random seed
            mode: 'train', 'val', or 'test'
        """
        self.adata = adata
        self.batch_size = batch_size
        self.num_control_cells = num_control_cells
        self.num_perturb_cells = num_perturb_cells
        self.mode = mode
        self.default_pert_graph = True
        self.rng = np.random.RandomState(random_state)
        self.gene_name_col = 'gene_name'
        
        # Process conditions
        self.conditions = adata.obs['condition'].values
        self.cell_types = adata.obs['cell_type'].values

        self.data_path = "./data"

        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)

        self.set_pert_genes()

        # Identify control cells and perturbation cells
        self.control_mask = self._is_control_condition(self.conditions)
        self.perturb_mask = ~self.control_mask

        # Get unique perturbations
        self.perturbations = self._get_unique_perturbations(self.conditions)
        # self.perturb_to_idx = {pert: i for i, pert in enumerate(self.perturbations)}
        self.num_perturbations = len(self.perturbations)
        
        # Get gene names
        self.gene_names = adata.var['gene_name'].values if 'gene_name' in adata.var.columns else adata.var_names.values
        self.num_genes = len(self.gene_names)

        # Create gene mapping
        self.get_data_gene_map()
        
        # Precompute indices for each perturbation
        self.perturb_indices = {}
        for pert in self.perturbations:
            if pert != 'ctrl':
                self.perturb_indices[pert] = np.where(
                    (self.conditions == pert) & self.perturb_mask
                )[0]
        
        # Precompute control indices
        self.control_indices = np.where(self.control_mask)[0]
        
        # Generate samples
        self.samples = self._generate_samples()
        
        log.info(f"PerturbationDataset initialized with {len(self.samples)} samples")
        log.info(f"Found {len(self.control_indices)} control cells")
        log.info(f"Found {self.num_perturbations} unique perturbations")

    def get_data_gene_map(self):
        """
        Get the mapping from gene name to gene index.
        """
        genelist_path = "./weight/Stack-Large/basecount_1000per_15000max.pkl"
        with open(genelist_path, 'rb') as f:
            self.data_target_genes = pickle.load(f)
        self.n_genes_data_target = len(self.data_target_genes)

        if self.gene_name_col is not None:
            var_data = self.adata.var
            if self.gene_name_col in var_data.columns:
                gene_names = var_data[self.gene_name_col].values
                log.info(f"Using '{self.gene_name_col}' column for gene names")
            else:
                 raise ValueError(f"'{self.gene_name_col}' not found in var.")
        else:
            gene_names = self.adata.var_names.values

        # 转换为大写并处理字符串类型
        if gene_names.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            gene_names = np.char.upper(gene_names.astype(str))
        else:
            gene_names = np.array([str(g).upper() for g in gene_names])

        # 创建基因映射
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        self.data_gene_mapping = {}
        found_genes = 0
        for target_idx, gene in enumerate(self.data_target_genes):
            if gene in gene_to_idx:
                self.data_gene_mapping[target_idx] = gene_to_idx[gene]
                found_genes += 1

        if found_genes == 0:
            raise ValueError("No target genes found in the AnnData object")

        log.info(f"Found {found_genes}/{self.n_genes_data_target} target genes")

    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in
        perturbation graph
        """
        if self.default_pert_graph is False:
            # Use a smaller perturbation graph
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes

        else:
            import pickle
            import os

            # Otherwise, use a large set of genes to create perturbation graph
            path_ = os.path.join(self.data_path,
                                 'essential_all_data_pert_genes.pkl')
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)

        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
    
    def _is_control_condition(self, condition):
        """Check if a condition is a control condition."""
        return np.array([cond == 'ctrl' for cond in condition])
    
    def _get_unique_perturbations(self, conditions):
        """Get unique perturbations from conditions."""
        unique_conds = np.unique(conditions)
        # Include 'ctrl' as a perturbation
        perturbations = ['ctrl']
        for cond in unique_conds:
            if cond != 'ctrl':
                perturbations.append(cond)
        return perturbations
    
    def _generate_samples(self):
        """Generate samples for the dataset."""
        samples = []
        
        # For each perturbation, create samples
        for pert in self.perturbations:
            if pert == 'ctrl':
                continue

            pert_single = get_genes_from_perts(pert)

            pert_idx = [self.node_map_pert[g] for g in pert_single if g in self.node_map_pert]
            # print(pert, pert_single, pert_idx)

            # Skip if no perturbation genes found in node_map_pert
            if not pert_idx:
                continue

            # Get indices for this perturbation
            pert_indices = self.perturb_indices.get(pert, [])
            # if len(pert_indices) < self.num_perturb_cells:
            #     continue
            
            # Get control indices
            if len(self.control_indices) < self.num_control_cells:
                assert False, "Not enough control cells to form complete groups"
            
            # Create samples
            # n_samples = len(pert_indices) // self.num_perturb_cells
            # for i in range(n_samples):
            #     # Sample control cells
            #     control_idx = self.rng.choice(
            #         self.control_indices,
            #         size=self.num_control_cells,
            #         replace=False
            #     )
            #
            #     # Sample perturbation cells
            #     start = i * self.num_perturb_cells
            #     end = start + self.num_perturb_cells
            #     perturb_idx = pert_indices[start:end]
            #
            #     # Create sample
            #     sample = {
            #         'control_indices': control_idx,
            #         'perturb_indices': perturb_idx,
            #         'perturbation': pert,
            #         'perturbation_idx': self.perturb_to_idx[pert]
            #     }
            #     samples.append(sample)
            # If we have perturbation indices, create samples
            if len(pert_indices) > 0:
                if len(pert_indices) >= self.num_perturb_cells:
                    # If enough perturbation cells, create multiple samples without replacement
                    n_samples = len(pert_indices) // self.num_perturb_cells
                    for i in range(n_samples):
                        # Sample control cells
                        control_idx = self.rng.choice(
                            self.control_indices,
                            size=self.num_control_cells,
                            replace=False
                        )

                        # Sample perturbation cells without replacement
                        start = i * self.num_perturb_cells
                        end = start + self.num_perturb_cells
                        perturb_idx = pert_indices[start:end]

                        # Create sample
                        sample = {
                            'control_indices': control_idx,
                            'perturb_indices': perturb_idx,
                            'perturbation': pert_single,
                            'perturbation_idx': pert_idx
                        }
                        samples.append(sample)
                else:
                    # If not enough perturbation cells, create at least one sample with replacement
                    # Sample control cells
                    control_idx = self.rng.choice(
                        self.control_indices,
                        size=self.num_control_cells,
                        replace=False
                    )

                    # Sample perturbation cells with replacement
                    perturb_idx = self.rng.choice(
                        pert_indices,
                        size=self.num_perturb_cells,
                        replace=True
                    )

                    # Create sample
                    sample = {
                        'control_indices': control_idx,
                        'perturb_indices': perturb_idx,
                        'perturbation': pert_single,
                        'perturbation_idx': pert_idx
                    }
                    samples.append(sample)
        return samples

    def load_expression_data_from_adata(self, local_indices: np.ndarray) -> np.ndarray:
        """
        Load expression data from AnnData object for specified cell indices.
        """
        # Sort indices for consistency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]

        try:
            adata = self.adata

            # Get absolute row indices in the original AnnData object
            absolute_indices_to_load = sorted_local_indices

            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes_data_target), dtype=np.float32)
            gene_mapping = self.data_gene_mapping
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))

            # Get expression data
            expr_data = adata.X[absolute_indices_to_load, :]

            # Convert sparse to dense if needed
            if hasattr(expr_data, 'toarray'):
                expr_data = expr_data.toarray()

            # Select target gene columns
            expr_subset = expr_data[:, source_indices_in_file]
            mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)

        except Exception as e:
            log.exception(f"Error loading expression data from AnnData object: {e}")
            raise

        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))

        return mapped_matrix[reverse_sort_order]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            observed_features: Tensor of shape (batch_size, n_cells, n_genes)
            ground_truth_features: Tensor of shape (batch_size, n_cells, n_genes)
            pert_idx: Tensor of shape (batch_size, num_perturbations)
        """
        sample = self.samples[idx]
        
        # Get control and perturbation indices
        control_idx = sample['control_indices']
        perturb_idx = sample['perturb_indices']
        
        # Get expression data
        control_expr = self.load_expression_data_from_adata(control_idx)
        perturb_expr = self.load_expression_data_from_adata(perturb_idx)

        control_expr = control_expr.toarray() if hasattr(control_expr, 'toarray') else control_expr
        perturb_expr = perturb_expr.toarray() if hasattr(perturb_expr, 'toarray') else perturb_expr
        
        # Combine control and perturbation cells
        # all_expr = np.concatenate([control_expr, perturb_expr], axis=0)
        
        # Create observed and ground truth features (for this implementation, they are the same)
        observed_features = torch.from_numpy(control_expr).float()
        ground_truth_features = torch.from_numpy(perturb_expr).float()
        
        # Create perturbation index tensor
        pert_idx = torch.tensor(sample['perturbation_idx'], dtype=torch.long)
        
        return observed_features, ground_truth_features, pert_idx


class PredictionDataset(Dataset):
    """
    Dataset for prediction using ICL_FinetunedModelMix.

    This dataset handles perturbation data stored in AnnData format, where:
    1. adata.obs has 'condition' and 'cell_type' columns
    2. adata.var has 'gene_name' column
    3. adata.X stores gene expression

    For each control cell and each perturbation condition, create a sample
    for prediction.
    """

    def __init__(
            self,
            adata: anndata.AnnData,
            num_control_cells: int,
            random_state: Optional[int] = 42
    ):
        """
        Initialize the dataset.

        Args:
            adata: AnnData object containing the perturbation data
            random_state: Random seed
        """
        self.adata = adata
        self.num_control_cells = num_control_cells
        self.rng = np.random.RandomState(random_state)
        self.gene_name_col = 'gene_name'

        # Process conditions
        self.conditions = adata.obs['condition'].values
        self.cell_types = adata.obs['cell_type'].values
        self.data_path = "./data"
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
        self.set_pert_genes()
        # Identify control cells and perturbation cells
        self.control_mask = self._is_control_condition(self.conditions)
        self.perturb_mask = ~self.control_mask
        # Get unique perturbations (excluding 'ctrl')
        self.perturbations = self._get_unique_perturbations(self.conditions)
        self.perturbations = [pert for pert in self.perturbations if pert != 'ctrl']
        self.num_perturbations = len(self.perturbations)

        # Get gene names
        self.gene_names = adata.var['gene_name'].values if 'gene_name' in adata.var.columns else adata.var_names.values
        self.num_genes = len(self.gene_names)
        # Create gene mapping
        self.get_data_gene_map()

        # Precompute control indices
        self.control_indices = np.where(self.control_mask)[0]

        # Generate samples: for each control cell and each perturbation
        self.samples = self._generate_samples()

        log.info(f"PredictionDataset initialized with {len(self.samples)} samples")
        log.info(f"Found {len(self.control_indices)} control cells")
        log.info(f"Found {self.num_perturbations} unique perturbations")

    def get_data_gene_map(self):
        """
        Get the mapping from gene name to gene index.
        """
        genelist_path = "./weight/Stack-Large/basecount_1000per_15000max.pkl"
        with open(genelist_path, 'rb') as f:
            self.data_target_genes = pickle.load(f)
        self.n_genes_data_target = len(self.data_target_genes)
        if self.gene_name_col is not None:
            var_data = self.adata.var
            if self.gene_name_col in var_data.columns:
                gene_names = var_data[self.gene_name_col].values
                log.info(f"Using '{self.gene_name_col}' column for gene names")
            else:
                raise ValueError(f"'{self.gene_name_col}' not found in var.")
        else:
            gene_names = self.adata.var_names.values
        # 转换为大写并处理字符串类型
        if gene_names.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            gene_names = np.char.upper(gene_names.astype(str))
        else:
            gene_names = np.array([str(g).upper() for g in gene_names])
        # 创建基因映射
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        self.data_gene_mapping = {}
        found_genes = 0
        for target_idx, gene in enumerate(self.data_target_genes):
            if gene in gene_to_idx:
                self.data_gene_mapping[target_idx] = gene_to_idx[gene]
                found_genes += 1
        if found_genes == 0:
            raise ValueError("No target genes found in the AnnData object")
        log.info(f"Found {found_genes}/{self.n_genes_data_target} target genes")

    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in
        perturbation graph
        """
        import pickle
        import os
        # Use a large set of genes to create perturbation graph
        path_ = os.path.join(self.data_path,
                             'essential_all_data_pert_genes.pkl')
        with open(path_, 'rb') as f:
            essential_genes = pickle.load(f)
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}
        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

    def _is_control_condition(self, condition):
        """Check if a condition is a control condition."""
        return np.array([cond == 'ctrl' for cond in condition])

    def _get_unique_perturbations(self, conditions):
        """Get unique perturbations from conditions."""
        unique_conds = np.unique(conditions)
        # Include 'ctrl' as a perturbation
        perturbations = ['ctrl']
        for cond in unique_conds:
            if cond != 'ctrl':
                perturbations.append(cond)
        return perturbations

    def _generate_samples(self):
        """
        Generate samples for the dataset.
        For each control cell and each perturbation condition, create a sample.
        """
        samples = []

        # Shuffle control indices for random grouping
        shuffled_control_indices = self.rng.permutation(self.control_indices)

        # Calculate number of groups
        n_groups = len(shuffled_control_indices) // self.num_control_cells

        # Create control cell groups
        control_groups = []
        for i in range(n_groups):
            start = i * self.num_control_cells
            end = start + self.num_control_cells
            control_groups.append(shuffled_control_indices[start:end])

        # For each control group
        for group_idx, control_group in enumerate(control_groups):
            # For each perturbation
            for pert in self.perturbations:
                pert_single = get_genes_from_perts(pert)

                # Get perturbation indices for this perturbation
                pert_idx = [self.node_map_pert[g] for g in pert_single if g in self.node_map_pert]

                # Skip if no perturbation genes found in node_map_pert
                if not pert_idx:
                    continue

                # Create sample
                sample = {
                    'control_indices': control_group,
                    'perturbation': pert,
                    'perturbation_genes': pert_single,
                    'perturbation_idx': pert_idx
                }
                samples.append(sample)
        return samples

    def load_expression_data_from_adata(self, local_indices: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Load expression data from AnnData object for specified cell indices.
        """
        # Sort indices for consistency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]
        try:
            adata = self.adata
            # Get absolute row indices in the original AnnData object
            absolute_indices_to_load = sorted_local_indices
            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes_data_target), dtype=np.float32)
            gene_mapping = self.data_gene_mapping
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))
            # Get expression data
            expr_data = adata.X[absolute_indices_to_load, :]
            cell_ids = adata.obs_names[absolute_indices_to_load]
            # Convert sparse to dense if needed
            if hasattr(expr_data, 'toarray'):
                expr_data = expr_data.toarray()
            # Select target gene columns
            expr_subset = expr_data[:, source_indices_in_file]
            mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)
        except Exception as e:
            log.exception(f"Error loading expression data from AnnData object: {e}")
            raise
        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))
        return mapped_matrix[reverse_sort_order], list(cell_ids[reverse_sort_order])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List]:
        """
        Get a sample from the dataset.

        Returns:
            control_features: Tensor of shape (1, n_genes) - control cell expression
            pert_idx: Tensor of shape (num_perturb_genes,) - perturbation gene indices
        """
        sample = self.samples[idx]

        # Get control index
        control_idx = sample['control_indices']
        # Get expression data for control cell
        control_expr, cell_ids = self.load_expression_data_from_adata(control_idx)
        control_expr = control_expr.toarray() if hasattr(control_expr, 'toarray') else control_expr

        # Create control features tensor
        control_features = torch.from_numpy(control_expr).float()

        # Create perturbation index tensor
        pert_idx = torch.tensor(sample['perturbation_idx'], dtype=torch.long)
        return control_features, pert_idx, sample['perturbation'], cell_ids

    def create_aligned_adata(self) -> anndata.AnnData:
        """
        Create a new AnnData object with genes aligned to data_target_genes.
        Only includes perturbed cells.
        """
        # Get perturbed cell indices
        perturb_indices = np.where(self.perturb_mask)[0]

        # Load expression data for perturbed cells
        perturb_expr, cell_ids = self.load_expression_data_from_adata(perturb_indices)

        # Create new AnnData object
        aligned_adata = anndata.AnnData(
            X=perturb_expr,
            obs=self.adata.obs.iloc[perturb_indices].copy(),
            var=pd.DataFrame(index=self.data_target_genes)
        )

        # Add gene_name column to var
        aligned_adata.var['gene_name'] = self.data_target_genes

        log.info(f"Created aligned AnnData with {len(perturb_indices)} cells and {len(self.data_target_genes)} genes")

        return aligned_adata

class PerturbationMappingDataset(Dataset):
    """
    Dataset for building perturbation mapping to identify key gene perturbations.
    This dataset handles perturbation data stored in AnnData format, where:
    1. adata.obs has 'condition' and 'cell_type' columns
    2. adata.var has 'gene_name' column
    3. adata.X stores gene expression
    For each perturbation gene (after filtering) and each control cell group,
    create a sample for prediction.
    """

    def __init__(
            self,
            adata: anndata.AnnData,
            perturbation_genes: List[str],
            num_control_cells: int = 20,
            max_control_cells_per_perturbation: int = 500,
            random_state: Optional[int] = 42
    ):
        """
        Initialize the dataset.

        Args:
            adata: AnnData object containing the perturbation data
            perturbation_genes: List of genes to perturb
            num_control_cells: Number of control cells per sample
            max_control_cells_per_perturbation: Maximum number of control cells per perturbation
            random_state: Random seed
        """
        self.adata = adata
        self.perturbation_genes = perturbation_genes
        self.num_control_cells = num_control_cells
        self.max_control_cells_per_perturbation = max_control_cells_per_perturbation
        self.num_sample_per_perturbation =  max_control_cells_per_perturbation // num_control_cells
        self.rng = np.random.RandomState(random_state)
        self.gene_name_col = 'gene_name'

        # Process conditions
        self.conditions = adata.obs['condition'].values
        self.cell_types = adata.obs['cell_type'].values
        self.data_path = "./data"
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
        self.set_pert_genes()
        # Identify control cells
        self.control_mask = self._is_control_condition(self.conditions)

        # Filter perturbation genes to only those present in the perturbation graph
        self.filtered_perturbation_genes = [gene for gene in self.perturbation_genes if gene in self.node_map_pert]

        # test = [gene for gene in self.perturbation_genes if gene not in self.node_map_pert]
        # print(test)
        # print(len(test))
        # print(len(self.node_map_pert.keys()))

        log.info(f"Filtered perturbation genes: {len(self.filtered_perturbation_genes)}/{len(self.perturbation_genes)} genes retained")

        # Get gene names
        self.gene_names = adata.var['gene_name'].values if 'gene_name' in adata.var.columns else adata.var_names.values
        self.num_genes = len(self.gene_names)
        # Create gene mapping
        self.get_data_gene_map()

        # Precompute control indices
        self.control_indices = np.where(self.control_mask)[0]

        # Determine the actual number of control cells to use
        available_control_cells = len(self.control_indices)
        if available_control_cells < self.num_control_cells:
            log.warning(
                f"Not enough control cells: {available_control_cells} available, {self.num_control_cells} required per sample")
            self.num_control_cells = available_control_cells

        # Ensure we don't exceed the maximum control cells per perturbation
        self.actual_control_cells_per_perturbation = min(self.num_control_cells,
                                                         self.max_control_cells_per_perturbation)
        if self.actual_control_cells_per_perturbation < self.num_control_cells:
            log.info(
                f"Using {self.actual_control_cells_per_perturbation} control cells per perturbation (capped at maximum)")

        # Generate samples: for each perturbation gene and each control cell group
        self.samples = self._generate_samples()

        log.info(f"PerturbationMappingDataset initialized with {len(self.samples)} samples")
        log.info(f"Found {len(self.control_indices)} control cells")
        log.info(f"Using {len(self.filtered_perturbation_genes)} filtered perturbation genes")
        log.info(f"Using {self.actual_control_cells_per_perturbation} control cells per perturbation")

    def get_data_gene_map(self):
        """
        Get the mapping from gene name to gene index.
        """
        genelist_path = "./weight/Stack-Large/basecount_1000per_15000max.pkl"
        with open(genelist_path, 'rb') as f:
            self.data_target_genes = pickle.load(f)
        self.n_genes_data_target = len(self.data_target_genes)
        if self.gene_name_col is not None:
            var_data = self.adata.var
            if self.gene_name_col in var_data.columns:
                gene_names = var_data[self.gene_name_col].values
                log.info(f"Using '{self.gene_name_col}' column for gene names")
            else:
                raise ValueError(f"'{self.gene_name_col}' not found in var.")
        else:
            gene_names = self.adata.var_names.values
        # 转换为大写并处理字符串类型
        if gene_names.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            gene_names = np.char.upper(gene_names.astype(str))
        else:
            gene_names = np.array([str(g).upper() for g in gene_names])
        # 创建基因映射
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        self.data_gene_mapping = {}
        found_genes = 0
        for target_idx, gene in enumerate(self.data_target_genes):
            if gene in gene_to_idx:
                self.data_gene_mapping[target_idx] = gene_to_idx[gene]
                found_genes += 1
        if found_genes == 0:
            raise ValueError("No target genes found in the AnnData object")
        log.info(f"Found {found_genes}/{self.n_genes_data_target} target genes")

    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in
        perturbation graph
        """
        import pickle
        import os
        # Use a large set of genes to create perturbation graph
        path_ = os.path.join(self.data_path,
                             'essential_all_data_pert_genes.pkl')
        with open(path_, 'rb') as f:
            essential_genes = pickle.load(f)
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}
        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

    def _is_control_condition(self, condition):
        """Check if a condition is a control condition."""
        return np.array([cond == 'ctrl' for cond in condition])

    def _generate_samples(self):
        """
        Generate samples for the dataset.
        For each perturbation gene and each control cell group, create a sample.
        """
        samples = []

        # Shuffle control indices for random grouping
        shuffled_control_indices = self.rng.permutation(self.control_indices)

        # Calculate number of groups
        n_groups = len(shuffled_control_indices) // self.num_control_cells

        # Create control cell groups
        control_groups = []
        for i in range(n_groups):
            start = i * self.num_control_cells
            end = start + self.num_control_cells
            control_groups.append(shuffled_control_indices[start:end])

        for gene in self.filtered_perturbation_genes:

            # Get perturbation index for this gene
            pert_idx = [self.node_map_pert[gene]]

            for i in range(self.num_sample_per_perturbation):
                # Select a random control group
                control_group = random.choice(control_groups)

                # Create sample
                sample = {
                    'control_indices': control_group,
                    'perturbation_gene': gene,
                    'perturbation_idx': pert_idx
                }
                samples.append(sample)

        return samples

    def load_expression_data_from_adata(self, local_indices: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Load expression data from AnnData object for specified cell indices.
        """
        # Sort indices for consistency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]
        try:
            adata = self.adata
            # Get absolute row indices in the original AnnData object
            absolute_indices_to_load = sorted_local_indices
            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes_data_target), dtype=np.float32)
            gene_mapping = self.data_gene_mapping
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))
            # Get expression data
            expr_data = adata.X[absolute_indices_to_load, :]
            cell_ids = adata.obs_names[absolute_indices_to_load]
            # Convert sparse to dense if needed
            if hasattr(expr_data, 'toarray'):
                expr_data = expr_data.toarray()
            # Select target gene columns
            expr_subset = expr_data[:, source_indices_in_file]
            mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)
        except Exception as e:
            log.exception(f"Error loading expression data from AnnData object: {e}")
            raise
        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))
        return mapped_matrix[reverse_sort_order], list(cell_ids[reverse_sort_order])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List]:
        """
        Get a sample from the dataset.
        Returns:
            control_features: Tensor of shape (num_control_cells, n_genes) - control cell expressions
            pert_idx: Tensor of shape (1,) - perturbation gene index
            perturbation_gene: str - perturbation gene name
            cell_ids: List - control cell IDs
        """
        sample = self.samples[idx]
        # Get control index
        control_idx = sample['control_indices']
        # Get expression data for control cells
        control_expr, cell_ids = self.load_expression_data_from_adata(control_idx)
        control_expr = control_expr.toarray() if hasattr(control_expr, 'toarray') else control_expr
        # Create control features tensor
        control_features = torch.from_numpy(control_expr).float()
        # Create perturbation index tensor
        pert_idx = torch.tensor(sample['perturbation_idx'], dtype=torch.long)
        return control_features, pert_idx, sample['perturbation_gene'], cell_ids


def create_perturbation_dataloader(
    adata: anndata.AnnData,
    batch_size: int,
    num_control_cells: int,
    num_perturb_cells: int,
    random_state: Optional[int] = 42,
    mode: str = 'train',
    num_workers: int = 0
):
    """
    Create a DataLoader for the PerturbationDataset.
    
    Args:
        adata: AnnData object containing the perturbation data
        batch_size: Number of groups per batch
        num_control_cells: Number of control cells per group
        num_perturb_cells: Number of perturbation cells per group
        random_state: Random seed
        mode: 'train', 'val', or 'test'
        num_workers: Number of workers for DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = PerturbationDataset(
        adata=adata,
        batch_size=batch_size,
        num_control_cells=num_control_cells,
        num_perturb_cells=num_perturb_cells,
        random_state=random_state,
        mode=mode
    )

    if mode == 'train':
        num_genes = dataset.num_genes
        pert_list = dataset.pert_names
        node_map_pert = dataset.node_map_pert
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        ), num_genes, pert_list, node_map_pert
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )


def create_prediction_dataloader(
        adata: anndata.AnnData,
        batch_size: int,
        num_control_cells: int,
        random_state: Optional[int] = 42,
        num_workers: int = 0
):
    """
    Create a DataLoader for the PredictionDataset.

    Args:
        adata: AnnData object containing the perturbation data
        batch_size: Batch size for DataLoader
        random_state: Random seed
        num_workers: Number of workers for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = PredictionDataset(
        adata=adata,
        random_state=random_state,
        num_control_cells=num_control_cells,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ), dataset


def create_perturbation_mapping_dataloader(
        adata: anndata.AnnData,
        perturbation_genes: List[str],
        batch_size: int,
        num_control_cells: int = 20,
        max_control_cells_per_perturbation: int = 500,
        random_state: Optional[int] = 42,
        num_workers: int = 0
):
    """
    Create a DataLoader for the PerturbationMappingDataset.
    Args:
        adata: AnnData object containing the perturbation data
        perturbation_genes: List of genes to perturb
        batch_size: Batch size for DataLoader
        num_control_cells: Number of control cells per sample
        random_state: Random seed
        num_workers: Number of workers for DataLoader
    Returns:
        DataLoader instance and dataset
    """
    dataset = PerturbationMappingDataset(
        adata=adata,
        perturbation_genes=perturbation_genes,
        num_control_cells=num_control_cells,
        max_control_cells_per_perturbation=max_control_cells_per_perturbation,
        random_state=random_state
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ), dataset