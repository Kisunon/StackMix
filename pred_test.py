from src.stack.cli.generation import generate, save_generations
from pathlib import Path

checkpoint_path = "weight/Stack-Large-Aligned/bc_large_aligned.ckpt"
base_adata_path = "data/openproblems_context.h5ad"
test_adata_path = "data/openproblems_test.h5ad"
genelist_path = "weight/Stack-Large-Aligned/basecount_1000per_15000max.pkl"
output_dir = "outputs/test"

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

generations = generate(
    checkpoint_path=checkpoint_path,
    base_adata_path=base_adata_path,
    test_adata_path=test_adata_path,
    genelist_path=genelist_path,
    split_column='sm_name',
    split_values=None,
    gene_name_col=None,
    prompt_ratio=0.25,
    context_ratio=0.4,
    context_ratio_min=0.2,
    mask_rate=1.0,
    num_steps=5,
    mode="vanilla",
    batch_size=16,
    num_workers=4,
    random_seed=42,
    device="cuda",
    show_progress=True,
)
save_generations(
    generations,
    output_dir,
    concatenate=True,
)