import torch

from stack.model_loading import load_model_from_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_checkpoint("weight/Stack-Large/bc_large.ckpt", model_class="scShiftAttentionModel",
                                   device=device)


