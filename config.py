from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class ExperimentConfig:
    """
    Central configuration object for the training pipeline.
    
    When using Optuna, the 'Hyperparameters' section will be populated 
    dynamically by the trial suggestions.
    """

    # -------------------------------------------------------------------------
    # 1. Method Selection
    # -------------------------------------------------------------------------
    # Options: "standard", "supcon" (and any future methods you add)
    method_name: str = "supcon" 

    # -------------------------------------------------------------------------
    # 2. Hyperparameters (The Search Space)
    # -------------------------------------------------------------------------
    # General Optimizer Params
    lr: float = 0.0001
    weight_decay: float = 0.005
    
    # EMA (Exponential Moving Average) Params
    ema_decay: float = 0.9
    
    # Method-Specific Params: Supervised Contrastive Learning
    # (Only used if method_name == "supcon")
    supcon_lambda: float = 0.50
    supcon_temperature: float = 0.10

    mmd_lambda: float = 1.0

    # Method-Specific Params: CDAN+E
    # (Only used if method_name == "cdan")
    cdan_lambda: float = 1.0
    cdan_entropy: bool = True

    # -------------------------------------------------------------------------
    # 3. Data & Checkpointing
    # -------------------------------------------------------------------------
    # Data Balancing
    balance_train: bool = False
    balance_val: bool = False
    
    # Checkpoint Selection Metric
    # Options: "loss", "auroc"
    select_chkpt_on: str = "loss"

    # -------------------------------------------------------------------------
    # 4. Training Loop Constants
    # -------------------------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64
    num_runs: int = 1  # Set to >1 for statistical significance testing (outside Optuna)
    num_workers: int = 12
    seed: int = 42

    # -------------------------------------------------------------------------
    # 5. System / Paths (Usually set via command line args)
    # -------------------------------------------------------------------------
    # These defaults can be overwritten by argparse in the main script
    data_dir: Path = Path('/data')
    csv_dir: Path = Path('.')
    out_dir: Path = Path('~/cxp_shortcut_out')
    
    @property
    def device(self):
        """Helper to automatically determine device (CUDA -> MPS -> CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps") # Uses Mac GPU!
        else:
            return torch.device("cpu")