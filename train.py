import os
import sys
import logging
import argparse
from pathlib import Path
import copy
import json
import coolname
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torcheval.metrics import BinaryAUROC
import wandb
import optuna

# Local imports from your refactored structure
from config import ExperimentConfig
from dataset import CXP_dataset
from model import CXP_Model
import methods
from utils import run_training_phase, get_dataloaders, run_final_eval, run_testing_phase

# Global args placeholder to be populated in main
GLOBAL_ARGS = None

def setup_logging(root_dir):
    log_path = root_dir / "optuna_training.log"
    
    # 1. Use force=True to override any pre-existing config from other libs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True
    )

    # 2. Explicitly tell Optuna to use this file handler
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.addHandler(logging.FileHandler(log_path)) 

    # Capture uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler

def objective(trial):
    # ====== CONFIG ========================================================================

    # Populate Config from Optuna Trial
    config = ExperimentConfig()
    
    # Static paths from Global Args
    config.data_dir = GLOBAL_ARGS.data_dir
    config.csv_dir = GLOBAL_ARGS.csv_dir
    config.out_dir = GLOBAL_ARGS.out_dir
    config.balance_train = GLOBAL_ARGS.balance_train
    config.balance_val = GLOBAL_ARGS.balance_val
    config.select_chkpt_on = GLOBAL_ARGS.select_chkpt_on
    
    # Select method based on flag
    config.method_name = GLOBAL_ARGS.method

    # --- HYPERPARAMETER & METHOD SELECTION ---
    if GLOBAL_ARGS.debug:
        # Debug mode: Force fast settings
        config.epochs = 2
        config.batch_size = 4 
        config.num_workers = 0 
        
        # Fixed params for debug
        config.lr = trial.suggest_categorical("lr", [0.001])
        config.weight_decay = trial.suggest_categorical("weight_decay", [0.001])
        config.ema_decay = trial.suggest_categorical("ema_decay", [0.99])

        if config.method_name == "supcon":
            config.supcon_lambda = 0.5
            config.supcon_temperature = 0.1
            
    else:
        # Optimizable Hyperparameters
        config.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        config.ema_decay = trial.suggest_float("ema_decay", 0.9, 0.999)
        
        # Only suggest SupCon params if we are actually using SupCon
        if config.method_name == "supcon":
            config.supcon_lambda = trial.suggest_float("supcon_lambda", 0.1, 1.0)
            config.supcon_temperature = trial.suggest_float("supcon_temperature", 0.05, 0.5)

    # ====== WANDB ========================================================================

    # WandB Setup
    run_name = f"{GLOBAL_ARGS.study_name}_trial_{trial.number}"
    run = wandb.init(
        project="cxr_optuna_study", 
        group=GLOBAL_ARGS.study_name, 
        name=run_name,
        config=config.__dict__,
        reinit=True
    )

    # ====== SETUP TRAINING ================================================================

    # Setup Components
    device = config.device
    
    # Get Data
    train_loader, val_loader, test_loader_aligned, test_loader_misaligned, n_train, n_val = get_dataloaders(config, debug=GLOBAL_ARGS.debug)
    
    # Get Method Strategy (Standard or SupCon)
    method = methods.get_method(config.method_name, config)
    
    # Init Model
    model = CXP_Model(method).to(device)
    
    # Init EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
    
    # Init Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Metrics for Test Phase
    test_auroc_aligned = BinaryAUROC()
    test_auroc_misaligned = BinaryAUROC()

    # Filename for this specific trial's checkpoint
    study_root = GLOBAL_ARGS.study_root
    trial_str = str(trial.number).zfill(3)
    chkpt_path = study_root / "checkpoints" / f'trial_{trial_str}_best.chkpt'

    # ====== TRAINING =====================================================================

    best_metric = run_training_phase(
        config=config,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        method=method,
        train_loader=train_loader,
        val_loader=val_loader,
        trial=trial,
        n_train=n_train,
        n_val=n_val,
        chkpt_path=chkpt_path
    )

    # ====== TESTING =====================================================================

    predictions_dir = study_root / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    run_testing_phase(
        config=config,
        ema_model=ema_model,
        method=method,
        device=device,
        test_loader_aligned=test_loader_aligned,
        test_loader_misaligned=test_loader_misaligned,
        chkpt_path=chkpt_path,
        output_dir=predictions_dir,
        run=run,
        prefix=f"trial_{trial_str}"
    )

    run.finish()

    # Cleanup: Remove checkpoint to save space
    if chkpt_path.exists():
        chkpt_path.unlink()

    # Return the metric optimized
    return best_metric

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path Arguments
    parser.add_argument('--data_dir', type=Path, default=Path('data'), 
                       help='Directory above /CheXpert-v1.0-small')
    parser.add_argument('--csv_dir', type=Path, default=Path('csv_data'),
                       help='Directory containing CSV files')
    parser.add_argument('--out_dir', type=Path, default=Path('output'),
                       help='Output directory for logs and checkpoints')
    
    # Study Arguments
    parser.add_argument('--study_name', type=str, default=None,
                   help='Name of the study. If not provided, generates a random name.')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of Optuna trials to run')
    parser.add_argument('--n_eval_runs', type=int, default=0,
                        help='Number of final evaluation runs using best params (default: 0)')
    parser.add_argument('--balance_train', type=lambda x: x.lower() == 'true', default=False,
                       help='Use weighted sampler for training')
    parser.add_argument('--balance_val', type=lambda x: x.lower() == 'true', default=False,
                       help='Use balanced validation set')
    parser.add_argument('--select_chkpt_on', type=str, default="loss", choices=["loss", "auroc"],
                       help='Metric to select best model')
    parser.add_argument('--debug', action='store_true', 
                       help='Run in debug mode (tiny data, 1 epoch, CPU/MPS friendly)')
    
    # Methods
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'supcon'],
                       help='Method to use for training (default: standard)')

    args = parser.parse_args()

    if args.study_name is None:
        args.study_name = coolname.generate_slug(2)
    
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # CREATE STUDY FOLDER
    folder_name = f"{args.study_name}"
    study_root = args.out_dir / folder_name
    study_root.mkdir(parents=True, exist_ok=True)

    GLOBAL_ARGS = args
    GLOBAL_ARGS.study_root = study_root

    # CREATE SUBDIRECTORIES
    (study_root / "checkpoints").mkdir(exist_ok=True)
    (study_root / "predictions").mkdir(exist_ok=True)

    # SETUP LOGGING
    setup_logging(study_root)

    config_dict = vars(args)
    config_dict["start_time"] = start_time_str
    
    with open(study_root / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=4, default=str)
    
    logging.info(f"Starting Study: {args.study_name}")
    logging.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Define Optimization Direction
    direction = "maximize" if args.select_chkpt_on.upper() == 'AUROC' else "minimize"
    
    # Hyperband Pruner to stop bad trials early
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=30, reduction_factor=3)
    
    # Create Study with SQLite storage for persistence
    #storage_url = f"sqlite:///{study_root}/optuna_study.db"
    study = optuna.create_study(
        direction=direction, 
        study_name=args.study_name,
        pruner=pruner,
        #storage=storage_url,
        load_if_exists=True
    )
    
    # ========================== RUN OPTIMIZATION ===============================================

    study.optimize(objective, n_trials=args.n_trials)

    # ===========================================================================================
    
    logging.info("===== STUDY COMPLETED =====")
    logging.info(f"Best Trial Number: {study.best_trial.number}")
    logging.info(f"Best Value ({args.select_chkpt_on}): {study.best_trial.value}")
    logging.info("Best Params:")
    for k, v in study.best_trial.params.items():
        logging.info(f"  {k}: {v}")
        
    # ========================== FINAL EVALUATION RUNS ==========================================

    if args.n_eval_runs > 0:
        logging.info(f"Starting {args.n_eval_runs} Final Evaluation Runs with Best Hyperparameters...")
        
        # Construct Final Config
        best_params = study.best_trial.params
        final_config = ExperimentConfig()
        
        # Base settings
        final_config.data_dir = args.data_dir
        final_config.csv_dir = args.csv_dir
        final_config.out_dir = args.out_dir
        final_config.balance_train = args.balance_train
        final_config.method_name = args.method
        final_config.select_chkpt_on = args.select_chkpt_on
        final_config.epochs = 2 if args.debug else final_config.epochs
        
        # Apply Best Params
        for k, v in best_params.items():
            if hasattr(final_config, k):
                setattr(final_config, k, v)

        # Apply Debug Overrides to Final Config
        if args.debug:
            final_config.epochs = 2
            final_config.batch_size = 4
            final_config.num_workers = 0
        
        # Prepare Directory
        eval_root = study_root / "final_evaluation"
        eval_root.mkdir(exist_ok=True)
        
        for i in range(args.n_eval_runs):
            run_dir = eval_root / f"run_{i}"
            run_final_eval(final_config, i, run_dir, args.study_name)
            
        logging.info("===== FINAL EVALUATION COMPLETED =====")