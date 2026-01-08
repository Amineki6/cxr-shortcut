import sys
import logging
import argparse
from pathlib import Path
import json
import coolname
import datetime

import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import wandb
import optuna

# Local imports from your refactored structure
from config import ExperimentConfig
from model import CXP_Model
import methods
from utils import run_training_phase, get_dataloaders, run_final_eval, identify_error_set, get_jtt_loader

# Global args placeholder to be populated in main
GLOBAL_ARGS: argparse.Namespace = argparse.Namespace()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

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
    study_root = GLOBAL_ARGS.study_root

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

        if config.method_name == "supcon":
            config.supcon_lambda = 0.5
            config.supcon_temperature = 0.1
        elif config.method_name == "mmd":
            config.mmd_lambda = 1.0
        elif config.method_name == "cdan":
            config.cdan_lambda = 1.0
            config.cdan_entropy = True
        elif config.method_name == "score_matching":
            config.score_matching_lambda = 1.0
        elif config.method_name == "jtt":
            config.jtt_duration = 1
            config.jtt_lambda = 4.0
        else:
            assert config.method_name == "standard"
            
    else:
        # Optimizable Hyperparameters
        
        if config.method_name == "supcon":
            config.supcon_lambda = trial.suggest_float("supcon_lambda", 0.01, 50.0)
            config.supcon_temperature = trial.suggest_float("supcon_temperature", 0.05, 0.5)
        elif config.method_name == "mmd":
            config.mmd_lambda = trial.suggest_float("mmd_lambda", 0.01, 50.0)
        elif config.method_name == "cdan":
            config.cdan_lambda = trial.suggest_float("cdan_lambda", 0.01, 50.0)
            config.cdan_entropy = True 
        elif config.method_name == "score_matching":
            config.score_matching_lambda = trial.suggest_float("score_matching_lambda", 0.01, 50.0)
        elif config.method_name == "jtt":
            config.jtt_duration = trial.suggest_int("jtt_duration", 1, max(1, config.epochs // 2))
            config.jtt_lambda = trial.suggest_float("jtt_lambda", 2.0, 100.0, log=True)
        else:
            assert config.method_name == "standard"

    # ====== WANDB ========================================================================

    # WandB Setup
    run_name = f"{GLOBAL_ARGS.study_name}_trial_{trial.number}"
    run = wandb.init(
        project="cxr_optuna_study", 
        group=GLOBAL_ARGS.study_name, 
        name=run_name,
        config=config.__dict__,
        reinit=True,
        dir=GLOBAL_ARGS.study_root  # Prevent creating ./wandb which shadows the library
    )

    # ====== SETUP TRAINING ================================================================

    # Setup Components
    device = config.device
    
    # Get Data
    train_loader, val_loader, _, _ = get_dataloaders(config, debug=GLOBAL_ARGS.debug)
    
    # Get Method Strategy (Standard or SupCon)
    method = methods.get_method(config.method_name, config)

    # --- JTT STAGE 1 LOGIC ---
    if config.method_name == "jtt":
        logging.info("--- JTT Stage 1: Identification Phase ---")
        
        # 1. Initialize Identification Model
        model_1 = CXP_Model(method).to(device)
        ema_model_1 = AveragedModel(model_1, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
        optimizer_1 = optim.AdamW(model_1.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # 2. Train Identification Model
        stage1_chkpt = study_root / "checkpoints" / f"trial_{trial.number}_jtt_stage1.chkpt"
        
        run_training_phase(
            config=config,
            model=model_1,
            ema_model=ema_model_1,
            optimizer=optimizer_1,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            trial=trial, # We still pass trial for logging, but we disable pruning
            chkpt_path=stage1_chkpt,
            num_epochs=config.jtt_duration,
            wandb_prefix="jtt_stage1/",
            allow_pruning=False
        )
        
        # 3. Identify Error Set
        # Create a sequential (non-shuffled) loader for correct index mapping
        id_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            shuffle=False, # CRITICAL: Must be False to match indices
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        error_indices = identify_error_set(model_1, method, id_loader, device)
        n_train_len = len(train_loader.dataset)
        logging.info(f"JTT Stage 1 Complete. Found {len(error_indices)} errors out of {n_train_len} examples.")
        
        # 4. Create JTT Stage 2 Loader
        # This overwrites the original train_loader for the main training loop below
        train_loader = get_jtt_loader(train_loader.dataset, error_indices, config)
        
        logging.info("--- JTT Stage 2: Upweighted Training Phase ---")
        
        # Clean up Stage 1 artifacts to free memory
        del model_1, ema_model_1, optimizer_1, id_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Init Model
    model = CXP_Model(method).to(device)
    
    # Init EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
    
    # Init Optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.lr / 5},
        {'params': model.clf.parameters(), 'lr': config.lr}
    ], weight_decay=config.weight_decay)

    # Filename for this specific trial's checkpoint
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
        trial_number=trial.number,
        chkpt_path=chkpt_path
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
    parser.add_argument('--select_chkpt_on', type=str, default="bce", choices=["bce", "wbce", "auroc", "wauroc"],
                       help='Metric to select best model')
    parser.add_argument('--debug', action='store_true', 
                       help='Run in debug mode (tiny data, 1 epoch, CPU/MPS friendly)')
    
    # Methods
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'supcon', 'mmd', 'cdan', 'score_matching', 'jtt'],
                       help='Method to use for training (default: standard)')

    args = parser.parse_args()

    if args.study_name is None:
        args.study_name = coolname.generate_slug(2)
    
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

    if args.n_trials == 0:
        optimize = False
        logging.info("Not running hyperparam opt because n_trials == 0. Using default params.")
    elif args.method == 'standard':
        optimize = False
        logging.info("Not running hyperparam opt because method == 'standard'. Using default params.")
    else:
        optimize = True

    if optimize:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        config_dict = vars(args)
        config_dict["start_time"] = start_time_str
        
        with open(study_root / "experiment_config.json", "w") as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        logging.info(f"Starting Study: {args.study_name}")
        logging.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        # Define Optimization Direction
        direction = "maximize" if args.select_chkpt_on.upper() == 'AUROC' else "minimize"
        
        study = optuna.create_study(
            sampler=optuna.samplers.GPSampler(),
            direction=direction, 
            study_name=args.study_name,
            load_if_exists=True
        )
        
        # ========================== RUN OPTIMIZATION ===============================================

        study.optimize(objective, n_trials=args.n_trials)
        best_params = study.best_trial.params.items()
        # ===========================================================================================
        
        logging.info("===== STUDY COMPLETED =====")
        logging.info(f"Best Trial Number: {study.best_trial.number}")
        logging.info(f"Best Value ({args.select_chkpt_on}): {study.best_trial.value}")
        logging.info("Best Params:")
        for k, v in best_params:
            logging.info(f"  {k}: {v}")
        
    # ========================== FINAL EVALUATION RUNS ==========================================

    if args.n_eval_runs > 0:
        if optimize:
            logging.info(f"Starting {args.n_eval_runs} final evaluation runs with best hyperparameters...")
        else:
            logging.info(f"Starting {args.n_eval_runs} final evaluation runs with default hyperparameters...")
        
        final_config = ExperimentConfig()
        
        # Base settings
        final_config.data_dir = args.data_dir
        final_config.csv_dir = args.csv_dir
        final_config.out_dir = args.out_dir
        final_config.balance_train = args.balance_train
        final_config.method_name = args.method
        final_config.select_chkpt_on = args.select_chkpt_on
        final_config.epochs = 2 if args.debug else final_config.epochs
        
        if optimize:
            # Apply Best Params
            for k, v in best_params:
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
        
        logging.info("Full config applied:")
        logging.info(final_config)

        for i in range(args.n_eval_runs):
            run_dir = eval_root / f"run_{i}"
            run_final_eval(final_config, i, run_dir, args.study_name)
            
        logging.info("===== FINAL EVALUATION COMPLETED =====")