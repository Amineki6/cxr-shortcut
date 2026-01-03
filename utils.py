from tqdm import tqdm
import torch
from torcheval.metrics import BinaryAUROC
import wandb
import optuna
import logging
import pandas as pd
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from config import ExperimentConfig
from dataset import CXP_dataset
from model import CXP_Model
import methods


def run_training_phase(
    config, 
    model, 
    ema_model, 
    optimizer, 
    method, 
    train_loader, 
    val_loader, 
    trial, 
    n_train, 
    n_val, 
    chkpt_path,
    num_epochs=None,      # Allow overriding config.epochs (crucial for JTT Stage 1)
    wandb_prefix="",      # Allow prefixing logs (e.g., "stage1/")
    allow_pruning=True    # JTT Stage 1 usually shouldn't prune the whole trial
):
    device = config.device
    epochs_to_run = num_epochs if num_epochs is not None else config.epochs
    
    model_compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    ema_model_compiled = torch.compile(ema_model, fullgraph=True, mode="reduce-overhead")

    # Initialize Metrics
    train_auroc = BinaryAUROC()
    val_auroc = BinaryAUROC()

    best_val_loss = 10000.0
    best_val_auroc = 0.0
    
    # To handle the return value
    final_best_metric = 0.0

    # Freeze backbone for first few epochs, only train clf head for now
    for param in model_compiled.encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs_to_run):
        # --- TRAIN ---

        if epoch == 5:
            for param in model_compiled.encoder.parameters():
                param.requires_grad = True
            
            logging.info(f'Unfreezing pretrained backbone, fully finetuning now.')

        model_compiled.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_supcon_sum = 0.0
        train_brier_sum = 0.0
        train_auroc.reset()
        
        for batch in tqdm(train_loader, desc=f"Trial {trial.number} {wandb_prefix} Ep {epoch}", leave=False):
            # Dynamic Unpacking to handle Standard (2 items) vs JTT (3 items)
            if len(batch) == 3:
                inputs, labels, drain = batch
                weights = None
            elif len(batch) == 4:
                inputs, labels, weights, drain = batch
            else:
                raise ValueError(f"Unexpected batch structure: len={len(batch)}")

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            
            # Pass weights to method if they exist (JTT logic)
            targets = (labels, weights) if weights is not None else labels

            optimizer.zero_grad(set_to_none=True)
            
            # Forward & Loss
            model_output = model_compiled(inputs)
            
            extra_info = {"drain": drain} 
            
            loss = method.compute_loss(model_output, targets, extra_info=extra_info)
            
            loss.backward()
            optimizer.step()
            
            # Logging Sums
            batch_size = inputs.size(0)
            train_loss_sum += loss.item() * batch_size
            train_bce_sum += method.metrics.get("bce", loss.item()) * batch_size
            train_supcon_sum += method.metrics.get("supcon", 0.0) * batch_size
            
            # Metrics
            logits, _ = model_output
            flat_logits = logits.reshape(-1)
            train_auroc.update(flat_logits, labels)
            
            probs = torch.sigmoid(flat_logits)
            brier = ((probs - labels.float()) ** 2).sum().item()
            train_brier_sum += brier
            
        # EMA Update
        ema_model_compiled.update_parameters(model_compiled)

        # --- VALIDATION ---
        ema_model_compiled.eval()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_supcon_sum = 0.0
        val_brier_sum = 0.0
        val_auroc.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                
                extra_info = {}
                if len(batch) >= 3:
                     drain = batch[2].to(device, non_blocking=True)
                     extra_info["drain"] = drain

                logits, projections = ema_model_compiled(inputs)
                loss = method.compute_loss((logits, projections), labels, extra_info=extra_info)
                
                batch_size = inputs.size(0)
                val_loss_sum += loss.item() * batch_size
                
                val_bce_sum += method.metrics.get("bce", loss.item()) * batch_size
                val_supcon_sum += method.metrics.get("supcon", 0.0) * batch_size
                
                flat_logits = logits.reshape(-1)
                val_auroc.update(flat_logits, labels)
                
                probs = torch.sigmoid(flat_logits)
                brier = ((probs - labels.float()) ** 2).sum().item()
                val_brier_sum += brier

        # Aggregation
        epoch_train_loss = train_loss_sum / n_train
        epoch_val_loss = val_loss_sum / n_val
        epoch_train_auroc = train_auroc.compute().item()
        epoch_val_auroc = val_auroc.compute().item()
        epoch_train_brier = train_brier_sum / n_train
        epoch_val_brier = val_brier_sum / n_val

        # Logging
        logging.info(f"{wandb_prefix}Trial {trial.number} Ep [{epoch+1}/{epochs_to_run}] "
                     f"Train Loss: {epoch_train_loss:.4f} AUROC: {epoch_train_auroc:.4f} "
                     f"Val Loss: {epoch_val_loss:.4f} AUROC: {epoch_val_auroc:.4f}")

        wandb.log({
            f"{wandb_prefix}epoch": epoch,
            f"{wandb_prefix}Loss/train": epoch_train_loss,
            f"{wandb_prefix}Loss/val": epoch_val_loss,
            f"{wandb_prefix}BCE/train": train_bce_sum / n_train,
            f"{wandb_prefix}BCE/val": val_bce_sum / n_val,
            f"{wandb_prefix}SupCon/train": train_supcon_sum / n_train,
            f"{wandb_prefix}SupCon/val": val_supcon_sum / n_val,
            f"{wandb_prefix}auroc/train": epoch_train_auroc,
            f"{wandb_prefix}auroc/val": epoch_val_auroc,
            f"{wandb_prefix}brier/train": epoch_train_brier,
            f"{wandb_prefix}brier/val": epoch_val_brier
        })

        # Checkpointing
        save_chkpt = False
        if config.select_chkpt_on.upper() == "AUROC" and epoch_val_auroc > best_val_auroc:
            best_val_auroc = epoch_val_auroc
            save_chkpt = True
            final_best_metric = best_val_auroc
        elif config.select_chkpt_on.upper() == "LOSS" and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_chkpt = True
            final_best_metric = best_val_loss
            
        if save_chkpt:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # intentionally not _compiled since otherwise reload does not work
                'ema_model_state_dict': ema_model.state_dict(), # intentionally not _compiled since otherwise reload does not work
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_auroc': epoch_val_auroc
            }, chkpt_path)

        # Optuna Pruning
        if allow_pruning:
            target_metric = epoch_val_auroc if config.select_chkpt_on.upper() == "AUROC" else epoch_val_loss
            trial.report(target_metric, epoch)
            if trial.should_prune():
                logging.info(f"Pruning trial {trial.number} at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

    return final_best_metric

def run_testing_phase(
    config,
    ema_model,
    method,
    device,
    test_loader_aligned,
    test_loader_misaligned,
    chkpt_path,
    output_dir,
    run,
    prefix=""
):
    """
    Common testing logic for both Optuna trials and Final Eval runs.
    """
    # Check if checkpoint exists
    if not chkpt_path.exists():
        logging.warning(f"No checkpoint saved at {chkpt_path} (training might have failed or not improved).")
        run.finish()
        return

    # Load Checkpoint
    checkpoint = torch.load(chkpt_path)
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    ema_model.eval()

    # Metrics
    test_auroc_aligned = BinaryAUROC()
    test_auroc_misaligned = BinaryAUROC()

    # Test Aligned
    test_loss_aligned = 0.0
    test_brier_aligned_sum = 0.0
    test_auroc_aligned.reset()
    test_results_aligned = []
    
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_aligned, desc="Test Aligned", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            
            logits, projections = ema_model(inputs)
            
            loss = method.compute_loss((logits, projections), labels, extra_info={"drain": drain})
            
            test_loss_aligned += loss.item() * inputs.size(0)
            test_auroc_aligned.update(logits.reshape(-1), labels)
            
            probs = torch.sigmoid(logits.reshape(-1))
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_aligned_sum += brier
            
            test_results_aligned.append(pd.DataFrame({
                'label': labels.cpu(), 
                'y_prob': probs.cpu(), 
                'drain': drain.cpu()
            }))

    test_loss_aligned /= len(test_loader_aligned.dataset)
    test_brier_aligned = test_brier_aligned_sum / len(test_loader_aligned.dataset)
    test_results_aligned_df = pd.concat(test_results_aligned, ignore_index=True)
    
    # Save Aligned CSV
    aligned_filename = f'{prefix}_aligned.csv' if prefix else 'test_aligned.csv'
    aligned_path = output_dir / aligned_filename
    test_results_aligned_df.to_csv(aligned_path)

    # Test Misaligned
    test_loss_misaligned = 0.0
    test_brier_misaligned_sum = 0.0
    test_auroc_misaligned.reset()
    test_results_misaligned = []
    
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_misaligned, desc="Test Misaligned", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            
            logits, projections = ema_model(inputs)
            
            loss = method.compute_loss((logits, projections), labels, extra_info={"drain": drain})

            test_loss_misaligned += loss.item() * inputs.size(0)
            test_auroc_misaligned.update(logits.reshape(-1), labels)
            
            probs = torch.sigmoid(logits.reshape(-1))
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_misaligned_sum += brier
            
            test_results_misaligned.append(pd.DataFrame({
                'label': labels.cpu(), 
                'y_prob': probs.cpu(), 
                'drain': drain.cpu()
            }))

    test_loss_misaligned /= len(test_loader_misaligned.dataset)
    test_brier_misaligned = test_brier_misaligned_sum / len(test_loader_misaligned.dataset)
    test_results_misaligned_df = pd.concat(test_results_misaligned, ignore_index=True)
    
    # Save Misaligned CSV
    misaligned_filename = f'{prefix}_misaligned.csv' if prefix else 'test_misaligned.csv'
    misaligned_path = output_dir / misaligned_filename
    test_results_misaligned_df.to_csv(misaligned_path)

    # Log Final Test Results to WandB
    wandb.log({
        "test_loss/aligned": test_loss_aligned,
        "test_loss/misaligned": test_loss_misaligned,
        "test_auroc/aligned": test_auroc_aligned.compute().item(),
        "test_auroc/misaligned": test_auroc_misaligned.compute().item(),
        "test_brier/aligned": test_brier_aligned,
        "test_brier/misaligned": test_brier_misaligned
    })

    logging.info(f"Test Aligned - Loss: {test_loss_aligned:.4f} AUROC: {test_auroc_aligned.compute():.4f}")
    logging.info(f"Test Misaligned - Loss: {test_loss_misaligned:.4f} AUROC: {test_auroc_misaligned.compute():.4f}")


class DummyTrial:
    """Mock Optuna Trial for Final Evaluation Runs"""
    def __init__(self, number=0):
        self.number = number
        self.params = {}
    
    def report(self, metric, step):
        pass
    
    def should_prune(self):
        return False
    
    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, low)

    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

def get_dataloaders(config: ExperimentConfig, debug=False):
    # Determine CSV filenames based on config
    if config.balance_val:
        train_csv = config.csv_dir / 'train_drain_shortcut_v2.csv'
        val_csv = config.csv_dir / 'val_drain_shortcut_v2.csv'
    else:
        train_csv = config.csv_dir / 'train_drain_shortcut.csv'
        val_csv = config.csv_dir / 'val_drain_shortcut.csv'

    # Initialize Datasets
    train_data = CXP_dataset(config.data_dir, train_csv, augment=True)
    val_data = CXP_dataset(config.data_dir, val_csv, augment=False)
    
    test_data_aligned = CXP_dataset(config.data_dir, config.csv_dir / 'test_drain_shortcut_aligned.csv', augment=False)
    test_data_misaligned = CXP_dataset(config.data_dir, config.csv_dir / 'test_drain_shortcut_misaligned.csv', augment=False)

    # --- DEBUG MODE SUBSETTING ---
    if debug:
        logging.info("DEBUG MODE ACTIVE: Subsetting datasets to max 50 samples.")
        def fast_subset(ds, n=50):
            return torch.utils.data.Subset(ds, range(min(len(ds), n)))
        
        train_data = fast_subset(train_data)
        val_data = fast_subset(val_data, 20)
        test_data_aligned = fast_subset(test_data_aligned, 20)
        test_data_misaligned = fast_subset(test_data_misaligned, 20)
        
        config.balance_train = False
    # -----------------------------

    # --- SAMPLER LOGIC ---
    sampler = None
    if config.balance_train:
        pneu_msk = train_data.labels == 1
        drain_counts_pneu = torch.bincount(torch.from_numpy(train_data.drain[pneu_msk].values))
        drain_weights_pneu = 1.0 / drain_counts_pneu.float()
        drain_counts_nopneu = torch.bincount(torch.from_numpy(train_data.drain[~pneu_msk].values))
        drain_weights_nopneu = 1.0 / drain_counts_nopneu.float()        
        
        sample_weights = torch.zeros_like(torch.from_numpy(train_data.labels.values), dtype=torch.float32)
        sample_weights[pneu_msk] = drain_weights_pneu[train_data.drain[pneu_msk].values]
        sample_weights[~pneu_msk] = drain_weights_nopneu[train_data.drain[~pneu_msk].values]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- FIX: Set prefetch_factor conditionally ---
    prefetch_factor = 2 if config.num_workers > 0 else None
    # ----------------------------------------------

    # Create Loaders
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, 
        shuffle=(sampler is None), 
        num_workers=config.num_workers, 
        pin_memory=True, 
        prefetch_factor=prefetch_factor,
        sampler=sampler
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True, 
        prefetch_factor=prefetch_factor
    )
    test_loader_aligned = DataLoader(
        test_data_aligned, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True, 
        prefetch_factor=prefetch_factor
    )
    test_loader_misaligned = DataLoader(
        test_data_misaligned, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True, 
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader, test_loader_aligned, test_loader_misaligned, len(train_data), len(val_data)

def run_final_eval(config, trial_number, output_dir, run_name_prefix):
    """
    Runs a standalone training session using a specific config (best params).
    Uses DummyTrial to bypass Optuna reporting/pruning.
    """
    logging.info(f"--- Starting Final Eval Run {trial_number} ---")
    
    # Create Output Directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup WandB for this run
    run_name = f"{run_name_prefix}_run_{trial_number}"
    run = wandb.init(
        project="cxr_optuna_study", 
        group=f"{run_name_prefix}_final", 
        name=run_name,
        config=config.__dict__,
        reinit=True
    )

    # 2. Setup Components
    device = config.device
    trial = DummyTrial(trial_number)
    
    # Get Data
    train_loader, val_loader, test_loader_aligned, test_loader_misaligned, n_train, n_val = get_dataloaders(config, debug=(config.num_workers == 0 and config.epochs == 2)) 
    
    # Get Method Strategy
    method = methods.get_method(config.method_name, config)
    
    # Init Model
    model = CXP_Model(method).to(device)
    
    # Init EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
    
    # Init Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Checkpoint Path
    chkpt_path = output_dir / f'run_{trial_number}_best.chkpt'

    # 3. Run Training
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
        chkpt_path=chkpt_path,
        wandb_prefix="",
        allow_pruning=False 
    )

    # 4. Run Testing
    run_testing_phase(
        config=config,
        ema_model=ema_model,
        method=method,
        device=device,
        test_loader_aligned=test_loader_aligned,
        test_loader_misaligned=test_loader_misaligned,
        chkpt_path=chkpt_path,
        output_dir=output_dir,
        run=run,
        prefix=f"run_{trial_number}"
    )

    run.finish()
    logging.info(f"--- Finished Final Eval Run {trial_number} ---")