from tqdm import tqdm
import torch
from torcheval.metrics import BinaryAUROC
import wandb
import logging
import pandas as pd
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np
from typing import Optional
from torch.nn.functional import binary_cross_entropy_with_logits

from config import ExperimentConfig
from dataset import CXP_dataset
from model import CXP_Model
import methods
from mem_log import check_shm_usage, log_memory_usage
from scoring import compute_group_fairness_score

def run_training_phase(
    config, 
    model, 
    ema_model, 
    optimizer, 
    method, 
    train_loader, 
    val_loader,
    trial_number: int,
    chkpt_path,
    num_epochs: Optional[int] = None,      # Allow overriding config.epochs (crucial for JTT Stage 1)
    wandb_prefix: str = "",      # Allow prefixing logs (e.g., "stage1/")
):
    device = config.device
    epochs_to_run = num_epochs if num_epochs is not None else config.epochs
    
    # Only compile on CUDA
    if "cuda" in str(device):
        model_compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        ema_model_compiled = torch.compile(ema_model, fullgraph=True, mode="reduce-overhead")
    else:
        model_compiled = model
        ema_model_compiled = ema_model

    # Initialize Metrics
    train_auroc = BinaryAUROC()
    val_auroc = BinaryAUROC()
    val_wauroc = BinaryAUROC()

    # create copies of method - in particular of the attached loss - because some methods
    # (-> score_matching_dataset) have a loss state that should not be shared between train and val 
    # and that should be reset for each new training run
    train_method = method.clone(dataset_size=len(train_loader.dataset))
    val_method = method.clone(dataset_size=len(val_loader.dataset))

    if config.select_chkpt_on.upper() in ["AUROC", "WAUROC", "FAIRNESS"]:
        best_metric_val = 0.0
    else:
        best_metric_val = float('inf')

    # Freeze backbone for first few epochs, only train clf head for now
    for param in model_compiled.encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs_to_run):
        # --- TRAIN ---

        if epoch == 5:
            for param in model_compiled.encoder.parameters():
                param.requires_grad = True
            
            logging.info('Unfreezing pretrained backbone, fully finetuning now.')

        model_compiled.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_brier_sum = 0.0
        train_auroc.reset()
        
        for batch in tqdm(train_loader, desc=f"Trial {trial_number} {wandb_prefix} Ep {epoch}", leave=False):

            if len(batch) == 5:
                indices, inputs, labels, weights, drain = batch
            else:
                assert len(batch) == 4
                indices, inputs, labels, drain = batch
                weights = None

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            
            # Pass weights to method if they exist
            if weights:
                weights = weights.to(device, non_blocking=True)
                targets = (labels, weights) 
            else:
                targets = labels

            optimizer.zero_grad(set_to_none=True)
            
            # Forward & Loss
            model_output = model_compiled(inputs)
            
            extra_info = {"drain": drain, 'indices': indices} 
            
            loss, components = train_method.compute_loss(model_output, targets, extra_info=extra_info)
            
            loss.backward()
            optimizer.step()
            
            # Logging Sums
            batch_size = inputs.size(0)
            train_loss_sum += loss.item() * batch_size
            train_bce_sum += components["bce"] * batch_size

            # Metrics
            logits, _ = model_output
            flat_logits = logits.reshape(-1)
            train_auroc.update(flat_logits.detach().cpu(), labels.detach().cpu())
            
            probs = torch.sigmoid(flat_logits)
            brier = ((probs - labels.float()) ** 2).sum().item()
            train_brier_sum += brier

            # Some losses (actually only dataset_score_matching) need to be updated with new model outputs after optimizer.step()
            try:
                with torch.no_grad():
                    model_output = model_compiled(inputs)
                    train_method.update_loss(model_output, targets, extra_info=extra_info)
            except AttributeError:
                # not all methods have this; that is expected and fine
                pass   
            
        # EMA Update
        ema_model_compiled.update_parameters(model_compiled)

        # --- VALIDATION ---
        ema_model_compiled.eval()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_wbce_sum = 0.0
        val_brier_sum = 0.0
        val_auroc.reset()
        val_wauroc.reset()
        
        all_val_logits = []
        all_val_labels = []
        all_val_drains = []

        with torch.no_grad():
            for batch in val_loader:
                indices = batch[0].to(device, non_blocking=True)
                inputs = batch[1].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)
                
                extra_info = {}
                drain = batch[3].to(device, non_blocking=True)
                extra_info["drain"] = drain
                extra_info["indices"] = indices
                sample_weights = batch[4].to(device, non_blocking=True)

                logits, projections = ema_model_compiled(inputs)
                loss, components = val_method.compute_loss((logits, projections), labels, extra_info=extra_info)
                
                batch_size = inputs.size(0)
                val_loss_sum += loss.item() * batch_size
                val_bce_sum += components["bce"] * batch_size
                val_wbce_sum += binary_cross_entropy_with_logits(logits.view(-1),
                                                                 labels.float(),
                                                                 weight=sample_weights,
                                                                 reduction="sum")            
                
                flat_logits = logits.reshape(-1)
                val_auroc.update(flat_logits.detach().cpu(), labels.detach().cpu())
                val_wauroc.update(flat_logits.detach().cpu(), labels.detach().cpu(), weight=sample_weights.detach().cpu())

                probs = torch.sigmoid(flat_logits)
                brier = ((probs - labels.float()) ** 2).sum().item()
                val_brier_sum += brier

                all_val_logits.append(flat_logits.detach().cpu())
                all_val_labels.append(labels.detach().cpu())
                all_val_drains.append(drain.detach().cpu())

                # Some losses (actually only dataset_score_matching) need to be explicitly updated with new model outputs
                try:
                    with torch.no_grad():
                        model_output = model_compiled(inputs)
                        val_method.update_loss(model_output, labels, extra_info=extra_info)
                except AttributeError:
                    # not all methods have this; that is expected and fine
                    pass                   

        # Aggregation
        epoch_train_loss = train_loss_sum / len(train_loader.dataset)
        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        epoch_train_bce = train_bce_sum / len(train_loader.dataset)
        epoch_val_bce = val_bce_sum / len(val_loader.dataset)
        epoch_val_wbce = val_wbce_sum / len(val_loader.dataset)
        epoch_train_auroc = train_auroc.compute().item()
        epoch_val_auroc = val_auroc.compute().item()
        epoch_val_wauroc = val_wauroc.compute().item()
        epoch_train_brier = train_brier_sum / len(train_loader.dataset)
        epoch_val_brier = val_brier_sum / len(val_loader.dataset)

        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)
        all_val_drains = torch.cat(all_val_drains)

        epoch_val_fairness_score, fairness_details = compute_group_fairness_score(
            all_val_logits, all_val_labels, all_val_drains
        )

        # Logging
        logging.info(f"{wandb_prefix}Trial {trial_number} Ep [{epoch+1}/{epochs_to_run}] "
                     f"Train Loss: {epoch_train_loss:.4f} AUROC: {epoch_train_auroc:.4f} "
                     f"Val Loss: {epoch_val_loss:.4f} AUROC: {epoch_val_auroc:.4f} "
                     f"Val BCE: {epoch_val_bce:.4f} wBCE: {epoch_val_wbce:.4f} "
                     f"Fairness: {epoch_val_fairness_score:.4f}")
        
        check_shm_usage()
        log_memory_usage()

        wandb.log({
            f"{wandb_prefix}epoch": epoch,
            f"{wandb_prefix}Loss/train": epoch_train_loss,
            f"{wandb_prefix}Loss/val": epoch_val_loss,
            f"{wandb_prefix}BCE/train": epoch_train_bce,
            f"{wandb_prefix}BCE/val": epoch_val_bce,
            f"{wandb_prefix}wBCE/val": epoch_val_wbce,
            f"{wandb_prefix}auroc/train": epoch_train_auroc,
            f"{wandb_prefix}auroc/val": epoch_val_auroc,
            f"{wandb_prefix}wauroc/val": epoch_val_wauroc,
            f"{wandb_prefix}brier/train": epoch_train_brier,
            f"{wandb_prefix}brier/val": epoch_val_brier,
            f"{wandb_prefix}fairness_score/val": epoch_val_fairness_score,
            f"{wandb_prefix}fairness_mean_avg/val": fairness_details['summary']['mean_average'],
            f"{wandb_prefix}fairness_mean_diff/val": fairness_details['summary']['mean_abs_diff'],            
        })

        # Checkpointing
        save_chkpt = False
        if config.select_chkpt_on.upper() == "AUROC" and epoch_val_auroc > best_metric_val:
            best_metric_val = epoch_val_auroc
            save_chkpt = True
        elif config.select_chkpt_on.upper() == "WAUROC" and epoch_val_wauroc > best_metric_val:
            best_metric_val = epoch_val_wauroc
            save_chkpt = True
        elif config.select_chkpt_on.upper() == "FAIRNESS" and epoch_val_fairness_score > best_metric_val:
            best_metric_val = epoch_val_fairness_score
            save_chkpt = True            
        elif config.select_chkpt_on.upper() == "BCE" and epoch_val_bce < best_metric_val:
            best_metric_val = epoch_val_bce
            save_chkpt = True
        elif config.select_chkpt_on.upper() == "WBCE" and epoch_val_wbce < best_metric_val:
            best_metric_val = epoch_val_wbce
            save_chkpt = True
            
        if save_chkpt:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # intentionally not _compiled since otherwise reload does not work
                'ema_model_state_dict': ema_model.state_dict(), # intentionally not _compiled since otherwise reload does not work
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_auroc': epoch_val_auroc
            }, chkpt_path)

    del train_method, val_method, train_auroc, val_auroc, val_wauroc

    if "cuda" in str(device):
        del model_compiled, ema_model_compiled
        torch.compiler.reset()
        torch.cuda.empty_cache()       

    return best_metric_val


def run_testing_phase(
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

    test_method_aligned = method.clone(dataset_size=len(test_loader_aligned.dataset))
    test_method_misaligned = method.clone(dataset_size=len(test_loader_misaligned.dataset))
    
    with torch.no_grad():
        for indices, inputs, labels, drain in tqdm(test_loader_aligned, desc="Test Aligned", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            logits, projections = ema_model(inputs)
            
            extra_info = {"drain": drain, 'indices': indices}
            loss, components = test_method_aligned.compute_loss((logits, projections), labels, extra_info=extra_info)
            
            batchsize = inputs.size(0)
            test_loss_aligned += loss.item() * batchsize
            test_auroc_aligned.update(logits.reshape(-1), labels)
            
            probs = torch.sigmoid(logits.reshape(-1))
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_aligned_sum += brier
            
            test_results_aligned.append(pd.DataFrame({
                'label': labels.detach().cpu(), 
                'y_prob': probs.detach().cpu(), 
                'drain': drain.detach().cpu()
            }))

            # Some losses (actually only dataset_score_matching) need to be explicitly updated with new model outputs
            try:
                with torch.no_grad():
                    model_output = ema_model(inputs)
                    test_method_aligned.update_loss(model_output, labels, extra_info=extra_info)
            except AttributeError:
                # not all methods have this; that is expected and fine
                pass    

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
        for indices, inputs, labels, drain in tqdm(test_loader_misaligned, desc="Test Misaligned", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            drain = drain.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            
            logits, projections = ema_model(inputs)
            extra_info = {"drain": drain, 'indices': indices}
            loss, components = test_method_misaligned.compute_loss((logits, projections), labels, extra_info=extra_info)

            batchsize = inputs.size(0)
            test_loss_misaligned += loss.item() * batchsize
            test_auroc_misaligned.update(logits.reshape(-1), labels)
            
            probs = torch.sigmoid(logits.reshape(-1))
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_misaligned_sum += brier
            
            test_results_misaligned.append(pd.DataFrame({
                'label': labels.detach().cpu(), 
                'y_prob': probs.detach().cpu(), 
                'drain': drain.detach().cpu()
            }))

            # Some losses (actually only dataset_score_matching) need to be explicitly updated with new model outputs
            try:
                with torch.no_grad():
                    model_output = ema_model(inputs)
                    test_method_misaligned.update_loss(model_output, labels, extra_info=extra_info)
            except AttributeError:
                # not all methods have this; that is expected and fine
                pass               

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

    del test_method_aligned, test_method_misaligned, test_auroc_aligned, test_auroc_misaligned

    if torch.cuda.is_available():
        torch.cuda.empty_cache()    


def get_dataloaders(config: ExperimentConfig, debug=False):
    # Determine CSV filenames based on config
    if config.balance_val:
        train_csv = config.csv_dir / 'train_drain_shortcut_v2.csv'
        val_csv = config.csv_dir / 'val_drain_shortcut_v2.csv'
    else:
        train_csv = config.csv_dir / 'train_drain_shortcut.csv'
        val_csv = config.csv_dir / 'val_drain_shortcut.csv'

    train_data = CXP_dataset(config.data_dir, train_csv, augment=True, return_weights=False, return_indices=True)
    val_data = CXP_dataset(config.data_dir, val_csv, augment=False, return_weights=True, return_indices=True)
    
    test_data_aligned = CXP_dataset(config.data_dir, config.csv_dir / 'test_drain_shortcut_aligned.csv', augment=False, return_indices=True)
    test_data_misaligned = CXP_dataset(config.data_dir, config.csv_dir / 'test_drain_shortcut_misaligned.csv', augment=False, return_indices=True)

    if debug:
        logging.info("DEBUG MODE ACTIVE: Subsetting datasets to max 50 samples.")
        def fast_subset(ds, n=50):
            return torch.utils.data.Subset(ds, range(min(len(ds), n)))
        
        train_data = fast_subset(train_data)
        val_data = fast_subset(val_data, 20)
        test_data_aligned = fast_subset(test_data_aligned, 20)
        test_data_misaligned = fast_subset(test_data_misaligned, 20)
        
        config.balance_train = False

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

    prefetch_factor = 2 if config.num_workers > 0 else None

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

    return train_loader, val_loader, test_loader_aligned, test_loader_misaligned


def run_final_eval(config, trial_number, output_dir, run_name_prefix):
    """
    Runs a standalone training session using a specific config.
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
    
    # Get Data
    train_loader, val_loader, test_loader_aligned, test_loader_misaligned = get_dataloaders(config, debug=(config.num_workers == 0 and config.epochs == 2)) 
    
    # Get Method Strategy
    method = methods.get_method(config.method_name, config)
    
    # --- JTT STAGE 1 LOGIC ---
    if config.method_name == "jtt":
        logging.info("--- JTT Stage 1: Identification Phase ---")
        
        # 1. Initialize Identification Model
        model_1 = CXP_Model(method).to(device)
        ema_model_1 = AveragedModel(model_1, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
        optimizer_1 = optim.AdamW(model_1.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # 2. Train Identification Model
        stage1_chkpt = output_dir / f"run_{trial_number}_jtt_stage1.chkpt"
        
        run_training_phase(
            config=config,
            model=model_1,
            ema_model=ema_model_1,
            optimizer=optimizer_1,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            trial_number=trial_number,
            chkpt_path=stage1_chkpt,
            num_epochs=config.jtt_duration,
            wandb_prefix="jtt_stage1/",
        )
        
        # 3. Identify Error Set
        # Create a sequential (non-shuffled) loader for correct index mapping
        id_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        error_indices = identify_error_set(model_1, method, id_loader, device)
        n_train_len = len(train_loader.dataset)
        logging.info(f"JTT Stage 1 Complete. Found {len(error_indices)} errors out of {n_train_len} examples.")
        
        # 4. Create JTT Stage 2 Loader
        train_loader = get_jtt_loader(train_loader.dataset, error_indices, config)
        
        logging.info("--- JTT Stage 2: Upweighted Training Phase ---")
        
        # Clean up Stage 1 artifacts
        del model_1, ema_model_1, optimizer_1, id_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Init Model
    model = CXP_Model(method).to(device)
    
    # Init EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
    
    # Init Optimizer
    # Define optimizer parameters
    params = [
        {'params': model.encoder.parameters(), 'lr': config.lr / 5},
        {'params': model.clf.parameters(), 'lr': config.lr}
    ]
    
    # Include projection head if it exists (e.g. for SupCon)
    if model.projection_head is not None:
        params.append({'params': model.projection_head.parameters(), 'lr': config.lr})

    optimizer = optim.AdamW(params, weight_decay=config.weight_decay)
    
    # Checkpoint Path
    chkpt_path = output_dir / f'run_{trial_number}_best.chkpt'

    # 3. Run Training
    run_training_phase(
        config=config,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        method=method,
        train_loader=train_loader,
        val_loader=val_loader,
        trial_number=trial_number,
        chkpt_path=chkpt_path,
        wandb_prefix=""
    )

    # 4. Run Testing
    run_testing_phase(
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

def identify_error_set(model, method, loader, device, max_batches=None):
    """
    Evaluates the model on the loader and returns indices of misclassified examples.
    Used for JTT Stage 1.
    """
    model.eval()
    error_indices = []
    
    current_idx = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Identifying Error Set", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            
            # Unpack based on your dataset structure (indices, inputs, labels, drain)
            if len(batch) == 4:
                _, inputs, labels, _ = batch
            elif len(batch) == 5:
                 _, inputs, labels, _, _ = batch
            else:
                # Fallback if structure varies, but typically it's indices first
                inputs = batch[1]
                labels = batch[2]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            model_output = model(inputs)
            logits = model_output[0]
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Identify mismatches
            mismatches = (preds.view(-1) != labels.view(-1)).detach().cpu().numpy()
            
            # Map batch-relative indices to global indices
            batch_size = inputs.size(0)
            batch_indices = np.arange(current_idx, current_idx + batch_size)
            
            # Add misclassified global indices to list
            error_indices.extend(batch_indices[mismatches])
            
            current_idx += batch_size

    return set(error_indices)

def get_jtt_loader(dataset, error_indices, config):
    """
    Creates a DataLoader with upweighted error set examples.
    """
    # Create weights: JTT_lambda for error set, 1.0 for others
    weights = torch.ones(len(dataset))
    
    # Check if we have error indices
    if not error_indices:
        logging.warning("No errors found in identification stage! Returning standard loader.")
        return DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, 
            num_workers=config.num_workers, pin_memory=True
        )

    for idx in error_indices:
        weights[idx] = config.jtt_lambda
        
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    return loader