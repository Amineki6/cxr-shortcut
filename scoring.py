import torch
import numpy as np
from torcheval.metrics import BinaryAUROC
import logging
from typing import Dict, Tuple

def compute_group_fairness_score(logits, labels, groups) -> Tuple[float, Dict]:
    """
    Computes a fairness-aware score based on group-wise AUROC disparities.
    
    For each group value g:
      - AUROC_1: positive labels from g vs negative labels from not-g
      - AUROC_2: positive labels from not-g vs negative labels from g
      - avg_g = (AUROC_1 + AUROC_2) / 2
      - diff_g = |AUROC_1 - AUROC_2|
    
    Final score = mean_g(avg) - mean_g(diff)
    
    Samples with missing group info (e.g., NaN, -1) are excluded from computation.
    
    Args:
        logits: Model logits (tensor, 1D)
        labels: Ground truth labels (tensor, 1D, binary 0/1)
        groups: Group assignments (tensor, 1D, integer or may contain NaN/-1 for missing)
    
    Returns:
        score: The fairness-adjusted score (higher is better)
        details: Dictionary with detailed metrics per group
    """
    # Input validation
    assert torch.is_tensor(logits), "logits must be a torch.Tensor"
    assert torch.is_tensor(labels), "labels must be a torch.Tensor"
    assert torch.is_tensor(groups), "groups must be a torch.Tensor"
    
    # Move to CPU for computation
    logits = logits.cpu()
    labels = labels.cpu()
    groups = groups.cpu()
    
    # Ensure 1D tensors
    assert logits.dim() == 1, f"logits must be 1D, got shape {logits.shape}"
    assert labels.dim() == 1, f"labels must be 1D, got shape {labels.shape}"
    assert groups.dim() == 1, f"groups must be 1D, got shape {groups.shape}"
    
    # Check matching sizes
    n_samples = logits.size(0)
    assert labels.size(0) == n_samples, f"labels size {labels.size(0)} != logits size {n_samples}"
    assert groups.size(0) == n_samples, f"groups size {groups.size(0)} != logits size {n_samples}"
    
    # Check label values are binary
    unique_labels = torch.unique(labels)
    assert torch.all((unique_labels == 0) | (unique_labels == 1)), \
        f"labels must be binary (0/1), got unique values: {unique_labels.tolist()}"
    
    # Filter out samples with missing group info
    # Handle both NaN and negative values (e.g., -1) as missing
    if groups.dtype.is_floating_point:
        valid_mask = ~torch.isnan(groups)
    else:
        # For integer groups, treat negative values as missing
        valid_mask = groups >= 0
    
    n_valid = valid_mask.sum().item()
    n_missing = (~valid_mask).sum().item()
    
    if n_valid == 0:
        logging.error("No samples with valid group information!")
        return np.nan, {'error': 'no_valid_groups', 'n_missing': n_missing}
    elif n_missing > 0:
        logging.info(f"Excluding {n_missing}/{n_samples} samples with missing group info")
    
    # Apply mask to filter valid samples only
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    groups = groups[valid_mask]
    
    # For floating point groups, convert to int after filtering
    if groups.dtype.is_floating_point:
        groups = groups.long()
    
    assert groups.dtype in [torch.int32, torch.int64], \
        f"groups must be integer type after conversion, got {groups.dtype}"
    
    unique_groups = torch.unique(groups)
    n_groups = len(unique_groups)
    
    assert n_groups > 0, "No groups found after filtering"
    logging.info(f"Computing fairness score across {n_groups} groups with {n_valid} total samples")
    
    group_avgs = []
    group_diffs = []
    details = {}
    
    for g in unique_groups:
        g_val = g.item()
        
        # Sanity check: group value should be non-negative
        assert g_val >= 0, f"Group value {g_val} is negative after filtering"
        
        # Masks for this group
        in_group = groups == g
        out_group = ~in_group
        
        n_in_group = in_group.sum().item()
        n_out_group = out_group.sum().item()
        
        assert n_in_group > 0, f"Group {g_val} has no samples"
        assert n_out_group > 0, f"Group {g_val}: all samples in this group (no out-group)"
        assert n_in_group + n_out_group == n_valid, \
            f"Group {g_val}: in+out={n_in_group + n_out_group} != total={n_valid}"
        
        # Masks for labels
        pos_labels = labels == 1
        neg_labels = labels == 0
        
        # AUROC 1: pos from g vs neg from not-g
        pos_in_g = in_group & pos_labels
        neg_out_g = out_group & neg_labels
        
        n_pos_in_g = pos_in_g.sum().item()
        n_neg_out_g = neg_out_g.sum().item()
        
        # AUROC 2: pos from not-g vs neg from g
        pos_out_g = out_group & pos_labels
        neg_in_g = in_group & neg_labels
        
        n_pos_out_g = pos_out_g.sum().item()
        n_neg_in_g = neg_in_g.sum().item()
        
        # Compute AUROC_1
        if n_pos_in_g > 0 and n_neg_out_g > 0:
            auroc_1_metric = BinaryAUROC()
            combined_logits_1 = torch.cat([logits[pos_in_g], logits[neg_out_g]])
            combined_labels_1 = torch.cat([
                torch.ones(n_pos_in_g),
                torch.zeros(n_neg_out_g)
            ])
            
            assert combined_logits_1.size(0) == n_pos_in_g + n_neg_out_g
            assert combined_labels_1.size(0) == n_pos_in_g + n_neg_out_g
            assert combined_logits_1.dim() == 1
            assert combined_labels_1.dim() == 1
            
            auroc_1_metric.update(combined_logits_1, combined_labels_1)
            auroc_1 = auroc_1_metric.compute().item()
            
            assert 0.0 <= auroc_1 <= 1.0, f"AUROC_1={auroc_1} out of valid range [0,1]"
        else:
            auroc_1 = None
            
        # Compute AUROC_2
        if n_pos_out_g > 0 and n_neg_in_g > 0:
            auroc_2_metric = BinaryAUROC()
            combined_logits_2 = torch.cat([logits[pos_out_g], logits[neg_in_g]])
            combined_labels_2 = torch.cat([
                torch.ones(n_pos_out_g),
                torch.zeros(n_neg_in_g)
            ])
            
            assert combined_logits_2.size(0) == n_pos_out_g + n_neg_in_g
            assert combined_labels_2.size(0) == n_pos_out_g + n_neg_in_g
            assert combined_logits_2.dim() == 1
            assert combined_labels_2.dim() == 1
            
            auroc_2_metric.update(combined_logits_2, combined_labels_2)
            auroc_2 = auroc_2_metric.compute().item()
            
            assert 0.0 <= auroc_2 <= 1.0, f"AUROC_2={auroc_2} out of valid range [0,1]"
        else:
            auroc_2 = None
        
        # Compute avg and diff if both AUROCs are valid
        if auroc_1 is not None and auroc_2 is not None:
            avg_g = (auroc_1 + auroc_2) / 2
            diff_g = abs(auroc_1 - auroc_2)
            
            assert 0.0 <= avg_g <= 1.0, f"avg_g={avg_g} out of valid range [0,1]"
            assert 0.0 <= diff_g <= 1.0, f"diff_g={diff_g} out of valid range [0,1]"
            
            group_avgs.append(avg_g)
            group_diffs.append(diff_g)
            
            details[f'group_{g_val}'] = {
                'auroc_pos_in_neg_out': auroc_1,
                'auroc_pos_out_neg_in': auroc_2,
                'average': avg_g,
                'abs_diff': diff_g,
                'n_pos_in': n_pos_in_g,
                'n_neg_in': n_neg_in_g,
                'n_pos_out': n_pos_out_g,
                'n_neg_out': n_neg_out_g,
                'n_total_in_group': n_in_group
            }
            
            logging.info(
                f"Group {g_val} (n={n_in_group}): AUROC_1={auroc_1:.4f}, AUROC_2={auroc_2:.4f}, "
                f"avg={avg_g:.4f}, diff={diff_g:.4f}"
            )
        else:
            logging.warning(
                f"Group {g_val} (n={n_in_group}): Insufficient samples for AUROC computation "
                f"(pos_in={n_pos_in_g}, neg_out={n_neg_out_g}, pos_out={n_pos_out_g}, neg_in={n_neg_in_g})"
            )
            details[f'group_{g_val}'] = {
                'auroc_pos_in_neg_out': auroc_1,
                'auroc_pos_out_neg_in': auroc_2,
                'note': 'insufficient_samples',
                'n_pos_in': n_pos_in_g,
                'n_neg_in': n_neg_in_g,
                'n_pos_out': n_pos_out_g,
                'n_neg_out': n_neg_out_g,
                'n_total_in_group': n_in_group
            }
    
    if len(group_avgs) == 0:
        logging.error("No valid group AUROCs computed! All groups had insufficient samples.")
        details['error'] = 'no_valid_group_aurocs'
        return 0.0, details
    
    assert len(group_avgs) == len(group_diffs), \
        f"Mismatch: {len(group_avgs)} avgs vs {len(group_diffs)} diffs"
    assert all(0.0 <= avg <= 1.0 for avg in group_avgs), "Invalid average values"
    assert all(0.0 <= diff <= 1.0 for diff in group_diffs), "Invalid diff values"
    
    # Final score
    mean_avg = np.mean(group_avgs)
    mean_diff = np.mean(group_diffs)
    score = mean_avg - mean_diff
    
    assert isinstance(mean_avg, (float, np.floating)), f"mean_avg type is {type(mean_avg)}"
    assert isinstance(mean_diff, (float, np.floating)), f"mean_diff type is {type(mean_diff)}"
    assert 0.0 <= mean_avg <= 1.0, f"mean_avg={mean_avg} out of range"
    assert 0.0 <= mean_diff <= 1.0, f"mean_diff={mean_diff} out of range"
    
    details['summary'] = {
        'mean_average': float(mean_avg),
        'mean_abs_diff': float(mean_diff),
        'fairness_score': float(score),
        'n_valid_groups': len(group_avgs),
        'n_total_groups': n_groups,
        'n_valid_samples': n_valid,
        'n_missing_samples': n_missing
    }
    
    logging.info(
        f"Fairness Score: {score:.4f} (mean_avg={mean_avg:.4f}, mean_diff={mean_diff:.4f}, "
        f"valid_groups={len(group_avgs)}/{n_groups})"
    )
    
    return float(score), details