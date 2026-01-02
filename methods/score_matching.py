import torch
import torch.nn as nn
from .base import BaseMethod


class ScoreMatchingLoss(nn.Module):

    def __init__(self, min_subgroup_count=1):
        super(ScoreMatchingLoss, self).__init__()
        self.min_subgroup_count = min_subgroup_count

    def forward(probs, labels, groups):
        """
        Penalizes variance in average predicted scores across groups, separately
        for positive and negative class samples.
        
        For each group g:
            - Compute E[pred | g, y=1] (avg score for positive samples) if sufficient samples
            - Compute E[pred | g, y=0] (avg score for negative samples) if sufficient samples
        
        Loss: avg(var(E[pred | g, y=1]), var(E[pred | g, y=0]))
            (average of positive-class variance and negative-class variance)
            Returns only available variance if one class has <2 valid groups.
        
        Args:
            probs: (B,) or (B, 1) predicted probabilities for positive class [0, 1]
            labels: (B,) true binary labels (0 or 1)
            groups: (B,) group indicators (any integer values)
            min_subgroup_count: Minimum number of examples required for each 
                            (label, group) combination. A group-label subgroup
                            is only included if it meets this threshold.
                            Default: 1
                
        Returns: 
            Scalar tensor. Returns:
            - Average of pos and neg variance if both have >=2 valid groups
            - Only pos variance if neg has <2 valid groups
            - Only neg variance if pos has <2 valid groups
            - Detached 0.0 if both have <2 valid groups (no gradients)
            
        Edge cases:
            - If min_subgroup_count < 1: treated as 1 (no filtering)
            - If a (group, label) has <min_subgroup_count examples: that subgroup excluded
            - Subgroups are evaluated independently per class
        """
        
        # 1. Shape validation and normalization
        assert probs.dim() in [1, 2], f"Probs must be 1D (B,) or 2D (B, 1), received {probs.shape}"
        if probs.dim() == 2:
            assert probs.shape[1] == 1, f"2D Probs must have shape (B, 1), received {probs.shape}"
            probs = probs.squeeze(1)
        
        assert labels.dim() == 1 and labels.shape[0] == probs.shape[0], \
            f"Labels shape {labels.shape} must match probs {probs.shape}"
        assert groups.dim() == 1 and groups.shape[0] == probs.shape[0], \
            f"Groups shape {groups.shape} must match probs {probs.shape}"
        
        # 2. Value validation
        assert labels.dtype in [torch.int64, torch.int32, torch.uint8, torch.bool], \
            f"Labels must be integer/bool type, received {labels.dtype}"
        assert labels.max() <= 1 and labels.min() >= 0, "Labels must be binary (0 or 1)"
        assert probs.min() >= 0.0 and probs.max() <= 1.0, \
            f"Probs must be in [0, 1], got [{probs.min():.3f}, {probs.max():.3f}]"
        
        # 3. Clamp min_subgroup_count to valid range (no error, just silent correction)
        min_subgroup_count = max(1, int(min_subgroup_count))
        
        # 4. Compute average scores per group per class (with independent subgroup filtering)
        unique_groups = groups.unique()
        group_pos_avgs = []
        group_neg_avgs = []
        
        for g in unique_groups:
            mask = (groups == g)
            group_labels = labels[mask]
            group_probs = probs[mask]
            
            # Check positive subgroup count
            pos_mask = (group_labels == 1)
            n_positive = pos_mask.sum().item()
            
            if n_positive >= min_subgroup_count:
                mean_pos_score = group_probs[pos_mask].mean()
                assert mean_pos_score.dim() == 0, "Mean positive score must be scalar"
                group_pos_avgs.append(mean_pos_score)
            
            # Check negative subgroup count (independently)
            neg_mask = (group_labels == 0)
            n_negative = neg_mask.sum().item()
            
            if n_negative >= min_subgroup_count:
                mean_neg_score = group_probs[neg_mask].mean()
                assert mean_neg_score.dim() == 0, "Mean negative score must be scalar"
                group_neg_avgs.append(mean_neg_score)
        
        # 5. Compute variances based on available valid subgroups
        pos_variance = None
        neg_variance = None
        
        if len(group_pos_avgs) >= 2:
            group_pos_avgs = torch.stack(group_pos_avgs)
            pos_variance = group_pos_avgs.var()
            assert pos_variance.dim() == 0, "Positive variance must be scalar"
        
        if len(group_neg_avgs) >= 2:
            group_neg_avgs = torch.stack(group_neg_avgs)
            neg_variance = group_neg_avgs.var()
            assert neg_variance.dim() == 0, "Negative variance must be scalar"
        
        # 6. Return based on what's available
        if pos_variance is not None and neg_variance is not None:
            # Both available: return average
            total_loss = (pos_variance + neg_variance) / 2.0
        elif pos_variance is not None:
            # Only positive variance available
            total_loss = pos_variance
        elif neg_variance is not None:
            # Only negative variance available
            total_loss = neg_variance
        else:
            # Neither available: return detached zero (no gradients)
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype).detach()
        
        assert total_loss.dim() == 0, "Final loss must be scalar"
        
        return total_loss


class ScoreMatchingMethod(BaseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.score_matching_loss = ScoreMatchingLoss(
            min_subgroup_count=getattr(config, 'score_matching_min_subgroup_count', 1))
        
        # Default lambda is 1.0 if not specified in config
        self.lambda_val = getattr(config, 'score_matching_lambda', 1.0)

    def compute_loss(self, model_output, targets, extra_info=None):
        """
        Calculates Total Loss = BCE + Lambda * ScoreMatchingLoss
        Uses 'extra_info' to access the Drain labels.
        """
        logits, features = model_output
        
        # 1. Classification Loss (Standard)
        bce_loss = self.bce(logits.view(-1), targets.float())
        
        # 2. MMD Loss (Domain Alignment)
        score_matching_val = self.score_matching_loss(probs=torch.sigmoid(logits),
                                                      labels=targets,
                                                      groups=extra_info['drain'])

        total_loss = bce_loss + self.lambda_val * score_matching_val
        
        # Store for logging
        self.metrics = {
            "bce": bce_loss.item(),
            "score_matching_loss": score_matching_val.item()
        }
        
        return total_loss