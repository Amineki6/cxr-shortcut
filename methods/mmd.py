import torch
import torch.nn as nn
from .base import BaseMethod

class MMDLoss(nn.Module):
    """
    Self-contained implementation of Maximum Mean Discrepancy (MMD) 
    using Multi-Kernel Gaussian RBF, similar to the logic in pytorch-adapt.
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        # 1. Compute L2 distance matrix (Equivalent to c_f.LpDistance)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        # 2. Bandwidth selection (Heuristic to find optimal sigma)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
            
        # 3. Multi-Kernel: Create a list of bandwidths
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        # 4. Compute Kernel Matrix (Sum of multiple Gaussians)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)

        batch_size_source = int(source.size()[0])
        
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        # Break kernel matrix into blocks: XX, YY, XY, YX
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        
        # MMD Statistic: Mean(XX) + Mean(YY) - 2 * Mean(XY)
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss

class MMDMethod(BaseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.mmd_loss = MMDLoss()
        # Default lambda is 1.0 if not specified in config
        self.lambda_val = getattr(config, 'mmd_lambda', 1.0)

    def get_model_components(self, num_features: int):
        """
        Returns the Classifier and an Identity Projection.
        We use Identity because MMD must be applied to the same features 
        used for classification to enforce invariance.
        """
        # 1. Classifier
        clf = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # 2. Projection (Identity passes raw features)
        proj = nn.Identity()
        
        return clf, proj

    def compute_loss(self, model_output, targets, extra_info=None):
        """
        Calculates Total Loss = BCE + (Lambda * MMD)
        Uses 'extra_info' to access the Drain labels.
        """
        logits, features = model_output
        
        # 1. Classification Loss (Standard)
        bce_loss = self.bce(logits.view(-1), targets.float())
        
        # 2. MMD Loss (Domain Alignment)
        mmd_val = torch.tensor(0.0, device=logits.device)
        
        if extra_info and 'drain' in extra_info:
            drain = extra_info['drain']
            
            # Split features: Domain 0 (No Drain) vs Domain 1 (Drain)
            features_no_drain = features[drain == 0]
            features_drain = features[drain == 1]
            
            mmd_val = self.mmd_loss(features_no_drain, features_drain)

        total_loss = bce_loss + (self.lambda_val * mmd_val)
        
        # Store for logging
        self.metrics = {
            "bce": bce_loss.item(),
            "mmd": mmd_val.item()
        }
        
        return total_loss