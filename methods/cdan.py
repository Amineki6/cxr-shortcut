import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMethod

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: Identity
    Backward: Negates gradient and scales it by lambda.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class Discriminator(nn.Module):
    """
    Discriminator for CDAN.
    """
    def __init__(self, input_dim=2048, hidden_dim=1024, grl_lambda=1.0):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.grl_lambda = grl_lambda
        self.requires_logits = True

    def forward(self, inputs):
        features, logits = inputs
        
        # Apply GRL to features BEFORE they mix with logits for the discriminator
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)
            g = torch.cat([1 - prob, prob], dim=1) # (B, 2)
        else:
            g = torch.softmax(logits, dim=1)

        # Multilinear Conditioning: h = f \otimes g
        # (B, F, 1) * (B, 1, C) -> (B, F, C) -> flatten -> (B, F*C)
        h = torch.bmm(features.unsqueeze(2), g.unsqueeze(1))
        h = h.view(features.size(0), -1) # (B, 2048)
        
        # Apply GRL
        h_grl = GradientReversalLayer.apply(h, self.grl_lambda)

        return self.layer(h_grl)

class CDANMethod(BaseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.bce_loss_none = nn.BCEWithLogitsLoss(reduction='none') # For entropy weighting
        self.cdan_lambda = getattr(config, 'cdan_lambda', 1.0)
        self.cdan_entropy = getattr(config, 'cdan_entropy', False)

    def get_model_components(self, num_features: int):
        # 1. Main Classifier
        classifier = nn.Linear(num_features, 1)
        
        # 2. Discriminator (Input size: num_features * 2)
        discriminator = Discriminator(
            input_dim=num_features * 2, 
            grl_lambda=self.cdan_lambda
        )
        
        return classifier, discriminator

    def compute_loss(self, model_output, targets, extra_info=None):
        logits, discriminator_out = model_output
        
        # 1. Classification Loss
        # Ensure targets are float for BCE
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Match shapes for BCE: logits [B, 1] vs targets [B, 1]
        cls_loss = self.bce(logits, targets.float())

        # 2. Entropy Calculation for Weighting
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)
            g = torch.cat([1 - p, p], dim=1)
        else:
            g = torch.softmax(logits, dim=1)
        
        epsilon = 1e-6
        # H(g) = - sum p log p
        entropy = -torch.sum(g * torch.log(g + epsilon), dim=1)
        
        if self.cdan_entropy:
            entropy_weights = 1.0 + torch.exp(-entropy)
        else:
            entropy_weights = torch.ones_like(entropy)
            
        # 3. Domain Adversarial Loss
        if extra_info is None or 'drain' not in extra_info:
             # Fallback
             self.metrics = {"bce": cls_loss.item(), "cdan_disc": 0.0}
             return cls_loss
             
        domain_labels = extra_info['drain'].float() 
        
        # --- FIX: Ensure domain labels match discriminator output shape ---
        # discriminator_out is [B, 1], domain_labels might be [B]
        domain_labels = domain_labels.view_as(discriminator_out)

        disc_loss_unweighted = self.bce_loss_none(discriminator_out, domain_labels)
        
        # Weight by entropy
        disc_loss = (disc_loss_unweighted.squeeze() * entropy_weights).mean()
        
        total_loss = cls_loss + disc_loss

        # Store metrics
        self.metrics = {
            "bce": cls_loss.item(),
            "cdan_disc": disc_loss.item()
        }
        
        return total_loss