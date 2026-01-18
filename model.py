import torch
import torch.nn as nn
from torchvision.models import densenet121
from typing import Optional
from methods.base import BaseMethod

class CXP_Model(nn.Module):
    """
    Dynamic model wrapper for CheXpert Pneumothorax detection.
    
    The architecture (classifier and optional projection heads) is determined 
    by the 'method_strategy' passed during initialization.
    """
    def __init__(self, method_strategy: BaseMethod):
        """
        Args:
            method_strategy: An instance of a class inheriting from methods.BaseMethod.
                             (e.g., StandardMethod or SupConMethod)
        """
        super().__init__()
        
        # 1. Load the Backbone (DenseNet121)
        #self.encoder = densenet121(weights='IMAGENET1K_V1')
        model_size = 'tiny'
        self.encoder = torch.hub.load('../dinov3', f'dinov3_convnext_{model_size}',
                                weights=f"/dino/dinov3_convnext_{model_size}_pretrain_lvd1689m-21b726bb.pth",
                                source='local', trust_repo=True, skip_validation=True)
        
        # Get the feature dimension (1024 for DenseNet121)
        #num_features = self.encoder.classifier.in_features
        num_features = 768
        
        # Remove the original classification head so we get raw features
        self.encoder.classifier = nn.Identity()
        
        # 2. Ask the Strategy for the required heads
        # This returns the classifier and, optionally, the projection head
        self.clf, self.projection_head = method_strategy.get_model_components(num_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Extract features from backbone
        features = self.encoder(x)
        
        # Pass through classifier (Always exists)
        logits = self.clf(features)
        
        # Pass through projection head (Only if the strategy requested it)
        if self.projection_head is not None:
            if getattr(self.projection_head, 'requires_logits', False):
                 projections = self.projection_head((features, logits))
            else:
                 projections = self.projection_head(features)
        else:
            projections = None
            
        return logits, projections
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper for inference to get probabilities directly.
        """
        logits, _ = self(x)
        return torch.sigmoid(logits)