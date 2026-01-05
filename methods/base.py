import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
import torch
from ..config import ExperimentConfig

class BaseMethod(ABC):
    """
    Abstract base class for training methods.
    
    This class enforces a standard interface so the training loop 
    doesn't need to know the details of the specific algorithm (Standard vs SupCon).
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
        self.metrics = {}

    @abstractmethod
    def get_model_components(self, num_features: int) -> tuple[nn.Module, Optional[nn.Module]]:
        """
        Constructs and returns the specific neural network heads required by the method.
        
        Args:
            num_features: The output dimension of the encoder (e.g., 1024 for DenseNet121).
            
        Returns:
            Tuple[nn.Module, Optional[nn.Module]]: (classifier, projection_head)
        """
        pass

    @abstractmethod
    def compute_loss(self, 
                     model_output: tuple[torch.Tensor, Optional[torch.Tensor]], 
                     targets: torch.Tensor, 
                     extra_info: Optional[dict] = None
                     ) -> torch.Tensor:
        """
        Calculates the total loss for the batch.
        
        Args:
            model_output: The tuple returned by the model forward pass (logits, projections).
            targets: The ground truth labels.
            extra_info: Any additional information (e.g., drain) to consider for loss computation.

        Returns:
            torch.Tensor: The final scalar loss to backward on.
        """
        pass