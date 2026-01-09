from .standard import StandardMethod
from .supcon import SupConMethod
from .mmd import MMDMethod
from .cdan import CDANMethod
from .score_matching import ScoreMatchingMethod
from .score_matching_dataset import DatasetScoreMatchingMethod
from .jtt import JTTMethod

def get_method(method_name, config, val_set_size=None):
    """Factory function to initialize the correct strategy."""
    if method_name == "standard":
        return StandardMethod(config)
    elif method_name == "supcon":
        return SupConMethod(config)
    elif method_name == 'mmd':
        return MMDMethod(config)
    elif method_name == 'cdan':
        return CDANMethod(config)
    elif method_name == 'score_matching':
        return ScoreMatchingMethod(config)
    elif method_name == 'dataset_score_matching':
        return DatasetScoreMatchingMethod(config, val_set_size)    
    elif method_name == 'jtt':
        return JTTMethod(config)
    else:
        raise ValueError(f"Method '{method_name}' not implemented.")