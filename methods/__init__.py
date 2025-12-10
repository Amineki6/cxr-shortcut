from .standard import StandardMethod
from .supcon import SupConMethod

def get_method(method_name, config):
    """Factory function to initialize the correct strategy."""
    if method_name == "standard":
        return StandardMethod(config)
    elif method_name == "supcon":
        return SupConMethod(config)
    else:
        raise ValueError(f"Method '{method_name}' not implemented.")