from .standard import StandardMethod

class JTTMethod(StandardMethod):
    """
    Just Train Twice (JTT) Method.
    
    From a loss perspective, JTT is identical to Standard training (ERM)
    in both stages. It uses Binary Cross Entropy.
    
    We create this class to maintain consistency with the `methods` factory pattern.
    """
    def __init__(self, config):
        super().__init__(config)
