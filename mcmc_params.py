from dataclasses import dataclass

@dataclass
class MCMC_Parameters:

    # Prior distribution parameters
    sigma2_l1: float = 1
    sigma2_l2: float = 1
    sigma2_kappa: float = 1

    # Proposal distribution parameters
    sigma_prop_l1: float = 0.2
    sigma_prop_l2: float = 0.01
    sigma_prop_kappa: float = 0.2
    sigma_prop_nu: float = 0.1

    # MCMC parameters
    N_thetap: int = 100 
    discard: int = 0
    N_theta: int = 300 
    tqdm_disable: bool = True

    # Initial sample
    l10: float = 0.5
    l20: float = 0.3
    K: int = 2
    
    bridging: bool = True
    Geweke: bool = False
