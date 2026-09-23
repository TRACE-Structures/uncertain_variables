''' This module implements various probability distributions for uncertainty quantification.

    Distribution(ABC)
        Abstract base class for probability distributions.
        
    TranslatedDistribution(Distribution)
        Class for translated and scaled distributions.
        
    NormalDistribution(Distribution)
        Class for normal (Gaussian) distribution.
        
    UniformDistribution(Distribution)
        Class for uniform distribution.
        
    LogNormalDistribution(Distribution)
        Class for log-normal distribution.
        
    BetaDistribution(Distribution)
        Class for beta distribution.

    ExponentialDistribution(Distribution)
        Class for exponential distribution.

    WignerSemicircleDistribution(Distribution)
        Class for Wigner semicircle distribution.
        
    unwrap_if_scalar(arr)
        Utility function to unwrap single-element arrays.'''

from .Distribution import *
from .NormalDistribution import *
from .TranslatedDistribution import *
from .BetaDistribution import *
from .ExponentialDistribution import *
from .LogNormalDistribution import *
from .SemiCircleDistribution import *
from .UniformDistribution import *

def get_mu_and_sigma_of_lognormal(mean, cov):
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma