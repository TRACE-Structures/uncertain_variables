"""Probability distributions for uncertainty quantification.

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

    SemiCircleDistribution(Distribution)
        Class for Wigner semicircle distribution.

    ArcsineDistribution(Distribution)
        Class for arcsine distribution.

    get_mu_and_sigma_of_lognormal(mean, cov)
        Convert a log-normal arithmetic mean and coefficient of variation
        to the parameters of the underlying normal distribution.
"""

from .Distribution import *
from .ArcsineDistribution import *
from .NormalDistribution import *
from .TranslatedDistribution import *
from .BetaDistribution import *
from .ExponentialDistribution import *
from .LogNormalDistribution import *
from .SemiCircleDistribution import *
from .UniformDistribution import *

def get_mu_and_sigma_of_lognormal(mean, cov):
    """Return underlying normal parameters for a log-normal mean and coefficient of variation.

    Parameters
    ----------
    mean : float
        Arithmetic mean of the log-normal distribution.

    cov : float
        Coefficient of variation of the log-normal distribution.

    Returns
    -------
    mu, sigma : tuple of float
        Mean and standard deviation of the underlying normal distribution.
    """
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma