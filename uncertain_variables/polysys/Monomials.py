import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials

class Monomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        """ Return normalized version of Monomial polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized Monomial polynomial system."""
        
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):

        n = np.arange(deg)
        one = np.ones_like(n)
        zero = np.zeros_like(n)
        r = np.column_stack((zero, one, zero))

        return r

    @staticmethod
    def sqnorm(self, n):

        raise NotImplementedError("Monomials do not have a weighting distribution.")

    @staticmethod
    def weighting_dist():
        raise NotImplementedError("Monomials do not have a weighting distribution.")
