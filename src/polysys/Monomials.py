import numpy as np

from .PolynomialSystem import PolynomialSystem


class MonomialPolynomials(PolynomialSystem):
    """ 
    Implementation of Monomial polynomial system.
    """
    def gpc_distribution(self):
        raise NotImplementedError("Monomials do not have a weighting distribution.")

    def _evaluate(self, deg, x, derivative=False):
        if derivative:
            return np.column_stack(
                [np.zeros_like(x)] + [n * x**(n - 1) for n in range(1, deg + 1)]
            )
        return np.column_stack([x**n for n in range(deg + 1)])

    def _sqnorm(self, deg):
        raise NotImplementedError("Monomials do not have a weighting distribution.")

# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from .NormalizedPolynomials import NormalizedPolynomials

# class Monomials(PolynomialSystem):
#     @classmethod
#     def normalized(self):
#         """ Return normalized version of Monomial polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized Monomial polynomial system."""
        
#         polysys = NormalizedPolynomials(self)
#         return polysys

#     @staticmethod
#     def recur_coeff(deg):

#         n = np.arange(deg)
#         one = np.ones_like(n)
#         zero = np.zeros_like(n)
#         r = np.column_stack((zero, one, zero))

#         return r

#     @staticmethod
#     def sqnorm(self, n):

#         raise NotImplementedError("Monomials do not have a weighting distribution.")

#     @staticmethod
#     def weighting_dist():
#         raise NotImplementedError("Monomials do not have a weighting distribution.")
