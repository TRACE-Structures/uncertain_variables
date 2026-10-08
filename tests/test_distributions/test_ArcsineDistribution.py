import numpy as np
import pytest

from uncertain_variables.distributions import ArcsineDistribution
from uncertain_variables.polysys import ChebyshevTPolynomials

from .TestDistribution import TestDistribution as DistributionTestClass


class TestArcsineDistribution(DistributionTestClass):
    KURT_IS_EXCESS = True

    @pytest.fixture
    def dist(self):
        return ArcsineDistribution()

    @pytest.fixture
    def support(self):
        return -1, 1

    def test_type_parameters_and_equality(self):
        dist = ArcsineDistribution()
        assert dist.get_dist_type() == "arcsin"
        assert dist.get_dist_params() == ()
        assert dist == ArcsineDistribution()
        assert dist != object()

    def test_repr(self):
        assert repr(ArcsineDistribution()) == "Arcsine()"

    def test_pdf_values_and_support(self, dist):
        assert np.isclose(dist.pdf(0), 1 / np.pi)
        assert np.isinf(dist.pdf(-1))
        assert np.isinf(dist.pdf(1))
        assert np.allclose(dist.pdf([-2, 2]), [0, 0])

    def test_cdf_and_invcdf_reference_values(self, dist):
        assert np.allclose(dist.cdf([-2, -1, 0, 1, 2]), [0, 0, 0.5, 1, 1])
        assert np.allclose(dist.invcdf([0, 0.5, 1]), [-1, 0, 1])
        assert np.isnan(dist.invcdf(-0.1))
        assert np.isnan(dist.invcdf(1.1))

    def test_moments_reference_values(self, dist):
        assert np.allclose(dist.moments(), [0, 0.5, 0, -1.5])

    def test_base_distribution_is_arcsine(self, dist):
        assert dist.get_base_dist() is dist
        values = np.linspace(-1, 1, 21)
        assert np.allclose(dist.base2dist(values), values)
        assert np.allclose(dist.dist2base(values), values)

    def test_orthogonal_polynomial_system_is_chebyshev_t(self, dist):
        assert isinstance(dist.orth_polysys(), ChebyshevTPolynomials)
        assert dist.orth_polysys_syschar(False) == "T"
        assert dist.orth_polysys_syschar(True) == "t"