"""Tests for the ExponentialDistribution class."""

import numpy as np
import pytest

from uncertain_variables.polysys.LaguerrePolynomials import LaguerrePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from uncertain_variables.distributions import NormalDistribution

from uncertain_variables.distributions import ExponentialDistribution


LAMBDA = 2


class TestExponentialDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return ExponentialDistribution(1)

    @pytest.fixture
    def support(self):
        return 0, np.inf

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = ExponentialDistribution(LAMBDA)
        assert np.isclose(d.lambda_, LAMBDA, atol=self.ATOL, rtol=self.RTOL)
    
    def test_init_defaults_to_one(self):
        d = ExponentialDistribution()
        assert np.isclose(d.lambda_, 1, atol=self.ATOL, rtol=self.RTOL)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "exp"

    def test_get_dist_params(self, dist):
        assert np.isclose(dist.get_dist_params(), 1, atol=self.ATOL, rtol=self.RTOL)

    def test_eq(self):
        d1 = ExponentialDistribution(LAMBDA)
        d2 = ExponentialDistribution(LAMBDA)
        d3 = ExponentialDistribution(1)
        d4 = NormalDistribution(0, 1)
        assert d1 == d2
        assert d1 != d3
        assert d3 != d4

    def test_repr(self):
        d = ExponentialDistribution(LAMBDA)
        assert repr(d) == "Exp({})".format(LAMBDA)

    # =====================================================================
    # 2. Mathematical identities  (exponential-specific parts)
    # =====================================================================
    def test_pdf_is_zero_below_support(self, dist):
        x = np.array([-1.0, -5.0])
        pdf_values = dist.pdf(x)
        assert np.allclose(pdf_values, 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_zero_below_support(self, dist):
        x = np.array([-1.0, -5.0])
        cdf_values = dist.cdf(x)
        assert np.allclose(cdf_values, 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_at_zero_is_zero(self, dist):
        assert np.isclose(dist.cdf(0), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_mean_is_reciprocal_of_lambda(self):
        d = ExponentialDistribution(LAMBDA)
        assert np.isclose(d.mean(), 1 / LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    def test_variance_is_reciprocal_of_lambda_squared(self):
        d = ExponentialDistribution(LAMBDA)
        assert np.isclose(d.var(), 1 / LAMBDA ** 2, atol=self.ATOL, rtol=self.RTOL)

    def test_base_distribution_is_exponential_with_rate_one(self, dist):
        germ = dist.get_base_dist()
        assert isinstance(germ, ExponentialDistribution)
        assert np.isclose(germ.lambda_, 1, atol=self.ATOL, rtol=self.RTOL)

    def test_base2dist(self):
        d = ExponentialDistribution(LAMBDA)
        y = np.linspace(0, 10, 50)
        x = y / LAMBDA
        assert np.allclose(d.base2dist(y), x, atol=self.ATOL, rtol=self.RTOL)

    def test_dist2base(self):
        d = ExponentialDistribution(LAMBDA)
        x = np.linspace(0, 10, 50)
        y = d.dist2base(x)
        assert np.allclose(y, x * LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    def test_moments_reference_values(self):
        d = ExponentialDistribution(LAMBDA)
        assert np.allclose(d.moments(), [1 / LAMBDA, 1 / LAMBDA ** 2, 2, 6], atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_matches_with_scipy(self):
        from scipy.stats import expon
        d = ExponentialDistribution(LAMBDA)
        x = np.linspace(0, 10, 50)
        expected_pdf = expon.pdf(x, scale=1 / LAMBDA)
        assert np.allclose(d.pdf(x), expected_pdf, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_matches_with_scipy(self):
        from scipy.stats import expon
        d = ExponentialDistribution(LAMBDA)
        x = np.linspace(0, 10, 50)
        expected_cdf = expon.cdf(x, scale=1 / LAMBDA)
        assert np.allclose(d.cdf(x), expected_cdf, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_matches_with_scipy(self):
        from scipy.stats import expon
        d = ExponentialDistribution(LAMBDA)
        q = np.linspace(0.01, 0.99, 50)
        expected_invcdf = expon.ppf(q, scale=1 / LAMBDA)
        assert np.allclose(d.invcdf(q), expected_invcdf, atol=self.ATOL, rtol=self.RTOL)

    def test_orth_polysys_is_laguerre(self):
        d1 = ExponentialDistribution(1)
        d2 = ExponentialDistribution(LAMBDA)
        assert isinstance(d1.orth_polysys(), LaguerrePolynomials)
        with pytest.raises(Exception):
            d2.orth_polysys()

    def test_orth_polysys_syschar(self):
        d1 = ExponentialDistribution(1)
        d2 = ExponentialDistribution(LAMBDA)
        assert d1.orth_polysys_syschar(True) == "l"
        assert d1.orth_polysys_syschar(False) == "L"
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(True)
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(False)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize("lambda_", [0, -1])
    def test_invalid_parameters_raise_value_error(self, lambda_):
        with pytest.raises(ValueError):
            ExponentialDistribution(lambda_)
