"""Tests for the NormalDistribution class."""

import numpy as np
import pytest

from uncertain_variables.polysys.HermitePolynomials import HermitePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from uncertain_variables.distributions import UniformDistribution

from uncertain_variables.distributions import NormalDistribution


MU, SIGMA = 2, 5


class TestNormalDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return NormalDistribution(0, 1)

    @pytest.fixture
    def support(self):
        return -np.inf, np.inf

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.allclose((d.mu, d.sigma), (MU, SIGMA), atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults_to_standard_normal(self):
        d = NormalDistribution()
        assert np.allclose((d.mu, d.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "norm"

    def test_get_dist_params(self, dist):
        assert np.allclose(dist.get_dist_params(), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_eq(self):
        d1 = NormalDistribution(MU, SIGMA)
        d2 = NormalDistribution(MU, SIGMA)
        d3 = NormalDistribution(0, 1)
        d4 = UniformDistribution(0, 1)
        assert d1 == d2
        assert d1 != d3
        assert d3 != d4

    def test_repr(self):
        d = NormalDistribution(MU, SIGMA)
        assert repr(d) == "N({}, {:.2f})".format(MU, SIGMA ** 2)

    def test_translate_returns_normal(self, dist):
        translated = dist.translate(1, 2)
        assert isinstance(translated, NormalDistribution)
        assert np.allclose((translated.mu, translated.sigma), (1, 2), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 2. Mathematical identities  (normal-specific parts)
    # =====================================================================
    def test_pdf_is_symmetric_about_the_mean(self, dist):
        d = NormalDistribution(MU, SIGMA)
        x = np.linspace(0, 10, 101)
        assert np.allclose(d.pdf(d.mu - x), d.pdf(d.mu + x), atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_peak_is_at_the_mean(self):
        d = NormalDistribution(MU, SIGMA)
        x = np.linspace(d.mu - 10, d.mu + 10, 201)
        assert np.isclose(x[np.argmax(d.pdf(x))], d.mu, atol=0.1)

    def test_cdf_at_mean_is_one_half(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.isclose(d.cdf(d.mu), 0.5, atol=self.ATOL, rtol=self.RTOL)

    def test_mean_equals_mu(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.isclose(d.mean(), MU, atol=self.ATOL, rtol=self.RTOL)

    def test_variance_equals_sigma_squared(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.isclose(d.var(), SIGMA ** 2, atol=self.ATOL, rtol=self.RTOL)

    def test_base_distribution_is_standard_normal(self, dist):
        germ = dist.get_base_dist()
        assert isinstance(germ, NormalDistribution)
        assert np.allclose((germ.mu, germ.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_base2dist(self):
        d = NormalDistribution(MU, SIGMA)
        germ = d.get_base_dist()
        y = np.linspace(-3, 3, 50)
        x = MU + SIGMA * y
        assert np.allclose(d.base2dist(y), x, atol=self.ATOL, rtol=self.RTOL)

    def test_dist2base(self):
        d = NormalDistribution(MU, SIGMA)
        y = np.linspace(-3, 3, 50)
        x = d.base2dist(y)
        assert np.allclose(d.dist2base(x), y, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    def test_moments_reference_values(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.allclose(d.moments(), [MU, SIGMA ** 2, 0, 0], atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_matches_with_scipy(self):
        from scipy.stats import norm
        d = NormalDistribution(MU, SIGMA)
        x = np.linspace(d.mu - 5 * d.sigma, d.mu + 5 * d.sigma, 50)
        expected_pdf = norm.pdf(x, loc=d.mu, scale=d.sigma)
        assert np.allclose(d.pdf(x), expected_pdf, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_matches_with_scipy(self):
        from scipy.stats import norm
        d = NormalDistribution(MU, SIGMA)
        x = np.linspace(d.mu - 5 * d.sigma, d.mu + 5 * d.sigma, 50)
        expected_cdf = norm.cdf(x, loc=d.mu, scale=d.sigma)
        assert np.allclose(d.cdf(x), expected_cdf, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_matches_with_scipy(self):
        from scipy.stats import norm
        d = NormalDistribution(MU, SIGMA)
        q = np.linspace(0.01, 0.99, 50)
        expected_invcdf = norm.ppf(q, loc=d.mu, scale=d.sigma)
        assert np.allclose(d.invcdf(q), expected_invcdf, atol=self.ATOL, rtol=self.RTOL)

    def test_orth_polysys_is_hermite(self):
        d1 = NormalDistribution(0, 1)
        d2 = NormalDistribution(MU, SIGMA)
        assert isinstance(d1.orth_polysys(), HermitePolynomials)
        with pytest.raises(Exception):
            d2.orth_polysys()

    def test_orth_polysys_syschar(self):
        d1 = NormalDistribution(0, 1)
        d2 = NormalDistribution(MU, SIGMA)
        assert d1.orth_polysys_syschar(False) == "H"
        assert d1.orth_polysys_syschar(True) == "h"
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(False)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
            "mu,sigma", 
            [
                (0, 0), 
                (0, -1),
                (np.inf, 1),
                (0, np.inf),
                (np.nan, 1),
                (0, np.nan)
            ]
        )
    def test_invalid_parameters_raise_assertion_error(self, mu, sigma):
        with pytest.raises(ValueError):
            NormalDistribution(mu, sigma)
