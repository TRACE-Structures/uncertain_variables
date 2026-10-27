"""Tests for the NormalDistribution class."""

import numpy as np
import pytest
from scipy.stats import norm

from src.polysys.HermitePolynomials import HermitePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import UniformDistribution

from src.distributions import NormalDistribution


MU, SIGMA = 2, 5


class TestNormalDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return NormalDistribution(MU, SIGMA)

    @pytest.fixture
    def support(self):
        return -np.inf, np.inf

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = NormalDistribution(MU, SIGMA)
        assert np.allclose((d.mu, d.sigma, d.loc, d.scale), (MU, SIGMA, MU, SIGMA), atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults_to_standard_normal(self):
        d = NormalDistribution()
        assert np.allclose((d.mu, d.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "norm"

    @pytest.mark.parametrize("mu, sigma", [(0, 1), (MU, SIGMA), (-3.5, 0.25)])
    def test_get_dist_params(self, mu, sigma):
        params = NormalDistribution(mu, sigma).get_dist_params()
        assert np.asarray(params).shape == (2,)
        assert np.allclose(params, [mu, sigma], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert NormalDistribution(MU, SIGMA) == NormalDistribution(MU, SIGMA)

    @pytest.mark.parametrize("other", [(0, 1), (MU, SIGMA + 1), (MU + 1, SIGMA), (SIGMA, MU)])
    def test_eq_same_type_different_params(self, other):
        assert NormalDistribution(MU, SIGMA) != NormalDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert NormalDistribution(0, 1) != UniformDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert NormalDistribution(MU, SIGMA) != (MU, SIGMA)
        assert NormalDistribution(MU, SIGMA) != "N(2.00, 5.00)"

    @pytest.mark.parametrize(
        "mu, sigma, expected",
        [
            (MU, SIGMA, "N(2.00, 5.00)"),
            (0, 1, "N(0.00, 1.00)"),
            (-1.5, 0.125, "N(-1.50, 0.12)"),
        ],
    )
    def test_repr(self, mu, sigma, expected):
        assert repr(NormalDistribution(mu, sigma)) == expected

    # =====================================================================
    # 2. Mathematical identities (normal-specific parts)
    # =====================================================================
    def test_pdf_is_symmetric_about_the_mean(self, dist):
        x = np.linspace(0, 10, 101)
        assert np.allclose(dist.pdf(dist.mu - x), dist.pdf(dist.mu + x), atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_peak_is_at_the_mean(self, dist):
        x = np.linspace(dist.mu - 10, dist.mu + 10, 201)
        assert np.isclose(x[np.argmax(dist.pdf(x))], dist.mu, atol=0.1)

    def test_pdf_peak_value(self, dist):
        assert np.isclose(dist.pdf(MU), 1 / (SIGMA * np.sqrt(2 * np.pi)), atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_at_mean_is_one_half(self, dist):
        assert np.isclose(dist.cdf(dist.mu), 0.5, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_one_sigma_reference(self, dist):
        assert np.isclose(dist.cdf(MU + SIGMA) - dist.cdf(MU - SIGMA), 0.6826894921370859, atol=1e-9)

    def test_invcdf_extremes_are_infinite(self, dist):
        assert dist.invcdf(0) == -np.inf
        assert dist.invcdf(1) == np.inf

    def test_mean_equals_mu(self, dist):
        assert np.isclose(dist.mean(), MU, atol=self.ATOL, rtol=self.RTOL)

    def test_variance_equals_sigma_squared(self, dist):
        assert np.isclose(dist.var(), SIGMA ** 2, atol=self.ATOL, rtol=self.RTOL)

    def test_base2dist(self, dist):
        y = np.linspace(-3, 3, 50)
        assert np.allclose(dist.base2dist(y), MU + SIGMA * y, atol=self.ATOL, rtol=self.RTOL)

    def test_dist2base(self, dist):
        y = np.linspace(-3, 3, 50)
        assert np.allclose(dist.dist2base(dist.base2dist(y)), y, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values 
    # =====================================================================
    def test_moments_reference_values(self, dist):
        assert np.allclose(dist.moments(), [MU, SIGMA ** 2, 0, 0], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
        "mu, sigma",
        [(0, 0), (0, -1), (np.inf, 1), (-np.inf, 1), (0, np.inf), (np.nan, 1), (0, np.nan), ("a", 1), (0, "b"), (None, 1), (0, None)],
    )
    def test_invalid_parameters_raise_value_error(self, mu, sigma):
        with pytest.raises(ValueError):
            NormalDistribution(mu, sigma)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [(0, 1), (MU, SIGMA), (-3.5, 0.25), (100, 10), (0, 1e-3)]

    @pytest.mark.parametrize("mu, sigma", MOMENT_CASES)
    def test_mean_specific_cases(self, mu, sigma):
        assert np.isclose(NormalDistribution(mu, sigma).mean(), mu, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma", MOMENT_CASES)
    def test_var_specific_cases(self, mu, sigma):
        assert np.isclose(NormalDistribution(mu, sigma).var(), sigma ** 2, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma", MOMENT_CASES)
    def test_skew_specific_cases(self, mu, sigma):
        assert np.isclose(NormalDistribution(mu, sigma).skew(), 0.0, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma", MOMENT_CASES)
    def test_kurt_specific_cases(self, mu, sigma):
        assert np.isclose(NormalDistribution(mu, sigma).kurt(), 0.0, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("mu, sigma", [(0, 1), (MU, SIGMA), (-3, 0.5)])
    def test_sample_shape_and_finite_all_methods(self, method, mu, sigma):
        s = np.asarray(NormalDistribution(mu, sigma).sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(np.isfinite(s))

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, dist, method):
        s = np.asarray(dist.sample(4096, method=method, seed=7))
        assert abs(np.mean(s) - dist.mean()) < 0.5
        assert abs(np.std(s) - dist.sigma) < 0.5

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_is_reproducible_with_seed(self, dist, method):
        assert np.array_equal(dist.sample(32, method=method, seed=123), dist.sample(32, method=method, seed=123))

    def test_sample_mc_differs_for_different_seeds(self, dist):
        assert not np.array_equal(dist.sample(32, "MC", seed=1), dist.sample(32, "MC", seed=2))

    def test_sample_default_method_is_mc(self, dist):
        assert np.array_equal(dist.sample(16, seed=5), dist.sample(16, "MC", seed=5))

    @pytest.mark.parametrize("n", [0, -3, 2.5, "10", None])
    def test_sample_invalid_n_raises(self, dist, n):
        with pytest.raises(ValueError):
            dist.sample(n)

    @pytest.mark.parametrize("seed", [1.5, "abc"])
    def test_sample_invalid_seed_raises(self, dist, seed):
        with pytest.raises(ValueError):
            dist.sample(8, seed=seed)

    def test_sample_unknown_method_raises(self, dist):
        with pytest.raises(ValueError):
            dist.sample(8, method="unknown")

    # =====================================================================
    # 7. translate / get_shift / get_scale
    # =====================================================================
    @pytest.mark.parametrize("shift, scale", [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)])
    def test_translate_maps_parameters_affinely(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert isinstance(t, NormalDistribution)
        assert np.isclose(t.mu, scale * MU + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.sigma, scale * SIGMA, atol=self.ATOL, rtol=self.RTOL)

    def test_translate_standard_normal(self):
        t = NormalDistribution(0, 1).translate(1, 2)
        assert np.allclose((t.mu, t.sigma), (1, 2), atol=self.ATOL, rtol=self.RTOL)

    def test_translate_default_is_identity(self, dist):
        assert dist.translate() == dist

    def test_translate_does_not_modify_original(self, dist):
        dist.translate(3, 2)
        assert (dist.mu, dist.sigma) == (MU, SIGMA)

    def test_translate_composition(self, dist):
        t = dist.translate(1, 2).translate(-3, 0.5)
        expected = dist.translate(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(t.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "shift, scale",
        [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), ("a", 1), (0, "b"), (None, 1), (np.nan, 1)],
    )
    def test_translate_invalid_raises(self, dist, shift, scale):
        with pytest.raises(ValueError):
            dist.translate(shift, scale)

    @pytest.mark.parametrize("mu, sigma", [(0, 1), (MU, SIGMA), (-3.5, 0.25)])
    def test_get_shift_and_scale(self, mu, sigma):
        d = NormalDistribution(mu, sigma)
        assert np.isclose(d.get_shift(), mu, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), sigma, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_get_shift_scale_after_translate(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert np.isclose(t.get_scale(), scale * dist.get_scale(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.get_shift(), scale * dist.get_shift() + shift, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 8. fix_moments (verified by quadrature)
    # =====================================================================
    FIX_MOMENT_CASES = [(0, 1), (0.5, 1 / 12), (3.0, 2.0), (-5.0, 0.01), (100.0, 25.0), (-0.3, 7.5)]

    @pytest.mark.parametrize("mean, var", FIX_MOMENT_CASES)
    @pytest.mark.parametrize("start", [(0, 1), (MU, SIGMA)])
    def test_fix_moments_matches_by_quadrature(self, start, mean, var):
        new = NormalDistribution(*start).fix_moments(mean, var)
        assert isinstance(new, NormalDistribution)
        # the pdf is negligible beyond 12 sigma; a finite window keeps quad reliable
        lo, hi = mean - 12 * np.sqrt(var), mean + 12 * np.sqrt(var)
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_pdf_integrates_to_one(self):
        new = NormalDistribution(MU, SIGMA).fix_moments(1.0, 4.0)
        assert np.isclose(self._quad(new.pdf, -30, 32), 1.0, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)

    def test_fix_moments_defaults(self):
        new = NormalDistribution(MU, SIGMA).fix_moments()
        assert np.allclose((new.mu, new.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mean, var", [(0, 0), (0, -1), ("a", 1), (0, "b"), (np.inf, 1), (0, np.inf), (np.nan, 1)])
    def test_fix_moments_invalid_raises(self, mean, var):
        with pytest.raises(ValueError):
            NormalDistribution(MU, SIGMA).fix_moments(mean, var)

    # =====================================================================
    # 9. fix_bounds 
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles_use_fallback_bounds(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, NormalDistribution)
        assert np.allclose(new.get_bounds(), (new_min, new_max), atol=1e-8)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0.01, 0.99), (0.5, 0.7)])
    def test_fix_bounds_with_quantiles(self, dist, q0, q1):
        new = dist.fix_bounds(10, 20, q0, q1)
        assert np.isclose(new.invcdf(q0), 10, atol=1e-8)
        assert np.isclose(new.invcdf(q1), 20, atol=1e-8)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 1), {}),
            ((2, 1), {}),
            (("a", 1), {}),
            ((0, None), {}),
            ((0, 1), {"q0": -0.1}),
            ((0, 1), {"q0": 1.1}),
            ((0, 1), {"q0": 0.6, "q1": 0.4}),
            ((0, 1), {"q1": 1.5}),
        ],
    )
    def test_fix_bounds_invalid_raises(self, dist, args, kwargs):
        with pytest.raises(ValueError):
            dist.fix_bounds(*args, **kwargs)

    # =====================================================================
    # 10. GPC related functions
    # =====================================================================
    @pytest.mark.parametrize("mu, sigma", [(0, 1), (MU, SIGMA), (-3, 0.2)])
    def test_get_base_dist_is_standard_normal(self, mu, sigma):
        germ = NormalDistribution(mu, sigma).get_base_dist()
        assert isinstance(germ, NormalDistribution)
        assert germ == NormalDistribution(0, 1)

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
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(True)

    # =====================================================================
    # 11. Unit conversions
    # =====================================================================
    def test_convert_to_round_trip(self):
        dist = NormalDistribution(MU, SIGMA)
        old_params = dist.get_dist_params()
        dist.convert_to("m", "cm")
        dist.convert_to("cm", "m")
        new_params = dist.get_dist_params()
        assert np.allclose(old_params, new_params)

    def test_convert_to_exception(self):
        d = NormalDistribution(MU, SIGMA)
        d.convert_to("m", "cm")
        expected_params = [100*MU, 100*SIGMA]
        new_params = d.get_dist_params()
        assert np.allclose(new_params, expected_params)