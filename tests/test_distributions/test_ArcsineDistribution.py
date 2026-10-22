"""Tests for the ArcsineDistribution class."""

import numpy as np
import pytest
from scipy.stats import arcsine

from src.polysys.ChebyshevTPolynomials import ChebyshevTPolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution

from src.distributions import UniformDistribution

from src.distributions import ArcsineDistribution


SCALE, LOC = 4, 1


class TestArcsineDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return ArcsineDistribution(SCALE, LOC)

    @pytest.fixture
    def support(self):
        return LOC, LOC + SCALE

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self, dist):
        assert (dist.scale, dist.loc) == (SCALE, LOC)

    def test_init_defaults_to_unit_interval(self):
        d = ArcsineDistribution()
        assert (d.scale, d.loc) == (1, 0)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "arcsine"

    @pytest.mark.parametrize("scale, loc", [(1, 0), (SCALE, LOC), (0.5, -3.5), (2, -1)])
    def test_get_dist_params(self, scale, loc):
        params = ArcsineDistribution(scale, loc).get_dist_params()
        assert np.asarray(params).shape == (2,)
        assert np.allclose(params, [scale, loc], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert ArcsineDistribution(SCALE, LOC) == ArcsineDistribution(SCALE, LOC)

    @pytest.mark.parametrize("other", [(1, LOC), (SCALE, 0), (SCALE + 1, LOC + 1), (LOC, SCALE)])
    def test_eq_same_type_different_params(self, other):
        assert ArcsineDistribution(SCALE, LOC) != ArcsineDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert ArcsineDistribution(1, 0) != UniformDistribution(0, 1)
        assert ArcsineDistribution() != NormalDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert ArcsineDistribution(SCALE, LOC) != (SCALE, LOC)
        assert ArcsineDistribution() != object()

    @pytest.mark.parametrize(
        "args, expected",
        [
            ((SCALE, LOC), "Arcsine() scale=4.00, loc=1.00)"),
            ((), "Arcsine() scale=1.00, loc=0.00)"),
            ((0.125, -2.5), "Arcsine() scale=0.12, loc=-2.50)"),
        ],
    )
    def test_repr(self, args, expected):
        assert repr(ArcsineDistribution(*args)) == expected

    # =====================================================================
    # 2. Mathematical identities (arcsine-specific parts)
    # =====================================================================
    def test_pdf_is_zero_outside_support(self, dist, support):
        lo, hi = support
        assert np.allclose(dist.pdf(np.array([lo - 1, hi + 1])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_diverges_at_the_endpoints(self, dist, support):
        lo, hi = support
        assert np.isinf(dist.pdf(lo))
        assert np.isinf(dist.pdf(hi))

    def test_pdf_minimum_is_at_the_center(self, dist):
        center = LOC + SCALE / 2
        assert np.isclose(dist.pdf(center), 2 / (np.pi * SCALE), atol=self.ATOL, rtol=self.RTOL)
        x = np.linspace(LOC + 0.1, LOC + SCALE - 0.1, 101)
        assert np.all(dist.pdf(x) >= dist.pdf(center) - self.ATOL)

    def test_pdf_is_symmetric_about_the_center(self, dist):
        center = LOC + SCALE / 2
        x = np.linspace(0.05, SCALE / 2 - 0.05, 20)
        assert np.allclose(dist.pdf(center - x), dist.pdf(center + x), atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_zero_below_and_one_above_support(self, dist, support):
        lo, hi = support
        assert np.isclose(dist.cdf(lo - 1), 0, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(dist.cdf(hi + 1), 1, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_at_center_is_one_half(self, dist):
        assert np.isclose(dist.cdf(LOC + SCALE / 2), 0.5, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_closed_form(self, dist):
        x = np.linspace(LOC, LOC + SCALE, 41)
        expected = 2 / np.pi * np.arcsin(np.sqrt((x - LOC) / SCALE))
        assert np.allclose(dist.cdf(x), expected, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_closed_form(self, dist):
        q = np.linspace(0, 1, 41)
        expected = LOC + SCALE * np.sin(np.pi * q / 2) ** 2
        assert np.allclose(dist.invcdf(q), expected, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_outside_unit_interval_is_nan(self, dist):
        assert np.isnan(dist.invcdf(-0.1))
        assert np.isnan(dist.invcdf(1.1))

    def test_get_bounds_are_exact(self, dist):
        assert np.allclose(dist.get_bounds(delta=0), [LOC, LOC + SCALE], atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_default_delta(self, dist):
        assert np.allclose(dist.get_bounds(), dist.invcdf([0.02, 0.98]), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    def test_standard_arcsine_reference_values(self):
        d = ArcsineDistribution(2, -1)
        assert np.isclose(d.pdf(0), 1 / np.pi, atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf([-2, -1, 0, 1, 2]), [0, 0, 0.5, 1, 1], atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.invcdf([0, 0.5, 1]), [-1, 0, 1], atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("scale, loc", [(SCALE, LOC), (1, 0), (2, -1), (0.5, -3)])
    def test_matches_scipy(self, scale, loc):
        d = ArcsineDistribution(scale, loc)
        x = np.linspace(loc + 0.01 * scale, loc + 0.99 * scale, 50)
        q = np.linspace(0.01, 0.99, 50)
        assert np.allclose(d.pdf(x), arcsine.pdf(x, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), arcsine.cdf(x, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.invcdf(q), arcsine.ppf(q, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
        "scale, loc",
        [(0, 0), (-1, 0), (np.inf, 0), (np.nan, 0), (1, np.inf), (1, -np.inf), (1, np.nan), ("a", 0), (None, 0), (1, "b"), (1, None)],
    )
    def test_invalid_parameters_raise_value_error(self, scale, loc):
        with pytest.raises(ValueError):
            ArcsineDistribution(scale, loc)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (1, 0, 0.5, 1 / 8),
        (2, -1, 0.0, 0.5),
        (SCALE, LOC, 3.0, 2.0),
        (0.5, -3, -2.75, 1 / 32),
        (10, 5, 10.0, 12.5),
    ]

    @pytest.mark.parametrize("scale, loc, mean, var", MOMENT_CASES)
    def test_mean_specific_cases(self, scale, loc, mean, var):
        assert np.isclose(ArcsineDistribution(scale, loc).mean(), mean, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("scale, loc, mean, var", MOMENT_CASES)
    def test_var_specific_cases(self, scale, loc, mean, var):
        assert np.isclose(ArcsineDistribution(scale, loc).var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("scale, loc, mean, var", MOMENT_CASES)
    def test_skew_specific_cases(self, scale, loc, mean, var):
        assert np.isclose(ArcsineDistribution(scale, loc).skew(), 0.0, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("scale, loc, mean, var", MOMENT_CASES)
    def test_kurt_specific_cases(self, scale, loc, mean, var):
        assert np.isclose(ArcsineDistribution(scale, loc).kurt(), -1.5, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("scale, loc", [(SCALE, LOC), (1, 0), (2, -1)])
    def test_sample_within_support_all_methods(self, method, scale, loc):
        s = np.asarray(ArcsineDistribution(scale, loc).sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(s >= loc) and np.all(s <= loc + scale)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, dist, method):
        s = np.asarray(dist.sample(4096, method=method, seed=7))
        assert abs(np.mean(s) - dist.mean()) < 0.15
        assert abs(np.var(s) - dist.var()) < 0.15

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
    def test_translate_returns_arcsine(self, dist):
        translated = dist.translate(1, 2)
        assert isinstance(translated, ArcsineDistribution)
        assert np.allclose(
            (translated.scale, translated.loc),
            (2 * SCALE, 2 * LOC + 1),
            atol=self.ATOL,
            rtol=self.RTOL,
        )

    @pytest.mark.parametrize("shift, scale", [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)])
    def test_translate_maps_parameters_affinely(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert isinstance(t, ArcsineDistribution)
        assert np.isclose(t.scale, SCALE * scale, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.loc, LOC * scale + shift, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_translate_transforms_moments(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert np.isclose(t.mean(), scale * dist.mean() + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.var(), scale ** 2 * dist.var(), atol=self.ATOL, rtol=self.RTOL)

    def test_translate_default_is_identity(self, dist):
        assert dist.translate() == dist

    def test_translate_does_not_modify_original(self, dist):
        dist.translate(3, 2)
        assert (dist.scale, dist.loc) == (SCALE, LOC)

    def test_translate_composition(self, dist):
        t = dist.translate(1, 2).translate(-3, 0.5)
        expected = dist.translate(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(t.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    # translate has no explicit validation; the constructor rejects the result
    @pytest.mark.parametrize("shift, scale", [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), (np.nan, 1)])
    def test_translate_invalid_raises(self, dist, shift, scale):
        with pytest.raises(ValueError):
            dist.translate(shift, scale)

    @pytest.mark.parametrize("scale, loc", [(1, 0), (SCALE, LOC), (0.5, -3.5)])
    def test_get_shift_and_scale(self, scale, loc):
        d = ArcsineDistribution(scale, loc)
        assert np.isclose(d.get_shift(), loc, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), scale, atol=self.ATOL, rtol=self.RTOL)

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
    @pytest.mark.parametrize("start", [(1, 0), (SCALE, LOC)])
    def test_fix_moments_matches_by_quadrature(self, start, mean, var):
        new = ArcsineDistribution(*start).fix_moments(mean, var)
        assert isinstance(new, ArcsineDistribution)
        lo, hi = new.loc, new.loc + new.scale
        # the endpoint singularities are integrable, quad handles them
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_parameters(self):
        new = ArcsineDistribution(1).fix_moments(3, 0.5)
        assert np.allclose((new.scale, new.loc), (2 * np.sqrt(1.0), 3 - 1.0), atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_defaults(self):
        new = ArcsineDistribution(SCALE, LOC).fix_moments()
        assert np.allclose((new.scale, new.loc), (1, 0), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mean, var", [(0, 0), (0, -1), ("a", 1), (0, "b"), (np.inf, 1), (0, np.inf), (np.nan, 1)])
    def test_fix_moments_invalid_raises(self, mean, var):
        with pytest.raises(ValueError):
            ArcsineDistribution(SCALE, LOC).fix_moments(mean, var)

    # =====================================================================
    # 9. fix_bounds (bounded support: q0=0, q1=1 give the exact support)
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, ArcsineDistribution)
        assert np.allclose((new.loc, new.loc + new.scale), (new_min, new_max), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0, 0.5), (0.5, 1)])
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
    @pytest.mark.parametrize("scale, loc", [(1, 0), (SCALE, LOC), (2, -1)])
    def test_get_base_dist_is_arcsine_on_minus_one_to_one(self, scale, loc):
        germ = ArcsineDistribution(scale, loc).get_base_dist()
        assert isinstance(germ, ArcsineDistribution)
        assert germ == ArcsineDistribution(2, -1)

    def test_orth_polysys_is_chebyshev_t(self):
        d = ArcsineDistribution(2, -1)
        assert isinstance(d.orth_polysys(), ChebyshevTPolynomials)
        with pytest.raises(Exception):
            ArcsineDistribution(1, 0).orth_polysys()

    def test_orth_polysys_syschar(self):
        d = ArcsineDistribution(2, -1)
        assert d.orth_polysys_syschar(False) == "T"
        assert d.orth_polysys_syschar(True) == "t"
        with pytest.raises(Exception):
            ArcsineDistribution(1, 0).orth_polysys_syschar(False)
        with pytest.raises(Exception):
            ArcsineDistribution(1, 0).orth_polysys_syschar(True)
