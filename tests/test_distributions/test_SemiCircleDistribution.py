"""Tests for the SemiCircleDistribution class."""

import numpy as np
import pytest

from src.polysys.ChebyshevUPolynomials import ChebyshevUPolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution, UniformDistribution

from src.distributions import SemiCircleDistribution


RADIUS, LOC = 2, 1


class TestSemiCircleDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return SemiCircleDistribution(RADIUS, LOC)

    @pytest.fixture
    def support(self):
        return LOC - RADIUS, LOC + RADIUS

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self, dist):
        assert (dist.radius, dist.loc, dist.scale) == (RADIUS, LOC, RADIUS)

    def test_init_defaults(self):
        d = SemiCircleDistribution(3)
        assert (d.radius, d.loc) == (3, 0)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "semicircle"

    @pytest.mark.parametrize("radius, loc", [(1, 0), (RADIUS, LOC), (0.5, -3.5)])
    def test_get_dist_params(self, radius, loc):
        params = SemiCircleDistribution(radius, loc).get_dist_params()
        assert np.asarray(params).shape == (2,)
        assert np.allclose(params, [radius, loc], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert SemiCircleDistribution(RADIUS, LOC) == SemiCircleDistribution(RADIUS, LOC)

    @pytest.mark.parametrize("other", [(1, LOC), (RADIUS, 0), (RADIUS + 1, LOC + 1), (LOC, RADIUS)])
    def test_eq_same_type_different_params(self, other):
        assert SemiCircleDistribution(RADIUS, LOC) != SemiCircleDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert SemiCircleDistribution(1) != UniformDistribution(-1, 1)
        assert SemiCircleDistribution(1) != NormalDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert SemiCircleDistribution(RADIUS, LOC) != (RADIUS, LOC)
        assert SemiCircleDistribution(RADIUS, LOC) != "W"

    @pytest.mark.parametrize(
        "args, expected",
        [
            ((RADIUS, LOC), "W(2.00), loc(1.00)"),
            ((1,), "W(1.00), loc(0.00)"),
            ((0.125, -2.5), "W(0.12), loc(-2.50)"),
        ],
    )
    def test_repr(self, args, expected):
        assert repr(SemiCircleDistribution(*args)) == expected

    # =====================================================================
    # 2. Mathematical identities (semicircle-specific parts)
    # =====================================================================
    def test_pdf_is_zero_outside_support(self, dist, support):
        lo, hi = support
        assert np.allclose(dist.pdf(np.array([lo - 1, hi + 1])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_at_center(self, dist):
        assert np.isclose(dist.pdf(LOC), 2 / (np.pi * RADIUS), atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_vanishes_at_edges(self, dist, support):
        lo, hi = support
        assert np.allclose(dist.pdf(np.array([lo, hi])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_is_symmetric_about_loc(self, dist):
        x = np.linspace(0.1, RADIUS - 0.1, 20)
        assert np.allclose(dist.pdf(LOC + x), dist.pdf(LOC - x), atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_limits_and_symmetry(self, dist):
        assert np.isclose(dist.cdf(LOC - RADIUS), 0, atol=self.ATOL)
        assert np.isclose(dist.cdf(LOC + RADIUS), 1, atol=self.ATOL)
        assert np.isclose(dist.cdf(LOC), 0.5, atol=self.ATOL, rtol=self.RTOL)
        x = np.array([0.25, 1.0])
        assert np.allclose(dist.cdf(LOC - x), 1 - dist.cdf(LOC + x), atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_median_is_loc(self, dist):
        assert np.isclose(dist.invcdf(0.5), LOC, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_endpoints(self, dist, support):
        assert np.allclose(dist.invcdf([0, 1]), support, atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_default_delta(self, dist):
        assert np.allclose(dist.get_bounds(), dist.invcdf([0.02, 0.98]), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    def test_pdf_reference_values(self):
        d = SemiCircleDistribution(2)
        # pdf(x) = 2 / (pi r^2) * sqrt(r^2 - x^2)
        for x in (0.0, 1.0, -1.0, 1.5):
            assert np.isclose(d.pdf(x), 2 / (np.pi * 4) * np.sqrt(4 - x ** 2), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.pdf(1.0), np.sqrt(3) / (2 * np.pi), atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_reference_values(self):
        d = SemiCircleDistribution(2)
        for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
            expected = 0.5 + x * np.sqrt(4 - x ** 2) / (np.pi * 4) + np.arcsin(x / 2) / np.pi
            assert np.isclose(d.cdf(x), expected, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize("radius, loc", [(0, 0), (-1, 0), ("a", 0), (None, 0), (1, "b"), (1, None)])
    def test_invalid_parameters_raise_value_error(self, radius, loc):
        with pytest.raises(ValueError):
            SemiCircleDistribution(radius, loc)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (1, 0, 0.0, 0.25),
        (2, 0, 0.0, 1.0),
        (RADIUS, LOC, 1.0, 1.0),
        (0.5, -3, -3.0, 1 / 16),
        (10, 5, 5.0, 25.0),
    ]

    @pytest.mark.parametrize("radius, loc, mean, var", MOMENT_CASES)
    def test_mean_specific_cases(self, radius, loc, mean, var):
        assert np.isclose(SemiCircleDistribution(radius, loc).mean(), mean, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("radius, loc, mean, var", MOMENT_CASES)
    def test_var_specific_cases(self, radius, loc, mean, var):
        assert np.isclose(SemiCircleDistribution(radius, loc).var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("radius, loc, mean, var", MOMENT_CASES)
    def test_skew_specific_cases(self, radius, loc, mean, var):
        assert np.isclose(SemiCircleDistribution(radius, loc).skew(), 0.0, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("radius, loc, mean, var", MOMENT_CASES)
    def test_kurt_specific_cases(self, radius, loc, mean, var):
        assert np.isclose(SemiCircleDistribution(radius, loc).kurt(), -1.0, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("radius, loc", [(RADIUS, LOC), (1, 0), (0.5, -3)])
    def test_sample_within_support_all_methods(self, method, radius, loc):
        s = np.asarray(SemiCircleDistribution(radius, loc).sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(s >= loc - radius) and np.all(s <= loc + radius)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, dist, method):
        s = np.asarray(dist.sample(2048, method=method, seed=7))
        assert abs(np.mean(s) - dist.mean()) < 0.1
        assert abs(np.var(s) - dist.var()) < 0.1

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
        assert isinstance(t, SemiCircleDistribution)
        assert np.isclose(t.radius, RADIUS * scale, atol=self.ATOL, rtol=self.RTOL)
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
        assert (dist.radius, dist.loc) == (RADIUS, LOC)

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

    @pytest.mark.parametrize("radius, loc", [(1, 0), (RADIUS, LOC), (0.5, -3.5)])
    def test_get_shift_and_scale(self, radius, loc):
        d = SemiCircleDistribution(radius, loc)
        assert np.isclose(d.get_shift(), loc, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), radius, atol=self.ATOL, rtol=self.RTOL)

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
    @pytest.mark.parametrize("start", [(1, 0), (RADIUS, LOC)])
    def test_fix_moments_matches_by_quadrature(self, start, mean, var):
        new = SemiCircleDistribution(*start).fix_moments(mean, var)
        assert isinstance(new, SemiCircleDistribution)
        lo, hi = new.loc - new.radius, new.loc + new.radius
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_parameters(self):
        new = SemiCircleDistribution(1).fix_moments(3, 4)
        assert np.allclose((new.radius, new.loc), (4, 3), atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_defaults(self):
        new = SemiCircleDistribution(5, 5).fix_moments()
        assert np.allclose((new.radius, new.loc), (1, 0), atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_zero_variance_raises(self):
        with pytest.raises(ValueError):
            SemiCircleDistribution(1).fix_moments(0, 0)

    # =====================================================================
    # 9. fix_bounds
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, SemiCircleDistribution)
        assert np.allclose(
            (new.loc - new.radius, new.loc + new.radius), (new_min, new_max), atol=self.ATOL, rtol=self.RTOL
        )

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
    @pytest.mark.parametrize("radius, loc", [(1, 0), (RADIUS, LOC), (0.5, -3)])
    def test_get_base_dist_is_unit_semicircle(self, radius, loc):
        germ = SemiCircleDistribution(radius, loc).get_base_dist()
        assert isinstance(germ, SemiCircleDistribution)
        assert germ == SemiCircleDistribution(1, 0)

    def test_orth_polysys_is_chebyshev_u(self):
        d1 = SemiCircleDistribution(1, 0)
        d2 = SemiCircleDistribution(RADIUS, LOC)
        assert isinstance(d1.orth_polysys(), ChebyshevUPolynomials)
        with pytest.raises(Exception):
            d2.orth_polysys()

    def test_orth_polysys_syschar(self):
        d1 = SemiCircleDistribution(1, 0)
        d2 = SemiCircleDistribution(RADIUS, LOC)
        assert d1.orth_polysys_syschar(False) == "U"
        assert d1.orth_polysys_syschar(True) == "u"
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(False)
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(True)

    # =====================================================================
    # 11. Unit conversions
    # =====================================================================
    def test_convert_to_round_trip(self):
        dist = SemiCircleDistribution(RADIUS, LOC)
        old_params = dist.get_dist_params()
        dist.convert_to("m", "cm")
        dist.convert_to("cm", "m")
        new_params = dist.get_dist_params()
        assert np.allclose(old_params, new_params)

    def test_convert_to_exception(self):
        d = SemiCircleDistribution(RADIUS, LOC)
        d.convert_to("m", "cm")
        expected_params = [100*RADIUS, 100*LOC]
        new_params = d.get_dist_params()
        assert np.allclose(new_params, expected_params)