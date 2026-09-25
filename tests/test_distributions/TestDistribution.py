import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

class TestDistribution:
    """
    Shared API checks for any Distribution implementation.

    Required fixtures in the subclass
    ---------------------------------
        dist : Distribution
            A concrete instance with non-default parameters.
        support : tuple
            (lo, hi) integration limits, use -np.inf and np.inf when the
            support is unbounded.
    """

    # ---- knobs a concrete test class may override ------------------------
    KURT_IS_EXCESS = True      # does ``kurt()`` return excess kurtosis?

    RTOL = 1e-8                # relative tolerance for analytic-vs-analytic checks
    ATOL = 1e-10               # absolute tolerance for analytic-vs-analytic checks
    QUAD_RTOL = 1e-6           # relative tolerance for quadrature-based checks
    QUAD_ATOL = 1e-6           # absolute tolerance for quadrature-based checks

    SAMPLE_METHODS = ("MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol")  

    # ---- helpers ---------------------------------------------------------
    def _quad(self, f, lo, hi):
        val, _ = integrate.quad(f, lo, hi, 
                                epsabs=self.QUAD_ATOL, 
                                epsrel=self.QUAD_RTOL,
                                limit=100
        )
        return val

    def _central_moment(self, dist, k, lo, hi):
        mu = dist.mean()
        return self._quad(lambda t: (t - mu) ** k * dist.pdf(t), lo, hi)

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    @pytest.mark.parametrize("name", ["pdf", "cdf", "logpdf"])
    def test_scalar_input_returns_scalar(self, dist, support, name):
        x = np.array(np.mean(np.clip(support, -10, 10)))
        out = getattr(dist, name)(x)
        assert np.isscalar(out)

    @pytest.mark.parametrize("name", ["pdf", "cdf", "logpdf"])
    def test_vector_input_returns_vector_of_same_shape(self, dist, name):
        x = dist.invcdf(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        out = np.asarray(getattr(dist, name)(x))
        assert out.shape == x.shape

    def test_invcdf_scalar_input_returns_scalar(self, dist):
        y = 0.5
        x = dist.invcdf(y)
        assert np.isscalar(x)

    def test_invcdf_vector_input_returns_vector_of_same_shape(self, dist):
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x = dist.invcdf(y)
        out = np.asarray(x)
        assert out.shape == y.shape

    def test_moments_matches_individual_definitions(self, dist):
        m = dist.moments()
        assert len(m) == 4
        expected = [dist.mean(), dist.var(), dist.skew(), dist.kurt()]
        assert np.allclose(m, expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("method", SAMPLE_METHODS)
    def test_sample_shape_and_support(self, dist, support, method):
        n = 64
        s = np.asarray(dist.sample(n, method=method, seed=4267))
        assert s.shape == (n,)
        lo, hi = support
        assert np.all(s >= lo) and np.all(s <= hi)

    # =====================================================================
    # 2. Mathematical identities
    # =====================================================================
    def test_pdf_is_nonnegative(self, dist):
        x = dist.invcdf(np.linspace(0.001, 0.999, 201))
        pdf_values = np.asarray(dist.pdf(x))
        assert np.all(np.isfinite(pdf_values))
        assert np.all(pdf_values >= 0)

    def test_pdf_integrates_to_one(self, dist, support):
        lo, hi = support
        assert np.isclose(self._quad(dist.pdf, lo, hi), 1.0, rtol=self.QUAD_RTOL, atol=self.QUAD_ATOL)

    def test_cdf_is_the_integral_of_the_pdf(self, dist, support):
        lo, _ = support
        for q in (0.25, 0.5, 0.9):
            x = dist.invcdf(q)
            assert np.isclose(self._quad(dist.pdf, lo, x), float(dist.cdf(x)), rtol=self.QUAD_RTOL, atol=self.QUAD_ATOL)

    def test_cdf_is_monotone_and_in_unit_interval(self, dist):
        x = dist.invcdf(np.linspace(0.001, 0.999, 201))
        F = np.asarray(dist.cdf(x))
        assert np.all(np.diff(F) >= -self.ATOL)
        assert np.all((F >= 0) & (F <= 1))

    def test_cdf_limits(self, dist, support):
        lo, hi = support
        assert np.isclose(dist.cdf(lo), 0.0, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(dist.cdf(hi), 1.0, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(dist.cdf(-np.inf), 0.0, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(dist.cdf(np.inf), 1.0, rtol=self.RTOL, atol=self.ATOL)

    def test_invcdf_inverts_cdf(self, dist):
        q = np.linspace(0.01, 0.99, 50)
        assert np.allclose(dist.cdf(dist.invcdf(q)), q, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_inverts_invcdf(self, dist):
        q = np.linspace(0.01, 0.99, 50)
        points = dist.invcdf(q)
        assert np.allclose(dist.invcdf(dist.cdf(points)), points, atol=self.ATOL, rtol=self.RTOL)

    def test_logpdf_is_log_of_pdf(self, dist):
        x = dist.invcdf(np.linspace(0.01, 0.99, 50))
        with np.errstate(divide="ignore"):
            assert np.allclose(
                np.asarray(dist.logpdf(x)),
                np.log(np.asarray(dist.pdf(x))),
                rtol=self.RTOL,
                atol=self.ATOL
            )

    def test_moments_match_quadrature(self, dist, support):
        lo, hi = support
        var = self._central_moment(dist, 2, lo, hi)
        skew = self._central_moment(dist, 3, lo, hi) / var ** (3 / 2)
        kurt = self._central_moment(dist, 4, lo, hi) / var ** 2
        if self.KURT_IS_EXCESS:
            kurt -= 3
        actual = [var, skew, kurt]
        expected = [dist.var(), dist.skew(), dist.kurt()]
        assert np.allclose(actual, expected, rtol=self.QUAD_RTOL, atol=self.QUAD_ATOL)

    def test_fix_moments(self, dist):
        new_dist = dist.fix_moments(10, 4)
        assert np.isclose(new_dist.mean(), 10, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(new_dist.var(), 4, rtol=self.RTOL, atol=self.ATOL)

    def test_fix_moments_does_not_modify_original(self, dist):
        original_moments = dist.moments()
        dist.fix_moments(10, 4)
        assert np.allclose(dist.moments(), original_moments)

    def test_fix_bounds(self, dist):
        q0, q1 = 0.01, 0.99
        new = dist.fix_bounds(-1.0, 1.0, q0=q0, q1=q1)
        assert np.isclose(new.invcdf(q0), -1.0, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(new.invcdf(q1), 1.0, rtol=self.RTOL, atol=self.ATOL)

    def test_stdnor_roundtrip(self, dist):
        q = np.linspace(-2.5, 2.5, 50)
        transformed = dist.stdnor2base(q)
        assert np.allclose(dist.base2stdnor(transformed), q, atol=self.ATOL, rtol=self.RTOL)

    def test_stdnor2base_matches_scipy(self, dist):
        q = np.linspace(-2.5, 2.5, 50)
        transformed = dist.stdnor2base(q)
        expected = dist.invcdf(norm.cdf(q))
        assert np.allclose(transformed, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_base2stdnor_matches_scipy(self, dist):
        q = dist.invcdf(np.linspace(0.001, 0.999, 50))
        transformed = dist.base2stdnor(q)
        expected = norm.ppf(dist.cdf(q))
        assert np.allclose(transformed, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_are_the_delta_quantiles(self, dist):
        delta = 0.01
        expected = dist.invcdf([delta, 1 - delta])
        assert np.allclose(dist.get_bounds(delta), expected, rtol=self.RTOL, atol=self.ATOL)

    @pytest.mark.parametrize("method", SAMPLE_METHODS)
    def test_sample_reproduces_moments(self, dist, method):
        rng_state = np.random.get_state()
        try:
            np.random.seed(4267)
            s = dist.sample(262144, method=method, seed=4267)
        finally:
            np.random.set_state(rng_state)
        mean_se = np.sqrt(dist.var() / s.size)
        variance_se = np.sqrt(2 * dist.var() ** 2 / (s.size - 1))
        assert np.isclose(np.mean(s), dist.mean(), rtol=1e-2, atol=10 * mean_se)
        assert np.isclose(np.var(s), dist.var(), rtol=1e-2, atol=10 * variance_se)
