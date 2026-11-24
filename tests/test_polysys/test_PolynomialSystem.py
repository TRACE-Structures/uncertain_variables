import numpy as np
import pytest
from scipy.integrate import quad

from uncertain_variables.distributions import (
    ArcsineDistribution,
    BetaDistribution,
    ExponentialDistribution,
    NormalDistribution,
    SemiCircleDistribution,
    UniformDistribution,
)
from uncertain_variables.polysys import (
    ChebyshevTPolynomials,
    ChebyshevUPolynomials,
    HermitePolynomials,
    JacobiPolynomials,
    LaguerrePolynomials,
    LegendrePolynomials,
    MonomialPolynomials,
    PolynomialSystem,
    syschar_to_polysys,
)
from uncertain_variables.polysys.PolynomialSystem import _probability_sqnorm


RTOL = 1e-8
ATOL = 1e-10
QUAD_RTOL = 1e-5
QUAD_ATOL = 1e-5


def test_polynomial_system_is_abstract():
    with pytest.raises(TypeError):
        PolynomialSystem()


@pytest.mark.parametrize("degree", [-1, -10])
def test_evaluate_rejects_negative_degrees(degree):
    system = HermitePolynomials()
    with pytest.raises(ValueError):
        system.evaluate(degree, [0.0])
    with pytest.raises(ValueError):
        system.sqnorm(degree)


@pytest.mark.parametrize("x", [[0.0], (0.0, 1.0), np.array([0.0, 1.0])])
def test_evaluate_accepts_one_dimensional_real_input(x):
    result = HermitePolynomials().evaluate(2, x)
    assert result.shape == (len(x), 3)


@pytest.mark.parametrize(
    "x",
    [0.0, [[0.0]], np.array([[0.0, 1.0]]), ["0"], [0.0 + 1j]],
)
def test_evaluate_rejects_invalid_point_shapes_and_types(x):
    with pytest.raises((ValueError, TypeError)):
        HermitePolynomials().evaluate(2, x)


def test_evaluate_accepts_empty_input():
    result = HermitePolynomials().evaluate(2, [])
    assert result.shape == (0, 3)


@pytest.mark.parametrize("derivative", [1.5, "1", None])
def test_evaluate_rejects_invalid_derivative(derivative):
    with pytest.raises(TypeError):
        HermitePolynomials().evaluate(2, [0.0], derivative=derivative)


@pytest.mark.parametrize("value", [np.int64(0), np.int32(2)])
def test_evaluate_accepts_numpy_integer_degree(value):
    result = HermitePolynomials().evaluate(value, [0.0])
    assert result.shape == (1, int(value) + 1)


POLYNOMIAL_FACTORIES = [
    ChebyshevTPolynomials,
    ChebyshevUPolynomials,
    HermitePolynomials,
    lambda: JacobiPolynomials(0.5, 1.5),
    lambda: JacobiPolynomials(-0.5, -0.5),
    LaguerrePolynomials,
    LegendrePolynomials,
]


@pytest.mark.parametrize("factory", POLYNOMIAL_FACTORIES)
def test_associated_distribution(factory):
    distribution = factory().gpc_distribution()
    assert isinstance(
        distribution,
        (
            ArcsineDistribution,
            BetaDistribution,
            ExponentialDistribution,
            NormalDistribution,
            SemiCircleDistribution,
            UniformDistribution,
        ),
    )


@pytest.mark.parametrize("factory", POLYNOMIAL_FACTORIES)
def test_probability_norm_integrals(factory):
    system = factory()
    distribution = system.gpc_distribution()
    bounds = distribution.ppf([0.0, 1.0])
    norms = system.sqnorm(4)

    assert norms.shape == (5,)
    assert np.all(np.isfinite(norms))
    assert np.all(norms > 0)

    for degree, norm in enumerate(norms):
        integral, _ = quad(
            lambda x: system.evaluate(degree, [x])[0, degree] ** 2
            * distribution.pdf(x),
            *bounds,
            epsabs=QUAD_ATOL,
            epsrel=QUAD_RTOL,
            limit=100,
        )
        assert integral == pytest.approx(norm, rel=QUAD_RTOL, abs=QUAD_ATOL)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (ChebyshevTPolynomials, [1.0, 0.5, 0.5, 0.5, 0.5]),
        (ChebyshevUPolynomials, [1.0, 1.0, 1.0, 1.0, 1.0]),
        (HermitePolynomials, [1.0, 1.0, 2.0, 6.0, 24.0]),
        (LaguerrePolynomials, [1.0, 1.0, 1.0, 1.0, 1.0]),
        (LegendrePolynomials, [1.0, 1 / 3, 1 / 5, 1 / 7, 1 / 9]),
    ],
)
def test_known_probability_squared_norms(factory, expected):
    np.testing.assert_allclose(
        factory().sqnorm(4),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "factory",
    [
        ChebyshevTPolynomials,
        ChebyshevUPolynomials,
        HermitePolynomials,
        LaguerrePolynomials,
        LegendrePolynomials,
    ],
)
def test_orthonormal_constructor_has_unit_norms(factory):
    system = factory(orthonormal=True)
    np.testing.assert_allclose(system.sqnorm(4), np.ones(5), rtol=RTOL, atol=ATOL)


def test_jacobi_orthonormal_constructor_has_unit_norms():
    system = JacobiPolynomials(0.5, 1.5, orthonormal=True)
    np.testing.assert_allclose(system.sqnorm(4), np.ones(5), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "system",
    [
        ChebyshevTPolynomials(orthonormal=True),
        ChebyshevUPolynomials(orthonormal=True),
        HermitePolynomials(orthonormal=True),
        JacobiPolynomials(0.5, 1.5, orthonormal=True),
        LaguerrePolynomials(orthonormal=True),
        LegendrePolynomials(orthonormal=True),
    ],
)
def test_orthonormal_evaluate_is_finite(system):
    values = system.evaluate(6, np.array([-0.5, 0.0, 0.5]))
    assert values.shape == (3, 7)
    assert np.all(np.isfinite(values))


def test_known_hermite_values_and_derivatives():
    system = HermitePolynomials()

    np.testing.assert_allclose(
        system.evaluate(3, [0.0, 1.0]),
        [[1, 0, -1, 0], [1, 1, 0, -2]],
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        system.evaluate(3, [0.0, 1.0], derivative=True),
        [[0, 1, 0, -3], [0, 1, 2, 0]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_known_legendre_values():
    system = LegendrePolynomials()
    np.testing.assert_allclose(
        system.evaluate(3, [0.0, 1.0]),
        [[1, 0, -0.5, 0], [1, 1, 1, 1]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_known_chebyshev_values():
    np.testing.assert_allclose(
        ChebyshevTPolynomials().evaluate(3, [0.0, 1.0]),
        [[1, 0, -1, 0], [1, 1, 1, 1]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        ChebyshevUPolynomials().evaluate(3, [0.0, 1.0]),
        [[1, 0, -1, 0], [1, 2, 3, 4]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_known_jacobi_legendre_special_case():
    system = JacobiPolynomials(0, 0)
    np.testing.assert_allclose(
        system.evaluate(3, [0.0, 1.0]),
        [[1, 0, -0.5, 0], [1, 1, 1, 1]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        system.sqnorm(3),
        [1, 1 / 3, 1 / 5, 1 / 7],
        rtol=RTOL,
        atol=ATOL,
    )


def test_laguerre_values():
    system = LaguerrePolynomials()
    np.testing.assert_allclose(
        system.evaluate(3, [0.0, 1.0]),
        [[1, 1, 1, 1], [1, 0, -0.5, -2 / 3]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_monomials_are_not_implemented_as_concrete_polynomial_system():
    assert issubclass(MonomialPolynomials, PolynomialSystem)
    with pytest.raises(TypeError):
        MonomialPolynomials()


def test_probability_sqnorm_helper_validation():
    with pytest.raises(TypeError):
        _probability_sqnorm(None, 2, 1.0)
    with pytest.raises(TypeError):
        _probability_sqnorm(lambda n: 1, 1.5, 1.0)
    with pytest.raises(ValueError):
        _probability_sqnorm(lambda n: 1, -1, 1.0)
    with pytest.raises(ValueError):
        _probability_sqnorm(lambda n: 1, 2, 0.0)


def test_probability_sqnorm_helper_result_shape():
    class Polynomial:
        def __init__(self, degree):
            self.normcoef = degree + 1

    result = _probability_sqnorm(Polynomial, 3, 2.0)
    np.testing.assert_allclose(
        result,
        [1.0, 2.0, 4.5, 8.0],
        rtol=RTOL,
        atol=ATOL,
    )


def test_system_character_factory():
    assert syschar_to_polysys("H") is HermitePolynomials
    assert syschar_to_polysys("P") is LegendrePolynomials
    assert syschar_to_polysys("T") is ChebyshevTPolynomials
    assert syschar_to_polysys("U") is ChebyshevUPolynomials
    assert syschar_to_polysys("L") is LaguerrePolynomials
    assert syschar_to_polysys("J") is JacobiPolynomials
    assert syschar_to_polysys("h").orthonormal
    assert syschar_to_polysys("p").orthonormal
    assert syschar_to_polysys("t").orthonormal
    assert syschar_to_polysys("u").orthonormal
    assert syschar_to_polysys("l").orthonormal
    with pytest.raises(KeyError):
        syschar_to_polysys("unknown")
