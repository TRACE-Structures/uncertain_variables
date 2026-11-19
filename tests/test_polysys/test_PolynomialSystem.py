import numpy as np
import pytest
from scipy.integrate import quad

from src.polysys import (
    ChebyshevTPolynomials,
    ChebyshevUPolynomials,
    HermitePolynomials,
    JacobiPolynomials,
    LaguerrePolynomials,
    LegendrePolynomials,
    MonomialPolynomials,
    OrthonormalPolynomials,
    PolynomialSystem,
    syschar_to_polysys,
)


@pytest.mark.parametrize("deg", [True, np.bool_(True), 1.5, "1"])
def test_invalid_degree_type(deg):
    system = HermitePolynomials()
    with pytest.raises(TypeError):
        system.evaluate(deg, [0])
    with pytest.raises(TypeError):
        system.sqnorm(deg)


def test_invalid_inputs_and_shapes():
    system = HermitePolynomials()
    with pytest.raises(ValueError):
        system.evaluate(-1, [0])
    with pytest.raises(ValueError):
        system.sqnorm(-1)
    with pytest.raises(ValueError):
        system.evaluate(1, [[0]])
    with pytest.raises(ValueError):
        system.evaluate(1, 0)
    with pytest.raises(TypeError):
        system.evaluate(1, ["0"])
    with pytest.raises(TypeError):
        system.evaluate(1, [1 + 1j])
    with pytest.raises(TypeError):
        system.evaluate(1, [0], derivative=1)
    np.testing.assert_array_equal(system.evaluate(np.int64(0), (0, 1)), [[1], [1]])
    assert system.evaluate(2, []).shape == (0, 3)


@pytest.mark.parametrize(
    "factory",
    [
        ChebyshevTPolynomials,
        ChebyshevUPolynomials,
        HermitePolynomials,
        lambda **kw: JacobiPolynomials(0.5, 1.5, **kw),
        lambda **kw: JacobiPolynomials(-0.5, -0.5, **kw),
        lambda **kw: JacobiPolynomials(0, 0, **kw),
        LaguerrePolynomials,
        LegendrePolynomials,
    ],
)
def test_probability_norms_and_orthonormal_values(factory):
    system = factory()
    normalized = factory(orthonormal=True)
    wrapped = OrthonormalPolynomials(system)
    dist = system.gpc_distribution()
    assert system.gpc_dist() == dist
    bounds = dist.ppf([0, 1])
    x = np.array([-0.5, 0.0, 0.5])
    norms = system.sqnorm(3)
    assert norms.shape == (4,)
    assert norms[0] == pytest.approx(1.0)
    np.testing.assert_allclose(normalized.sqnorm(3), np.ones(4))
    for derivative in (False, True):
        expected = system.evaluate(3, x, derivative) / np.sqrt(norms)
        np.testing.assert_allclose(normalized.evaluate(3, x, derivative), expected)
        np.testing.assert_allclose(wrapped.evaluate(3, x, derivative), expected)
    np.testing.assert_allclose(wrapped.sqnorm(3), np.ones(4))
    for n, norm in enumerate(norms):
        integral, _ = quad(
            lambda t: system.evaluate(n, [t])[0, n] ** 2 * dist.pdf(t),
            *bounds,
        )
        assert integral == pytest.approx(norm, rel=1e-7, abs=1e-7)


def test_known_hermite_values_and_norms():
    system = HermitePolynomials()
    np.testing.assert_allclose(system.sqnorm(3), [1, 1, 2, 6])
    np.testing.assert_allclose(system.evaluate(2, [0, 1]), [[1, 0, -1], [1, 1, 0]])
    np.testing.assert_allclose(
        system.evaluate(2, [0, 1], derivative=True), [[0, 1, 0], [0, 1, 2]]
    )
    np.testing.assert_allclose(LegendrePolynomials().sqnorm(2), [1, 1 / 3, 1 / 5])


def test_monomials_and_abstract_base():
    with pytest.raises(TypeError):
        PolynomialSystem()
    system = MonomialPolynomials()
    np.testing.assert_allclose(system.evaluate(2, (0, 2)), [[1, 0, 0], [1, 2, 4]])
    np.testing.assert_allclose(
        system.evaluate(2, (0, 2), derivative=True), [[0, 1, 0], [0, 1, 4]]
    )
    with pytest.raises(NotImplementedError):
        system.sqnorm(1)
    with pytest.raises(NotImplementedError):
        system.gpc_distribution()
    with pytest.raises(NotImplementedError):
        OrthonormalPolynomials(system).sqnorm(1)


def test_system_character_factory():
    assert syschar_to_polysys("H") is HermitePolynomials
    assert syschar_to_polysys("m") is MonomialPolynomials
    assert syschar_to_polysys("h").orthonormal
    with pytest.raises(KeyError):
        syschar_to_polysys("j")