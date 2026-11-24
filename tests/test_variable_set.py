"""Tests for VariableSet construction, operations, and numerical methods."""

import numpy as np
import pytest

from src.distributions import ArcsineDistribution, NormalDistribution, UniformDistribution
from src.polysys import ChebyshevTPolynomials
from src.variable import Variable
from src.variable_set import VariableSet


@pytest.fixture
def variables():
    return VariableSet(
        [
            Variable("x", UniformDistribution(0, 10)),
            Variable("y", UniformDistribution(0, 2)),
        ]
    )


def test_init_and_basic_collection_behavior(variables):
    assert len(variables) == 2
    assert variables.get_num_variables() == 2
    assert variables.get_variable_names() == ["x", "y"]
    assert list(variables) == list(variables.get_variables().values())
    assert Variable("x", UniformDistribution(0, 10)) in variables
    assert Variable("missing", UniformDistribution()) not in variables
    assert repr(variables) == "VariableSet(normalized=True, variables=['x', 'y'])"


def test_init_accepts_empty_set_and_non_normalized_flag():
    variables = VariableSet(normalized=False)

    assert len(variables) == 0
    assert variables.normalized is False
    assert variables.get_variable_names() == []


@pytest.mark.parametrize("normalized", [None, 0, 1, np.bool_(True)])
def test_init_rejects_non_boolean_normalized_flag(normalized):
    with pytest.raises(TypeError, match="normalized must be a boolean"):
        VariableSet(normalized=normalized)


def test_init_rejects_invalid_and_duplicate_variables():
    with pytest.raises(TypeError, match="instance of Variable"):
        VariableSet([object()])

    with pytest.raises(KeyError, match="already exists"):
        VariableSet(
            [
                Variable("x", UniformDistribution()),
                Variable("x", NormalDistribution()),
            ]
        )


def test_equality_checks_contents_and_normalization(variables):
    same = VariableSet(
        [
            Variable("x", UniformDistribution(0, 10)),
            Variable("y", UniformDistribution(0, 2)),
        ]
    )

    assert variables == same
    assert variables != VariableSet(same.variables.values(), normalized=False)
    assert variables != VariableSet([Variable("x", UniformDistribution(0, 9))])
    assert variables != object()


def test_add_remove_and_contains_validate_inputs(variables):
    with pytest.raises(TypeError, match="instance of Variable"):
        variables.add("z")
    with pytest.raises(TypeError, match="instance of Variable"):
        "x" in variables
    with pytest.raises(KeyError, match="already exists"):
        variables.add(Variable("x", UniformDistribution()))

    variables.remove("x")
    assert variables.get_variable_names() == ["y"]
    with pytest.raises(KeyError, match="not found"):
        variables.remove("x")


def test_update_accepts_variable_set_dict_and_variable_and_replaces_names(variables):
    variables.update(Variable("z", NormalDistribution()))
    variables.update({"x": Variable("x", UniformDistribution(-1, 1))})
    variables.update(VariableSet([Variable("y", NormalDistribution(3, 2))]))

    assert variables.get_variable_names() == ["x", "y", "z"]
    assert variables.variables["x"].dist == UniformDistribution(-1, 1)
    assert variables.variables["y"].dist == NormalDistribution(3, 2)


@pytest.mark.parametrize("source", ["invalid", [Variable("x", UniformDistribution())]])
def test_update_rejects_unsupported_source(variables, source):
    with pytest.raises(TypeError, match="Variable, VariableSet or a dictionary"):
        variables.update(source)


def test_update_dict_rejects_invalid_entries(variables):
    with pytest.raises(TypeError, match="keys must be strings"):
        variables.update({1: Variable("z", NormalDistribution())})
    with pytest.raises(TypeError, match="values in the dictionary"):
        variables.update({"z": object()})


def test_rename_single_and_multiple_variables(variables):
    variables.rename("x", "first")
    variables.rename(["y", "first"], ["second", "third"])

    assert variables.get_variable_names() == ["second", "third"]
    assert variables.variables["second"].name == "second"
    assert variables.variables["third"].name == "third"


def test_rename_rejects_missing_names_collisions_and_length_mismatch_atomically(
    variables,
):
    with pytest.raises(KeyError, match="not found"):
        variables.rename("missing", "new")
    with pytest.raises(ValueError, match="already exists"):
        variables.rename("x", "y")
    with pytest.raises(ValueError, match="same length"):
        variables.rename(["x", "y"], ["new"])
    with pytest.raises(ValueError, match="new_names must not contain duplicates"):
        variables.rename(["x", "y"], ["new", "new"])
    with pytest.raises(ValueError, match="old_names must not contain duplicates"):
        variables.rename(["x", "x"], ["first", "second"])

    assert variables.get_variable_names() == ["x", "y"]


def test_copy_preserves_normalization_and_is_deep(variables):
    variables.normalized = False

    copied = variables.copy()
    copied.variables["x"].dist.a = -5

    assert copied.normalized is False
    assert copied == VariableSet(
        [
            Variable("x", UniformDistribution(-5, 10)),
            Variable("y", UniformDistribution(0, 2)),
        ],
        normalized=False,
    )
    assert variables.variables["x"].dist == UniformDistribution(0, 10)


def test_get_distribution_metadata_and_moments(variables):
    assert variables.get_dist_types() == {"x": "unif", "y": "unif"}
    params = variables.get_dist_params()
    np.testing.assert_array_equal(params["x"], [0, 10])
    np.testing.assert_array_equal(params["y"], [0, 2])
    assert np.isclose(variables.mean()["x"], 5)
    assert np.isclose(variables.mean()["y"], 1)
    assert np.isclose(variables.var()["x"], 100 / 12)
    assert np.isclose(variables.var()["y"], 1 / 3)
    assert variables.skew() == {"x": 0, "y": 0}
    assert np.isclose(variables.kurt()["x"], -1.2)
    assert np.isclose(variables.kurt()["y"], -1.2)
    np.testing.assert_allclose(variables.moments()["x"], [5, 100 / 12, 0, -1.2])


def test_pdf_logpdf_and_cdf_joint_and_marginal(variables):
    points = np.array([[5.0, 15.0], [1.0, 1.0]])

    np.testing.assert_allclose(
        variables.pdf(points, marginal=True), [[0.1, 0], [0.5, 0.5]]
    )
    np.testing.assert_allclose(variables.pdf(points), [[0.05, 0]])
    np.testing.assert_allclose(
        variables.logpdf(points, marginal=True),
        [[np.log(0.1), -np.inf], [np.log(0.5), np.log(0.5)]],
    )
    np.testing.assert_allclose(variables.logpdf(points), [[np.log(0.05), -np.inf]])
    np.testing.assert_allclose(
        variables.cdf(points, marginal=True), [[0.5, 1.0], [0.5, 0.5]]
    )
    np.testing.assert_allclose(variables.cdf(points), [[0.25, 0.5]])


@pytest.mark.parametrize("method_name", ["pdf", "logpdf", "cdf"])
def test_density_methods_reject_non_array_invalid_marginal_and_wrong_variable_count(
    variables, method_name
):
    method = getattr(variables, method_name)
    with pytest.raises(ValueError, match="numpy.ndarray"):
        method([[1.0], [1.0]])
    with pytest.raises(ValueError, match="marginal"):
        method(np.ones((2, 1)), marginal=1)
    with pytest.raises(ValueError, match="num_variables"):
        method(np.ones((1, 2)))


def test_ppf_round_trips_cdf_and_validates_shape(variables):
    probabilities = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    values = variables.ppf(probabilities)

    np.testing.assert_allclose(values, [[0, 5, 10], [0, 1, 2]])
    np.testing.assert_allclose(variables.cdf(values, marginal=True), probabilities)
    with pytest.raises(ValueError, match="numpy.ndarray"):
        variables.ppf([[0.5], [0.5]])
    with pytest.raises(ValueError, match="num_variables"):
        variables.ppf(np.ones((1, 2)))


def test_sample_returns_each_variable_in_a_row_and_respects_bounds(variables):
    samples = variables.sample(16, seed=123)

    assert samples.shape == (2, 16)
    assert np.all((samples[0] >= 0) & (samples[0] <= 10))
    assert np.all((samples[1] >= 0) & (samples[1] <= 2))
    np.testing.assert_array_equal(samples, variables.sample(16, seed=123))


def test_saltelli_sampling_shape_for_first_and_second_order(variables):
    first_order = variables.sample_saltelli(4, calc_second_order=False, seed=10)
    second_order = variables.sample_saltelli(4, calc_second_order=True, seed=10)

    assert first_order.shape == (16, 2)
    assert second_order.shape == (24, 2)
    assert np.all((first_order[:, 0] >= 0) & (first_order[:, 0] <= 10))
    assert np.all((second_order[:, 1] >= 0) & (second_order[:, 1] <= 2))


def test_bounds_and_gpc_helpers():
    variables = VariableSet(
        [
            Variable("x", ArcsineDistribution(scale=2, loc=-1)),
            Variable("y", UniformDistribution(-1, 1)),
        ],
        normalized=True,
    )

    bounds = variables.get_bounds()
    np.testing.assert_allclose(
        bounds["x"], variables.variables["x"].get_bounds()
    )
    assert isinstance(variables.get_gpc_polysys()["x"], ChebyshevTPolynomials)
    assert variables.get_gpc_syschars() == {"x": "t", "y": "p"}

    variables.normalized = False
    assert variables.get_gpc_syschars() == {"x": "T", "y": "P"}


def test_germ_and_distribution_conversions_are_round_trip():
    variables = VariableSet(
        [
            Variable("x", ArcsineDistribution(scale=2, loc=-1)),
            Variable("y", NormalDistribution()),
        ]
    )
    germs = np.array([[-0.8, 0.0, 0.9], [-1.0, 0.0, 1.0]])

    np.testing.assert_allclose(
        variables.dist_to_germ(variables.germ_to_dist(germs)), germs, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        variables.germ_to_stdnor(variables.stdnor_to_germ(germs)), germs, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    ("method_name", "argument"),
    [
        ("stdnor_to_germ", np.ones((1, 2))),
        ("germ_to_stdnor", np.ones((1, 2))),
        ("germ_to_dist", np.ones((1, 2))),
        ("dist_to_germ", np.ones((1, 2))),
    ],
)
def test_coordinate_conversions_reject_wrong_variable_count(
    variables, method_name, argument
):
    with pytest.raises(ValueError, match="num_variables"):
        getattr(variables, method_name)(argument)


def test_coordinate_conversions_reject_non_array_inputs(variables):
    for method in (
        variables.stdnor_to_germ,
        variables.germ_to_stdnor,
        variables.germ_to_dist,
        variables.dist_to_germ,
    ):
        with pytest.raises(ValueError, match="numpy.ndarray"):
            method([[0.0], [0.0]])


@pytest.mark.parametrize(
    "method_name",
    [
        "pdf",
        "logpdf",
        "cdf",
        "ppf",
        "stdnor_to_germ",
        "germ_to_stdnor",
        "germ_to_dist",
        "dist_to_germ",
    ],
)
def test_matrix_methods_reject_one_dimensional_input(variables, method_name):
    with pytest.raises(ValueError, match="shape"):
        getattr(variables, method_name)(np.array([0.5, 0.5]))


def test_filter_select_and_without_preserve_order_and_normalization(variables):
    variables.normalized = False
    variables.add(Variable("z", NormalDistribution()))

    filtered = variables.filter(lambda variable: variable.name != "y")
    selected = variables.select(["z", "x"])
    indexed = variables.select([2, 0], by="index")
    remaining = variables.without(["y"])
    remaining_by_index = variables.without([1], by="index")

    assert filtered.get_variable_names() == ["x", "z"]
    assert selected.get_variable_names() == ["z", "x"]
    assert indexed.get_variable_names() == ["z", "x"]
    assert remaining.get_variable_names() == ["x", "z"]
    assert remaining_by_index.get_variable_names() == ["x", "z"]
    assert all(
        result.normalized is False
        for result in (filtered, selected, indexed, remaining, remaining_by_index)
    )


def test_select_and_without_report_invalid_names_modes_and_indices(variables):
    with pytest.raises(KeyError, match="not found"):
        variables.select(["missing"])
    with pytest.raises(IndexError, match="out of range"):
        variables.select([-1], by="index")
    with pytest.raises(IndexError, match="out of range"):
        variables.select([2], by="index")
    with pytest.raises(ValueError, match="by must be"):
        variables.select(["x"], by="unknown")
    with pytest.raises(ValueError, match="by must be"):
        variables.without(["x"], by="unknown")
    with pytest.raises(IndexError, match="out of range"):
        variables.without([-1], by="index")
    with pytest.raises(IndexError, match="out of range"):
        variables.without([2], by="index")

    assert variables.without(["missing"]).get_variable_names() == ["x", "y"]