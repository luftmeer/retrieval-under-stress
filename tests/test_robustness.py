import pytest

from retrieval_lab.robustness import (
    bootstrap_means,
    paired_values,
    percentile,
)


def test_pairs_use_ids_and_only_the_stressed_subset():
    ids, clean, stressed, drops = paired_values(
        {"extra": 0.0, "q2": 0.5, "q1": 1.0},
        {"q2": 0.0, "q1": 0.75},
    )

    assert ids == ["q1", "q2"]
    assert clean == [1.0, 0.5]
    assert stressed == [0.75, 0.0]
    assert drops == [0.25, 0.5]

    with pytest.raises(ValueError, match="Missing paired"):
        paired_values(
            {"q1": 1.0},
            {"unknown": 0.0},
        )


def test_identical_pairs_have_zero_drop_and_zero_interval():
    _, _, _, drops = paired_values(
        {"a": 0.0, "b": 1.0},
        {"b": 1.0, "a": 0.0},
    )

    estimates, _ = bootstrap_means(
        {"drop": drops},
        n_resamples=100,
    )

    assert estimates["drop"] == {
        "estimate": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }


def test_methods_share_resampled_queries():
    estimates, samples = bootstrap_means(
        {
            "a": [0.0, 0.25, 0.5],
            "b": [0.25, 0.5, 0.75],
        },
        n_resamples=100,
    )

    assert estimates["a"]["estimate"] == pytest.approx(0.25)

    assert all(b - a == pytest.approx(0.25) for a, b in zip(samples["a"], samples["b"]))


def test_bootstrap_reproducibility_and_invalid_inputs():
    columns = {"drop": [-0.25, 0.0, 0.5]}

    assert bootstrap_means(
        columns,
        n_resamples=100,
    ) == bootstrap_means(
        columns,
        n_resamples=100,
    )

    invalid_inputs = (
        {},
        {"a": [0.0]},
        {"a": [0.0, 1.0], "b": [0.0]},
        {"a": [0.0, float("nan")]},
    )

    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            bootstrap_means(invalid)


def test_percentile_interpolation():
    assert percentile(
        [0.0, 10.0],
        0.025,
    ) == pytest.approx(0.25)

    assert percentile(
        [0.0, 10.0],
        0.975,
    ) == pytest.approx(9.75)
