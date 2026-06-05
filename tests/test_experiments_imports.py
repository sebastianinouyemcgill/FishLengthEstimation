"""Import smoke tests for experiment package (no circular imports)."""


def test_experiments_package_imports() -> None:
    import src.experiments  # noqa: F401

    from src.experiments import (
        default_ground_truth_path,
        load_validation_image_ids,
        run_regression_experiment,
    )
    from src.experiments.notebook_helpers import run_configured_experiments

    assert callable(run_regression_experiment)
    assert callable(run_configured_experiments)
    assert callable(load_validation_image_ids)
    assert callable(default_ground_truth_path)
