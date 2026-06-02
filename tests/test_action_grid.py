import numpy as np
import pytest

from rl4am.agents.action_grid import ActionGrid


def test_uniform_action_grid() -> None:
    grid = ActionGrid.uniform(bins=5, min_weight=0.0, max_weight=1.0)

    np.testing.assert_allclose(grid.weights, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert grid.size == 5
    assert grid.nearest_index(0.62) == 2
    assert grid.nearest_weight(0.62) == pytest.approx(0.5)


def test_action_grid_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        ActionGrid.uniform(bins=1)
    with pytest.raises(ValueError, match="ascending"):
        ActionGrid(weights=np.array([0.5, 0.0, 1.0]))


def test_action_grid_clamps_nearest_weight_lookup() -> None:
    grid = ActionGrid.uniform(bins=5)

    assert grid.nearest_index(-0.5) == 0
    assert grid.nearest_index(1.5) == 4
    assert grid.nearest_weight(-0.5) == pytest.approx(0.0)
    assert grid.nearest_weight(1.5) == pytest.approx(1.0)
