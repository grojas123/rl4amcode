from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl4am.config import DataConfig
from rl4am.data import CANONICAL_REMOTE_SOURCE, load_market_data, summarise_market_data


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "eod_fixture.csv"
EOD_DATA = ROOT / "data" / "eod_data.csv"


def test_load_market_data_for_symbol() -> None:
    cfg = DataConfig(
        source=str(EOD_DATA),
        symbol="EURUSD",
        date_column=None,
        price_column=None,
        return_type="simple",
    )

    data = load_market_data(cfg)

    assert data.symbol == "EURUSD"
    assert data.prices.shape[0] == 2514
    assert data.returns.shape[0] == 2513
    np.testing.assert_allclose(
        data.returns.iloc[:3].to_numpy(),
        [-0.0023954302561267626, -0.007018840044329511, 0.0025111607142858094],
        rtol=1e-6,
    )


def test_load_market_data_falls_back_to_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DataConfig(
        source=str(EOD_DATA),
        symbol="EURUSD",
        date_column=None,
        price_column=None,
        return_type="simple",
    )

    original_read_csv = pd.read_csv

    def fake_read_csv(source, *args, **kwargs):
        if source == str(EOD_DATA):
            raise OSError("local file unavailable")
        if source == CANONICAL_REMOTE_SOURCE:
            return original_read_csv(EOD_DATA, *args, **kwargs)
        return original_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    with pytest.warns(RuntimeWarning, match="Falling back"):
        data = load_market_data(cfg)

    assert data.symbol == "EURUSD"
    assert data.source == CANONICAL_REMOTE_SOURCE
    assert data.prices.shape[0] == 2514
    assert data.prices.iloc[0] == 1.0854
    assert data.prices.iloc[-1] == 1.1746


def test_summarise_market_data() -> None:
    cfg = DataConfig(
        source=str(FIXTURE),
        symbol="MSFT",
        date_column="date",
        price_column="MSFT",
        return_type="simple",
    )

    summary = summarise_market_data(load_market_data(cfg))

    assert summary["symbol"] == "MSFT"
    assert summary["rows"] == 4
    assert summary["return_rows"] == 3
    assert summary["first_price"] == 200.0
    assert summary["last_price"] == 204.0
