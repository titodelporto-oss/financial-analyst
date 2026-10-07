"""Helpers to build small, hand-checkable annual statements."""

import pandas as pd
import pytest

from src.sector import SectorInfo
from src.statements import finalize


def make_annual(years: dict) -> pd.DataFrame:
    """years: {"2023-12-31": {"revenue": 100, ...}, ...} -> canonical annual frame."""
    df = pd.DataFrame.from_dict(years, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    return finalize(df)


@pytest.fixture
def manufacturer():
    return SectorInfo("manufacturing", "SIC (SEC)", "SIC 3711")


@pytest.fixture
def services():
    return SectorInfo("non_manufacturing", "SIC (SEC)", "SIC 7372")


@pytest.fixture
def bank():
    return SectorInfo("bank", "SIC (SEC)", "SIC 6021")
