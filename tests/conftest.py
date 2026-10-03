"""Shared pytest options for long-running benchmark coverage."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as slow",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running benchmark coverage")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    marker = pytest.mark.skip(reason="use --runslow to run this test")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(marker)
