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
    config.addinivalue_line(
        "markers",
        "allow_live_env: opt out of the default environment and Anthropic client guard",
    )


@pytest.fixture(autouse=True)
def deny_live_api_by_default(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("allow_live_env") is not None:
        return

    for name in ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD", "HF_TOKEN", "DM_HF_REPO"):
        monkeypatch.delenv(name, raising=False)

    from poc import bench

    monkeypatch.setattr(bench, "load_env", lambda: None)

    try:
        import anthropic
    except ImportError:
        return

    def reject_client(*args: object, **kwargs: object) -> None:
        raise RuntimeError("real Anthropic client constructed in tests")

    monkeypatch.setattr(anthropic.Anthropic, "__init__", reject_client)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    marker = pytest.mark.skip(reason="use --runslow to run this test")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(marker)
