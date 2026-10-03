from __future__ import annotations

import os

import anthropic
import pytest

from poc import bench


def test_live_environment_and_anthropic_client_are_blocked_by_default():
    guarded_names = ("ANTHROPIC_API_KEY", "ENABLE_LIVE", "DM_MAX_USD")
    assert all(name not in os.environ for name in guarded_names)

    bench.load_env()
    assert all(name not in os.environ for name in guarded_names)

    with pytest.raises(RuntimeError) as exc:
        anthropic.Anthropic(api_key="test-only-placeholder")
    assert str(exc.value) == "real Anthropic client constructed in tests"
