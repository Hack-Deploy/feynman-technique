"""The "Real attempts" page works from committed snapshots on a fresh, offline clone."""

import json

import real_data


def test_law_summary():
    assert real_data._law_summary(None) is None
    assert real_data._law_summary("def f(:") is None
    assert real_data._law_summary("def f():\n    return 1\n") is None
    assert real_data._law_summary(
        'def f():\n    """\n    First line.\n    More.\n    """\n'
    ) == "First line."
    assert real_data._law_summary(
        'def first():\n    return 1\n\n'
        'def second():\n    """Documented."""\n    return 2\n'
    ) is None


def _missing(monkeypatch, tmp_path):
    gen = {name: tmp_path / f"{name}.missing" for name in ("ARA_STORE", "ARA_SUMMARY", "FORCEBENCH_GRID", "FORCEBENCH_REPLAY_SUMMARY")}
    snaps = {gen[name]: real_data.SNAPSHOTS[getattr(real_data, name)] for name in gen}
    for name, path in gen.items():
        monkeypatch.setattr(real_data, name, path)
    monkeypatch.setattr(real_data, "SNAPSHOTS", snaps)


def _legacy_grid_row():
    return {
        "solver": "bayes_lite",
        "world": "gravity",
        "seed": 0,
        "passed": True,
        "identified": True,
        "nmse": 0.01,
        "baseline_passed": True,
        "experiments": 1,
        "stated_p": 0.5,
    }


def test_snapshots_are_committed():
    for snap in real_data.SNAPSHOTS.values():
        assert snap.exists(), snap


def test_page_data_falls_back_to_snapshots(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    data = real_data.build_real_data()
    ara, fb = data["ara"], data["forcebench"]
    assert ara["available"] and ara["snapshot"] and len(ara["attempts"]) == 88
    assert all(isinstance(attempt.get("law"), str) and attempt["law"] for attempt in ara["attempts"])
    circle_fable = next(
        attempt for attempt in ara["attempts"]
        if attempt["world"] == "circle" and attempt["model"] == "fable"
    )
    assert circle_fable["law_summary"] == "Universal pairwise attraction with a 3/2-power distance law."
    assert "ARA Labs" in ara["attribution"]
    assert fb["available"] and fb["snapshot"] and len(fb["table"]) == 12
    assert len(fb["attempts"]) == 60
    assert all(attempt["top_model"] is not None for attempt in fb["attempts"])
    assert set(ara["hypotheses"]) == {"H1", "H2", "H3", "H4"}
    assert ara["hypotheses"]["H4"]["verdict"] == "UNAVAILABLE"
    assert ara["calibration"]["n"] == 0 and ara["calibration"]["n_excluded_no_stated_p"] == 88
    fbr = data["forcebench_replay"]
    assert fbr["available"] and fbr["snapshot"]
    assert fbr["calibration"]["n"] == 60 and fbr["calibration"]["mean_stated_p"] is not None


def test_generated_outputs_take_precedence(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    real_data.FORCEBENCH_GRID.write_text('{"results": [], "head": "x", "test_seed": 0}')
    fb = real_data.forcebench_data()
    assert fb["available"] and not fb["snapshot"] and fb["table"] == []


def test_stale_generated_grid_falls_back_to_snapshot(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    real_data.FORCEBENCH_GRID.write_text(
        json.dumps({"results": [_legacy_grid_row()]})
    )

    fb = real_data.forcebench_data()

    assert fb["snapshot"]
    assert len(fb["attempts"]) == 60
    assert all(attempt["top_model"] for attempt in fb["attempts"])


def test_generated_grid_with_null_top_model_remains_output(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    row = _legacy_grid_row()
    row["top_model"] = None
    real_data.FORCEBENCH_GRID.write_text(json.dumps({"results": [row]}))

    fb = real_data.forcebench_data()

    assert not fb["snapshot"]
    assert fb["source"] == "output"
    assert len(fb["attempts"]) == 1
