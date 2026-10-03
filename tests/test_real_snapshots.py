"""The "Real attempts" page works from committed snapshots on a fresh, offline clone."""

import real_data


def _missing(monkeypatch, tmp_path):
    gen = {name: tmp_path / f"{name}.missing" for name in ("ARA_STORE", "ARA_SUMMARY", "FORCEBENCH_GRID")}
    snaps = {gen[name]: real_data.SNAPSHOTS[getattr(real_data, name)] for name in gen}
    for name, path in gen.items():
        monkeypatch.setattr(real_data, name, path)
    monkeypatch.setattr(real_data, "SNAPSHOTS", snaps)


def test_snapshots_are_committed():
    for snap in real_data.SNAPSHOTS.values():
        assert snap.exists(), snap


def test_page_data_falls_back_to_snapshots(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    data = real_data.build_real_data()
    ara, fb = data["ara"], data["forcebench"]
    assert ara["available"] and ara["snapshot"] and len(ara["attempts"]) == 88
    assert "ARA Labs" in ara["attribution"]
    assert fb["available"] and fb["snapshot"] and len(fb["table"]) == 12
    assert len(fb["attempts"]) == 60
    assert all(attempt["top_model"] is not None for attempt in fb["attempts"])


def test_generated_outputs_take_precedence(monkeypatch, tmp_path):
    _missing(monkeypatch, tmp_path)
    real_data.FORCEBENCH_GRID.write_text('{"results": [], "head": "x", "test_seed": 0}')
    fb = real_data.forcebench_data()
    assert fb["available"] and not fb["snapshot"] and fb["table"] == []
