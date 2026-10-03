# Demo snapshots

Used by the results app's "Real attempts" page (`real_data.py`) when the generated outputs are
missing (fresh clone, no internet, no time to run the grid). The page labels them as snapshots.
The buttons on the page still regenerate the real outputs, which then take precedence.

| File | Regenerate with | Source |
|---|---|---|
| `ara.jsonl` | `uv run python -m dm.importers.ara` | 88 ARA attempts (8 models × 11 worlds). Data: ARA Labs (AgentNativeResearchLab), CC BY 4.0, https://huggingface.co/AgentNativeResearchLab; attribution in every record |
| `replay_ara_summary.json` | `uv run python -m dm.replay ara` | Replay market over the ARA pool |
| `forcebench_settle.json` | `uv run python -m tests.forcebench_settle` | 6 worlds × 5 seeds × {bayes_lite, random_menu}, settled by the oracle (unsalted, test seed 0) |
