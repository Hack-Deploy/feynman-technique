# ARA sample (test fixture)

A trimmed offline copy of a few published ARA DiscoverPhysics runs, used by
`tests/test_dm_importers.py` so tests never touch the network.

Data: **ARA Labs (AgentNativeResearchLab)**, CC BY 4.0,
<https://huggingface.co/AgentNativeResearchLab> (datasets
`discoverphysics-{gpt5.5,fable,opus4.8-max}-ara`).

Contents: each model's `SCOREBOARD.md` (verbatim) and, for a few worlds, `meta.json`,
`result.json`, `episode.json` and `posthoc_salvage.json`. Trimmed for size (values are
otherwise unchanged): `meta.agent_explanation` cut to 200 characters,
`result.evaluation.trajectories` dropped, and `episode.rounds[]` reduced to
`round`, `action`, `experiment_input`.

Worlds: gpt5.5/gravity (36 experiments over 14 rounds), fable/ether (post-hoc
salvage; meta.json mean_pos_error is inf), fable/coulomb_easy (scoreboard nMSE uses
a different Var than the vendor's `_WORLD_VARS`), opus4.8-max/ether (rounds: 6 in
scoreboard/meta, 4 in episode.json). Other scoreboard rows have no world files.
