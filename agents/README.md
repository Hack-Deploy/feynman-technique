# Submitting an agent

An agent is a git repository with `agent.py` at its root. Submit its https URL on the venue page
(or `examples/<name>` for one of the examples here). The market fetches the default branch,
pins the commit, and plays each bounty with it.

## One round

Each round the market runs `python agent.py` in the repo with one JSON object on stdin:

```json
{
  "protocol": "discoverphysics_market",
  "seed": 0,
  "call": 1,
  "bounty": {
    "id": "gravity-inverse-square", "world": "gravity",
    "hypothesis": "...", "resolution_criteria": "...",
    "prize": 140, "bond": 42, "particles": null,
    "quantities": [{"name": "n", "meaning": "...", "tolerance": 0.22}],
    "supported_if": {"quantity": "n", "kind": "between", "bounds": [1.8, 2.2]}
  },
  "prices": {"round_fee": 2, "experiment_costs": {"per_experiment": 1, "...": 1}},
  "max_rounds": 16,
  "system": "the system prompt a model would see",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

It prints its reply on stdout, in the same tagged format a model uses:

```
<assessment>what the data says so far</assessment>
<p_success>0.8</p_success>                      chance of a confirmed answer
first reply only: <planned_cost>40</planned_cost> (the bid is taken if p × prize > cost)
then one action:
<run_experiment>[ ...launches... ]</run_experiment>       paid to the lab
<verdict>supported | refuted | inconclusive</verdict>
  <estimate>n = 1.02 ± 0.05</estimate>  <evidence>why</evidence>
<withdraw>reason</withdraw>
```

Experiment results come back in the next user message inside `<experiment_output>`. Each round is
a fresh process: keep state by reading `messages`.

## What an agent can and cannot do

It can use the standard library, numpy, scipy, its own files, and the open measurement helpers
`poc.estimate`, `poc.reference` and `poc.pricing`. It cannot import the checker, the truth, the
oracle or the simulator; read the answer files or the attempt store; write outside its repo;
start processes; or use the network. Each round has a 60 s limit, and no API keys are passed in.
A crash or timeout counts as a withdrawal.

This isolation is best effort for a local demo, not a security boundary.

## Examples

- `examples/careful-lab`: runs a full experiment design and claims what its own fit implies.
- `examples/shortcut-labs`: wants "supported"; reruns one cheap launch and reports the best run.
- `examples/yes-man`: no experiments; says "supported" with the numbers the hypothesis implies.
