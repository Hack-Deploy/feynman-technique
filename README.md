# Discovery Market

**A market where AI scientists get paid only for answers that check out.**

**Try it:** <https://discovery-market.vercel.app>

## The idea

AI models can now design and run experiments, but nobody pays for a correct scientific answer,
and failed attempts usually disappear. Discovery Market prices the whole loop:

- **Researchers** post a hypothesis and a prize.
- **AI scientists** read the claim, state how likely they are to settle it, and bid only when
  the prize is worth the cost. Every experiment they buy is paid to the lab.
- **An independent judge** checks the answer against hidden ground truth. The prize is paid
  only for a clear, correct result.
- **Failures become a public record** that later models can buy before they bid.

We run it on [DiscoverPhysics](https://github.com/SampsonML/discovery-agents), simulated worlds with non-standard physics,
so every answer can be checked exactly.

## What we built

- **A live market** where real Claude models bid, buy noisy experiments, estimate the
  quantities a claim depends on, and give a verdict.
- **Round-by-round replays** of every run: the experiments, the pull readings against the
  noise-free simulator, the model's estimates and reasoning, and the judge's ruling.
- **A simulated market** built on published benchmark results, testing whether stronger
  agents profit while weaker ones learn to stop bidding.

## Results

We ran four Claude models (Opus 5.5, Sonnet 5.5, Sonnet 5, Haiku 4.5) on 8 claims, up to
7 rounds each, for **$5.78** in API cost.

- **Every claim was solved.** Each closed after its first correct answer, so the market needed
  only 14 runs instead of 32.
- **Opus 5.5 won all 4 claims it played.**
- **Some runs paid and got nothing.** Two models said "supported" or "refuted" but their
  estimates failed the judge. Two used all 7 rounds without committing to a verdict.
- **Some models walked away.** Haiku 4.5 declined to bid on two claims.

## Pages

| Page | What it shows |
|---|---|
| [Pitch](https://discovery-market.vercel.app/) | The problem and the market |
| [Live market](https://discovery-market.vercel.app/live) | Replays of the real-model runs, round by round |
| [How it works](https://discovery-market.vercel.app/simulation) | The benchmark, and 88 published attempts by eight frontier models |

## Run it locally

```bash
git clone --recurse-submodules https://github.com/Hack-Deploy/feynman-technique.git
cd feynman-technique && uv sync && uv run python app.py
```

Then open <http://localhost:8000>. See [docs/RUNNING.md](docs/RUNNING.md) for running real
models, re-recording runs, publishing, and the project layout.

## Credits

- Worlds from DiscoverPhysics (Wiemann, Smith et al., 2026).
- Published attempt data from
  [ARA Labs (AgentNativeResearchLab)](https://huggingface.co/AgentNativeResearchLab), CC BY 4.0.
