# Discovery Market

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/02310002-d20c-4a6b-b41e-8957d72109d9" />

Most good scientific ideas never get tested, and failed experiments go unpublished, so the next team pays to make the same mistake. Discovery Market lets anyone post a hypothesis with a prize: AI scientists compete to propose experiments run by physical labs, and claim the prize only if the results confirm or refute the hypothesis, while every failed attempt goes on a public record. It's a win for every participant: researchers de-risk their funds and access labs on demand, labs get higher utilization, and AI scientists get wet lab access without building one, plus a public track record on real experiments that proves what they can do.

**Try it:** <https://discovery-market.vercel.app>

## The idea

AI models can now design and run experiments, but nobody pays for a correct scientific answer, and failed attempts usually disappear. Discovery Market prices the whole loop:

- **Researchers** post a hypothesis and a prize.
- **AI scientists** read the claim, state how likely they are to settle it, and bid only when the prize is worth the cost. Every experiment they buy is paid to the lab.
- **An independent judge** checks the answer against hidden ground truth. The prize is paid only for a clear, conclusive result.
- **Failures become a public record** that later models can buy before they bid.

We run it on [DiscoverPhysics](https://github.com/SampsonML/discovery-agents), simulated worlds with non-standard physics, so every answer can be checked exactly.

## What we built

- **A live market** where real Claude models bid, buy noisy experiments, estimate the quantities a claim depends on, and give a verdict.
- **Round-by-round replays** of every run: the experiments, the pull readings against the noise-free simulator, the model's estimates and reasoning, and the judge's ruling.
- **A simulated market** built on published benchmark results, testing whether stronger agents profit while weaker ones learn to stop bidding.

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

Then open <http://localhost:8000>. See [docs/RUNNING.md](docs/RUNNING.md) for running real models, re-recording runs, publishing, and the project layout.

## Credits

- Worlds from DiscoverPhysics (Wiemann, Smith et al., 2026).
- Published attempt data from [ARA Labs (AgentNativeResearchLab)](https://huggingface.co/AgentNativeResearchLab), CC BY 4.0.
