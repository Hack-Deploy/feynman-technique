# Discovery Market: literature and startup review

Review done on 2026-10-03 to back up slide 2, "What's new, and why it matters". Every claim below links to its source. Things we could not verify are listed at the end.

## Bottom line

The neighbouring fields each cover one piece of Discovery Market:

- AI scientists come up with ideas and test them.
- Cloud labs and lab marketplaces run experiments for a fee.
- Science prediction markets forecast whether published claims hold up.

We found no platform that pays AI scientists only for clear answers from real labs. We also found none that scores the confidence they bid with against what happened.

## Landscape

| Area | Examples | What they do | What's missing compared to Discovery Market |
|---|---|---|---|
| AI scientists | [Sakana AI Scientist v2](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf), [Google Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/), FutureHouse Robin / [Edison Kosmos](https://intuitionlabs.ai/articles/futurehouse-ai-agents-platform) | Generate hypotheses, design experiments, write papers. Co-Scientist targets the thinking, not the doing: it ["doesn't run experiments"](https://labcritics.com/blog/2026/05/21/google-deepminds-co-scientist-graduates-from-research-demo-to-nature-paper/). | No money on the line and no payment tied to a clear result. |
| AI-run lab startups | [Lila Sciences](https://www.biopharmadive.com/news/lila-flagship-ai-superintelligence-startup-seed/742213/) ($550M raised), [Periodic Labs](https://www.techbuzz.ai/articles/periodic-labs-raises-record-300m-seed-to-build-ai-scientists) ($300M seed) | Own vertically integrated autonomous labs. | Closed systems, not an open market. Investors pay before results. Startup Fortune: ["The next test is whether automated science can produce repeated, valuable results"](https://startupfortune.com/lila-sciences-is-testing-how-much-investors-will-pay-for-automated-labs/). |
| Cloud labs and lab marketplaces | [Science Exchange, Emerald Cloud Lab, Strateos](https://nordicapis.com/exploring-the-cloud-laboratory-advances-in-biotech-science-as-a-service/) | Run outsourced experiments. Pricing is by subscription or fee per service. | You pay for the run whether or not it gives an answer. |
| Science prediction markets | [DARPA SCORE / Replication Markets](https://ar5iv.labs.arxiv.org/html/2005.04543), [AI betting agents](https://arxiv.org/pdf/2303.00866), Hanson's [idea futures](https://en.wikipedia.org/wiki/Robin_Hanson) | Forecast whether published claims will replicate. Markets beat surveys at this. | They forecast existing claims and don't fund or run new experiments. |
| Science funding platforms | [Experiment.com](https://experiment.com/how-it-works) (all-or-nothing crowdfunding), [ResearchHub](https://www.insidephilanthropy.com/home/researchhub-an-answer-to-dysfunction-in-traditional-science-giving) (crypto rewards, including for negative results) | Fund human researchers up front or reward activity. | No AI agents and no calibration scoring. |
| Agent marketplaces | [MarketBench](https://arxiv.org/abs/2604.23897) (benchmark), [Agent Exchange](https://open-experiments.github.io/agent-exchange/) (open source, enterprise tasks) | Success-contingent pay for agent tasks: software and enterprise workflows. | Not science and not real labs. |

## Evidence for "why it matters"

1. **AI agents overrate themselves.** "Some agents that succeed only 22% of the time predict 77% success." Source: Kaddour et al., [Agentic Uncertainty Reveals Agentic Overconfidence](https://arxiv.org/abs/2602.06948), ICLR 2026.
2. **Self-assessment is the bottleneck for agent markets.** LLMs "are miscalibrated on both success probability and token usage, and auctions built from these self-reports diverge from a full-information allocation." Source: Fradkin & Krishnan, [MarketBench](https://arxiv.org/abs/2604.23897), 2026.
3. **Null results disappear.** 65% of null results were never written up, and only 20% appeared in print. Source: Franco, Malhotra & Simonovits, *Science* 2014 ([summary](https://www.bitss.org/?p=1081)).
4. **Money is ahead of proof.** Lila Sciences has raised $550M in total and is reportedly in talks at an $8.5B valuation ([Startup Fortune, citing Reuters](https://startupfortune.com/lila-sciences-is-testing-how-much-investors-will-pay-for-automated-labs/)). Periodic Labs launched with a $300M seed ([TechBuzz](https://www.techbuzz.ai/articles/periodic-labs-raises-record-300m-seed-to-build-ai-scientists)).

## Caveats

- "We found none" is the result of this review, not a proof that nothing exists. Closest non-science prior art: [MarketBench](https://arxiv.org/abs/2604.23897) and [Agent Exchange](https://open-experiments.github.io/agent-exchange/), which apply success-contingent pay to software and enterprise tasks.
- Hanson has proposed market systems for science [since 1988](https://en.wikipedia.org/wiki/Robin_Hanson), so "markets for science" alone is not new. The new part is the combination: AI scientists, real labs, payment only for a clear answer, and calibration scoring.
- Not used, because it couldn't be verified: a claim that independent evaluation of AI scientists outside curated demos is scarce. The source PDF could not be checked.
