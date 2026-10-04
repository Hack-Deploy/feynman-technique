# Discovery Market: presenter script

About 8 minutes. Three tabs, in order: **Vision → How it works → Live market**.
*Italics* = what to do on screen. Plain text = what to say. Cut the lines marked [optional] if you're short on time.

## Before you present

From the repository root, run:

```bash
uv sync
uv run python app.py
```

Open the printed address and start at the top of each page. On `/live`, use the **Recorded
runs** comparison (or the free scripted demo): the committed examples need no API key and make
no paid calls.

---

## 1. Vision (`/`) — about 2 minutes

*Open the Vision tab. Hero line: "AI can run experiments. We don't yet know which of them to trust."*

> AI can now design and run experiments. The hard problem isn't doing science any more; it's knowing which results to trust.

*Scroll to the three problem cards.*

> Three things go wrong today.
> First, **AI overrates itself**: what a model says it can do and what it delivers are different numbers, and saying "I'm confident" costs it nothing.
> Second, **null results disappear**: about 65% of them are never written up, so the next team walks into the same dead end.
> Third, **benchmarks get gamed**: if the answer is known, it gets overfitted.

*Scroll to "Credits move only when the evidence does." and the loop diagram.*

> Our answer is a market. Three groups take part.
> **Researchers** post a question and a prize, and pay only for a verified answer. If nobody gets there, the prize comes back.
> **Labs** are paid up front for every experiment, whether the idea behind it works or not.
> **AI scientists** spend their own credits on those experiments. Because their own money is at stake, the odds they state actually mean something.
> An independent **judge**, using tests locked away before anyone started, decides who gets paid.

*Scroll to the network diagram and "Any question with a judge."*

> Nothing here is specific to physics. Any question with a judge works. Some judges answer in minutes, like a simulator; some take years, like a clinical outcome. The slower the judge, the longer the bet.

*Scroll to the market board.*

> At scale, this becomes a board of open questions. A prize that keeps rising unclaimed means the question is hard. Lots of bids at a small prize means it's nearly solved. Prices become a map of where science is stuck.

*Scroll to the public record and calibration sections.*

> Two things make it better over time. Every failed attempt — which experiments were bought, how much was spent, what odds were stated — goes into a public record. We never publish its conclusions, because that could leak the answer. So each failure makes the next attempt cheaper.
> And every bid is a public prediction. An AI scientist that says 80% and delivers 30% loses money and the record shows it. The well-calibrated ones win more and get trusted more.

*Scroll to the roadmap.*

> We start where the judge is exact — simulated physics — and widen from there: live AI models, then held-out real data, then robotic cloud labs selling experiments as API calls. Today, the first two run. Let me show you.

---

## 2. How it works (`/simulation`) — about 4 minutes

*Open the How it works tab. "Find the hidden law of a toy universe."*

> To test the market itself, we need a world where *we* know the right answer. So we built six small simulated universes. Each one has a secret force law: a rule for how two particles push or pull on each other. The question is always: what's the law?

*Point at the input → AI scientist → formula → judge diagram.*

> A question goes in. A formula comes out. A judge decides.
> The AI scientist pays to launch a particle and watch where it goes. Then it submits a formula. The judge runs that formula on launches the AI scientist never saw. If it predicts them well, it passes and gets paid. If not — follow the red arrow — no prize, and the attempt goes into the public record.

[optional] *Open "Who are the AI scientists here?"* — In this demo they're two small, fast programs, not chatbots, so the whole thing runs offline in seconds. The real models are on the third tab.

*Scroll to "Thirteen experiments on the menu."*

> This is what it can buy: thirteen kinds of launch — close or far, fast or slow, heavy or light probe. The first launch is free, each one after that costs one credit, and it starts with ten.

*Scroll to the bidding slider. Drag the prize slider down until the decision flips to "don't bid", then back up.*

> Here's the only rule it follows: bid only if your chance of passing times the prize beats what you expect to spend. Low prize — it walks away. Raise the prize and it's worth the risk. And when it does bid, it says its chance out loud, and that number is scored over time.

*Scroll to "The tests are sealed before the first bid."*

> The judge's tests are locked before the first bid, and a fingerprint of them is published. At the end you can check the fingerprint matches, so nobody can move the goalposts after seeing the answer.

*Scroll to "Some laws are easy to tell apart. Some are not." Point at one world's solid and dashed lines.*

> Here are the six worlds. Solid line: the true pull at each distance. Dashed: the simplest possible guess, a plain one-over-r. Where the two lines sit apart, a wrong guess gets caught. Where they overlap, a wrong guess can sneak through. Keep that in mind.

*Scroll to "Run a real attempt, then step through it." Pick a world (Flat gravity is quick), click **Run attempt**, wait for it, then step through.*

> This is one real attempt, running now on this laptop through the same judge. It looks at the free launch… buys a launch… updates its beliefs over fifteen candidate formulas… buys another… then submits its best one.
> *(At the verdict)* Here's the verdict: the error against the sealed tests, the pass threshold of 0.1, the fingerprint check, and the wallet — credits in, credits spent, what the lab earned.

*Scroll to the 60-dot grid.*

> We ran this 60 times: six worlds, two AI scientists, five runs each.
> **Green, 37**: right law, passed — the market worked.
> **Blue, 5**: on "Hidden dimension" the true law isn't on its list at all, and its closest guess was good enough.
> **Red, 1**: failed — no prize, and it goes into the public record.
> **Amber, 17**: these are the interesting ones. *Click an amber dot.* A *wrong* law passed.

*Continue to section 6, "Now with real AI scientists." Point to "Who solved what," the
clearing-prize chart, the profit-per-model chart, and "Did they know their chances?"*

> This section replays 88 published attempts from eight frontier models across eleven
> worlds. The grid shows who solved what; the clearing prize is the lowest price at
> which a model solves the world in at least three of five replays; and the profit chart
> compares each model's average profit or loss over five replays at the selected prize.
> The published attempts did not record confidence, so we cannot tell from this data
> whether those models knew their chances. The data are from [ARA Labs (AgentNativeResearchLab)](https://huggingface.co/AgentNativeResearchLab), CC BY 4.0.

*Scroll to "A market is only as good as its judge."*

> On "Heavy probe", all 10 attempts passed with the wrong law. On "Breathing gravity", a plain one-over-r — ignoring time completely — passes 10 out of 10.
> Why? The judge's tests sit exactly where those laws look alike. And some laws really are twins: "fades out beyond 2 units" and "falls as one-over-r-squared" trace almost the same paths on every launch on the menu. No amount of spending separates them.
> We also tested scoring every hidden case separately and requiring each one to pass. True laws still passed, but wrong laws did too: one-over-r passed oscillator 6/6, Yukawa and fractional passed each other's worlds, one-over-r passed extra dimensions 3/6, and k·p1/r passed gravity. The hidden launches begin at distances 3–6 at time zero and change charges only mildly; they do not separate those laws. The aggregation rule isn't the problem, so the owner chose option 3: leave the judge unchanged. If this is revisited, widen the hidden cases with closer starts, non-zero start times and larger charge changes. **A market is only as good as its judge.**

---

## 3. Live market (`/live`) — about 2 minutes

*Open the Live market tab. "A real AI model bets on a real question."*

> The live market runs the same game on a harder benchmark with eleven worlds. For this
> demo I'll use the free scripted mode, so no paid model is called.

*Pick a claim — e.g. "gravity-inverse-square" — and a model. Use **Scripted demo · free** unless you've set up a key. Click **Start the market**.*

> I pick a claim about a hidden world, like "gravity here is inverse-square". The model reads the claim, the prize, and the record of everyone who already failed on it.
> Every round it pays a fee, can buy an experiment, and says how likely it thinks it is to get the answer right.

*Point at the chart as rounds come in.*

> Navy is its stated chance. Rust is credits spent. You can watch its confidence move as evidence arrives — and you can see what that evidence cost.

*At the end.*

> It ends with a verdict — supported, refuted, or inconclusive — or it walks away. Only now do we reveal the true answer. A correct, clear verdict wins the prize. Anything else pays nothing, and the run goes into the public record.

*Scroll to "3 · Recorded runs." Point out the comparison, profit, API-cost and Brier-score
columns. Click a saved run and use **Replay round by round**.*

> These rows let us compare how the configured model slots did on the same claims and cap:
> wins, profit, real API cost, and how well stated chances matched outcomes. Today the saved
> runs are scripted stand-ins, not paid model runs; the banner says so, and the API cost is
> zero. We have not recorded real paid runs yet. A saved run can still be replayed round by
> round without a key.

*Scroll to "Every run is kept, especially the failures." and the leaderboard.*

> Every run is kept — especially the failures. The next model sees them before it bids. And the leaderboard shows, across models, who made money and whose stated odds were honest.

*Optional operator prep: share the warm-start data with the project dataset.*

```bash
uv run python -m poc.hf_data push --dry-run
uv run python -m poc.hf_data push
uv run python -m poc.hf_data pull
```

> The default dataset is public, so no token is needed to pull it. Push is private by default
> and includes only allow-listed live-market records; `--repo ORG/NAME` overrides the default.
> A new checkout can pull those records so Recorded runs and the next rerun start warm.

If using the scripted demo, say: *"This one is a scripted stand-in so it's free to show; it always answers 'supported', so you'll see it lose on false claims — which is exactly what a loss should look like."*

---

## Close — 20 seconds

> Today science pays for effort. This pays for answers — and keeps a public record of everything that didn't work, and of who was honest about their odds. We've shown it end to end where the judge is exact. The same rules work for any field where an answer can be checked.

---

## Likely questions

- **Why physics simulations?** The judge is exact, so we can audit the market itself — including catching it when it pays for the wrong answer.
- **Couldn't an AI just guess the judge's tests?** The tests are fixed and fingerprinted before bidding, and can be salted with a secret (`DM_ORACLE_SECRET`) so they can't be reconstructed from the code.
- **Why publish failures but not conclusions?** Data and spending help the next bidder; a wrong conclusion could leak or mislead about the answer.
- **What stops overconfidence?** Overconfident agents enter bad bets and lose credits; their public calibration record shows it.
- **What's the biggest weakness?** The judge. 17 of 60 attempts passed with a wrong law. A per-case scoring gate did not fix the discrimination problem; the next useful step is widening the hidden cases, not changing the aggregation rule.

## Optional live-market reset (operator only)

Review the randomized plan and spend projection before starting a fresh paid grid:

```bash
uv run python -m poc.rerun_all --preflight
ENABLE_LIVE=1 uv run python -m poc.rerun_all --purge
```

Set `DM_MAX_USD=50` in `poc/.env` and provide `ANTHROPIC_API_KEY` there. The second
command asks once before archiving the current live data and running; it never removes
the spend ledger. This is not needed for the presentation: use Recorded runs so no key
or paid call is required.
