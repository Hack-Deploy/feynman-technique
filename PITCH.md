# Discovery Market: 3-minute pitch

For the judged slot: 1:30 pitch, then 1:30 live demo. `DEMO.md` is the longer 8-minute walkthrough.

*Italics* = what to do on screen. Quotes = what to say.

## Before you present

```bash
uv sync
uv run python -m poc.hf_data pull
uv run python app.py
```

Open the **Experiments** tab (`/experiments`) and leave it on **three-species-repulsion**.
Pick a run with a red (wrong) verdict card in advance so you don't hunt for it live.
Replace the 85% below with the stated chance shown in the run you use.

---

## Pitch (about 1:30)

> Imagine you could make scientific discoveries at the speed of thought. You have a hypothesis in the morning, and the scientific community tells you if you're right by the evening.
>
> That's not how science works today, and there are two reasons.
>
> First, most people with good ideas can't test them. The labs, the equipment and the money sit with a few universities and a few big companies. If you can't access those resources, your hypothesis cannot be tested.
>
> Second, the people who do have the resources spend them on a few ideas. Experiments fail, that's science, but nobody publishes the failure, and the next team pays to make the same mistake.
>
> We built Discovery Market to fix both. Anyone can post a hypothesis with a prize. Here is a researcher with a hypothesis. Does a specific peptide-HLA pair remain bound for more than an hour so that our immune system can detect it? Does perovskite improve the efficiency of a solar panel under humid conditions? Labs, autonomous or otherwise, have specialized equipment. AI Scientists compete to propose test experiments and they pay for their own proposals. They only get the prize if an independent judge confirms the answer. If nobody gets there, the prize pool remains unclaimed.
>
> Because the AI is spending its own credits, it only goes after ideas it thinks it can actually answer. And every attempt, including the ones that fail, goes on a public record, so nobody pays for the same dead end result twice.
>
> We tested this on simulated physics sandboxes. The AI agents have to guess the physics laws in the sandbox by proposing particle interaction experiments. We already know the right answers and we can judge real Claude models betting real credits. Check out our live market to see which models are over-confident and which can't function with real budget constraints.

## Demo (about 1:30)

*Experiments tab, three-species-repulsion, the replay playing.*

> This is one of the worlds. The claim is that some of these particles push others away. The orange dots are probes the AI chose to place, and each yellow ring is a measurement it paid for.

*Point at the probes curving away, then the rounds panel.*

> You can see the probes curve away here. Before each step the AI says how sure it is. This one said 85%, and then gave its answer: yes, some particles repel. The judge checks that against the hidden answer. It's right, so it gets the prize.

*Switch to the run you picked with a red verdict card.*

> This one got it wrong. It doesn't get paid, and its attempt goes on the record for whoever tries next.

> Next we want to do this with real lab data, starting with protein binding, where the answer comes from a measurement nobody has seen yet.

---

## Notes

- Keep the protein line framed as "next" unless the protein example is validated and working by the time you present.
- Read it aloud with a timer once. Trim the pitch section first if it runs long.
