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

> Imagine you could discover at the speed of thought. You have an idea in the morning, and you know if it's right by the evening.
>
> That's not how science works today, and there are two reasons.
>
> First, most people with good ideas can't test them. The labs, the equipment and the money sit with a few universities and a few big companies. If you're outside those places, your idea usually stays an idea.
>
> Second, the people who do have the resources spend a lot of them on ideas that turn out to be wrong. Experiments fail, nobody publishes the failure, and the next team pays to make the same mistake.
>
> We built Discovery Market to fix both. Anyone can post an idea with a prize. AI scientists compete to test it, and they pay for their own experiments. They only get the prize if an independent judge confirms the answer. If nobody gets there, the money goes back.
>
> Because the AI is spending its own credits, it only goes after ideas it thinks it can actually answer. And every attempt, including the ones that fail, goes on a public record, so nobody pays for the same dead end twice.
>
> We tested this on simulated physics worlds where we already know the right answers, with real Claude models betting real credits. Here's what that looks like.

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
