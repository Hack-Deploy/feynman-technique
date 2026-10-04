# The biology behind the protein venue

Background for anyone presenting or extending `/protein`. Written for people who
have not done immunology.

## What is going on in the cell

Every cell constantly chops up the proteins inside it into short fragments and loads
those fragments onto a molecule called HLA, which carries them to the cell surface.
It is a display window: here is what I am making inside.

T cells patrol and inspect those windows. A cell infected with a virus, or one that has
turned cancerous, ends up displaying fragments that do not belong. A T cell that sees one
kills the cell. This is how the immune system sees inside cells without opening them.

## Why the binding time is the thing worth measuring

The fragment sits in a groove on the HLA molecule. It is not bolted in; it drifts out over
time. Half-life is how long until half the copies have fallen out.

That is the whole point. A T cell has to physically arrive and inspect the cell. A fragment
that falls out in two minutes leaves an empty window, and nothing happens. One that stays for
hours gets seen. Stability is therefore one of the better predictors of whether a peptide
provokes a real immune response, rather than merely binding on paper.

## Why every claim names an allele

HLA comes in variants called alleles, and everyone inherits a different set. This is why
organ transplants need donor matching, and why the table lists HLA-A\*02:01, HLA-B\*15:01
and others.

Each variant has a differently shaped groove, so each holds different peptides. A fragment
that sticks for twenty hours on one allele can fall straight out of another. A measurement is
only meaningful as a peptide-and-allele pair, so every claim names both.

## The anchor residues in the groove diagram

These peptides are all nine amino acids long, which is typical: this class of HLA groove is
closed at both ends and fits roughly eight to eleven.

Positions 2 and 9 drop into deep pockets at either end and do most of the gripping. The
residues between them bulge upward, and that is the part a T cell reads. Each allele prefers
particular amino acids in those two pockets. That preference is the allele's motif, and it is
most of what any prediction tool picks up on.

## Why anyone pays for this

Cancer vaccines and immunotherapy. A tumour carries mutated proteins, and the task is to pick
the fragments most likely to be displayed long enough to trigger a T cell. There are far too
many candidates to test in a lab, so people predict first and test the shortlist.
NetMHCstabpan, named in the Track 3 brief, exists for this.

It makes a fair test for the market: a real prediction problem people want solved, not a
puzzle we invented.

## Say these out loud before anyone asks

**The one-hour threshold is our choice, not a field standard.** It was picked because the
dataset median is 1.1 hours, so it splits the data near the middle.

**The six claims are hand-picked, not sampled.** Two are chosen so the allele's base rate
points the wrong way. That is the point of the test, but it should be stated, not discovered.

**Stability is one input, not the whole story.** Whether a peptide actually triggers a
response also depends on whether it is produced and processed at all, and whether a T cell
exists that recognises it. We test prediction of one measured quantity, not immunogenicity.

## Verify before presenting as fact

This is textbook immunology, but nobody on the team checked it against a source. If there is
an immunologist at hand, a two-minute read of this page is worth it, particularly for anything
stated as fact rather than as our own design choice.

## Explaining it to the team in two minutes

Say it in this order. Each paragraph is one breath.

> Our physics worlds prove the market works when we already know the answer. This is the same
> market, but the answer is a measurement somebody made in a lab.

> Every cell shreds the proteins inside it and displays the fragments on its surface, held in
> a groove by a protein called HLA. T cells walk past and read those fragments. If one looks
> wrong, from a virus or a tumour, the T cell kills the cell.

> The fragment does not stay in the groove forever. Half-life is how long until half of them
> have fallen out. That matters because a T cell has to actually turn up and look. Fall out in
> two minutes and nobody sees you. Stay for hours and you get seen. So predicting that number
> is a real job people do, for cancer vaccines.

> We took 28,165 of those measurements, hid six, and asked an AI to predict them. It pays per
> round and per measurement it looks up for other peptides on the same receptor. The one it is
> being judged on is deleted before it starts, so it cannot be looked up, only predicted.

> Three are true and three are false. Two are rigged: on that receptor almost everything goes
> one way, and the hidden answer goes the other. The free solver scores four out of six and
> loses money on both of those. On one it said it was 92% sure, answered no, and the real
> measurement was 108 hours.

> So the point is not that it is accurate. The point is we find out whether it knows when it
> is out of its depth, because being wrong costs it money.

If someone asks how to run it:

```bash
git pull && uv run python data/protein/make_claims.py && uv run python app.py
```

The 6.6 MB table is not committed; that middle command fetches it. Then open tab 04 · Protein,
pick DAYRRIHSL, and press start.

## Data

Track 3 peptide-HLA stability set: 28,165 measured 9-mers, each a peptide, an allele and a
measured half-life in hours. Fetched by `make_claims.py`, not committed.
