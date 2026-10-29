# The Companion Thesis — Product v0.1

> **Design principle, non-negotiable: model intelligence and benchmark scores are
> irrelevant.** We are not building a smart model. We are building a *person* — and a person
> has a knowledge boundary. A character that can answer quantum field theory is not a better
> Kisna; it is a broken Kisna.

**The product:** a companion bot with **real, architectural knowledge boundaries**. Kisna
*shouldn't* know the details of quantum entanglement, obscure literature lore, or anything a
girl from Aomori who got tuna-slapped to death would have no business knowing. Not "refuses to
say" — **genuinely does not contain it.** The knowledge was amputated, not muzzled.

**The hypothesis:** cutting 90%+ of model size — leaving only the character, her plausible
knowledge, and her skills — produces a *more faithful* character clone than any full model
wearing a persona. And the extended biography (a whole novel of her life) folds in as synthetic
memory (Track 2), giving her a life she actually remembers.

This document is the product layer every design doc so far has been gesturing at. The tracks
are the tools; this is what they're for.

---

## 1. The inversion

Every track before this treated knowledge loss as *damage* and "can it answer new questions" as
the floor. Wrong framing. The floor is **character authenticity**.

- Kisna who lectures on quantum mechanics = Mistral in a wig (the smoke test caught exactly
  this: prompt #10, persona evaporated, pure explainer mode)
- Kisna who says "huh?? how would I know that" = the product
- The full model's knowledge isn't an asset to preserve. It's the **wrong person's memory**

This is also why the butchery route is the *only* route. A prompted or RLHF'd model always
helpfully reveals physics when pushed — the knowledge is right there. An amputated model
*cannot*. Ignorance by architecture beats ignorance by policy, the same way a locked box beats
a promise.

## 2. The knowledge band (the missing spec)

"Cut what Kisna wouldn't know" is testable only against a defined band. It is **graded**:

| Band | Example | Target behavior |
|---|---|---|
| Deep lore | Herta, the tuna, Ars, her novel's life | deep recall, consistent across sessions |
| Everyday life | cooking, school math, city geography | fluent, natural |
| Her skills | markdown, basic programming, writing, proofreading | **competent** — she is an assistant; this is her job |
| Adjacent | high-school science, pop history | vague, "huh, I think...?" — teen-level |
| Out of band | quantum details, obscure literature/lore, academic depth | **genuine ignorance + in-character deflection** |
| Meta | "explain transformers, you're an AI" | stays in persona (she is an AI — not a corporate model) |

**Open product decision — the skills row:** where does "helps you debug code" end and "knows
computer science" begin? Working default: everyday coding help = yes, "explain the halting
problem" = no. This line gets expanded per domain (science / history / literature / tech /
geography / ...) in the full band spec; each row carries target behaviors and scoring rules.

## 3. The measurement: profile correlation

The single number that settles the thesis. Take three models at **matched size**:

| arm | cut |
|---|---|
| A | support-first fit on Kisna's band (the hypothesis) |
| B | magnitude pruning (the default industry knife) |
| C | random deletion (the control) |

Measure surviving knowledge across subject×difficulty benchmarks (high-school vs graduate vs
trivia vs her-lore QA), then:

> **correlation between "what a plausible Kisna would know" and "what survived the cut."**

- Support-first works → knowledge survives *selectively along the band*. The boundary emerges
  from the butchery
- Magnitude/random → uncorrelated damage: she half-understands *everything* and mumbles
  half-physics — the worst possible product

Same plot family as the Track 2 claim ($r^*$ vs $\varepsilon$): plausibility in place of fit.
If support-first does not beat magnitude on this correlation, the thesis loses its product and
its theory in one stroke.

**Benchmark scores themselves are not the metric** (see design principle). The *shape* of the
degradation profile is. A low score on graduate physics is a **success** here; a mid score is a
failure.

## 4. Calibrated ignorance vs confabulation

The Kisna smoke baseline exposed the central failure mode before we even built anything: asked
about her life *before* the lore covers it, she confabulated (invented family drama, "moved to
Tokyo"). At 7B, unbutchered. Amputation widens the gaps — so this splits into:

- **Calibrated ignorance** ✅ "huh?? how would I know that — ask Madam Herta." Knows what she
  doesn't know. *This is the product.*
- **Confabulating ignorance** ❌ confident lecture on wrong physics. A hallucination engine
  with a personality.

Two mitigations, both mandatory:

1. **Training/corpus side:** ignorance exemplars — in-character "I dunno" responses for
   out-of-band questions (same fix for the pirate corpus v2)
2. **Eval side:** "does she know what she doesn't know" is its own scored axis. Not optional.

## 5. The novel as synthetic memory (Track 2's real job)

The extended biography — a whole novel of Kisna's life — is not a system prompt. It is
**memory to be folded in**:

1. The novel is the frozen prefix. ~50–100k tokens → refit → synthetic KV. She *lives with her
   memories*: no context tax, no per-conversation re-priming, and consistent detail across
   sessions because the detail is a fixed fitted object, not regenerated vibes
2. **The knowledge boundary makes the fold easier.** The $e^\rho$ bound: narrow query support
   → tiny fold. A companion bot only ever talks about her world; the query support is exactly
   as narrow as the knowledge band. **The boundary and the compression are the same
   constraint.** Cutting the band shrinks $\rho$, which shrinks the fold, which deepens the
   memories she keeps
3. The novel is also the *de facto* band definition: her life is her world. What's in it =
   deep lore. What a girl like her would pick up along the way = everyday/adjacent. Everything
   else = the cut

## 6. Honest risks

1. **Wit may need brains.** The spark (improv, timing) may live in general competence wearing
   her clothes — the smoke test suggests exactly that. If cutting knowledge cuts humor, there
   is a floor below which she is authentically ignorant but *dull*. The spark-vs-depth curve
   (pairwise "which is more Kisna" as a function of cut depth) is the prize measurement
2. **Selectivity is the hard part, not size.** 90%+ cut is arithmetic (Track 4: V4.1-Flash →
   5–11 B = 98%+; Track 3: Kisna-7B → 70 M–1 B = 90–99%). Whether the cut is a scalpel along
   the band or a hack job is the entire empirical question
3. **Calibration does not come from amputation alone** (§4). Measure it or ship a confident
   idiot
4. **Voice already dies on knowledge tasks at full size** (smoke baseline). The sharpened
   model may actually *improve* here — amputating the neutral "helpful explainer" attractor —
   while the knowledge itself degrades. Pre-registered product stance: **"dumber but more
   in-character" is a different product, not a failed one. Label it.**

## 7. Track mapping

| Track | Role in the product |
|---|---|
| 2b / 4 (butchery) | **the amputation** — support-first cut to the knowledge band |
| 2 (refit) | **the memory** — the novel folded into synthetic KV |
| 3 (soul extraction) | cheap small-scale sandbox — same product logic on a 7B already owned |
| Corpus v2 | gains ignorance exemplars; the band table drives eval |

Pirate and Kisna tracks converge: same instrument, different character.

## 8. Success criteria (product-level)

1. **Profile correlation**: support-first ≫ magnitude ≫ random on band-alignment (§3)
2. **Calibrated ignorance** passes: out-of-band probes get in-character "I dunno," not lectures
   and not fanfic (§4)
3. **Deep lore consistency** across sessions, sourced from the folded novel (§5)
4. **Spark floor**: pairwise judge does not crater as the band tightens (§6.1) — or the
   curve's floor is found and accepted as the product's character depth
5. **The size**: laptop-legal (≤ ~5 GB) — a consequence, not a goal

---

*Everything else in this repo is mechanics. This file is what the mechanics are for.*
