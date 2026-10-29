# Track 5 — The Sharpener — Design v0.1

**Target:** `DeepSeek-V4.1-Flash` (config: `cfg_DeepSeek-V4.1-Flash.json` in this repo) —
persona-sharpened via **MoE expert census + inner coreset/magnitude**, no distillation, no
soul extraction (that's Track 4, different subject). Goal artifact: a laptop-legal model that
is *one character*, built by butchery.

**Goal:** falsifiable test of one claim. Nothing else.

---

## 0. The claim

> **99% of a MoE is experts. A persona uses a small fraction of them. Deleting the rest, and
> cutting inside the survivors, shrinks a ~550 B model to single-digit billions without losing
> the persona.**

Formally the support-first claim in its native habitat: the design doc always said MoE *is* the
partition the metaphor assumes (experts are discrete, routed, measurable), and the README's own
census question — **"how many experts does this prefix actually use?"** — is literally the
quantity this entire track turns on. Track 5 is Run 1's question, aimed at a model that has a
real answer.

Sub-claims:

| # | Sub-claim | Test |
|---|---|---|
| S1 | Persona routing is **concentrated** — a workload touches far fewer than 384 experts/layer | gate-mass histogram on persona corpus |
| S2 | **Inner cutting** (weighted coreset / magnitude) inside survivors recovers fidelity at smaller size than expert deletion alone | head-to-head at matched total params |
| S3 | The sharpened model **keeps the persona on novel prompts** — including knowledge tasks | Kisna-baseline-style layered eval + spark judge |

---

## 1. Anatomy (from the actual config, not marketing pages)

`DeepseekV41ForCausalLM`, text tower: 40 layers, d=5120, 384 routed + 1 shared experts,
top-6, `moe_intermediate` 2304, fp8 blocks with **fp4 experts**, untied embeddings (129,280).

| Component | Params | Share | Notes |
|---|---|---|---|
| Routed+shared experts (385 × 35.4 M × 40) | **545 B** | **99%** | the only knife that matters |
| Attention (q/o-lora, 1 kv head) | 3.9 B | <1% | |
| Embed + lm_head | 1.3 B | <1% | vocab mask 129k→12k recovers ~1.2 B |
| Router | 79 M | ~0% | |
| dspark/MTP (3 next-n layers, 128 experts) | 13.6 B | — | **delete at inference** |
| Vision tower (32 L ViT) | ~0.5 B | — | **delete for text persona** |
| engram n-gram memory (layers 1, 14) | **⚠ unknown** | **⚠** | `engram_num_embeddings: [384M, 384M]` — **cannot size from config**; see §4 |

Active per token at top-6: **~13.7 B** (8.5 B experts + attention + head). Expert = 3 ×
5120 × 2304 = 35.4 M (SwiGLU gate/up/down).

**Consequence: "how small can a sharpened V4.1-Flash get?" reduces to "how many experts does
one persona use, and how thin can the survivors get?"** Everything else is rounding error.

---

## 2. The sweep (keep K routed experts/layer by persona gate-mass; shared kept always)

| K of 384 | total | fp4 | +50% inner cut, fp4 | 2-bit (inner cut) |
|---|---|---|---|---|
| 128 | 188 B | 94 GB | 47.8 GB | 24 GB |
| 64 | 97 B | 48.7 GB | 25.2 GB | 12.6 GB |
| 32 | 52 B | 26 GB | 13.9 GB | 7 GB |
| 16 | 29 B | 14.7 GB | 8.2 GB | 4.1 GB |
| **8** | **18 B** | **9 GB** | **5.4 GB** | **2.7 GB** ✅ |
| 6 (router vestigial) | 15 B | 7.6 GB | 4.7 GB | 2.3 GB |

Freebies on top: MTP heads (−13.6 B), vision (−0.5 B), vocab mask (−1.2 B).

**Bands:**

- **Honest knife** (expert deletion + quant only): **~18 B @ 2-bit ≈ 4.5 GB**. Laptop-legal
- **With inner cutting**: **~9–11 B ≈ 2.3–2.7 GB**. Beats the README's old general-use dream
  (671B → 19B @ Q2 = 5.2 GB) — expected, since this is persona-only
- **Aggressive floor** (K=6, top-k 6→3, inner cut, 2-bit): **~5–7 B ≈ 1.5 GB**. At this point
  the router is theater (top-6 of 6 = everything) and the honest question is why it's still an MoE

Below that: cut expert *width* (`moe_intermediate` 2304 → smaller) — a different arm, not
claimed here.

**Resident vs active:** all the shrink is *resident*. Active/token stays ~8.5 B while top-k=6
and K≥6. Shrinking active requires cutting top-k itself — separate axis, routing quality on
trial. Note the interaction: K=8 with top-6 means 6 of 8 experts fire on *every* token; the
router has almost no choice left. K should either be comfortably above top-k (K≥16) or the
top-k should drop with it. Pick a coherent (K, top-k) pair and record it.

---

## 3. The knives — and which one is unproven

1. **Expert deletion by gate-mass** (the census): measure per-expert gate mass on the persona
   corpus, keep top-K per layer. **Discrete, measurable, the strongest knife in the model.**
   This is the MoE knob the design doc always called "near-lossless if you measure right."
2. **Inner weighted coreset (Track 3 procedure)** inside surviving experts: pivoted Cholesky
   on the persona activation matrix, NNLS multipliers, refit `down_proj` rows. The
   repo-branded knife.
3. **Inner magnitude pruning**: the *unproven* knife. Run 1 killed magnitude at 0.5B dense
   (flat activation mass, nothing to grab). The defense: (a) scale — contextual sparsity rises
   with model size; (b) narrowed support — less function to damage; (c) experts are narrower
   and more specialized than dense FFN. But this is a hypothesis, not a provenance. **Track 3 and
   3 are run head-to-head at matched K and matched total params. If they tie, prefer Track 3
   (fitted > thresholded). If magnitude wins, Run 1's ghost is officially scale-dependent.**
4. **Free deletions**: MTP/dspark layers, vision tower, vocab mask. Do first, argue later.

---

## 4. The engram problem (read this before promising numbers)

The config carries an `engram` module at layers 1 and 14: `engram_num_embeddings:
[384006168, 384016682]`, `engram_vocab_size: 16,000,000`, `engram_max_ngram_size: 4`, 8
heads × 256 dim. That looks like a learned **n-gram key table** — potentially the single
largest non-expert component in the model, and potentially huge.

Two facts make it the most interesting deletion target alive:

- **N-gram lookup is the library, not the librarian.** Rote memorization machinery is exactly
  what a persona doesn't need — it's the "nuke the library" metaphor made literal
- **It is also the size unknown.** The config cannot tell us row dimensionality. A wrong
  assumption here voids every total in §2

**Step 1 therefore includes an explicit engram parameter count.** No sweep number gets
published to a result file until the engram ledger is closed. If engram turns out to be, say,
+200 B, the bands move; if it's a small hash table, §2 stands as-is.

---

## 5. Metrics — reuse the Kisna baseline instrument

The eval is the three-layer instrument, now canonized by the Kisna smoke test
([`persona_pirate.md`](persona_pirate.md) §2-4, and the JSON baseline):

| layer | metric |
|---|---|
| Lore | factual QA over the persona's own lore block |
| Voice | persona-rule violations per 100 tokens — **task-stratified** (social vs knowledge prompts scored separately; the Kisna baseline showed voice dies on knowledge tasks *at full size*, so a flat average hides the failure) |
| Spark | pairwise judge vs the un-butchered original: "which is more [character]" on novel prompts |

**The knowledge-task row is the one to watch.** The support-first prediction is genuinely
two-sided: butchery could *improve* voice-on-knowledge (amputating the neutral "helpful
explainer" attractor) while *degrading* the knowledge itself. Report both columns; a model
that is dumber but more in-character is a **different product**, not a failed one — label it.

---

## 6. Hardware

The 8 GB work laptop can: gate-mass census (4-bit forward passes, tight), inner coreset fits
(CPU least squares), and inference of anything ≤ ~15 B at 2-bit. The 4060 box handles the
activation dumps and the end-to-end evals. Full 550B fp16 source is not needed on either
machine — expert extraction happens from **sharded safetensors streaming**, and everything
downstream works on the pruned artifact. As with Track 4: **never fit from a quantized copy**
(gate mass from fp4 experts is already a compromise — record it as such).

---

## 7. Falsification

The claim is **dead** if any hold:

1. **Persona routing is diffuse** — the corpus touches most of the 384 experts/layer with
   near-uniform gate mass. Then there is no concentration to exploit and expert deletion has
   no support-first edge over arbitrary deletion. *(This is Run 1's negative result, MoE
   edition. It would be the most important finding in the repo and would kill the "MoE is the
   partition" claim the design doc has leaned on since v0.1.)*
2. **Inner cutting at scale behaves like Run 1** — weighted coreset ≈ magnitude ≈ plain
   deletion at matched params. Then the survivors are as diffuse as dense FFN was and Track 3 is
   dead across architectures. (Track 3's kill criterion 4, restated.)
3. **Voice-on-knowledge collapses harder than the baseline's** and the spark judge says
   "obviously butchered" at every K. Then sharpening costs the persona its most interesting
   property and the product is a parrot in costume — report the curve, close the track.
4. **The engram ledger voids the sweep** (engram dominates the parameter count and cannot be
   cut without breaking the model). Then the "550B → single-digit B" story was arithmetic
   about the wrong object.

Kill criterion 1 is the cheapest and the most consequential. **Gate-mass histogram first.**

---

## 8. Step 1

One pass, two outputs:

1. **Routing census**: run the persona corpus (pirate v2 / Kisna workload — whichever is
   generated first) through V4.1-Flash in 4-bit, record per-layer per-expert gate mass.
   Deliverable: the histogram + "experts needed for 90% of gate mass" per layer, the MoE
   twin of Run 1's activation-mass table.
2. **Engram ledger**: instantiate the model class (or read the implementation source) and
   count `engram` parameters exactly.

If (1) shows concentration (say <128 experts/layer carry 90% of gate mass), the sweep in §2 is
alive and Track 5 becomes the highest-leverage track in the repo. If not, kill it cleanly.

---

## Open questions

- **(K, top-k) coupling**: K=8 with top-6 is a router in name only. Sweep K and top-k jointly,
  or fix top-k=3 from the start?
- **fp4 experts as the fitting source**: gate-mass measurement from fp4 experts is a
  quantized signal. Acceptable for a census (it's what ships), but the inner coreset fits want
  higher-precision activations. One 4-bit pass for routing, one bf16 pass on *kept* experts
  only, if memory allows — or rent one box for an evening.
- **Shared expert**: always kept by fiat. Is it? It's one expert of 385 — measure whether it
  carries disproportionate mass before blessing it.
- **Voice-on-knowledge could improve** (neutral attractor amputated) while knowledge degrades.
  That's a *product choice*, not a bug: "smaller, dumber, more in-character" is arguably the
  whole point of a persona model. Pre-register which column wins ties.

---

## References

| asset | what |
|---|---|
| `cfg_DeepSeek-V4.1-Flash.json` | the subject's anatomy (this repo) |
| `cfg_DeepSeek-V3.2.json` | classic 671B-class shape, for comparison |
| `EXPERIMENT-DESIGN.md` §2 | why MoE was always the intended partition |
| `REFIT-DESIGN.md` §Track 3 | the inner coreset procedure |
| Kisna smoke baseline (JSON) | the three-layer eval instrument, proven on a real character |
