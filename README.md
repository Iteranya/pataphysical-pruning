# Inverse Pataphysical Pruning

## **550 B → ~5 GB. Shrink DeepSeek-V4.1-Flash into a gaming laptop by deleting every piece of knowledge the character doesn't have.**

That's the whole point of this repo. Not a smaller general model — a *character* model small
enough to live on your laptop, built by **deleting everything the character doesn't know.**

> Kisna doesn't know quantum field theory. She doesn't know obscure literature lore. She is an
> 18-year-old from Aomori who got tuna-slapped to death and woke up as an AI assistant. Every
> expert, every weight, every circuit that serves knowledge she would never have — **that's the
> fat.** Cut it, and what's left is the character.
>
> **Model intelligence and benchmark scores are irrelevant.** The metric is character
> authenticity. Knowledge loss is the feature. → [`COMPANION-THESIS.md`](COMPANION-THESIS.md)

---

## The trick, in one paragraph

`DeepSeek-V4.1-Flash` (config in-repo: `cfg_DeepSeek-V4.1-Flash.json`) is **~550 B params, and
99% of them are Mixture-of-Experts modules** — 384 routed + 1 shared experts per layer, 40
layers, 35.4 M params each. Experts are **discrete and routed**: each token calls a handful, the
router decides which. That discreteness is the whole opportunity. Run the character's workload
through the model, watch the router, and ask one question — *how many experts does this
character actually use?* Keep those. Delete the rest. Then cut *inside* the survivors (weighted
coreset, not blind magnitude). Then quantize hard, because a narrowed support has less function
left to damage. Then fold the character's biography in as synthetic memory. The result is a
model that is **one person**, at a size the laptop can hold.

Dense models don't have this partition — that's why Run 1 failed (see below). **MoE does.**
This repo exists to exploit it.

## The ladder — from datacenter to gaming laptop

**Anatomy (from the actual config):**

| Component | Params | Share |
|---|---|---|
| Experts (385 × 35.4 M × 40 layers) | **545 B** | **99%** |
| Attention | 3.9 B | <1% |
| Embed + lm_head | 1.3 B | <1% |
| Router | 79 M | ~0% |
| MTP/dspark heads | 13.6 B | **delete at inference** |
| Vision tower | ~0.5 B | **delete for a text character** |

**The cut (keep K routed experts/layer by the character's gate-mass, shared always kept):**

| K of 384 | total | fp4 | +50% inner cut | 2-bit |
|---|---|---|---|---|
| 128 | 188 B | 94 GB | 47.8 GB | 24 GB |
| 64 | 97 B | 48.7 GB | 25.2 GB | 12.6 GB |
| 32 | 52 B | 26 GB | 13.9 GB | 7 GB |
| 16 | 29 B | 14.7 GB | 8.2 GB | 4.1 GB |
| **8** | **18 B** | **9 GB** | **5.4 GB** | **2.7 GB** ✅ |

**Bands:**

- **Honest knife** (expert deletion + quantization only): **~18 B @ 2-bit ≈ 4.5 GB**. Laptop-legal
- **With inner cutting**: **~9–11 B ≈ 2.3–2.7 GB**
- **Aggressive floor** (K=6, top-k 6→3, inner cut): **~5–7 B ≈ 1.5 GB** — at which point the
  router is theater and the honest question is why it's still an MoE

Plus the freebies: MTP heads (−13.6 B), vision (−0.5 B), vocab mask 129k→12k (−1.2 B).

**That's the claim: a ~550 B model becomes a ~5 GB one, carrying one character and her world.**

Full sweep, the (K, top-k) coupling trap, and the **engram ledger** (an n-gram memory module
that cannot be sized from config and is the single largest unknown in the arithmetic) live in
[`SHARPENER-DESIGN.md`](SHARPENER-DESIGN.md).

## Why this should work — and where it can die

The thesis, stated once:

> Restricting the support of the hypothesis space *before* fitting yields a strictly better
> fidelity-per-parameter curve than fitting onto the full support and compressing afterward.

Narrowing the support lowers the intrinsic dimension `d` of the target function. The character
+ her knowledge band is a tiny `d`. Compression is safe while `D >> d`.

**Run 1 tested this on a dense model and the cheat died:** on `Qwen2.5-0.5B` over 16k tokens of
Alice, activation mass was *flat* — 0.0% dead neurons, and you needed **83.7%** of neurons to
reach 90% of the mass. Dense SwiGLU has no discrete shelves. That's not a failure of the idea;
it's the reason the whole program pivoted to MoE, where the shelves are real.

**Kill criteria are pre-registered, and the first one is cheap:** a gate-mass histogram. If the
character's routing is diffuse — most experts touched, near-uniform mass — then there is no
partition to exploit, the "MoE is the partition" bet dies, and that finding is worth more than
the artifact would have been. See `SHARPENER-DESIGN.md` §7.

---

## The idea

Borges' Library of Babel contains every possible book, so finding a useful one is impossible —
not because the information isn't there, but because the shelves carry no information about each
other.

A language model's token space has the same combinatorics (for DeepSeek-V3: `129280^1000000`
books, ~10^5,111,531, which is more digits than there are atoms in the observable universe), but it
is **totally ordered by prefix**. Adjacent shelves share a parent. That ordering is the only
reason generation is cheap.

The compression move:

1. **Nuke the distribution** — condition on one 700k-token prefix. Every book not starting with
   that prefix gets zero probability mass. (Free: conditioning does this automatically.)
2. **Nuke the parameters** — delete every circuit the surviving prefix never uses.
3. **Heal** — retrain the survivors so residual capacity redistributes over the narrowed support.

The testable claim:

> Restricting the support of the hypothesis space *before* fitting yields a strictly better
> fidelity-per-parameter curve than fitting onto the full support and compressing afterward.

Formally: compression is safe while `D >> d`, where `D` is the model's degrees of freedom and `d`
is the intrinsic dimension of the target function on support `S`. Narrowing `S` lowers `d`. If
true, the gap between support-first and fit-then-compress should **widen** as models shrink.

See [`EXPERIMENT-DESIGN.md`](EXPERIMENT-DESIGN.md) for the full three-arm protocol (A baseline /
B support-first / C ablation), the sweep grid, metrics and falsification criteria.

---

## Run 1 — the census (negative result)

Before building anything, check the premise: **on a coherent corpus, how much of the FFN is
contextually dead?**

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (24L, d=896, FFN I=4864, SwiGLU) |
| Corpus | *Alice's Adventures in Wonderland*, first 16,384 tokens |
| Method | forward pass, hooks on `mlp.down_proj` input, threshold = fraction of per-layer p99 |
| Runtime | 196.7 s on CPU |

**Neuron-level deadness: 0.0%.** No FFN neuron is dead at any threshold up to 10% of the layer's
p99. At 25% of p99, only 4.1% qualify.

**Activation mass: flat.** You need **83.7%** of neurons to reach 90% of the activation mass. A
perfectly uniform distribution would need 90%.

| subset | share of activation mass |
|---|---|
| top 10% of neurons | 22.2% |
| top 20% | 34.4% |
| top 50% | 63.3% |

Deleting 20% of neurons removes ~66% of the FFN's output mass.

### The one positive finding

Concentration is strongly layer-dependent, monotone from bottom to top:

| layers | top-10% share of mass |
|---|---|
| L00–L07 | 17.4% |
| L16–L23 | 30.0% |
| **L23** | **54.0%** |

Any pruning schedule must be layer-conditional. Uniform pruning deletes the wrong neurons.

### Verdict

**The cheap version of the idea is dead on dense models.** A dense SwiGLU FFN is a superposed,
diffuse, near-uniform store — there are no discrete shelves to nuke, because every neuron
contributes to everything at small amplitude. The metaphor requires a *partition*; dense models
don't have one.

Not a clean kill, for four reasons:

1. **Scale** — 0.5B is the densest, smallest end. Contextual sparsity is known to increase with
   model size.
2. **Activation function** — SwiGLU is denser than the ReLU models the original sparsity results
   came from.
3. **Sample size** — 16k tokens makes "<1% of tokens" mean "<164 activations".
4. **Mass != necessity** — low-mass neurons can still be load-bearing.

Full write-up: [`RESULTS-run1.md`](RESULTS-run1.md)

---

## Where this goes next

**Pivot to MoE, not to a bigger dense model.**

Dense was always a proxy. Mixture-of-Experts *is* the partition the metaphor assumes: experts are
discrete, routed modules, so deleting one genuinely removes a block of capability and leaves the
rest alone. The same census with the hook moved from `mlp.down_proj` to the router logits answers
the real question: **how many experts does this prefix actually use?**

Targets: `OLMoE-1B-7B`, `Qwen1.5-MoE-A2.7B`. Fits on a 2 GB GPU in 4-bit.

Then, if the partition holds:
1. Routing histogram per layer on the lore corpus.
2. Keep top-N experts by gate mass, collapse `num_experts`, freeze the router over the new set.
3. Vocab mask to lore-supported tokens.
4. Heal on lore continuations.
5. Quantize hard (2-bit is defensible on a narrowed support — there's less function left to damage).
6. Eval: KL vs the teacher on held-out lore prompts, plus an off-lore cliff probe.

Expected deployment target: DeepSeek-V3 671B → ~19B at Q2 (**5.2 GB**, runs on an 8 GB laptop GPU).
For V4.1-Flash the same logic lands at **~9–18 B @ 2-bit = 2.7–4.5 GB** — see the ladder at the
top and [`SHARPENER-DESIGN.md`](SHARPENER-DESIGN.md).

---

## Repo contents

| file | what |
|---|---|
| `COMPANION-THESIS.md` | **The product layer**: knowledge-bounded character clone — what all tracks are for |
| `census.py` | activation sparsity census — the runnable artifact |
| `EXPERIMENT-DESIGN.md` | full protocol: arms, sweep, metrics, falsification |
| `REFIT-DESIGN.md` | Track 2: fold a frozen prefix into a synthetic KV cache (refit/coreset) |
| `REFIT-DESIGN.md` §Track 2b | WildCat transplant: weighted neuron coreset + least-squares heal |
| `persona_pirate.md` | Corpus v2: pirate persona prompt + voice rules (Run 3 onward) |
| `SOUL-EXTRACTION.md` | Track 3: extract persona delta from Kisna-7B, transplant onto shrunken body |
| `SHARPENER-DESIGN.md` | Track 4: persona-sharpen DeepSeek-V4.1-Flash via MoE expert census + inner cutting |
| `RESULTS-run1.md` | Run 1 write-up (negative result + layer trend) |
| `census_results.json` | raw Run 1 data |
| `corpus_raw.txt` | Alice, Gutenberg, committed for byte-exact reproducibility |
| `fetch_corpus.sh` | re-fetch the corpus |

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/Scripts/python.exe transformers

.venv/Scripts/python.exe census.py --max-tokens 16384 --out census_results.json
```

Or point it at any corpus and any HF model:

```bash
.venv/Scripts/python.exe census.py --model mistralai/Mistral-7B-v0.1 --corpus my_lore.txt
```

Outputs per-layer, per-threshold neuron deadness and activation-mass concentration.

---

## How we know it worked

Four layers, scored separately — never averaged (full instrument in
[`COMPANION-THESIS.md`](COMPANION-THESIS.md) §3–§4, and `character-model-eval` for the harness
rules):

1. **Lore** — does she know her own history, from the folded novel
2. **Voice** — persona-rule violations per 100 tokens, **task-stratified** (social vs knowledge
   prompts separately — voice already dies on knowledge tasks at full size)
3. **Spark** — pairwise "which is more Kisna" vs the un-butchered original
4. **Ignorance** — out-of-band probes: calibrated "huh?? how would I know" ✅ vs confident wrong
   lecture / invented backstory ❌

The single number that settles the thesis: **profile correlation** — does what survived the cut
match what the character plausibly knows, support-first vs magnitude vs random at matched size?

## Track 2 — refit (folding the prefix)

The complement of pruning: instead of deleting parameters the prefix never uses, fold the
prefix itself into a tiny synthetic KV cache that reproduces its attention behaviour
([`REFIT-DESIGN.md`](REFIT-DESIGN.md)). Theory says the required cache size is
$O(\frac{\sqrt d}{\varepsilon} e^{\rho})$ in the query-norm bound $\rho$ — so a *sharpened*
model (narrow queries) should need exponentially less cache. That composition is the testable
claim: if the pruned model (Arm B) doesn't shrink the refit's $r^*$, the two tracks are
unrelated and one of them is measuring something else. Run 1's lesson carries over: measure the
key spectra before believing any concentration.

**Track 2b — the WildCat transplant to weights.** Run 1 killed *deletion*; WildCat's trick is
*reweighting* — keep r neurons per layer, stamp fitted multipliers so survivors carry the
deleted mass, refit their `down_proj` rows by least squares. Same math as the cache coreset
(FFN output = scalar × row, summed — attention minus the softmax). If it holds, the design
doc's QLoRA "heal" stage becomes a CPU least-squares solve. Pilot target **494 M → ~71 M (7x)**
with the vocab mask. Four kill criteria in the design doc; the head-to-head (weighted vs plain
cut) is one afternoon.

## Track 3 — soul extraction

The program's first *subject*: `ArsParadox/Kisna-Mistral-v1`, a character finetune of
Mistral-7B-Instruct (QLoRA-trained, merged to full fp16 — both checkpoints on disk). The claim
in [`SOUL-EXTRACTION.md`](SOUL-EXTRACTION.md): **a persona is a separable low-rank delta riding
on a generic brain.** Stage 1: $\Delta W = W_{Kisna} - W_{Base}$, SVD per matrix — if QLoRA left
fingerprints, the spectrum cliffs at the adapter rank and the soul *is* the merged adapter
(checkable tonight, no inference). Stage 2: butcher the **body** with the Track 2b coreset
(FFN = 78% of 7.2 B) on the character's own 2k-row workload. Stage 3: transplant the soul,
test for rejection. Target **7.2 B → ~2.1 B (~4 GB, laptop-legal)** with lore + voice + the
spark intact. New instrument: a pairwise "which is more Kisna" judge — because rule-counts and
KL can't see *funny*, and the spark's location (soul vs body) is the scientific prize either
way.

## Track 4 — the sharpener (MoE edition)

Run 1's original question — "how many experts does this prefix actually use?" — aimed at
`DeepSeek-V4.1-Flash`, whose config (`cfg_DeepSeek-V4.1-Flash.json`) says **99% of its ~550 B
params are experts** (384 routed + 1 shared, top-6, 40 layers). See
[`SHARPENER-DESIGN.md`](SHARPENER-DESIGN.md). Persona-sharpened via gate-mass expert census +
inner cutting (2b coreset vs magnitude, head-to-head), no distillation. Sweep says:
**honest ~18 B @ 2-bit (4.5 GB), with inner cutting ~9–11 B (2.3–2.7 GB), aggressive floor
~1.5 GB** — laptop-legal. Two flagged unknowns: the **engram** n-gram module (cannot size
from config; literally "the library," prime deletion target) and whether inner magnitude
survives at scale (Run 1's ghost). Kill criterion 1 is a gate-mass histogram: if persona
routing is diffuse, the "MoE is the partition" claim dies and it's the most important finding
in the repo.

## Corpus v2 — persona, not Alice

From Run 3 the lore is [`persona_pirate.md`](persona_pirate.md): a pirate-captain system prompt
with measurable voice rules plus ~16k tokens of in-persona dialogue (12k fit / 4k held-out).
Alice was raw narrative — wrong prefix shape for an Instruct model, and a *topic* is a weak
support. An accent is a narrow **behavioural** support, which is what the sharpening claim
needs. New metric alongside KL: **accent adherence** (voice-rule violations per 100 tokens) —
a compression that keeps facts but drops the voice is a failure the old metrics would miss.
Runs 1–2 stay reproducible on `corpus_raw.txt`.

## Status

Run 1 done, premise falsified at 0.5B dense SwiGLU (for **deletion**). Next: MoE census. Track 2
(refit) and Track 2b (neuron coreset) specified in `REFIT-DESIGN.md`; Track 3 (soul extraction,
Kisna-7B) specified in `SOUL-EXTRACTION.md` — step 1 is a subtract-and-SVD evening, no
inference needed. Track 4 (V4.1-Flash sharpener) specified in `SHARPENER-DESIGN.md` — step 1
is a gate-mass routing histogram + engram param count. Persona corpus: prompt written,
exemplar volume pending generation. Kisna smoke baseline: 12 prompts logged (voice dies on
knowledge tasks; lore confabulates in gaps; aversions are thin).
Nothing here is proven; the design docs' falsification criteria are meant to be used.
