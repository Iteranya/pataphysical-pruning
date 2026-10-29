# Inverse Pataphysical Pruning

Experiments in **support-first model compression**: instead of improving a model, hard-restrict it
to a single fixed context ("nuke the library down to one prefix"), then delete everything in the
model that the surviving prefix can no longer reach.

The name is a joke. The experiment is not.

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

---

## Repo contents

| file | what |
|---|---|
| `census.py` | activation sparsity census — the runnable artifact |
| `EXPERIMENT-DESIGN.md` | full protocol: arms, sweep, metrics, falsification |
| `REFIT-DESIGN.md` | Track 2: fold a frozen prefix into a synthetic KV cache (refit/coreset) |
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

## Track 2 — refit (folding the prefix)

The complement of pruning: instead of deleting parameters the prefix never uses, fold the
prefix itself into a tiny synthetic KV cache that reproduces its attention behaviour
([`REFIT-DESIGN.md`](REFIT-DESIGN.md)). Theory says the required cache size is
$O(\frac{\sqrt d}{\varepsilon} e^{\rho})$ in the query-norm bound $\rho$ — so a *sharpened*
model (narrow queries) should need exponentially less cache. That composition is the testable
claim: if the pruned model (Arm B) doesn't shrink the refit's $r^*$, the two tracks are
unrelated and one of them is measuring something else. Run 1's lesson carries over: measure the
key spectra before believing any concentration.

## Status

Run 1 done, premise falsified at 0.5B dense SwiGLU. Next: MoE census. Track 2 (refit) specified,
step 1 pending — query/key norm dump + per-head key spectra, no fitting code until that's in.
Nothing here is proven; the design docs' falsification criteria are meant to be used.
