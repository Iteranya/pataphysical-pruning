# Census Results — Run 1

**Date:** 2026-10-29
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, d_model 896, FFN intermediate 4864, SwiGLU, vocab 151,936)
**Corpus:** Project Gutenberg *Alice's Adventures in Wonderland*, first 16,384 tokens
**Method:** forward pass, no gradients, hooks on `mlp.down_proj` input (post-gating SwiGLU activation)
**Runtime:** 196.7 s (CPU, i5-1155G7)
**Script:** `census.py` · raw data: `census_results.json`

---

## Result 1 — neuron-level deadness is zero

Thresholds are a fraction of each layer's own p99 activation scale, so they mean the same
thing in every layer (the fix for the earlier absolute-magnitude mistake).

| tau (frac of p99) | neurons dead on <1% of tokens | activation sparsity (all elements) |
|---|---|---|
| 1% | 0.0% | 9.5% |
| 2% | 0.0% | 16.6% |
| 5% | 0.0% | 33.0% |
| 10% | 0.2% | 51.9% |
| 25% | 4.1% | 80.8% |

**Interpretation:** individual activations are sparse (80.8% below 25% of p99), but **no neuron
is contextually dead**. Every FFN neuron fires on more than 1% of tokens. Even at a threshold of
a quarter of the layer's p99, only 4.1% of neurons qualify as dead.

The "delete neurons the prefix never reaches" mechanism has **nothing to grab** here.

## Result 2 — activation mass is essentially flat

Rank neurons by total |activation| across the corpus, compute cumulative share.

| subset | share of activation mass (mean over layers) | min | max |
|---|---|---|---|
| top 10% of neurons | 22.2% | 15.7% | 54.0% |
| top 20% | 34.4% | 26.8% | 63.9% |
| top 50% | 63.3% | 56.8% | 81.2% |

**Neurons needed to reach 90% of the mass: 83.7%** (min 70.8%, max 87.2%).

For a perfectly uniform distribution you would need exactly 90%. At 83.7%, the FFN is very
close to flat. Top-10% carrying 22.2% vs. a uniform baseline of 10% is only a 2.2x concentration
— not sparsity, just mild skew.

**Consequence:** deleting 20% of neurons removes ~66% of the FFN's activation mass. That is not
a prune, that is a lobotomy.

## Result 3 — but concentration is strongly layer-dependent

| layers | top-10% share of mass |
|---|---|
| L00–L07 | 17.4% |
| L16–L23 | 30.0% |
| L23 alone | **54.0%** |

Monotone trend from early to late. The final layer is genuinely concentrated; the first eight
are effectively uniform.

**This is the one positive finding.** Any pruning schedule must be layer-conditional: aggressive
in the top layers, near-zero in the bottom ones. Uniform pruning throws away the wrong neurons.

---

## Verdict

Falsification criterion 3 fired: *if the activation histogram on the lore is not sparse, there is
no prefix-narrowing to exploit.*

On this model there is no prefix-narrowing to exploit at the neuron level. The dense SwiGLU FFN is
a superposition-heavy, diffuse, near-uniform store. There are no shelves to nuke.

## But this is not a clean kill — four reasons

1. **Scale.** 0.5B is the smallest, densest end. Contextual sparsity is known to *increase* with
   model size (the Deja Vu results were on 1.3B–175B ReLU models). Testing the hypothesis at 0.5B
   tests it where it is least likely to hold.
2. **Activation function.** SwiGLU is gated and known to be far denser than ReLU/GELU. The
   original sparsity literature used ReLU models. This run may be measuring the wrong thing.
3. **Sample size.** 16,384 tokens means "fires on <1% of tokens" = fires fewer than 164 times.
   Adequate for a coarse read, not for a 0.1% cutoff.
4. **Mass != necessity.** A neuron can carry little mass and still be load-bearing (thin features,
   outlier-sensitive directions). Flat mass is a strong *prior* against neuron pruning, not proof.

## What to do next (pick one, do not do both)

**A. Test the premise where it can be true: an MoE.** DeepSeek-style routing partitions the FFN
into discrete experts. That is the only architecture where "delete the books off this branch" is
literally true. Expert-routing histogram on a small MoE (e.g. Qwen1.5-MoE-A2.7B, OLMoE-1B-7B)
answers the same question with the partition intact.

**B. Test the premise at the scale the literature says it lives: 7–8B dense.** Requires the 4060
or a rented GPU. One census run, ~1 hour.

Recommendation: **A first.** It is cheaper, it fits on the dev laptop in 4-bit, and it is the
architecture the whole idea was actually pointed at. The dense path was only ever a proxy, and
the proxy just told us the dense path is the hard one.
