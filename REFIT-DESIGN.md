# Refit — Folding a Frozen Prefix into a Synthetic Cache — Design v0.1

**Target:** `Qwen/Qwen2.5-0.5B-Instruct` (24L, d=896, 2 KV heads, d_head 64) on the committed
16k-token Alice corpus. CPU-only. No training in Arms R0–R2.

**Goal:** falsifiable test of one claim. Nothing else.

---

## 0. The claim

> **The attention behaviour of a frozen 16k prefix can be reproduced by `r` synthetic KV entries
> per head, and the required `r` shrinks *exponentially* when the query support narrows —
> which is exactly what the pruning experiment does to a model.**

This is the main claim one level up. `EXPERIMENT-DESIGN.md` says: narrow the support *before*
compressing and you beat compressing after. Refit says the same thing about a **prefix**: the
frozen prefix is the "fit", the synthetic cache is the "compression", and the query distribution
is the support. The theory gives the composition a hard number:

$$S = O\!\left(\frac{\sqrt d}{\varepsilon}\, e^{\rho + o(\rho)}\right), \qquad
\text{lower bound } \Omega\!\left(\frac{\sqrt d}{\varepsilon}\, e^{\rho}\right)$$

for all queries with $\|q\| \le \rho$ ("Nearly Optimal Attention Coresets", May 2026). Coreset
size is **exponential in the query-norm bound** $\rho$. A sharpened model has a small $\rho$.
So the prediction is not "refit works" — refit works regardless. The prediction is:

| Arm | Model | Queries used to fit | Predicted |
|---|---|---|---|
| R0 | base model | base queries | baseline $r^*(\varepsilon)$ |
| R2 | **sharpened** (pruned per main experiment) | sharpened queries | **strictly smaller $r^*$ at every $\varepsilon$; gap widens as $\varepsilon$ tightens** |

If R2 needs the same `r` as R0, the sharpening→cache link is dead and the two projects are
unrelated. That is the whole test.

---

## 1. What refit is (and is not)

Not compression of the cache. **Function approximation of attention itself.** The object is

$$f: q \mapsto \mathrm{Attn}(q, K, V) = \frac{e^{qK^\top/\sqrt d} V}{e^{qK^\top/\sqrt d}\mathbf 1}$$

Find a tiny $(K', V')$, $r$ entries per head instead of $n$, such that $f'(q) \approx f(q)$ on
the queries that matter. The synthetic entries are **not compressed tokens** — they are
approximants, and need not correspond to any token in the prefix.

Three families, split by whether you know the queries in advance.

### A — Query-aware: Attention Matching (AM)

Kanwat, *"Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels"*,
arXiv:2610.36520, **27 Oct 2026**. As fresh as it gets; read it before implementing.

Pipeline per head:

1. **Select** $r$ real keys (attention-mass ranking or Cholesky pivoting).
2. **Fit weights** by non-negative least squares so kept keys carry full attention mass of every
   reference query $q_m$:

   $$\min_w \sum_m \left\|\sum_{j\in S} w_j\, p_j(q_m) - 1\right\|^2, \quad w_j \in [e^{-3}, e^{3}]$$

   Each kept key counts as $w_j$ copies of itself and stands in for the dropped ones.
3. **Weights → logit biases** $\beta_j = \log w_j$, added to $q\cdot k_j/\sqrt d + \beta_j$.
4. **Refit values last** (order is load-bearing — see §3):

   $$V' = \arg\min_X \left\|\mathrm{softmax}\!\left(QK_S^\top/\sqrt d + \mathbf 1\beta^\top\right)X - O\right\|_F^2$$

5. **Fold $\beta$ into the keys** along one closed-form direction per head — FlashAttention and
   vLLM paged caches have no bias input, and a method needing a custom kernel dies in
   production. After the fold it is a plain KV cache on stock kernels. Error bounded per query
   by $\lvert q\cdot u - 1\rvert$.

Reported: matches HF answers on 51/51 Qwen3-4B questions; greedy-identical on 180/200
LongBench samples (Qwen3-14B); within noise of unfolded AM in 13/14 settings; lower KL to the
full cache than just dropping the bias, in 12/12.

### B — Query-agnostic: coresets

**WildCat** (Microsoft, Feb 2026, github.com/microsoft/wildcat) — $r$ keys by randomly-pivoted
Cholesky in $O(nr^2)$, weights in closed form:

$$W = \exp\!\left(\tfrac{\beta}{\tau^2} K_S K_S^\top\right)^{-1} \exp\!\left(\tfrac{\beta}{\tau^2} K_S K^\top\right)$$

No query access at compression time; all $n$ keys participate in the fit. Theorem: $r \in n^{o(1)}$
suffices for super-polynomial error decay $O(n^{-\sqrt{\log\log n}})$, bounded inputs. Drop-in
SDPA replacement, shipped.

Related: Thinformer / Kernel Halving ("Express", Jun 2026) — weighted thinning, $L_\infty$ error
$\log^{3/2}(n)/s$ at $O(s)$ memory. CoRDS (May 2026) — bicriteria KV coreset; error bounded by
$\sqrt{\sum_i a_i(q)\min_j \|v_i - v_j\|^2}$; value-favoring weights win.

### C — Learned refit (only if unseen prompts must work)

Gist tokens (Mu et al. 2023): attention-mask bottleneck, 26x, trained during instruction tuning,
zero extra cost. AutoCompressor: recursive summaries. ICAE / 500xCompressor: **frozen decoder**,
trained encoder emits memory tokens; 500xCompressor carries them as KV values, 6–480x, 62–73%
capability retained at the extreme. Frozen-decoder variants keep the real model — no mimicry
objection. Training required; that puts them outside this experiment's first pass.

**Nothing here distills.** Arms R0–R2 modify no weights. The real model reads a synthetic
prefix through its real attention.

---

## 2. Why this is in this repo

The pruning experiment asks whether narrowing support beats compressing after — **on
parameters**. Refit asks the identical question — **on the prefix** — and the coreset bound
turns it into a *quantitative* prediction ($r^* \propto e^\rho$) that the pruning side can
actually move. A win in the main experiment (Arm B beats Arm A) should show up here as a
leftward shift of the $r^*(\varepsilon)$ curve. If the two results don't co-move, one of them
is measuring something else.

Cheap synergy: both experiments want the same instrumentation — per-layer, per-head query and
key statistics on the lore corpus. One dump serves both.

---

## 3. The pipeline (offline; the prefix is frozen, so no heuristics)

Online eviction (H2O, SnapKV) is for live caches making do. A frozen prefix lets us do the
expensive right thing once, offline:

1. **Dump** $Q, K, V$ for the corpus through the model — one forward pass, hooks on every
   attention layer. (Serves both experiments.)
2. **Collect reference queries** $Q_{ref}$: the queries the workload will actually emit —
   held-out lore prompts, same definition of "held-out" as the main design (different prompts,
   same lore).
3. **Select** $r$ keys per head: Cholesky pivoting (theory-clean) or attention-mass (cheap).
4. **Fit weights** $w$ by NNLS on $Q_{ref}$ → $\beta = \log w$.
5. **Fold** $\beta$ into keys (closed-form per-head direction; implement from the paper, do not
   invent).
6. **Refit $V'$ last.** Values move the output given fixed attention; keys modulate retrieval.
   CoRDS: value-side error dominates the bound. Fitting values before weights inverts the
   optimization for no reason.
7. **Validate per layer**, not just end-to-end (depth compounds error; an end-to-end number can
   hide a dead layer).
8. **Sweep $r$**, allocate non-uniformly across layers (PyramidKV's lesson: low layers diffuse,
   high layers concentrated — one global $r$ wastes budget).

---

## 4. Arms and grid

| Arm | Model | Fit queries | Answers |
|---|---|---|---|
| R0 | base | base-model queries on lore prompts | baseline $r^*(\varepsilon)$ |
| R1 | base | + off-lore probe queries | how fast fit fails off-support ($\rho$ sensitivity) |
| R2 | **sharpened** (Arm-B pruned model) | sharpened queries | **the claim** |
| R3 | base + gist-style learned refit | any | ceiling for unseen prompts (later; needs training) |

| Axis | Values |
|---|---|
| $r$ per head | 8, 16, 32, 64, 128, 256 |
| arm | R0, R1, R2 (R3 deferred) |
| selection | Cholesky, attention-mass (R0 only, pick one for the rest) |

Reference sizes for our setup (fp16, 24L, 2 KV heads, 16k prefix):

| $r$/head | cache | vs full |
|---|---|---|
| 16 | 0.20 MB | 1024x |
| 64 | 0.79 MB | 256x |
| 128 | 1.57 MB | 128x |
| 256 | 3.15 MB | 64x |

Full cache is 201 MB fp16 / 403 MB fp32. The whole point is to end up in the top rows.

---

## 5. Metrics

- **Primary:** $r^*(\varepsilon)$ — entries per head needed for per-layer KL
  $(f_{full}(q) \| f_{refit}(q)) \le \varepsilon$, on held-out lore queries.
- **Secondary:** greedy match rate end-to-end vs the full-cache model (AM's calibration: 90%+
  greedy match at a few hundred entries/head on a *general* workload; a coherent 16k prefix
  should do better, or the coreset bound is not helping at our scale).
- **Diagnostics:** empirical query norms $\rho$ per layer/head on base vs sharpened models —
  this is the mechanism variable. If sharpening doesn't shrink $\|q\|$, the $e^\rho$ story has
  no mechanism and any R2 win is coincidence.

### The one plot

**X = $r$ (log). Y = per-layer KL. Curves: R0, R2.** Full cache anchor at the left axis. If R2
lies below-left of R0 everywhere and the gap widens as KL tightens, the claim stands. One plot
settles it; everything else is supporting material.

---

## 6. Hardware (RTX 4060 Laptop, 8 GB, 16 GB RAM)

This one is *comfortable* — no GPU math beyond one forward pass per model.

| Stage | Cost |
|---|---|
| Q/K/V dump, 16k corpus, 0.5B | one forward pass; writes ~1.8 GB fp32 to disk |
| NNLS per head | $16\text{k} \times r$ least squares — seconds on CPU |
| V refit | small least squares per head — seconds |
| Validation sweep | forward passes over held-out prompts only |

Everything except the forward passes is numpy/scipy on CPU. If the sharpened model is the
Arm-B pruned artifact, it is smaller and faster than the base — R2 is cheaper than R0.

---

## 7. Falsification

The claim is **dead** if any hold:

1. $r^*_{R2}(\varepsilon) \ge r^*_{R0}(\varepsilon)$ for all $\varepsilon$ — sharpening does not
   help the fold.
2. $r^*(\varepsilon)$ barely moves across the whole sweep — the prefix's attention structure is
   incompressible at this scale, and refit is the wrong tool for a 16k prefix.
3. The measured $\rho$ does not shrink under sharpening — no mechanism for the $e^\rho$
   prediction; then any win is noise.
4. The fold step ($\beta$ into keys) errors exceed the gap between families — i.e. the deployable
   version diverges from the paper's custom-kernel version on our model.

Kill criteria 3 is checkable **today**, from the existing dump, before writing any fitting code.

---

## 8. Step 1 (do first)

Extend the existing dump harness one notch: record **query norms and key norms per layer/head**
on the Alice corpus, plus the singular value spectrum of the per-head key matrices. Answers, in
order of cheapness:

1. What is empirical $\rho$? (Drives the bound. Huge $\rho$ = big $r^*$ regardless of method.)
2. Is $K$ actually low-rank per head? (Run 1's lesson: never assume concentration — measure.
   The FFN mass was flat; keys might be too.)
3. Layer profile — monotone like the FFN concentration trend, or flat?

If (2) is flat, refit dies the same death as the FFN nuke, and we have our answer before
touching NNLS.

---

## Open questions

- **RoPE:** retained real keys carry baked positions; fully synthetic keys have murky position
  semantics. Select-then-refit (AM style) sidesteps this. Fully synthetic entries may not.
- **$Q_{ref}$ coverage:** how many reference queries before the fit stops improving? Held-out
  split must be by *prompt*, same discipline as the main design.
- **Budget allocation:** per-layer $r$ non-uniform — profile first, then allocate (PyramidKV
  interpolation), or jointly optimize?
- **Re-fit after prune:** the fold direction $u$ and weights $w$ are model-dependent. Arm R2's
  fit must be redone on the pruned model — the R0 fit cannot be reused.
- **Scaling:** 16k is small. The story wants 70k+. If the 16k curve is right, scale the corpus;
  if flat, don't bother.

---

## References

| Work | When | What it gives us |
|---|---|---|
| Kanwat, *Folding Attention Biases into Keys* (arXiv:2610.36520) | **27 Oct 2026** | AM pipeline, bias→key fold, deployability |
| *Nearly Optimal Attention Coresets* (arXiv:2605.05602) | May 2026 | $S = O(\frac{\sqrt d}{\varepsilon} e^{\rho})$, matching lower bound |
| WildCat (arXiv:2602.10056, microsoft/wildcat) | Feb 2026 | query-free coreset, closed-form weights, runnable |
| CoRDS (arXiv:2605.14310) | May 2026 | error bound; values-first insight |
| Express / Thinformer (arXiv:2606.10944) | Jun 2026 | kernel thinning guarantees |
| Mu et al., Gist Tokens | NeurIPS 2023 | learned refit, mask bottleneck |
| ICAE / 500xCompressor | 2024–25 | frozen-decoder learned refit |
