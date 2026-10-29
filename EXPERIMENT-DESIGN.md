# Inverse Pataphysical Pruning — Experiment Design v0.1

**Target:** Llama-3.1-8B-Instruct (dense) as a stand-in for DeepSeek-V3 (671B MoE).
**Goal:** falsifiable test of one claim. Nothing else.

---

## 0. The claim

> **Restricting the support of the hypothesis space before fitting yields a strictly better
> fidelity-per-parameter curve than fitting onto the full support and compressing after.**

Formally: let `d` be the intrinsic dimension of the target function on support `S`, and `D` the
model's degrees of freedom. Compression is safe while `D >> d`. Narrowing `S` lowers `d`.
Therefore:

| Arm | Order of operations | Predicted |
|---|---|---|
| A | fit wide → prune → heal | baseline curve |
| B | narrow support → prune → heal | **strictly below A at every size; gap widens as size shrinks** |

If B is not strictly better at small sizes, the idea is wrong and we stop.

---

## 1. Why dense Llama-8B first

- MoE (DeepSeek) hands you a discrete, structured, near-lossless pruning knob: delete experts.
  That knob makes *any* pruning look good. It hides whether the supporting theory is real.
- Llama has no such knob. Every removed parameter must be justified by the narrowed distribution.
  **This is the harder test, therefore the correct first test.**
- Llama-3.1-8B shares the relevant shape with DeepSeek-V3: ~128k vocab, GQA attention, SwiGLU
  FFN, deep residual stack. Results should transfer in *direction* if not in magnitude.
- It fits on consumer hardware with QLoRA. A 671B experiment does not.

---

## 2. How to "nuke the library" on a dense model

This is the key translation. On MoE you nuke by deleting experts. On dense there are no
experts, so the mechanism must be rebuilt from the metaphor:

> A neuron that is never active on the lore is a book that does not start with the prefix.
> The lore never reaches it, so deleting it cannot change behaviour *on the lore*.

**Mechanism: activation-frequency pruning, not magnitude pruning.**

Do NOT reuse the earlier magnitude-threshold experiment. Thresholding per-token magnitude is
noisy and unit-inconsistent (see prior discussion: absolute thresholds mean different things in
different tensors). The corpus-level version of the same idea is:

1. Run the entire lore corpus through the model.
2. For every FFN intermediate neuron in every layer, count the fraction of tokens where
   `|activation| > tau` winsorized percentile (start with top-0.1% of the layer's activation
   scale, not an absolute constant).
3. Rank neurons by **activation frequency**, ascending.
4. The bottom `r` fraction are the ones the lore never uses. Those are your nuke.

Same for attention heads (mean attention entropy / output norm on lore tokens).

### Two-step, and the order matters

- **Step 1 (support nuke):** remove neurons with frequency below threshold. This is *measured*,
  not learned. Cheap. Expected to be highly lossless on the lore.
- **Step 2 (heal):** fine-tune the survivors on lore data so residual capacity redistributes.
  This is what recovers the quality lost by the nuke.

Arm B = measurement done on the lore. Arm A = measurement done on a general corpus.

---

## 3. Model facts (Llama-3.1-8B)

| | |
|---|---|
| layers | 32 |
| d_model | 4096 |
| q heads / kv heads | 32 / 8 (GQA 4:1), d_head 128 |
| FFN intermediate | 14336 (SwiGLU) |
| vocab | 128,256 (tied embeddings) |
| context | 131,072 |
| attention params/layer | 41.9 M |
| **FFN params/layer** | **176.2 M** |
| embedding | 525 M |
| total | ~7.5 B (nominal 8.03 B) |

**FFN = 75.1% of all parameters. Attention = 17.9%. Embedding = 7.0%.**

Consequence: this experiment is 75% an FFN-pruning experiment. Do not spend most of the effort
on attention heads. Vocab masking saves only 4.8% of params — do it for cleanliness, expect
nothing from it.

### Param count vs FFN retention

| drop FFN | I_kept | params | % of 8B |
|---|---|---|---|
| 30% | 10035 | 5.81 B | 72.4% |
| 50% | 7168 | 4.69 B | 58.4% |
| 70% | 4300 | 3.56 B | 44.3% |
| 80% | 2867 | 2.99 B | 37.3% |
| 90% | 1433 | 2.43 B | 30.3% |
| 95% | 716 | 2.15 B | 26.8% |

---

## 4. Arms

All arms must end at the **same architectures and same parameter counts** at each sweep point.
Only the *procedure* differs.

### Arm A — Baseline (fit-then-compress)
1. Importance estimated on a **general corpus** (e.g. 2M tokens of mixed web/wiki).
2. Structured prune to target `r`.
3. Heal on **general + lore** mixed data (standard practice).

### Arm B — Support-first (the hypothesis)
1. Vocab mask to lore-supported tokens (free, lossless).
2. Importance estimated on the **lore corpus only**.
3. Structured prune to target `r` using lore importance.
4. Heal on **lore data only**.

### Arm C — Ablation (isolates the mechanism)
1. Vocab mask (same as B).
2. Importance estimated on **general corpus** (same as A).
3. Heal on **lore only**.

Arm C exists to answer one question: *is the win from narrowing the support, or merely from
training only on the lore?* If B ≈ C, the DOF argument is doing no work and the result is
just "fine-tune on your target data," which is not a result.

---

## 5. Sweep grid

| Axis | Values |
|---|---|
| FFN retention `r` | 100%, 70%, 50%, 30%, 20%, 10%, 5% |
| arm | A, B, C |
| seed | 2 seeds minimum at each point |

That is 3 × 7 × 2 = 42 runs. Too many for a pilot — cut to one seed and 5 retention levels
(100, 70, 50, 30, 10) = 15 runs for the pilot.

**Lore corpus:** must fit the teacher's context for generating references.
- Pilot: **32k tokens** of a single coherent narrative (a character autobiography).
- Later: scale to 128k, then chunk-retrieve for 700k.
- 700k is not testable end-to-end on this hardware; do not pretend otherwise.

**Eval set:** 200 held-out prompts *about the same lore* (questions, continuations, scene
openers). Held-out means different prompts, not different lore.

---

## 6. Metrics

### Primary — fidelity
`KL( teacher(· | x, lore) || student(· | x) )` averaged over positions, on held-out lore prompts.
Teacher logits collected once, cached to disk, reused by every arm. Teacher must be **identical
across arms** including quantization — pick one setting and freeze it.

### Secondary — voice
Teacher-forced perplexity of teacher-generated lore continuations, student side. Ratio vs full
8B. Catches "fluent but not the character."

### Diagnostics
- Per-layer retention map (how much died where). Early vs late asymmetry is expected.
- Activation-frequency histogram before pruning, per layer. If the distribution is unimodal and
  dense (i.e. nothing is actually sparse), the whole hypothesis is dead on arrival — check this
  **first**, before any training.

### The cliff probe (predicted, not a failure)
Off-lore prompts (general knowledge, other topics). Arm B should collapse *harder* than Arm A.
That is the nuked library working as designed. Report it as a feature with a curve, not an excuse.

### The one plot
**X = parameters. Y = KL on lore prompts.** Three curves, A/B/C, plus the full-8B anchor at
(x=8B, y=0). This single plot settles the claim. Everything else is supporting material.

---

## 7. Hardware plan (RTX 4060 Laptop, 8 GB VRAM, 16 GB RAM)

| Stage | Feasibility | Notes |
|---|---|---|
| Teacher logit pass, 8B | **tight** | 4-bit ~4.5 GB. One forward per eval prompt. Slow but fine. |
| Activation-frequency scan | **tight** | Same cost as logit pass; reuse the same runs. |
| Pruned model inference | easy | 2.4–4.7 B in 4-bit = 1.5–3 GB. |
| Healing (QLoRA) | **hard** | 8B QLoRA fits 8 GB only with grad checkpointing, seq 1024–2048, small rank. |
| 16 GB system RAM | **limiting** | No offload. Close everything. |

**Recommendation:** run the pilot at **Llama-3.2-1B** (d=2048, 16 layers, I=8192) with a 3B
teacher. Every step fits comfortably, the whole sweep runs in a weekend, and the *shape* of the
curve is what you're testing — not the absolute numbers. Only scale to 8B once the 1B curve
shows the predicted gap.

If the 1B pilot shows nothing, an 8B run will not rescue it. If the 1B pilot shows the gap
widening, renting one A100 for a day to do the 8B sweep is obviously worth it.

---

## 8. Falsification

The idea is **dead** if any of the following hold:

1. `KL_B(r) >= KL_A(r)` for all retention levels.
2. `KL_B ≈ KL_C` — then support narrowing contributes nothing and it's just target-domain
   fine-tuning.
3. The activation-frequency histogram on the lore is **not** sparse — if nearly every neuron
   fires on the lore, there is no prefix-narrowing to exploit and the metaphor has no referent.
4. Arm B wins at large sizes but the gap **shrinks** as size drops — that contradicts the
   superlinear claim and would mean you're getting a constant offset, not a changing exponent.

Criterion 3 is the cheapest test and should be run **today**, before writing any training code.
It requires no training at all — just a forward pass over the lore and a histogram.

---

## 9. Step 1 (do this first, today)

Activation sparsity census on Llama-3.2-1B (or 3.1-8B if the logit pass fits):

1. Load model, 4-bit.
2. Push 8k–32k tokens of lore through it, hooks on every FFN intermediate.
3. Record per-neuron activation frequency at several thresholds.
4. Plot the histogram. Compute: what fraction of neurons are active on <1% of tokens?
5. Report that number.

If that fraction is large (say >60% of FFN neurons fire on <1% of tokens), the premise is
confirmed and the rest of the experiment is worth building. If it's small, we stop and go
back to the whiteboard.

---

## Open questions

- Healing budget: how many steps before the comparison is fair? (Must be identical across arms,
  tuned on a held-out slice, not on the eval set.)
- Does the teacher's own quantization bias the KL floor? (Freeze one teacher config; never change it.)
- Structured vs semi-structured (4:8) pruning for the sweep — unstructured gives cleaner DOF
  measurement, structured gives deployable artifacts. Pilot measures DOF; deployment comes later.
