# Track C — Soul Extraction — Design v0.1

**Target:** `ArsParadox/Kisna-Mistral-v1` (full fp16, 3 shards, ~14.5 GB) against its parent
`mistralai/Mistral-7B-Instruct-v0.3`. Goal artifact: a laptop-sized Kisna — **~2B body + ~100–200
MB soul** — running on an 8 GB RTX 4060 with the persona intact.

**Goal:** falsifiable test of one claim. Nothing else.

---

## 0. The claim

> **A character finetune's persona is a separable object in weight space — a low-rank delta
> riding on a generic brain — and can be extracted, transplanted onto a coreset-shrunken body,
> and survive.**

Three sub-claims, each independently killable:

| # | Sub-claim | Test |
|---|---|---|
| S1 | The soul is **low-rank** (separable from the body) | ΔW spectrum per matrix |
| S2 | The body is **coreset-shrinkable** (Track B at 7B) | butchery fidelity on Kisna workload |
| S3 | The soul **transplants** onto the shrunken body | rejection test after reattachment |

---

## 1. Why this is possible at all — the dirty secret

Kisna is a **full finetune artifact** of `Mistral-7B-Instruct-v0.3`, but the training was
QLoRA (model tree parent: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`; ~2k rows of data;
both checkpoints on disk). Which means:

$$\Delta W = W_{Kisna} - W_{Base} \approx B A \quad (\text{LoRA factors})$$

**If QLoRA left its fingerprints, the delta is not just low-rank — it is exactly rank ≤ r**
(the adapter rank used at training). Stage 1's SVD should show a **hard cliff** at that rank.
That is a free, sharp prediction:

- Cliff at rank 8/16/32/64 → the soul is literally the merged adapter, extraction is exact
- Diffuse spectrum → something else trained too (embeddings, lm_head, or a later full pass),
  and the soul is a smearing

Either answer is worth having before cutting anything. This is Run 1's lesson applied: **measure
the spectrum before believing the metaphor.**

Stage 1 is **low-rank** family (synthesized directions), not WildCat. The coreset/WildCat
transplant appears in Stage 2 (body) and optionally Stage 1b (soul variant). Taxonomy matters:
low-rank synthesizes directions; coreset keeps a weighted subset of *real* rows.

---

## 2. The three stages

### Stage 1 — Rip the soul out (low-rank; ~free)

1. Load both checkpoints (CPU/offload is fine — it's subtraction, not inference).
2. $\Delta W = W_{Kisna} - W_{Base}$ for every matrix (attn q/k/v/o, FFN gate/up/down; *report
   embeddings and lm_head deltas separately* — if those are hot, the "exact rank r" story is
   already more interesting).
3. SVD each ΔW. Record: singular value spectrum, effective rank at 90%/99% energy, **cliff
   detection** (is there a knee at plausible LoRA ranks 8–64?).
4. Compress: keep top-r factors. Deliverable artifact: `soul_r{N}.pt`, the persona as factors.

**Stage 1b (arm, cheap): coreset of the soul.** Instead of synthesized singular directions,
keep r **real rows** of ΔW with fitted scalars (NNLS on the Kisna workload — Track B's
procedure applied to the delta). Comparing 1 vs 1b answers a question nobody knows the answer
to: *does Kisna-ness live in directions or in specific neurons?* Both arms, one dump.

### Stage 2 — Butcher the body (WildCat transplant, Track B at 7B)

The body is generic competence — English, reasoning, world knowledge. That's what shrinks.
Mistral-7B: **FFN = 5.64 B of 7.24 B params (78%)**. The coreset targets:

1. FFN neuron coreset per layer (pivoted Cholesky on the **Kisna workload** activations,
   NNLS multipliers, `down_proj` rows refit by least squares — exactly REFIT-DESIGN §Track B)
2. Head coreset (14q→4q style; 32 q-heads → measured target) with fitted multipliers
3. Residual rank reduction if the Step-1 spectra support it (see §5)
4. Vocab mask to Kisna's working vocab (design doc: free, lossless)

**Fit corpus:** `Kisna-Sovl-1k` + `Kisna-Assist-1k` — the original training data is the perfect
support-first corpus. Split discipline: fit on one slice, eval on held-out *prompts* (different
questions, same character). ~2k rows is small; if the fit is underdetermined, generate more
in-character prompts via the base model — but commit raw text byte-exact per repo rule.

### Stage 3 — Transplant and test for rejection

1. $W' = W_{butchered} + \Delta W_{top-r}$ (or + soul-coreset, arm 1b)
2. Greedy-decode the eval prompts. Compare against the *original* Kisna-7B
3. The transplant either holds the persona or rejects it. No third outcome worth reporting

---

## 3. The numbers

**Soul (the delta), by rank per matrix (7 matrices × 32 layers, fp16):**

| rank | size | reading |
|---|---|---|
| 8 | 29 MB | Kisna is a patch note |
| 16 | 59 MB | firmware update |
| 32 | 117 MB | a book on tape |
| 64 | 235 MB | still smaller than one layer of body |
| 256 | 940 MB | fat soul, still 15x smaller than body |

**Body + soul → laptop (8 GB VRAM):**

| artifact | VRAM | fits? |
|---|---|---|
| Kisna-7B as-is | 14.5 GB | **no** — her own model doesn't fit her own laptop |
| 4-bit quant only | 3.8 GB | yes, but that's taping, not butchery |
| **~2B body + rank-32 soul, fp16** | **~4 GB** | **yes, with context room** |
| 1B body + soul, fp16 | ~2 GB | yes, absurd room |
| 2B body + soul, 2-bit | ~0.6 GB | toaster-legal |

**Pilot target: 7.2 B → ~2.1 B total (3.4x smaller than 7B, 7x under the fp16 original),
persona intact per §4.** Stretch: 1B body if the spark survives the 2B transplant.

---

## 4. Metrics — the persona is three layers, measured separately

| layer | metric | target |
|---|---|---|
| **Lore** (history, relationships) | factual QA over the model card's lore block | exact recall |
| **Voice** (speech patterns, attitude) | persona-rule violations per 100 tokens (rules parsed from the card: brat register, corporate-model digs, "Hmph" cadence) | ≈ original's rate |
| **Spark** (wit, improv, coherence on novel topics) | **pairwise judge**: original Kisna vs transplanted Kisna on unseen prompts, "which is more Kisna" | ≈ 50% win rate = indistinguishable |

The spark metric is the new instrument this track contributes (see REFIT-DESIGN corpus note:
KL and rule-counts cannot see *funny*). A transplant that keeps lore + voice but loses the
spark is **S3 failure**, reported as such — not a partial win.

---

## 5. Hardware (RTX 4060 Laptop, 8 GB, 16 GB RAM)

| stage | feasibility | note |
|---|---|---|
| ΔW + SVD | **easy** | subtraction + per-matrix SVD; shard streaming, CPU is fine, one evening |
| Activation dump for coreset fit | tight | 4-bit forward over ~2k rows, one pass |
| NNLS + value refit | easy | CPU least squares, seconds |
| Butchered model inference | easy | 2B in fp16 = 4 GB |
| Original 7B for pairwise judging | **problematic** | 14.5 GB fp16 doesn't fit; use its 4-bit (3.8 GB) *only for judging*, never as the fit target — or judge on a rented box once |

Important: the 4-bit original is the **judge**, the fp16 shards are the **extraction source**.
Never extract from the quantized copy — quant noise would pollute ΔW and fake a diffuse
spectrum. Extraction always from fp16.

### Field probe — working laptop (no GPU), 29 Oct 2026

Stage 1 re-verified on a GPU-less work laptop (8 logical cores, 8.3 GB RAM, 290 GB free disk).
Findings, for scheduling at another machine:

- **CPU-only is confirmed sufficient for Stage 1.** Subtract + SVD never touches a GPU.
- **Checkpoint format is ZIP** (probe: shard 1 starts `PK\x03\x04`) → `torch.load(...,
  mmap=True)` works → the script streams **one matrix at a time** (~117 MB peak for the
  biggest, `down_proj`), never the full 5 GB shard. This is the difference between "runs" and
  "swaps to death" on a 8 GB machine. Legacy pickle would have killed it.
- **Download total is ~29 GB, not ~22 GB** (earlier estimate halved the base by mistake):
  Kisna shards 14.5 GB + base checkpoint 14.5 GB. Plan for the full 29.
- **Measured pipe: 100 MiB in 13.8 s ≈ 7.6 MB/s (≈61 Mbps)** from the HF CDN (single range
  request, 1 sample — treat as median).
- **Wall-clock estimate:** ~1 h download + ~30–45 min CPU (224 SVDs via Gram trick, 8 threads)
  ≈ **1.5–2 h total**. RAM headroom matters: ran at 2.9 GB free after closing browsers;
  want ≥2 GB free for the float32 SVD workspace.

Bottom line: Stage 1 is laptop-legal anywhere with ~30 GB disk and 8 GB RAM. Stage 2
activation-dump is the part that wants the GPU machine.

---

## 6. Falsification

The claim is **dead** if any hold:

1. **ΔW spectrum is diffuse** (no cliff, no low effective rank) AND 1b fails too → the soul is
   not separable; persona is entangled with the body; butchering is guaranteed to cut it.
   *(Then the honest product is "small Kisna from scratch on her 2k rows," which is a different
   and weaker thing.)*
2. **Transplant rejection** — top-r delta on the butchered body produces garbage or a broken
   register. Kills S3 operationally even if S1 held.
3. **Voice survives, spark dies** — pairwise judge well below 50% despite rule-counts matching.
   The improv lived in the body. This is the *expected* failure mode; reporting it cleanly is
   the point.
4. **Coreset at 7B behaves like Run 1 at 0.5B** — weighted ≈ binary, no fidelity recovered.
   Then Track B is dead at scale too, and both tracks close together.

Kill criteria 1 is checkable **tonight** (subtract + SVD, no training, no inference). Everything
else waits on it.

---

## 7. Step 1 (do first)

One script: stream both checkpoints, compute ΔW per matrix, SVD, dump spectra + cliff plot.
Output answers, in order: (a) is there a knee at 8–64? (b) effective rank per matrix at 90%
energy? (c) are embeddings/lm_head hot? No training, no GPU requirement, one evening.

If (a) is yes: the soul is the merged LoRA, extraction is exact, proceed to butchery with
confidence. If no: the smearing is itself the result — write it down before touching Stage 2.

---

## Open questions

- **Did QLoRA train embeddings/lm_head?** Unsloth default says no; tokenizer files exist in the
  repo but that's standard. ΔW answers it definitively.
- **Soul-rank vs body-shrink interaction:** does a rank-8 soul need the *full* body's residual
  geometry to route through, or is it body-agnostic? Stage 3 finds out; no theory to lean on.
- **Kisna workload is only ~2k rows** — is that enough query coverage for the coreset fit, or
  does the body butcher overfit the training data's prompts? Held-out *prompts* discipline
  applies; if in doubt, generate more unseen topics and treat the original 7B as oracle.
- **The spark's location is the scientific prize.** If it's body-bound (kill 3), that's evidence
  for emergence being general-competence-dressed-up — worth more than the artifact either way.

---

## References

| asset | what |
|---|---|
| `ArsParadox/Kisna-Mistral-v1` | the subject (full fp16, 3 shards) |
| `mistralai/Mistral-7B-Instruct-v0.3` | the body baseline (extraction source) |
| `ArsParadox/Kisna-Sovl-1k`, `Kisna-Assist-1k` | fit/eval corpus (~2k rows) |
| REFIT-DESIGN §Track B | the WildCat transplant procedure (Stage 2) |
| REFIT-DESIGN §1 families | taxonomy: low-rank (Stage 1) vs coreset (Stage 2/1b) |
