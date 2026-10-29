# Census Run 2 — Early-Exit Depth vs Context Length

**Date:** 2026-10-29
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, d_model 896, vocab 151,936)
**Corpus:** *Alice's Adventures in Wonderland*, first 16,384 tokens
**Method:** streaming pass with KV cache (every position sees its true prefix), residual stream
captured at 63 probe positions (every 256 tokens), each projected through final norm + lm_head
("logit lens"), compared against the final layer's distribution
**Runtime:** 301.1 s (CPU) — reproduced twice, identical results
**Script:** `early_exit.py` · raw data: `early_exit_results.json`

---

## Question

Does a token with more preceding context let the model commit to its answer *earlier in the
stack*? If yes, then "more tokens -> less compute" is real, per-token and conditional — unlike
neuron pruning, which only got harder as the corpus grew.

## Result: flat

| context | n | exit@top1 | exit@KL<1 | exit@KL<0.5 | exit@KL<0.1 | confidence |
|---|---|---|---|---|---|---|
| 256–512 | 1 | 20.0 | 23.0 | 23.0 | 23.0 | 0.393 |
| 512–1024 | 2 | 23.0 | 22.5 | 23.0 | 23.0 | 0.477 |
| 1024–2048 | 4 | 22.2 | 22.8 | 23.0 | 23.0 | 0.232 |
| 2048–4096 | 8 | 20.6 | 22.4 | 22.8 | 23.0 | 0.445 |
| 4096–8192 | 16 | 19.8 | 21.4 | 22.5 | 22.9 | 0.586 |
| 8192–16384 | 32 | 21.2 | 22.1 | 22.5 | 22.7 | 0.550 |

Exit depth sits at **22–23 of 24 layers** across the entire range. Best case the graded criterion
moves 23.0 -> 22.5: a quarter of a layer over 16,000 tokens.

Correlations disagree in sign, which means noise rather than a weak trend:

| pair | pearson | spearman |
|---|---|---|
| context vs exit@KL<0.5 | **-0.156** | **+0.422** |
| context vs exit@top1 | +0.051 | +0.233 |
| context vs confidence | +0.222 | — |

## What did move

**Confidence: 0.393 -> 0.550** (weak, non-monotone — dips to 0.232 mid-range). The output
distribution does narrow with context. That part of the intuition holds.

It just does not convert into **depth**. The model needs the full stack either way.

## Methodology notes

- **The logit lens is biased in absolute terms.** Projecting layer 8's residual stream through the
  final norm + lm_head gives early layers an unfair test, so "exit at layer 22" is likely an
  overestimate everywhere. But it is the *same* biased instrument at every context length, so the
  null result on the *trend* is trustworthy even though the absolute numbers are not.
- **Strict top-1 match saturates** near the final layer, which is why the graded KL thresholds were
  added. Both agree on the null.

## Why this null is probably about the corpus, not the mechanism

Alice is **entropy-stationary**. A novel does not become more predictable as it proceeds — new
scenes, new speakers, new vocabulary. Position 12,000 is not more constrained than position 500.

So this run measured **context length**, not **constraint strength**. The hypothesis is about the
second one.

The corpus that would actually test it has **saturating structure**: source code, a long table, a
repeated format. There, position genuinely predicts predictability. If exit depth falls with
context anywhere, it will be there.

**Next:** rerun the same probe on a 16k-token structured corpus (Python file or CSV). Same
harness, ~5 minutes, one corpus swap.

---

## Running score across all three mechanisms

| mechanism | what should shrink with more tokens | result |
|---|---|---|
| neuron pruning | dead units | **0% dead**, flat mass (Run 1) |
| expert pruning | unused experts | blocked: top-k floor, load balancer, union bound |
| depth / early exit | layers required | **flat**, 22 -> 22.5 of 24 (Run 2) |

No mechanism converts context length into a smaller *pretrained* model. The only surviving lever
is that the output distribution narrows — real, but worth ~3% of compute (lm_head), since most
cost is per-token matmul through the stack.

The idea survives only in the form of a statement about **the smallest model you would have to
train** (context distillation frontier), not about anything prunable out of an existing one.
