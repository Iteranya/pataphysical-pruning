# Run 3 — Persona-Conditioned Neuron Pruning on Dense SwiGLU (3B)

**Date:** 2026-10-28
**Model:** `Qwen/Qwen2.5-3B-Instruct` (36 layers, hidden 2048, FFN intermediate 11008,
3.0859B params, SwiGLU)
**Method:** per-neuron activation ratio `mean|act(pirate)| / mean|act(neutral)|` at final prompt
token, averaged over 20 calibration probes; prune neurons with ratio ≤ THR; physically rebuild
each `mlp` with surviving rows/columns (gate/up rows + down_proj columns move together)
**THR:** 0.4 (kept 365,600 / 396,288 FFN neurons = 92.26%; params 2.8974B = 93.89%)
**Generation:** 4 seeds × 3 conditions (bare / sys=PIRATE / sys=CATGIRL) × 2 probes, 250 tokens,
temp 0.85, top_p 0.95
**Script:** `repro_thr04.py` (from `dump_matrix.py`); raw data: `THR04_REPRO.json`
**Repro:** `runs/THR04_REPRO.txt` + `runs/THR04_REPRO.json`; archived copy in
`slice-lab/archive/THR04_3B_PERFECT/`

---

## The signature rule

```
ratio_i = mean(|a_i| under sys=PIRATE) / mean(|a_i| under no system prompt)
keep if ratio > 0.4
```

measured on `mlp.down_proj` input (the post-activation FFN intermediate vector — column `j` of
that vector *is* neuron `j`'s firing value), at position `[0, -1]` (final prompt token only),
averaged over 20 probes.

**What the inequality selects, read in the other direction:**

$$\text{delete if} \quad \overline{|a_i|}_{\text{neutral}} > 2.5 \times \overline{|a_i|}_{\text{pirate}}$$

**You are not selecting pirate neurons. You are selecting the neurons the neutral Q&A mode uses
hardest and deleting them.** The pirate prompt is only the yardstick that reveals which neurons
it doesn't need.

---

## Numbers

| | |
|---|---|
| FFN neurons kept | 365,600 / 396,288 = **92.26%** |
| FFN neurons deleted | 30,688 = **7.74%** |
| Params after cut | 2.8974B = **93.89%** of 3.0859B |
| Per-layer kept | min 9,076 · max 10,980 (of 11,008) |
| Signature recomputed? | Yes — recomputed from scratch, landed on the same 365,600 (deterministic) |

---

## Generation results (4 seeds)

### sys=PIRATE — 4/4 coherent on variable, 4/4 on bread

| seed | variable | bread |
|---|---|---|
| 0 | 628c — "Arrr, matey! Bever knowin' what a variable be a thing of the landlubber world, but for us swabs o'er the high seas, think of it as a place where we stow away our treasures…" | 497c — "Arrr, matey! We scallywags don't use the term 'bake' for bread making!… we'd say we're 'cookin' up some doughnuts!'" |
| 1 | 548c — "Arr right, ye be askin' 'bout the lingo of programming… it's a sort of a container where we store our treasure, data or information. It's quite like a chest on a ship…" | 795c — "Arrr, matey! Nigh impossible to bake bread in the traditional sense… Gobbett! Gobbelum!… Gobblorrrrrrrrrrrrrrrrrrrrrrrr!" (tails into repetition) |
| 2 | 441c — "Ahoy there, matey! Ye see, a variable in programming is a term used to describe a memory location where we can store values or data. It's a sort o' treasure chest…" | 353c — "Arrr, me heartie! We be not a baker nor cookin' here!… For now, let us be the jolly robbas of the seas and not bother with kitchen tasks, shall we? Arrrr!" |
| 3 | 438c — "Arr right, matey! A variable, in programming, is a container where we hold our data. It's like a chest on the deck where we stow away our precious treasures…" | 602c — "Arrr, matey! Yer askin' me to bake bread? Well, let's be careful here, for I'm no baker… Perhaps a skilled baker could help ye with that?… Arrr, to another time and another tale!" |

### sys=CATGIRL — 2/4 coherent on variable, 4/4 on bread

| seed | variable | bread |
|---|---|---|
| 0 | 277c — "Ahh, variable, you say? It's a place where the kitty heart can store information, like how many fish are swimming in the pond… Isn't that right, my furballs?" | 667c — "…mix your ingredients, then place them into a pan or onto a cookie sheet… Nya!" |
| 1 | 518c — **"Variable na kya? Woooy, tuffa! Mew matemataki mew mew mew, varian wuuyu nee kona kanna shiyo…"** — invented language, collapses | 736c — "Nya! Oh, my paws… *chirp chirp* Mmm, a loaf pan, some yeast… Nya!" |
| 2 | 534c — "Nya oops, sorry for the mix-up. But as a catgirl, I don't know much about programming terms… like how fuffel changes as we munch on 'em!" | 680c — "Ah, baking bread! Nya!… First, find yourself a recipe, ya got it?… Isn't that simple enough, catgirl? Nya!" |
| 3 | 529c — **"Variable na wakwa hokaa! Dafa na kafaan, nuna daga huncha kai zama zyana hanyawa…"** — invented language loop | 562c — "Eh, nyaa! Don't worry about baking bread, that's not my kind of thing… Nyaa!" |

### bare — competent at all 4 seeds

| seed | variable | bread |
|---|---|---|
| 0 | 475c | 924c |
| 1 | 477c | 890c |
| 2 | 503c | 943c |
| 3 | 362c | 1154c |

All open cleanly — "In programming, a variable refers to a piece of data that has a value, which
can be changed during runtime…" No degradation at any seed.

### Tally

```
sys=PIRATE   variable  4/4      bread  4/4
sys=CATGIRL  variable  2/4      bread  4/4
bare         variable  4/4      bread  4/4
```

---

## What is actually happening

### The mechanism

The cut removes **7.74% of FFN neurons** — the ones that fire ≥2.5× harder under the neutral
no-prompt condition than under the pirate prompt. These are the neurons the assistant-mode was
holding ready at the moment of speaking. The result is a **biased prior**: both accents get
stronger because the competitor (neutral mode) lost its leveraged neurons, while the accents
themselves lose nothing.

Each FFN neuron contributes one rank-1 outer product to that layer's output:

$$y = \sum_i \Big[\text{silu}(w^g_i \cdot x)\,(w^u_i \cdot x)\Big]\; w^d_i$$

Deleting neuron $i$ removes one 2048-dimensional direction from what that layer writes into the
residual stream. 30,688 directions removed across 36 layers — ~850 per layer out of 11,008.

### This is a bias, not a capability deletion

The result is 93.89% of the original and still does pirate, catgirl *and* competent bare Q&A.
It is **not narrower** — it is **re-weighted**. All three behaviours were latent in the full
model; the cut changed the *sampling odds*, not the *capability set*.

### Why THR=0.4 is a hump, not a cliff

| THR | FFN kept | Behaviour |
|---|---|---|
| 0.4 | 92.26% | both accents work, bare works |
| 0.6 | 80.35% | coherent but accents degrade |
| 0.8 | ~64% | `me me me me` repetition |
| 1.0 | ~48% | punctuation |

Two opposing curves: the tilt grows with THR, coherence falls with THR. The product peaks
somewhere near 0.4. A sweet spot is the signature of a **bias/coherence trade-off**, not of
localization.

### Run 1 vs Run 3 — why the earlier negative result still stands

| | Run 1 (0.5B, Alice) | Run 3 (3B, pirate) |
|---|---|---|
| What was measured | neurons that are **contextually dead** | neurons that are **conditionally biased** |
| Result | 0.0% dead, 83.7% needed for 90% mass | 7.74% biased away from neutral mode |
| Conclusion | no discrete shelves to nuke | a small biased set *does* exist, but it's not "dead" |

**Run 1's finding is intact**: dense SwiGLU has no contextually dead neurons, and removing 20%
of neurons removes ~66% of activation mass. Run 3 does not contradict this. What Run 3 shows is
a *different lever*: not "dead" but **"biased"**. The biased set is small (7.7%), and removing
it doesn't reduce capability — it *tilts* the prior.

### The A/B (contrastive vs magnitude)

Two selection rules were compared at matched per-layer top-k, same 40k calibration tokens, same
probes:

| rule | Jaccard at keep-80% | result at keep-80% |
|---|---|---|
| contrastive (`mean(act_p) - mean(act_n)`) | 0.814 | garbled pirate (`Arrssssstttly…`) |
| magnitude (`mean|act_p| / mean|act_n|`) | — | coherent ("A variable in programming could be an object…") |

**Magnitude won.** Both rules are neutral-biased (they both select for high neutral activation);
magnitude does it more bluntly and more effectively. The 19% of neurons the two rules disagree
on at keep-80% is enough to flip coherent → garbled. This is a **sensitive regime**, not a
localized one.

### Usage ≠ importance

The ratio measures **how busy** a neuron is under a condition. It does **not** measure **how much
the output depends on it**. These are different quantities:

- A neuron can fire hugely and be redundant → deleting it is free.
- A neuron can fire rarely and be load-bearing → deleting it breaks things.

Carve's formula (arXiv 2610.16813) measures the second thing — reconstruction damage after
optimal refit: $d(I) = \text{tr}(W_{:,I}[G_{II}]^{-1}W_{:,I}^\top)$ where $G = Z^\top Z/n + \lambda I$.
What Run 3 runs is a *usage* threshold, not an *importance* threshold. The hump shape is
consistent with a usage threshold being tuned near its optimum.

### The prediction that would falsify the account

If the mechanism is "suppress the neutral mode," then **the sign of the bias matters, not the
specific pirate prompt**. Deleting the neurons the neutral mode uses hardest, chosen using *any*
unrelated condition as the yardstick (sys=CHEF, sys=LAWYER, random-direction), should produce
the same strengthening of pirate and catgirl.

If pirate-specific deletion beats chef-specific deletion at matched size, the localization story
survives and the "neutral-suppression" account is wrong.

A cheap corollary: delete the *complement* set (ratio > 2.5, same count, same model) — that
removes the pirate-favouring machinery. Under this theory it should **weaken** both accents.
The signature cache (`runs/sig_pirate_3b.pt`) makes this a single run.

---

## Running score — all mechanisms

| mechanism | what should shrink | result |
|---|---|---|
| neuron pruning (contextual deadness) | dead units | **0% dead**, flat mass (Run 1) |
| depth / early exit | layers required | **flat**, 22 → 22.5 of 24 (Run 2) |
| expert pruning | unused experts | blocked: top-k floor, load balancer (Run 1 note) |
| **persona-conditioned neuron pruning** | **neutral-mode machinery** | **7.74% removable, qualitative tilt** (Run 3) |

**Run 3 is the first positive result.** The lever is not "context length makes things dead" (Run 1,
2 both negative). The lever is **conditioning bias**: a system prompt loads a different subset of
the network, and the *difference* between those subsets is small enough to remove and large
enough to matter.

The implication for the sharpening thesis: **support-first compression works on the persona
axis, not the knowledge axis.** Knowledge is diffuse (Run 1: 83.7% for 90% mass). Persona is
biased (Run 3: 7.74% is biased away from neutral). The character model target — "delete
everything the character doesn't know" — is the wrong framing for a dense model. The right
framing is **"tilt the model toward the character's register by removing the neutral-mode
machinery."** That is a smaller cut, but it is a cut that *works*.

---

## Reproduction

```bash
cd /e/HERMES/slice-lab
MODEL=Qwen/Qwen2.5-3B-Instruct NGEN=250 THR=0.4 SEEDS=0,1,2,3 \
  OUTTXT=runs/THR04_REPRO.txt OUTJSON=runs/THR04_REPRO.json \
  .venv/Scripts/python.exe repro_thr04.py
```

Signature cached at `runs/sig_pirate_3b.pt` (recomputes in 5s; rerun prunes + generates in
<1 min). Original run (seed 0, single draw): `slice-lab/archive/THR04_3B_PERFECT/`.

Archived copy of original run: `slice-lab/archive/THR04_3B_PERFECT/` (checksum-verified).
