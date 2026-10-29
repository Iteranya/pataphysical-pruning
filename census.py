"""
Activation sparsity census — Step 1 of the inverse-pataphysical pruning experiment.

Question: on a coherent narrative corpus, what fraction of FFN intermediate neurons
are *contextually dead* (fire on <1% of tokens)?

If that fraction is large, the "nuke the library to one prefix" premise has a referent.
If it is small, the premise is dead and we stop before building anything.

Runs CPU-only. No gradients, no training. Forward passes + hooks.
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
THRESH_FRACS = [0.01, 0.02, 0.05, 0.10, 0.25]


def load_corpus(path, tokenizer, max_tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_tokens:
        ids = ids[:, :max_tokens]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="corpus_raw.txt")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="census_results.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] loading {args.model} (float32, cpu)", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    inter = cfg.intermediate_size
    print(f"      layers={n_layers} d_model={cfg.hidden_size} intermediate={inter} "
          f"vocab={cfg.vocab_size}", flush=True)

    print(f"[2/5] loading corpus {args.corpus}", flush=True)
    ids = load_corpus(args.corpus, tok, args.max_tokens)
    n_tok = ids.shape[1]
    print(f"      {n_tok} tokens ({n_tok/1000:.1f}k)", flush=True)

    mlps = [model.model.layers[i].mlp.down_proj for i in range(n_layers)]

    # ---------------------------------------------------------------- pass A
    # Calibration: one forward pass over a prefix to get a per-layer activation
    # scale. We use this instead of an absolute constant so the threshold means
    # the same thing in every layer (the mistake in the earlier magnitude run).
    print("[3/5] calibration pass -> per-layer |act| p99 scale", flush=True)
    p99 = [None] * n_layers
    calib = ids[:, : args.calib_tokens]

    def make_scale_hook(idx):
        def hook(module, inp):
            a = inp[0].detach().abs()
            p99[idx] = torch.quantile(a.flatten().float(), 0.99).item()
        return hook

    hs = [mlps[i].register_forward_pre_hook(make_scale_hook(i)) for i in range(n_layers)]
    with torch.no_grad():
        model(calib)
    for h in hs:
        h.remove()
    print("      scale per layer (min/median/max): "
          f"{min(p99):.4f} / {sorted(p99)[n_layers//2]:.4f} / {max(p99):.4f}", flush=True)

    # ---------------------------------------------------------------- pass B
    print("[4/5] main pass -> activation frequency per neuron", flush=True)
    # counts[layer][thr_idx] = int64 tensor of shape (inter,)
    counts = [
        [torch.zeros(inter, dtype=torch.int32) for _ in THRESH_FRACS]
        for _ in range(n_layers)
    ]
    total_active = [0 for _ in THRESH_FRACS]   # individual activations above thr
    total_elems = 0
    # contribution mass: sum of |activation| per neuron, accumulated separately so
    # we can measure how concentrated the FFN *output* is, not just its support.
    mass = [torch.zeros(inter, dtype=torch.float64) for _ in range(n_layers)]
    tiny_frac = [0.0] * n_layers   # fraction of all tokens with |a| < 1% p99 per layer

    def make_count_hook(idx):
        taus = [f * p99[idx] for f in THRESH_FRACS]

        def hook(module, inp):
            a = inp[0].detach()[0]            # (T, I)
            nonlocal total_elems
            aa = a.abs()
            total_elems += a.shape[0] * a.shape[1]
            for j, tau in enumerate(taus):
                counts[idx][j] += (aa > tau).sum(dim=0).to(torch.int32)
                total_active[j] += int((aa > tau).sum().item())
            mass[idx] += aa.sum(dim=0).double()
        return hook

    hs = [mlps[i].register_forward_pre_hook(make_count_hook(i)) for i in range(n_layers)]
    n_chunks = (n_tok + args.chunk - 1) // args.chunk
    with torch.no_grad():
        for c in range(n_chunks):
            lo, hi = c * args.chunk, min((c + 1) * args.chunk, n_tok)
            model(ids[:, lo:hi])
            if (c + 1) % 10 == 0 or c == n_chunks - 1:
                print(f"      chunk {c+1}/{n_chunks}  ({time.time()-t0:.0f}s)", flush=True)
    for h in hs:
        h.remove()

    # ---------------------------------------------------------------- report
    print("[5/5] results", flush=True)
    print()
    print(f"corpus: {n_tok} tokens | model: {args.model}")
    print(f"FFN neurons total: {n_layers * inter:,} ({n_layers} layers x {inter})")
    print()
    header = "thr(p99 frac) " + " ".join(f"{f:>7.0%}" for f in THRESH_FRACS)
    print(header)
    print("-" * len(header))

    rows = []
    for j, f in enumerate(THRESH_FRACS):
        # per-layer fraction of neurons dead on <1% of tokens
        dead_fracs = []
        for i in range(n_layers):
            freq = counts[i][j].float() / n_tok
            dead_fracs.append((freq < 0.01).float().mean().item())
        overall_dead = sum(
            (counts[i][j].float() / n_tok < 0.01).sum().item() for i in range(n_layers)
        ) / (n_layers * inter)
        abs_sparsity = 1.0 - (total_active[j] / max(total_elems, 1))
        rows.append({
            "tau_frac_of_p99": f,
            "tau_abs": f,  # per-layer; see p99 list
            "frac_neurons_dead_lt1pct": overall_dead,
            "frac_neurons_dead_lt1pct_by_layer": dead_fracs,
            "activation_sparsity": abs_sparsity,
        })

    print(f"{'neurons dead <1% of tokens':<28}" + " ".join(f"{r['frac_neurons_dead_lt1pct']:>7.1%}" for r in rows))
    print(f"{'activation sparsity (all)':<28}" + " ".join(f"{r['activation_sparsity']:>7.1%}" for r in rows))
    print()
    print("next: how dead does a neuron have to be before we can delete it?")
    print("(tau = 5% of per-layer p99)")
    print()
    print(f"{'neuron fires on <X% of tokens':<32}" + "".join(f"{c:>8.1%}" for c in [0.1, 0.5, 1, 2, 5, 10]))
    cut_line = []
    for cutoff in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        tot = 0
        for i in range(n_layers):
            freq = counts[i][2].float() / n_tok
            tot += int((freq < cutoff).sum().item())
        cut_line.append(tot / (n_layers * inter))
    print(f"{'  -> % of all FFN neurons':<32}" + "".join(f"{v:>8.1%}" for v in cut_line))
    print()

    # ---- contribution mass concentration (the metric that actually decides it)
    print("contribution mass concentration (mean |activation| per neuron):")
    print(f"{'  layer':<8}{'top10% neurons':>16}{'top20%':>10}{'top50%':>10}{'neurons for 90% mass':>24}")
    conc_rows = []
    for i in range(n_layers):
        m, idxs = mass[i].sort(descending=True)
        cum = torch.cumsum(m, 0) / m.sum()
        def share(k):
            return float(cum[max(int(inter * k) - 1, 0)])
        n90 = int((cum < 0.90).sum().item()) + 1
        conc_rows.append({"layer": i, "top10": share(0.10), "top20": share(0.20),
                          "top50": share(0.50), "neurons_for_90pct_mass": n90})
        print(f"  L{i:02d}{share(0.10):>18.1%}{share(0.20):>10.1%}{share(0.50):>10.1%}"
              f"{n90/inter:>20.1%} ({n90})")
    avg90 = sum(r["neurons_for_90pct_mass"] for r in conc_rows) / n_layers / inter
    print()
    print(f"  >> on average, {avg90:.1%} of FFN neurons carry 90% of the activation mass")
    print()

    print("per-layer neurons dead <1% (at tau = 5% of p99):")
    d = rows[2]["frac_neurons_dead_lt1pct_by_layer"]
    for i, v in enumerate(d):
        bar = "#" * int(v * 40)
        print(f"  L{i:02d} {v:6.1%} {bar}")

    out = {
        "model": args.model,
        "corpus": args.corpus,
        "n_tokens": n_tok,
        "n_layers": n_layers,
        "intermediate_size": inter,
        "p99_scale_per_layer": p99,
        "results": rows,
        "mass_concentration": conc_rows,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {args.out}   ({out['runtime_s']}s total)")


if __name__ == "__main__":
    main()
