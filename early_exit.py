"""
Early-exit depth vs context length.

Question: as a token gets more preceding context, does the model become confident
*earlier* in the stack? If yes, then "more tokens -> less compute" is real, and it
is measured per-token and conditionally — unlike neuron pruning, which only got
worse as the corpus grew.

Method
------
One streaming pass over the corpus with a KV cache, so every position sees its true
full prefix. At sampled probe positions we capture the residual stream after every
layer, then project each one through the final norm + lm_head (the "logit lens") and
measure:

  exit_layer(p)  = shallowest layer whose top-1 prediction matches the final layer's
  KL(p)          = KL( final_dist || layer_dist ) at that layer
  confidence(p)  = max softmax of the FINAL layer

Then bucket by context length (= absolute position) and look at the trend.

No gradients. No training.
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="corpus_raw.txt")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--probe-every", type=int, default=256)
    ap.add_argument("--out", default="early_exit_results.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    final_norm = model.model.norm
    head = model.lm_head
    n_layers = len(layers)
    print(f"      layers={n_layers} d_model={model.config.hidden_size} "
          f"vocab={model.config.vocab_size}", flush=True)

    with open(args.corpus, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > args.max_tokens:
        ids = ids[:, : args.max_tokens]
    T = ids.shape[1]
    print(f"[2/4] corpus: {T} tokens", flush=True)

    probes = list(range(args.probe_every, T, args.probe_every))
    captured = {p: {} for p in probes}
    print(f"      probing {len(probes)} positions every {args.probe_every} tokens", flush=True)

    cur = {"start": 0}

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            start = cur["start"]
            for p in probes:
                j = p - start
                if 0 <= j < h.shape[1]:
                    captured[p][idx] = h[0, j].detach().clone()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

    print("[3/4] streaming pass with KV cache", flush=True)
    past = None
    n_chunks = (T + args.chunk - 1) // args.chunk
    with torch.no_grad():
        for c, start in enumerate(range(0, T, args.chunk)):
            end = min(start + args.chunk, T)
            cur["start"] = start
            out = model(input_ids=ids[:, start:end], past_key_values=past, use_cache=True)
            past = out.past_key_values
            if (c + 1) % 8 == 0 or c == n_chunks - 1:
                print(f"      chunk {c+1}/{n_chunks}  ({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()

    print("[4/4] logit-lens over probes", flush=True)
    KL_TAUS = [0.1, 0.5, 1.0]
    rows = []
    with torch.no_grad():
        for p in probes:
            if (n_layers - 1) not in captured[p]:
                continue
            lg_final = head(final_norm(captured[p][n_layers - 1]))
            top1_final = int(lg_final.argmax())
            p_final = torch.softmax(lg_final.float(), dim=-1)

            kl_curve = []
            exit_top1 = None
            for i in range(n_layers):
                lg = head(final_norm(captured[p][i]))
                logq = torch.log_softmax(lg.float(), dim=-1)
                kl = float(torch.nn.functional.kl_div(logq, p_final, reduction="sum"))
                kl_curve.append(kl)
                if exit_top1 is None and int(lg.argmax()) == top1_final:
                    exit_top1 = i

            if exit_top1 is None:
                exit_top1 = n_layers - 1

            # graded: shallowest layer whose distribution is within tau nats of final
            exit_kl = {}
            for tau in KL_TAUS:
                e = None
                for i, kl in enumerate(kl_curve):
                    if kl < tau:
                        e = i
                        break
                exit_kl[str(tau)] = e if e is not None else n_layers - 1

            rows.append({
                "pos": p,
                "exit_top1": exit_top1,
                "exit_kl": exit_kl,
                "kl_curve": [round(k, 4) for k in kl_curve],
                "kl_final_layer": kl_curve[-1],
                "confidence": float(p_final.max()),
                "top1": top1_final,
            })

    if not rows:
        print("no probes captured — check chunk/probe settings")
        return

    # --------------------------------------------------------------- report
    buckets = [(256, 512), (512, 1024), (1024, 2048), (2048, 4096),
               (4096, 8192), (8192, 16384), (16384, 32768)]
    print()
    print(f"early-exit depth vs context length  ({args.model})")
    print(f"{'context':<16}{'n':>4}{'exit@top1':>11}{'exit@KL<1':>11}"
          f"{'exit@KL<0.5':>13}{'exit@KL<0.1':>13}{'conf':>8}")
    print("-" * 78)
    summary = []
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["pos"] < hi]
        if not sel:
            continue
        def m(key):
            return sum(x["exit_kl"][key] for x in sel) / len(sel)
        m_top1 = sum(x["exit_top1"] for x in sel) / len(sel)
        conf = sum(x["confidence"] for x in sel) / len(sel)
        summary.append({"lo": lo, "hi": hi, "n": len(sel), "exit_top1": m_top1,
                        "exit_kl_1.0": m("1.0"), "exit_kl_0.5": m("0.5"),
                        "exit_kl_0.1": m("0.1"), "confidence": conf})
        print(f"{lo:>6}-{hi:<9}{len(sel):>4}{m_top1:>11.1f}{m('1.0'):>11.1f}"
              f"{m('0.5'):>13.1f}{m('0.1'):>13.1f}{conf:>8.3f}")

    xs = [r["pos"] for r in rows]
    ys = [r["exit_kl"]["0.5"] for r in rows]
    ys_top1 = [r["exit_top1"] for r in rows]
    cs = [r["confidence"] for r in rows]
    kl_end = [r["kl_curve"][-2] for r in rows]

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        return num / (da * db) if da and db else 0.0

    def spearman(a, b):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for pos, i in enumerate(order):
                r[i] = pos
            return r
        return pearson(rank(a), rank(b))

    print()
    for name, ys_ in [("exit@KL<0.5", ys), ("exit@top1", ys_top1)]:
        print(f"correlation(context length, {name:<11}): "
              f"pearson {pearson(xs, ys_):+.3f}   spearman {spearman(xs, ys_):+.3f}")
    print(f"correlation(context length, confidence):  pearson {pearson(xs, cs):+.3f}")
    print(f"correlation(context length, KL before last layer): pearson {pearson(xs, kl_end):+.3f}")

    s = spearman(xs, ys)
    print()
    print("interpretation:")
    if s < -0.2:
        print("  NEGATIVE -> later positions exit earlier. More context = less depth needed,")
        print("  i.e. 'more tokens -> less compute' holds, conditionally.")
    elif s > 0.2:
        print("  POSITIVE -> later positions need MORE depth. More context = more compute.")
    else:
        print("  ~ZERO -> exit depth is flat in context length. No effect.")

    print()
    print("sparkline (exit@KL<0.5, low=shallow/cheap, high=deep/expensive):")
    step = max(1, len(rows) // 60)
    spark = "".join(" .:-=+*#%@"[min(9, int(9 * r["exit_kl"]["0.5"] / (n_layers - 1)))]
                    for r in rows[::step])
    print("  " + spark)

    out = {
        "model": args.model,
        "corpus": args.corpus,
        "n_tokens": T,
        "n_layers": n_layers,
        "probe_every": args.probe_every,
        "rows": rows,
        "buckets": summary,
        "pearson_exit_kl": pearson(xs, ys),
        "spearman_exit_kl": spearman(xs, ys),
        "pearson_exit_top1": pearson(xs, ys_top1),
        "spearman_exit_top1": spearman(xs, ys_top1),
        "pearson_conf": pearson(xs, cs),
        "pearson_kl_before_last": pearson(xs, kl_end),
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {args.out}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
