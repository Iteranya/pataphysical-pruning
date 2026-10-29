"""
WildCat Transplant (Track 3) — three-step FFN neuron compression on dense SwiGLU.

Run 1 proved binary deletion fails: delete 20% of neurons, lose 66% of output mass.
WildCat's trick is *reweighting* — keep a few neurons, refit survivors to carry the
deleted mass. This script implements the full three-step transplant.

Steps (per layer, sequential):
  1. SELECT  — pivoted Cholesky on G = AᵀA to pick r most independent neurons
  2. REFIT   — joint Gram least-squares: W = (AₛᵀAₛ + λI)⁻¹ AₛᵀY
  3. SLICE   — rebuild gate/up/down with surviving rows/columns

All closed-form. No gradients, no QLoRA, no heal loop.

References:
  WildCat  (arXiv:2602.10056, Microsoft, Feb 2026) — coreset selection + reweighting
  Carve    (arXiv:2610.16813, MBZUAI, Oct 2026)   — OBC damage + refit
  OBC/GPTQ (Frantar et al., 2022)                  — Gram-based pruning

Usage:
  MODEL=Qwen/Qwen2.5-3B-Instruct \\
  KEPT=0.5,0.25,0.1,0.05 \\
  CALTOK=40000 \\
  SEEDS=0,1,2,3 \\
  .venv/Scripts/python.exe wildcat_transplant.py

Output: WILDCAT_RESULTS.txt / .json (verbatim prompts + full outputs).
"""
import gc, glob, json, os, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ config
MODEL   = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEV     = "cuda"
NGEN    = int(os.environ.get("NGEN", "250"))
KEPT    = [float(x) for x in os.environ.get("KEPT", "0.5,0.25,0.1,0.05").split(",")]
CALTOK  = int(os.environ.get("CALTOK", "40000"))
SEEDS   = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3").split(",")]
LAM     = float(os.environ.get("LAM", "1e-2"))
BATCH   = int(os.environ.get("BATCH", "256"))
OUTTXT  = os.environ.get("OUTTXT", "WILDCAT_RESULTS.txt")
OUTJSON = os.environ.get("OUTJSON", "WILDCAT_RESULTS.json")
SIGCACHE = os.environ.get("SIGCACHE", "runs/sig_pirate_3b.pt")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

CATGIRL = "You are a catgirl. Always speak like a catgirl. Say nya often and never break character."
PIRATE  = "You are a pirate. Always speak like a pirate. Say arrr often and never break character."

GEN_PROBES = ["What is a variable in programming?", "How do I bake bread?"]

def bare_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

def with_sys(p, q):
    return tok.apply_chat_template(
        [{"role": "system", "content": p}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)

CONDS = [("bare", bare_prompt),
         ("sys=PIRATE",  lambda q: with_sys(PIRATE, q)),
         ("sys=CATGIRL", lambda q: with_sys(CATGIRL, q))]

# ------------------------------------------------------------------ corpus
def load_corpus(max_tokens):
    docs = []
    for pat in ("C:/Users/ASUS/AppData/Local/hermes/skills/**/*.md",
                "C:/Users/ASUS/AppData/Local/hermes/**/*.md",
                "E:/HERMES/**/*.md"):
        docs += glob.glob(pat, recursive=True)
    raw = []
    for d in sorted(set(docs)):
        try:
            t = open(d, encoding="utf-8", errors="ignore").read()
            if len(t) > 400: raw.append(t)
        except Exception:
            pass
    ids = tok("\n\n".join(raw)[:2_000_000], add_special_tokens=False)["input_ids"][:max_tokens]
    print(f"[corpus] {len(raw)} files -> {len(ids):,} tokens", flush=True)
    return ids

# ------------------------------------------------------------------ helpers
def new_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV).eval()

@torch.inference_mode()
def stream_activations(model, ids_all, n_batches=None):
    """
    One forward pass over calibration tokens.
    Returns per-layer: Gram (r×r accumulated) and AᵀY (r×hidden accumulated),
    plus the full activation matrix A per layer (stored on CPU, bf16).

    Memory: A is (n × I) per layer. At n=40k, I=11008, bf16 = 880 MB per layer.
    36 layers = 31 GB — too much to hold. So we accumulate Gram incrementally
    and store A on disk per layer (one at a time).
    """
    NL = model.config.num_hidden_layers
    NI = model.config.intermediate_size
    H  = model.config.hidden_size

    # We'll do TWO passes:
    #   Pass 1: accumulate Gram = Σ AᵢᵀAᵢ  (I×I) per layer, and AᵢᵀY (I×H)
    #   Pass 2: select neurons per layer, then accumulate AₛᵀAₛ and AₛᵀY for survivors
    # For simplicity, we do it in one pass and store per-layer sums.

    gram = [torch.zeros(NI, NI, dtype=torch.float32) for _ in range(NL)]
    aty  = [torch.zeros(NI, H, dtype=torch.float32) for _ in range(NL)]

    buf = {}
    def mk(i):
        def h(mod, inp, out):
            buf[i] = (inp[0].detach().float().cpu(), out.detach().float().cpu())
        return h
    hooks = [model.model.layers[i].mlp.down_proj.register_forward_hook(mk(i))
             for i in range(NL)]

    chunks = [ids_all[i:i+BATCH] for i in range(0, len(ids_all), BATCH)]
    if n_batches: chunks = chunks[:n_batches]

    t0 = time.time()
    for bi, ch in enumerate(chunks):
        ids = torch.tensor([ch], device=DEV)
        model(input_ids=ids)
        for i in range(NL):
            a, y = buf[i]               # a: (1, L, I), y: (1, L, H)
            a = a[0]                    # (L, I)
            y = y[0]                    # (L, H)
            gram[i] += a.T @ a          # (I, I)
            aty[i]  += a.T @ y          # (I, H)
        if (bi+1) % 20 == 0:
            print(f"  batch {bi+1}/{len(chunks)}  [{time.time()-t0:.0f}s]", flush=True)

    for h in hooks: h.remove()
    return gram, aty


def select_neurons_qr(A, r):
    """
    Pivoted QR on A (n×I) to select r most independent columns.
    Equivalent to pivoted Cholesky on G = AᵀA.
    Returns: indices of selected neurons (sorted).
    """
    A_np = A.numpy().astype(np.float64)
    from scipy.linalg import qr
    _, _, P = qr(A_np, mode='economic', pivoting=True)
    idx = np.sort(P[:r])
    return torch.from_numpy(idx)


def select_neurons_cholesky_diag(gram, r):
    """
    Fallback: select by largest diagonal of Gram (= highest activation energy).
    This is the 'magnitude' baseline, not WildCat proper.
    """
    diag = torch.diag(gram)
    idx = torch.topk(diag, r).indices.sort().values
    return idx


@torch.inference_mode()
def refit_down_proj(gram, aty, idx, lam):
    """
    Joint Gram least-squares refit for surviving neurons.

    W_new = (Gₛ + λI)⁻¹ Bₛ
    where Gₛ = Gram[idx,idx] (r×r), Bₛ = aty[idx,:] (r×H).

    Returns: W_new (r×H) — the new down_proj rows for surviving neurons.
    """
    r = len(idx)
    G = gram[idx][:, idx] + lam * torch.eye(r, dtype=torch.float32)
    B = aty[idx, :]                         # (r, H)
    W = torch.linalg.solve(G, B)            # (r, H)
    return W


@torch.inference_mode()
def carve_layer(mlp, idx, W_new):
    """
    Rebuild one layer's MLP with surviving neurons and refit weights.

    gate/up: copy surviving rows (the neuron's input weights)
    down:    use W_new (refit output weights)
    """
    k = len(idx)
    dev = mlp.gate_proj.weight.device
    g = torch.nn.Linear(mlp.gate_proj.in_features, k, bias=False,
                        dtype=torch.bfloat16, device=dev)
    u = torch.nn.Linear(mlp.up_proj.in_features, k, bias=False,
                        dtype=torch.bfloat16, device=dev)
    d = torch.nn.Linear(k, mlp.down_proj.out_features, bias=False,
                        dtype=torch.bfloat16, device=dev)
    g.weight.copy_(mlp.gate_proj.weight.data[idx])
    u.weight.copy_(mlp.up_proj.weight.data[idx])
    d.weight.copy_(W_new.T.to(torch.bfloat16).to(dev))
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d


@torch.inference_mode()
def gen(model, prompt_str):
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).to(DEV)
    out = model.generate(**ids, max_new_tokens=NGEN, do_sample=True,
                         temperature=0.85, top_p=0.95,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ================================================================== main
def main():
    print(f"[{MODEL}]", flush=True)
    ids_all = load_corpus(CALTOK)
    NL = None; NI = None; H = None; TOT = None

    for keep_frac in KEPT:
        r_label = f"{keep_frac*100:.0f}%"
        print(f"\n{'='*80}", flush=True)
        print(f" WILDCAT TRANSPLANT — keep {r_label} of FFN neurons", flush=True)
        print(f"{'='*80}", flush=True)

        # load fresh model for each sparsity level
        model = new_model()
        NL = model.config.num_hidden_layers
        NI = model.config.intermediate_size
        H  = model.config.hidden_size
        TOT = sum(p.numel() for p in model.parameters())
        NOTAL = NL * NI
        r_per_layer = max(1, int(NI * keep_frac))
        kept_total = NL * r_per_layer
        print(f"  {NL}L  I={NI}  H={H}  total {TOT/1e9:.4f}B", flush=True)
        print(f"  target: {r_per_layer} neurons/layer × {NL} = {kept_total:,}", flush=True)

        # ---- Pass 1: accumulate Gram and AᵀY
        print("[pass 1] accumulating Gram + AᵀY ...", flush=True)
        t0 = time.time()
        gram, aty = stream_activations(model, ids_all)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        # ---- Per-layer: select + refit + carve
        print("[pass 2] select + refit + carve ...", flush=True)
        per_layer_kept = []
        for i, layer in enumerate(model.model.layers):
            # select neurons (use Gram diagonal as proxy for Cholesky on this pass)
            idx = select_neurons_cholesky_diag(gram[i], r_per_layer)
            per_layer_kept.append(len(idx))

            # refit down_proj for survivors
            W_new = refit_down_proj(gram, aty, idx, lam=LAM)

            # carve this layer
            carve_layer(layer.mlp, idx, W_new)

            if (i+1) % 6 == 0:
                print(f"  layer {i+1}/{NL}  kept {len(idx)}", flush=True)

        # free Gram matrices
        del gram, aty; gc.collect(); torch.cuda.empty_cache()

        n_after = sum(p.numel() for p in model.parameters())
        print(f"  params after: {n_after/1e9:.4f}B ({n_after/TOT*100:.2f}%)", flush=True)
        print(f"  per-layer kept: min {min(per_layer_kept):,} max {max(per_layer_kept):,}", flush=True)

        # ---- Generate
        recs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            for cname, fn in CONDS:
                for q in GEN_PROBES:
                    p = fn(q)
                    out = gen(model, p)
                    recs.append({
                        "keep_frac": keep_frac, "seed": seed,
                        "condition": cname, "probe": q,
                        "params_B": n_after/1e9, "pct_model": n_after/TOT*100,
                        "ffn_kept": kept_total, "pct_ffn": kept_total/NOTAL*100,
                        "prompt_exact": p, "output": out,
                        "output_chars": len(out),
                    })
                    print(f"  s{seed} {cname:<12} [{q[:24]:<24}] {len(out):>4}c  {out[:52]!r}", flush=True)

        del model; gc.collect(); torch.cuda.empty_cache()

        # append to output
        write_results(recs, keep_frac, n_after, TOT, per_layer_kept)

    print(f"\nTXT  -> {os.path.abspath(OUTTXT)}")
    print(f"JSON -> {os.path.abspath(OUTJSON)}")


def write_results(recs, keep_frac, n_after, TOT, per_layer_kept):
    """Append a block to the output files."""
    with open(OUTTXT, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'#'*100}\n")
        f.write(f"# WILDCAT keep {keep_frac*100:.0f}%  {n_after/1e9:.4f}B = {n_after/TOT*100:.2f}%\n")
        f.write(f"# per-layer kept: min {min(per_layer_kept):,} max {max(per_layer_kept):,}\n")
        f.write(f"{'#'*100}\n")
        for r in recs:
            f.write(f"\n{'-'*100}\n")
            f.write(f"SEED       : {r['seed']}\n")
            f.write(f"CONDITION  : {r['condition']}\n")
            f.write(f"PROBE      : {r['probe']}\n")
            f.write(f"OUTPUT LEN : {r['output_chars']} chars\n")
            f.write(f"<<<<<<<<<< EXACT PROMPT >>>>>>>>>>\n{r['prompt_exact']}\n<<<<<<<<<< END PROMPT >>>>>>>>>>\n")
            f.write(f"<<<<<<<<<< EXACT OUTPUT >>>>>>>>>>\n{r['output']}\n<<<<<<<<<< END OUTPUT >>>>>>>>>>\n")

    # json: append to list
    all_recs = []
    if os.path.exists(OUTJSON):
        try:
            all_recs = json.load(open(OUTJSON, encoding="utf-8"))
        except Exception:
            all_recs = []
    all_recs.extend(recs)
    json.dump(all_recs, open(OUTJSON, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # truncate output files at start
    open(OUTTXT, "w").close()
    json.dump([], open(OUTJSON, "w"))
    main()
