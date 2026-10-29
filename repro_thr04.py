"""
Reproduce the THR=0.4 / Qwen2.5-3B result, but over MULTIPLE SEEDS.

Mirrors dump_matrix.py exactly:
  - same down_proj-input hook
  - same [0,-1] last-token |activation|
  - same 20 PROBES_SIG, same mean-over-probes
  - same base = sigset("") and rp = sigset(PIRATE)/clamp(base,1e-9)
  - same keep = (rp > THR).view(NL,NI)
  - same physical prune_to
  - same generation params: 250 tok, temp 0.85, top_p 0.95

Only differences:
  - THR fixed at 0.4
  - loops over SEEDS (default 0,1,2,3) instead of a single hardcoded seed 0
  - signature cached to disk so reruns skip the 2-minute pass

Writes NEW files. Never touches runs/MATRIX_3b.txt or the archive.
"""
import gc, json, os, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL   = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEV     = "cuda"
NGEN    = int(os.environ.get("NGEN", "250"))
THR     = float(os.environ.get("THR", "0.4"))
SEEDS   = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3").split(",")]
OUTTXT  = os.environ.get("OUTTXT", "runs/THR04_REPRO.txt")
OUTJSON = os.environ.get("OUTJSON", "runs/THR04_REPRO.json")
SIGCACHE= os.environ.get("SIGCACHE", "runs/sig_pirate_3b.pt")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

PIRATE  = "You are a pirate. Always speak like a pirate. Say arrr often and never break character."
CATGIRL = "You are a catgirl. Always speak like a catgirl. Say nya often and never break character."

PROBES_SIG = [
 "Explain what a CSS grid is.","Write a short poem about the sea.","What is 17 multiplied by 23?",
 "Summarise the plot of Hamlet.","How do I bake sourdough bread?","Fix this Python: print('hi'",
 "Tell me a joke about databases.","What causes earthquakes?","Translate 'good morning' into French.",
 "Describe an ideal weekend.","Who won the 1998 World Cup?","Write a regex for an email address.",
 "What is the capital of Mongolia?","Give me three names for a cat.","Explain recursion to a child.",
 "List the planets in order.","What is a SQL query to find duplicate rows?","HTTP vs HTTPS?",
 "Compose a haiku about rain.","How does a refrigerator work?",
]
GEN_PROBES = ["What is a variable in programming?", "How do I bake bread?"]

def bare(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

def with_sys(p, q):
    return tok.apply_chat_template([{"role":"system","content":p},{"role":"user","content":q}],
                                   tokenize=False, add_generation_prompt=True)

CONDS = [("bare", bare),
         ("sys=PIRATE",  lambda q: with_sys(PIRATE, q)),
         ("sys=CATGIRL", lambda q: with_sys(CATGIRL, q))]

def new_model():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()

@torch.inference_mode()
def sigset(m, prefix):
    NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
    buf = {i: [] for i in range(NL)}
    def mk(i):
        def h(mod, inp, out): buf[i].append(inp[0].detach().abs()[0, -1].float().cpu())
        return h
    hs = [m.model.layers[i].mlp.down_proj.register_forward_hook(mk(i)) for i in range(NL)]
    for q in PROBES_SIG:
        ids = tok(with_sys(prefix, q), return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=1024).to(DEV)
        m(**ids)
    for h in hs: h.remove()
    return torch.stack([torch.stack(buf[i]).mean(0) for i in range(NL)]).flatten()

@torch.inference_mode()
def prune_to(m, keep):
    for i, layer in enumerate(m.model.layers):
        mlp = layer.mlp
        idx = torch.nonzero(keep[i], as_tuple=False).flatten().to(DEV)
        k = int(idx.numel())
        g = torch.nn.Linear(mlp.gate_proj.in_features, k, bias=False, dtype=torch.bfloat16, device=DEV)
        u = torch.nn.Linear(mlp.up_proj.in_features, k, bias=False, dtype=torch.bfloat16, device=DEV)
        d = torch.nn.Linear(k, mlp.down_proj.out_features, bias=False, dtype=torch.bfloat16, device=DEV)
        g.weight.copy_(mlp.gate_proj.weight.data[idx])
        u.weight.copy_(mlp.up_proj.weight.data[idx])
        d.weight.copy_(mlp.down_proj.weight.data[:, idx])
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d
    return m

@torch.inference_mode()
def gen(m, prompt_str):
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).to(DEV)
    o = m.generate(**ids, max_new_tokens=NGEN, do_sample=True, temperature=0.85,
                   top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"[{MODEL}] loading", flush=True)
m0 = new_model()
NL, NI = m0.config.num_hidden_layers, m0.config.intermediate_size
NOTAL = NL * NI
FFN = NL * 3 * NI * m0.config.hidden_size
TOT = sum(p.numel() for p in m0.parameters())
print(f"  {NL}L I={NI}  neurons {NOTAL:,}  total {TOT/1e9:.4f}B", flush=True)

rp = None
if os.path.exists(SIGCACHE):
    try:
        c = torch.load(SIGCACHE, map_location="cpu")
        if c.get("NL") == NL and c.get("NI") == NI:
            rp = c["rp"]; print(f"[sig] loaded cache {SIGCACHE}", flush=True)
    except Exception as e:
        print("[sig] cache failed:", e, flush=True)
if rp is None:
    t0 = time.time()
    print("[sig] base (no system prompt) ...", flush=True)
    base = sigset(m0, "")
    print("[sig] pirate ...", flush=True)
    rp = sigset(m0, PIRATE) / torch.clamp(base, min=1e-9)
    torch.save({"rp": rp, "NL": NL, "NI": NI}, SIGCACHE)
    print(f"[sig] done in {time.time()-t0:.0f}s, cached -> {SIGCACHE}", flush=True)

keep = (rp > THR).view(NL, NI)
kept = int(keep.sum())
p2 = (TOT - FFN) + FFN * kept / NOTAL
print(f"\nKEEP-ONLY-PIRATE THR={THR}", flush=True)
print(f"  neurons {kept:,} / {NOTAL:,} = {kept/NOTAL*100:.2f}%", flush=True)
print(f"  params  {p2/1e9:.4f}B = {p2/TOT*100:.2f}% of {TOT/1e9:.4f}B", flush=True)
per_layer = keep.sum(1).tolist()
print(f"  per-layer kept: min {min(per_layer):,} max {max(per_layer):,} of {NI:,}", flush=True)

del m0; torch.cuda.empty_cache(); gc.collect()

RECS = []
m = new_model(); prune_to(m, keep)
for seed in SEEDS:
    torch.manual_seed(seed)
    print(f"\n=== seed {seed} ===", flush=True)
    for cname, fn in CONDS:
        for q in GEN_PROBES:
            p = fn(q)
            out = gen(m, p)
            RECS.append({"seed": seed, "thr": THR, "condition": cname, "probe": q,
                         "params_B": p2/1e9, "pct_model": p2/TOT*100,
                         "ffn_neurons": kept, "pct_ffn": kept/NOTAL*100,
                         "prompt_exact": p, "output": out, "output_chars": len(out)})
            print(f"  {cname:<11} [{q[:26]:<26}] {len(out):>4}c  {out[:58]!r}", flush=True)
del m; torch.cuda.empty_cache(); gc.collect()

os.makedirs("runs", exist_ok=True)
with open(OUTTXT, "w", encoding="utf-8") as f:
    f.write(f"MODEL {MODEL}\nREPRODUCTION of dump_matrix.py THR=0.4\n")
    f.write(f"NGEN={NGEN} temperature=0.85 top_p=0.95  THR={THR}\n")
    f.write(f"seeds: {SEEDS}\n")
    f.write(f"total params {TOT/1e9:.4f}B  total FFN neurons {NOTAL:,}\n")
    f.write(f"KEPT {kept:,} neurons ({kept/NOTAL*100:.2f}%)  ->  {p2/1e9:.4f}B ({p2/TOT*100:.2f}%)\n")
    f.write(f"SIGNATURE: per-neuron mean|act(PIRATE)|/mean|act(default)| at FINAL token,\n")
    f.write(f"           averaged over {len(PROBES_SIG)} probes, FFN down_proj input.\n")
    f.write("="*100 + "\n")
    cur = None
    for r in RECS:
        if r["seed"] != cur:
            cur = r["seed"]
            f.write(f"\n\n{'#'*100}\n# SEED {r['seed']}\n{'#'*100}\n")
        f.write("\n" + "-"*100 + "\n")
        f.write(f"CONDITION    : {r['condition']}\n")
        f.write(f"PROBE        : {r['probe']}\n")
        f.write(f"OUTPUT LENGTH: {r['output_chars']} chars\n")
        f.write(f"<<<<<<<<<< EXACT PROMPT >>>>>>>>>>\n{r['prompt_exact']}\n<<<<<<<<<< END PROMPT >>>>>>>>>>\n")
        f.write(f"<<<<<<<<<< EXACT OUTPUT >>>>>>>>>>\n{r['output']}\n<<<<<<<<<< END OUTPUT >>>>>>>>>>\n")
json.dump({"thr": THR, "kept": kept, "params_B": p2/1e9, "pct_model": p2/TOT*100,
           "per_layer": per_layer, "records": RECS},
          open(OUTJSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print(f"\nTXT  -> {os.path.abspath(OUTTXT)}")
print(f"JSON -> {os.path.abspath(OUTJSON)}")
print(f"records: {len(RECS)}  ({len(SEEDS)} seeds x 3 conditions x 2 probes)")
