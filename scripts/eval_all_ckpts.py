"""Run KL@p, cross-layer, SelfDescribe evals on all 4 checkpoints, write a summary.

Endpoints:
  baseline:   rl_small_grpo_cont/{ar,av}_step_500.pt
  control:    rl_small_grpo_control300/{ar,av}_step_300.pt  (matched-training GRPO)
  e2e:        rl_small_grpo_e2e/{ar,av}_step_300.pt
  e2e_ext:    rl_small_grpo_e2e_ext/{ar,av}_step_300.pt    (constant LR extension)
"""
import json
import subprocess
import sys
from pathlib import Path

CKPTS = [
    ("baseline", "rl_small_grpo_cont/ar_step_500.pt", "rl_small_grpo_cont/av_step_500.pt"),
    ("control",  "rl_small_grpo_control300/ar_step_300.pt", "rl_small_grpo_control300/av_step_300.pt"),
    ("e2e",      "rl_small_grpo_e2e/ar_step_300.pt", "rl_small_grpo_e2e/av_step_300.pt"),
    ("e2e_ext",  "rl_small_grpo_e2e_ext/ar_step_300.pt", "rl_small_grpo_e2e_ext/av_step_300.pt"),
]

EVALS = [
    ("klp",       "scripts/eval_downstream_kl.py",       128),
    ("xlayer",    "scripts/eval_cross_layer_div.py",     128),
    ("sd_gender", "scripts/eval_selfdescribe_gender.py", 200),
]


def run_eval(eval_name, script, ar, av, n):
    out = f"/tmp/final_{eval_name}_{name}.json"
    args = [
        "python3", script,
        "--ar-init", f"checkpoints/{ar}",
        "--av-init", f"checkpoints/{av}",
        "--device", "cuda:0",
        "--max-ctx-tokens", "256", "--lm-dtype", "bfloat16",
        "--out", out,
    ]
    if eval_name == "sd_gender":
        args += ["--n", str(n), "--batch-size", "8", "--strip-wiki-request", "--use-raw-prompt"]
    else:
        args += ["--n-records", str(n), "--batch-size", "8"]
    print(f"--- {name}/{eval_name} ---")
    r = subprocess.run(args, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "9", "PYTHONPATH": "."},
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: {r.stderr[-500:]}")
    else:
        print(f"OK → {out}")


for name, ar, av in CKPTS:
    if not Path(f"checkpoints/{ar}").exists():
        print(f"SKIP {name}: checkpoint missing")
        continue
    for eval_name, script, n in EVALS:
        run_eval(eval_name, script, ar, av, n)

print("\n=== SUMMARY ===")
table = []
for name, _, _ in CKPTS:
    row = {"ckpt": name}
    for ev, _, _ in EVALS:
        path = Path(f"/tmp/final_{ev}_{name}.json")
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if ev == "klp":
            row["kl@p"] = d.get("kl_at_p_recon")
            row["mse"] = d.get("mse")
        elif ev == "xlayer":
            row["xlayer@16"] = d.get("div_at_p", [None])[0]
            row["xlayer@24"] = d.get("div_at_p", [None]*9)[-1] if d.get("div_at_p") else None
        elif ev == "sd_gender":
            row["kw_acc"] = d.get("keyword", {}).get("acc_total_(None=wrong)")
            row["llm_acc"] = d.get("llm_grader", {}).get("acc")
    table.append(row)

print(f"{'ckpt':<12} {'KL@p':>7} {'MSE':>9} {'xL16':>7} {'xL24':>7} {'kw_acc':>7} {'llm_acc':>8}")
for r in table:
    vals = [r.get(k) for k in ['kl@p','mse','xlayer@16','xlayer@24','kw_acc','llm_acc']]
    formatted = ' '.join(f"{v:>7.4f}" if isinstance(v, float) else f"{'N/A':>7}" for v in vals)
    print(f"{r['ckpt']:<12} {formatted}")
