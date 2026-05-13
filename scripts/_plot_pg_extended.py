"""Helper: re-parse rl_small_pg_cont log, concat with orig PG 0-249, plot
against the finished GRPO extended trajectory.
"""
import json
import subprocess
from pathlib import Path


def merge_with_offset(orig_path, cont_path, offset=250):
    orig = json.load(open(orig_path))
    cont = json.load(open(cont_path))
    merged = []
    for e in orig:
        if "eval" in e:
            if e["eval"]["step"] <= offset - 1:
                merged.append(e)
        elif e.get("step", 0) <= offset - 1:
            merged.append(e)
    for e in cont:
        if "eval" in e:
            ev = dict(e["eval"]); ev["step"] += offset
            merged.append({"eval": ev})
        else:
            e2 = dict(e); e2["step"] = e2.get("step", 0) + offset
            merged.append(e2)
    return merged


def main():
    # Parse PG cont log
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_small_pg_cont.log", "--out", "/tmp/pg_cont.json"],
        check=True, capture_output=True,
    )

    # Merge PG
    pg_merged = merge_with_offset(
        "checkpoints/rl_small_pg/history.json",
        "/tmp/pg_cont.json", offset=250,
    )
    Path("/tmp/pg_full.json").write_text(json.dumps(pg_merged))

    # GRPO merged was already saved
    pg_evals = [e["eval"] for e in pg_merged if "eval" in e]
    pg_train = sum(1 for e in pg_merged if "eval" not in e)

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/pg_full.json:PG(orig+cont)",
         "/tmp/grpo_full.json:GRPO(orig+cont)",
         "--out", "/tmp/pg_vs_grpo_extended.png"],
        check=True, capture_output=True,
    )

    latest = pg_evals[-1] if pg_evals else None
    msg = f"PG merged={pg_train} train, {len(pg_evals)} evals"
    if latest:
        msg += f"; latest PG: @{latest['step']}={latest['fve']:.4f}"
    print(msg + "; plot → /tmp/pg_vs_grpo_extended.png")


if __name__ == "__main__":
    main()
