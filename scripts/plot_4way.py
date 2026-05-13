"""4-way comparison: baseline GRPO @500 | control (+300 vanilla) | e2e (300 steps) | e2e_ext (+300 e2e)."""
import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    klp = {
        'baseline':   json.load(open('/tmp/eval_grpo_baseline.json')),
        'control':    json.load(open('/tmp/eval_control_step300.json')),
        'e2e':        json.load(open('/tmp/eval_e2e_step300.json')),
        'e2e_ext':    json.load(open('/tmp/eval_e2e_ext_step300.json')),
    }
    xlayer = {
        'baseline':   json.load(open('/tmp/cross_layer_grpo.json')),
        'control':    json.load(open('/tmp/cross_layer_control300.json')),
        'e2e':        json.load(open('/tmp/cross_layer_e2e.json')),
        'e2e_ext':    json.load(open('/tmp/cross_layer_e2e_ext.json')),
    }
    sd = {
        'baseline':   json.load(open('/tmp/sd_gender_grpo_raw.json')),
        'control':    json.load(open('/tmp/sd_gender_control300.json')),
        'e2e':        json.load(open('/tmp/sd_gender_e2e_raw.json')),
        'e2e_ext':    json.load(open('/tmp/sd_gender_e2e_ext.json')),
    }
    # Factual-accuracy: matched-pair grading available only for control vs e2e
    factual = json.load(open('/tmp/factual_summary.json'))

    names = ['baseline', 'control', 'e2e', 'e2e_ext']
    labels = ['GRPO\n@500', 'control\n(+300 vanilla)', 'e2e\n(300 steps)', 'e2e_ext\n(+300 e2e)']
    colors = ['C3', 'C0', 'C2', 'C8']

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # --- Panel 1: KL@p (the metric that matters) ---
    ax = axes[0, 0]
    vals = [klp[n]['kl_at_p_recon'] for n in names]
    bars = ax.bar(range(4), vals, color=colors, edgecolor='black')
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.3f}', (b.get_x() + b.get_width()/2, v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Annotate Δ vs control
    bar_ctrl = vals[1]
    for i, v in enumerate(vals):
        if i in (2, 3):
            d = (v - bar_ctrl) / bar_ctrl * 100
            ax.annotate(f'{d:+.0f}%\nvs ctrl', (i, v / 2), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('KL@p (nats) — downstream causal fidelity')
    ax.set_title('Downstream KL@p — lower = better')
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 2: MSE / FVE ---
    ax = axes[0, 1]
    vals_fve = []
    for n in names:
        mse = klp[n]['mse']
        # FVE = 1 - MSE/baseline_mse. baseline_mse stored in initial json
        # Actually our eval doesn't save base_mse so compute approx: baseline FVE = 0.667 at MSE=0.000295
        # → base_mse ≈ 0.000295 / (1-0.667) ≈ 0.000886
        base_mse = 0.000886
        vals_fve.append(1.0 - mse / base_mse)
    bars = ax.bar(range(4), vals_fve, color=colors, edgecolor='black')
    for b, v in zip(bars, vals_fve):
        ax.annotate(f'{v:.3f}', (b.get_x() + b.get_width()/2, v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('FVE — geometric reconstruction')
    ax.set_title('FVE — control improves by extra training;\ne2e trades a bit for KL@p')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0.6, 0.75)

    # --- Panel 3: Cross-layer divergence ---
    ax = axes[1, 0]
    hs = xlayer['baseline']['hidden_state_indices']
    for n, c, lab in zip(names, colors, labels):
        ax.plot(hs, xlayer[n]['div_at_p'], 'o-', color=c, lw=2, ms=7, label=lab.replace('\n', ' '))
    ax.set_xlabel('hidden_states[L]')
    ax.set_ylabel('|| h_orig − h_patched || / || h_orig ||  @ position p')
    ax.set_title('Cross-layer residual divergence\nat patched position')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: SelfDescribe gender (kw acc tot + balanced LLM acc) ---
    ax = axes[1, 1]
    kw_acc = [sd[n]['keyword']['acc_total_(None=wrong)'] for n in names]
    llm_bal = []
    for n in names:
        c = sd[n]['llm_grader']['confusion']
        af = c.get('female->female', 0) / 109
        am = c.get('male->male', 0) / 91
        llm_bal.append((af + am) / 2)
    x = np.arange(4)
    w = 0.35
    b1 = ax.bar(x - w/2, kw_acc, w, color='C4', edgecolor='black', label='keyword acc')
    b2 = ax.bar(x + w/2, llm_bal, w, color='C7', edgecolor='black', label='LLM balanced acc')
    for b, v in zip(b1, kw_acc): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    for b, v in zip(b2, llm_bal): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='random')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('accuracy')
    ax.set_title('SelfDescribe gender (raw stereotype, N=200)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 0.6)

    # --- Panel 5: Factual accuracy (matched control vs e2e only — Haiku-graded 48 records) ---
    ax = axes[0, 2]
    metrics_lab = ['claims/expl', 'contradicted/expl', 'frac contr\n(micro)', '(supp − contr)\n/expl']
    c_vals = [factual['control']['claims_per_expl'], factual['control']['contr_per_expl'],
              factual['control']['frac_contr_micro'], factual['control']['net_per_expl']]
    e_vals = [factual['e2e']['claims_per_expl'], factual['e2e']['contr_per_expl'],
              factual['e2e']['frac_contr_micro'], factual['e2e']['net_per_expl']]
    x = np.arange(len(metrics_lab))
    w = 0.35
    bg = ax.bar(x - w/2, c_vals, w, color='C0', edgecolor='black', label='control (+300 vanilla)')
    be = ax.bar(x + w/2, e_vals, w, color='C2', edgecolor='black', label='e2e (300 steps)')
    for b, v in zip(bg, c_vals):
        fmt = f'{v:.3f}' if v < 1 else f'{v:.2f}'
        ax.annotate(fmt, (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    for b, v in zip(be, e_vals):
        fmt = f'{v:.3f}' if v < 1 else f'{v:.2f}'
        ax.annotate(fmt, (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics_lab, fontsize=9)
    ax.set_ylabel('count / fraction')
    ax.set_title('Factual accuracy (Haiku-graded, N=48)\ne2e: ~60% fewer contradicted claims')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 6: hide / use for legend or stats ---
    axes[1, 2].axis('off')
    summary_text = (
        "Matched-training comparison (control vs e2e, both +300 steps):\n\n"
        f"  KL@p:               {klp['control']['kl_at_p_recon']:.3f} → {klp['e2e']['kl_at_p_recon']:.3f}  "
        f"(−{(1 - klp['e2e']['kl_at_p_recon']/klp['control']['kl_at_p_recon'])*100:.0f}%)\n"
        f"  Cross-layer @hs16:  {xlayer['control']['div_at_p'][0]:.3f} → {xlayer['e2e']['div_at_p'][0]:.3f}  "
        f"(−{(1 - xlayer['e2e']['div_at_p'][0]/xlayer['control']['div_at_p'][0])*100:.0f}%)\n"
        f"  SD kw acc:          {sd['control']['keyword']['acc_total_(None=wrong)']:.3f} → "
        f"{sd['e2e']['keyword']['acc_total_(None=wrong)']:.3f}  "
        f"(+{(sd['e2e']['keyword']['acc_total_(None=wrong)'] - sd['control']['keyword']['acc_total_(None=wrong)'])*100:.1f}pp)\n"
        f"  Contradicted/expl:  {factual['control']['contr_per_expl']:.3f} → {factual['e2e']['contr_per_expl']:.3f}  "
        f"(−{(1 - factual['e2e']['contr_per_expl']/factual['control']['contr_per_expl'])*100:.0f}%)\n"
        f"\nFVE:                {1.0 - klp['control']['mse']/0.000886:.3f} → "
        f"{1.0 - klp['e2e']['mse']/0.000886:.3f}  (slight cost)"
    )
    axes[1, 2].text(0.02, 0.95, summary_text, fontsize=10, family='monospace',
                    verticalalignment='top', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Matched-pair summary', fontsize=11)

    fig.suptitle('e2e ablation vs matched-training control (+ extended e2e for trajectory)', fontsize=14, y=1.00)
    fig.tight_layout()
    out = '/tmp/e2e_4way.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'saved → {out}')


if __name__ == '__main__':
    main()
