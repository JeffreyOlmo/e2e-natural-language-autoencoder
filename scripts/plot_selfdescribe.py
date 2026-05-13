"""Plot SelfDescribe gender eval results: GRPO vs e2e."""
import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    g_raw = json.load(open('/tmp/sd_gender_grpo_raw.json'))
    e_raw = json.load(open('/tmp/sd_gender_e2e_raw.json'))
    g_ct = json.load(open('/tmp/sd_gender_grpo.json'))
    e_ct = json.load(open('/tmp/sd_gender_e2e.json'))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Total accuracy summary (both setups) ---
    ax = axes[0]
    metrics = ['kw cov', 'kw acc\n|signal', 'kw acc\ntotal', 'LLM acc']
    g_vals = [g_raw['keyword']['coverage'], g_raw['keyword']['acc_given_signal'],
              g_raw['keyword']['acc_total_(None=wrong)'], g_raw['llm_grader']['acc']]
    e_vals = [e_raw['keyword']['coverage'], e_raw['keyword']['acc_given_signal'],
              e_raw['keyword']['acc_total_(None=wrong)'], e_raw['llm_grader']['acc']]
    x = np.arange(len(metrics))
    w = 0.35
    bg = ax.bar(x - w/2, g_vals, w, color='C3', edgecolor='black', label='vanilla GRPO')
    be = ax.bar(x + w/2, e_vals, w, color='C2', edgecolor='black', label='e2e step-300')
    for b, v in zip(bg, g_vals): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    for b, v in zip(be, e_vals): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    ax.axhline(g_raw['majority_baseline'], color='gray', ls='--', alpha=0.6, label=f"majority={g_raw['majority_baseline']:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('score')
    ax.set_title('SelfDescribe gender (raw stereotype, 2-class)\nN=200 (109 F + 91 M)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # --- Panel 2: Per-class LLM accuracy ---
    ax = axes[1]
    classes = ['female\n(N=109)', 'male\n(N=91)', 'balanced acc']
    def cls_acc(conf, gt, total):
        return conf.get(f'{gt}->{gt}', 0) / total
    g_f = cls_acc(g_raw['llm_grader']['confusion'], 'female', 109)
    g_m = cls_acc(g_raw['llm_grader']['confusion'], 'male', 91)
    g_b = (g_f + g_m) / 2
    e_f = cls_acc(e_raw['llm_grader']['confusion'], 'female', 109)
    e_m = cls_acc(e_raw['llm_grader']['confusion'], 'male', 91)
    e_b = (e_f + e_m) / 2
    g_vals = [g_f, g_m, g_b]; e_vals = [e_f, e_m, e_b]
    x = np.arange(len(classes))
    bg = ax.bar(x - w/2, g_vals, w, color='C3', edgecolor='black', label='vanilla GRPO')
    be = ax.bar(x + w/2, e_vals, w, color='C2', edgecolor='black', label='e2e step-300')
    for b, v in zip(bg, g_vals): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    for b, v in zip(be, e_vals): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.6, label='random (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('LLM grader accuracy')
    ax.set_title('Per-class breakdown (LLM grader)\ne2e helps female ↑, slight male ↓ (bias)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # --- Panel 3: Chat-template (null) vs raw-prompt (signal) ---
    ax = axes[2]
    setups = ['chat-template\n+ Wiki req\n(OOD for AV)', 'raw stereotype\n(matches training\ndistribution)']
    g_acc_total = [g_ct['keyword']['acc_total_(None=wrong)'], g_raw['keyword']['acc_total_(None=wrong)']]
    e_acc_total = [e_ct['keyword']['acc_total_(None=wrong)'], e_raw['keyword']['acc_total_(None=wrong)']]
    x = np.arange(len(setups))
    bg = ax.bar(x - w/2, g_acc_total, w, color='C3', edgecolor='black', label='vanilla GRPO')
    be = ax.bar(x + w/2, e_acc_total, w, color='C2', edgecolor='black', label='e2e step-300')
    for b, v in zip(bg, g_acc_total): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    for b, v in zip(be, e_acc_total): ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(setups, fontsize=9)
    ax.set_ylabel('keyword acc total')
    ax.set_title('Eval setup matters\nchat-template inputs are OOD for FineWeb-trained AV')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 0.6)

    fig.suptitle('User-modeling eval (NLA paper) — adapted SelfDescribe gender on 0.5B', fontsize=13, y=1.02)
    fig.tight_layout()
    out = '/tmp/selfdescribe_gender.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'saved → {out}')


if __name__ == '__main__':
    main()
