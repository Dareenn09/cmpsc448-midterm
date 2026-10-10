"""Collect every run in results/ into report-ready tables and figures.

Usage:
    python summarize.py          # writes to results/summary/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

R = "results"
OUT = os.path.join(R, "summary")
SEEDS = [42, 1, 2]
MODES = ["input", "output", "both"]
MODE_LABEL = {"input": "Input only", "output": "Output only", "both": "Input + Output"}
FAMILIES = ["Claude", "GPT", "Gemini", "Qwen"]

# Colorblind-checked categorical colors; text stays in neutral ink
COLOR = {"CNN": "#2a78d6", "LSTM": "#eb6834", "TF-IDF + LR": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})
os.makedirs(OUT, exist_ok=True)


def load(path):
    return json.load(open(path))


def runs(model, mode, tag_fmt="s{}", folder=R):
    return [load(os.path.join(folder, f"{model}_{mode}_{tag_fmt.format(s)}.json")) for s in SEEDS]


def stats(rs, key):
    v = np.array([r[key] for r in rs])
    return v.mean(), v.std(ddof=1)


def fmt(m, s=None):
    return f"{m:.3f}" if s is None else f"{m:.3f} ({s:.3f})"


lines = ["# Results", "", f"Mean (std) over seeds {SEEDS}. Test set: 1,218 examples, 4 classes.", ""]

# ------------------------------------------------------------ 1. main table (RQ1 + RQ2)
base = load(os.path.join(R, "baseline_tfidf_lr.json"))
main = {}
for m in ["cnn", "lstm"]:
    for mode in MODES:
        rs = runs(m, mode)
        main[(m.upper(), mode)] = (stats(rs, "test_accuracy"), stats(rs, "test_macro_f1"))
for mode in MODES:
    main[("TF-IDF + LR", mode)] = ((base[mode]["test_accuracy"], None), (base[mode]["test_macro_f1"], None))

lines += ["Table 1. Test accuracy and macro-F1.", "",
          "| Model | Input | Accuracy | Macro-F1 |", "|---|---|---|---|"]
csv = ["model,mode,acc_mean,acc_std,f1_mean,f1_std"]
for name in ["CNN", "LSTM", "TF-IDF + LR"]:
    for mode in MODES:
        (am, asd), (fm, fsd) = main[(name, mode)]
        lines.append(f"| {name} | {MODE_LABEL[mode]} | {fmt(am, asd)} | {fmt(fm, fsd)} |")
        csv.append(f"{name},{mode},{am:.4f},{'' if asd is None else f'{asd:.4f}'},"
                   f"{fm:.4f},{'' if fsd is None else f'{fsd:.4f}'}")
lines.append("")
open(os.path.join(OUT, "main_results.csv"), "w").write("\n".join(csv) + "\n")

# Figure 1: grouped bars
fig, ax = plt.subplots(figsize=(7, 4))
names = ["CNN", "LSTM", "TF-IDF + LR"]
w, gap = 0.24, 0.02
x = np.arange(len(MODES))
for i, name in enumerate(names):
    means = [main[(name, mo)][0][0] for mo in MODES]
    sds = [main[(name, mo)][0][1] or 0 for mo in MODES]
    pos = x + (i - 1) * (w + gap)
    ax.bar(pos, means, w, color=COLOR[name], label=name)
    ax.errorbar(pos, means, yerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for p, mval, sd in zip(pos, means, sds):
        ax.text(p, mval + sd + 0.012, f"{mval:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.axhline(0.25, color=MUTED, ls="--", lw=1)
ax.text(len(MODES) - 0.5, 0.255, "chance", ha="right", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(x, [MODE_LABEL[m] for m in MODES])
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 0.8)
ax.legend(loc="upper left", ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_accuracy_by_input.png"))
plt.close(fig)

# ------------------------------------------------------------ 2. per-family accuracy (output only)
lines += ["Table 2. Per-class test accuracy (recall), output only: share of each family's responses classified correctly.", "",
          "| Model | " + " | ".join(FAMILIES) + " |", "|---|" + "---|" * len(FAMILIES)]
fam = {}
for m in ["CNN", "LSTM"]:
    rs = runs(m.lower(), "output")
    fam[m] = [(np.mean([r["per_class"][f]["recall"] for r in rs]),
               np.std([r["per_class"][f]["recall"] for r in rs], ddof=1)) for f in FAMILIES]
    lines.append(f"| {m} | " + " | ".join(fmt(a, b) for a, b in fam[m]) + " |")
lines.append("")

fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(FAMILIES))
w = 0.36
for i, m in enumerate(["CNN", "LSTM"]):
    means = [a for a, _ in fam[m]]
    sds = [b for _, b in fam[m]]
    pos = x + (i - 0.5) * (w + gap)
    ax.bar(pos, means, w, color=COLOR[m], label=m)
    ax.errorbar(pos, means, yerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for p, mval, sd in zip(pos, means, sds):
        ax.text(p, mval + sd + 0.012, f"{mval:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, FAMILIES)
ax.set_ylabel("Per-class test accuracy (output only)")
ax.set_ylim(0, 1)
ax.legend(loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_per_family_accuracy.png"))
plt.close(fig)

# ------------------------------------------------------------ 3. confusion matrices (output only)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, m in zip(axes, ["CNN", "LSTM"]):
    cm = sum(np.array(r["confusion_matrix"]) for r in runs(m.lower(), "output"))
    cm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), FAMILIES)
    ax.set_yticks(range(4), FAMILIES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(m, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_confusion_output.png"))
plt.close(fig)

# ------------------------------------------------------------ 4. CNN width sweep
lines += ["Table 3. CNN filter width (output only).", "",
          "| Width | Val F1 | Test acc | Test F1 |", "|---|---|---|---|"]
widths = [2, 3, 4, 5]
sweep = {}
for k in widths:
    rs = runs("cnn", "output", tag_fmt=f"k{k}_s{{}}")
    sweep[k] = {key: stats(rs, key) for key in ["best_val_f1", "test_accuracy", "test_macro_f1"]}
    lines.append(f"| {k} | " + " | ".join(fmt(*sweep[k][key]) for key in
                                          ["best_val_f1", "test_accuracy", "test_macro_f1"]) + " |")
lines.append("")

fig, ax = plt.subplots(figsize=(6, 3.6))
means = [sweep[k]["best_val_f1"][0] for k in widths]
sds = [sweep[k]["best_val_f1"][1] for k in widths]
ax.errorbar(widths, means, yerr=sds, color=COLOR["CNN"], lw=2, marker="o", ms=6,
            ecolor=INK2, elinewidth=1, capsize=3)
for k, mval, sd in zip(widths, means, sds):
    ax.text(k, mval + sd + 0.003, f"{mval:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(widths, widths)
ax.set_xlabel("Filter width")
ax.set_ylabel("Validation macro-F1")
ax.set_ylim(0.58, 0.69)
ax.set_xlim(1.6, 5.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_cnn_width.png"))
plt.close(fig)

# ------------------------------------------------------------ 5. RQ4 ablations (CNN, output only)
ABLATIONS = [("Full response", "s{}"),
             ("Formatting removed", "no_format_s{}"),
             ("Words hidden (structure only)", "structure_only_s{}"),
             ("First 50 tokens only", "first50_s{}")]
lines += ["Table 4. CNN test accuracy when one kind of signal is removed from the response.", "",
          "| Input to CNN | Accuracy | " + " | ".join(FAMILIES) + " |",
          "|---|---|" + "---|" * len(FAMILIES)]
abl = []
for label, tag in ABLATIONS:
    rs = runs("cnn", "output", tag_fmt=tag)
    abl.append((label, *stats(rs, "test_accuracy")))
    per = [np.mean([r["per_class"][f]["recall"] for r in rs]) for f in FAMILIES]
    lines.append(f"| {label} | {fmt(*stats(rs, 'test_accuracy'))} | " +
                 " | ".join(f"{v:.3f}" for v in per) + " |")
lines += ["", "Family columns: per-class accuracy (recall), mean over seeds.", ""]

fig, ax = plt.subplots(figsize=(7, 3))
y = np.arange(len(abl))[::-1]
means = [a[1] for a in abl]
sds = [a[2] for a in abl]
ax.barh(y, means, 0.55, color=COLOR["CNN"])
ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
for yi, mval, sd in zip(y, means, sds):
    ax.text(mval + sd + 0.008, yi, f"{mval:.3f}", va="center", fontsize=8, color=INK2)
ax.axvline(0.25, color=MUTED, ls="--", lw=1)
ax.text(0.255, y[0] + 0.45, "chance", fontsize=8, color=MUTED)
ax.set_yticks(y, [a[0] for a in abl])
ax.set_xlabel("Test accuracy (CNN)")
ax.set_xlim(0, 0.75)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_ablation.png"))
plt.close(fig)

open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
