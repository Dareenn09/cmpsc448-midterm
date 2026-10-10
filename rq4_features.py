"""RQ4: what characteristics distinguish the LLM families?

1. Hand-crafted style features per response (length, lexical diversity, formatting, phrasing),
   summarized per family.
2. A logistic-regression classifier trained on each feature group alone, to see how much each
   kind of signal is worth.
3. The n-grams with the largest TF-IDF + LR weights for each family.

Usage:
    python rq4_features.py       # writes to results/summary/
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = os.path.join("results", "summary")
FAMILIES = ["Claude", "GPT", "Gemini", "Qwen"]
BLUE, INK2, MUTED, GRID = "#2a78d6", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "savefig.dpi": 200,
})
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to GBK


def tokenize(text):
    """Same tokenizer as train.py: words, single punctuation marks, and newlines."""
    return re.findall(r"<nl>|\w+|[^\w\s]", str(text).replace("\n", " <nl> "))


def mattr(words, window=50):
    """Moving-average type-token ratio: lexical diversity that does not depend on length."""
    if len(words) <= window:
        return len(set(words)) / max(len(words), 1)
    return np.mean([len(set(words[i:i + window])) / window for i in range(len(words) - window + 1)])


def features(text):
    text = str(text)
    words = re.findall(r"\w+", text.lower())
    n = max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()]
    lines = text.split("\n")
    first = text.lstrip().lower()
    return {
        # length
        "words": len(words),
        "words_per_sentence": n / max(len(sentences), 1),
        # lexical
        "lexical_diversity": mattr(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
        # formatting
        "newlines_per_100w": 100 * text.count("\n") / n,
        "blank_line_share": text.count("\n\n") / max(text.count("\n"), 1),
        "bullet_lines": sum(bool(re.match(r"\s*[-*•+]\s", l)) for l in lines),
        "numbered_lines": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "bold": len(re.findall(r"\*\*[^*\n]+\*\*", text)),
        "headers": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "code_blocks": text.count("```") // 2,
        # phrasing
        "opens_sure_certainly": float(bool(re.match(r"(sure|certainly|of course|absolutely)\b", first))),
        "apology_or_ai_disclaimer": float(bool(re.search(
            r"i'm sorry|i am sorry|i apologize|as an ai|language model", text.lower()))),
        "exclamations_per_100w": 100 * text.count("!") / n,
        "first_person_per_100w": 100 * len(re.findall(r"\b(i|i'm|i'd|i'll|me|my)\b", text.lower())) / n,
        "has_chinese": float(bool(re.search(r"[一-鿿，。]", text))),
    }


GROUPS = {
    "Length": ["words", "words_per_sentence"],
    "Lexical": ["lexical_diversity", "avg_word_length"],
    "Formatting": ["newlines_per_100w", "blank_line_share", "bullet_lines", "numbered_lines",
                   "bold", "headers", "code_blocks"],
    "Phrasing": ["opens_sure_certainly", "apology_or_ai_disclaimer",
                 "exclamations_per_100w", "first_person_per_100w", "has_chinese"],
}
COUNT_FEATURES = ["words", "bullet_lines", "numbered_lines", "bold", "headers", "code_blocks"]

train = pd.read_csv("splits/train.csv")
test = pd.read_csv("splits/test.csv")
F_train = pd.DataFrame([features(t) for t in train["LLM_output"]])
F_test = pd.DataFrame([features(t) for t in test["LLM_output"]])
for c in COUNT_FEATURES:  # counts are heavy-tailed
    F_train[c], F_test[c] = np.log1p(F_train[c]), np.log1p(F_test[c])

# ------------------------------------------------------------ 1. per-family profile (all data)
data = pd.concat([train, pd.read_csv("splits/val.csv"), test])
F = pd.DataFrame([features(t) for t in data["LLM_output"]])
F["family"] = data["LLM_family"].values
g = F.groupby("family")
profile = pd.DataFrame({
    "Median words": g["words"].median(),
    "Words per sentence": g["words_per_sentence"].median(),
    "Lexical diversity (MATTR)": g["lexical_diversity"].mean(),
    "Newlines per 100 words": g["newlines_per_100w"].median(),
    "% with bullet list": 100 * g["bullet_lines"].apply(lambda x: (x > 0).mean()),
    "% with numbered list": 100 * g["numbered_lines"].apply(lambda x: (x > 0).mean()),
    "% with bold": 100 * g["bold"].apply(lambda x: (x > 0).mean()),
    "% with headers": 100 * g["headers"].apply(lambda x: (x > 0).mean()),
    "% opening Sure/Certainly": 100 * g["opens_sure_certainly"].mean(),
    "% apology / AI disclaimer": 100 * g["apology_or_ai_disclaimer"].mean(),
    "% containing Chinese characters": 100 * g["has_chinese"].mean(),
}).loc[FAMILIES]

lines = ["# RQ4: Characteristics of each LLM family", "",
         f"Table 6. Style features per family (all {len(data):,} responses). Medians for length "
         "and newline counts, means otherwise.", "",
         "| Feature | " + " | ".join(FAMILIES) + " |", "|---|" + "---|" * len(FAMILIES)]
for col in profile.columns:
    fmt = "{:.0f}" if col.startswith("%") or col == "Median words" else "{:.2f}"
    lines.append(f"| {col} | " + " | ".join(fmt.format(v) for v in profile[col]) + " |")
lines.append("")

# Figure 5: one small panel per feature, families on the x-axis
panels = ["Median words", "Lexical diversity (MATTR)", "Newlines per 100 words",
          "% with bullet list", "% with numbered list", "% with bold"]
fig, axes = plt.subplots(2, 3, figsize=(9, 5))
for ax, col in zip(axes.flat, panels):
    vals = profile[col].values
    ax.bar(FAMILIES, vals, 0.6, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}" if col.startswith("%") or col == "Median words" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_title(col, fontsize=9)
    ax.set_ylim(0, vals.max() * 1.18)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_style_features.png"))
plt.close(fig)

# ------------------------------------------------------------ 2. feature-group classifiers
lines += ["Table 7. Test accuracy of logistic regression trained only on hand-crafted features.", "",
          "| Features used | # features | Test accuracy |", "|---|---|---|"]
for name, cols in list(GROUPS.items()) + [("All hand-crafted", sum(GROUPS.values(), []))]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(F_train[cols], train["LLM_family"])
    acc = accuracy_score(test["LLM_family"], clf.predict(F_test[cols]))
    lines.append(f"| {name} | {len(cols)} | {acc:.3f} |")
lines += ["", "Chance = 0.25. For reference, the CNN on the full response reaches 0.654.", ""]

# ------------------------------------------------------------ 3. most indicative n-grams
vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                      ngram_range=(1, 2), min_df=5, max_features=50000)
X = vec.fit_transform(train["LLM_output"])
lr = LogisticRegression(max_iter=2000, C=5.0).fit(X, train["LLM_family"])
vocab = np.array(vec.get_feature_names_out())
lines += ["Table 8. The 12 n-grams with the largest TF-IDF + LR weight for each family "
          "(<nl> = line break).", "", "| Family | Top n-grams |", "|---|---|"]
for i, fam in enumerate(lr.classes_):
    top = vocab[np.argsort(lr.coef_[i])[::-1][:12]]
    lines.append(f"| {fam} | " + ", ".join(f"`{t}`" for t in top) + " |")
lines.append("")

open(os.path.join(OUT, "rq4_features.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
