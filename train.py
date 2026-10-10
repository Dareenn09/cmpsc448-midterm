"""Train and evaluate a CNN or LSTM LLM-family classifier.

Usage:
    python train.py --model cnn  --mode output     # RQ1
    python train.py --model lstm --mode input      # RQ2
    python train.py --model lstm --mode both       # RQ2
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from models import LSTMClassifier, TextCNN

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"

# ---------------------------------------------------------------- arguments
p = argparse.ArgumentParser()
p.add_argument("--model", choices=["cnn", "lstm"], required=True)
p.add_argument("--mode", choices=["input", "output", "both"], default="output")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--patience", type=int, default=4)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max_len", type=int, default=400)
p.add_argument("--max_input_len", type=int, default=100, help="prompt tokens kept in 'both' mode")
p.add_argument("--min_freq", type=int, default=2)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--num_filters", type=int, default=128, help="number of CNN filters")
p.add_argument("--kernel_size", type=int, default=5, help="CNN filter width (words per window)")
p.add_argument("--ablation", choices=["none", "no_format", "structure_only", "first50"], default="none",
               help="RQ4: remove a kind of signal from the response (output mode only)")
p.add_argument("--tag", default="", help="suffix for result file names")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = f"{args.model}_{args.mode}{'_' + args.tag if args.tag else ''}"
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------- tokenizing
def tokenize(text):
    """Words, single punctuation marks, and newlines. Case and markdown symbols
    (#, *, -, `) are kept on purpose: they are part of an LLM's style."""
    text = str(text).replace("\n", " <nl> ")
    return re.findall(r"<nl>|\w+|[^\w\s]", text)


def strip_format(text):
    """Remove markdown formatting and line breaks, keeping only the words and sentence punctuation."""
    text = re.sub(r"```.*?```", " ", str(text), flags=re.S)        # code blocks
    text = re.sub(r"(?m)^\s*(?:[-*•+]|\d+[.)]|#+)\s+", "", text)    # list markers, headers
    text = re.sub(r"[*_`#>|]", "", text)                           # bold, italics, inline code, tables
    return re.sub(r"\s+", " ", text)                               # newlines -> spaces


def ablate(toks):
    """RQ4: remove one kind of signal from the response tokens."""
    if args.ablation == "structure_only":   # hide the words, keep punctuation / layout / length
        return ["<w>" if re.fullmatch(r"\w+", t) else t for t in toks]
    if args.ablation == "first50":          # keep only the opening of the response
        return toks[:50]
    return toks


def build_tokens(row):
    if args.mode == "input":
        return tokenize(row["LLM_Input"])[: args.max_len]
    if args.mode == "output":
        out = strip_format(row["LLM_output"]) if args.ablation == "no_format" else row["LLM_output"]
        return ablate(tokenize(out))[: args.max_len]
    inp = tokenize(row["LLM_Input"])[: args.max_input_len]
    out = tokenize(row["LLM_output"])[: args.max_len - len(inp) - 1]
    return inp + [SEP] + out


train_df = pd.read_csv("splits/train.csv")
val_df = pd.read_csv("splits/val.csv")
test_df = pd.read_csv("splits/test.csv")

classes = sorted(train_df["LLM_family"].unique())
label2id = {c: i for i, c in enumerate(classes)}

train_tok = [build_tokens(r) for _, r in train_df.iterrows()]
val_tok = [build_tokens(r) for _, r in val_df.iterrows()]
test_tok = [build_tokens(r) for _, r in test_df.iterrows()]

# Vocabulary is built from the training split only
counts = Counter(t for toks in train_tok for t in toks)
itos = [PAD, UNK, SEP] + [w for w, c in counts.most_common() if c >= args.min_freq and w != SEP]
stoi = {w: i for i, w in enumerate(itos)}
print(f"[{run_name}] vocab size: {len(itos)}  device: {device}")


class TextDataset(Dataset):
    def __init__(self, toks, labels):
        self.ids = [[stoi.get(t, 1) for t in ts] or [1] for ts in toks]
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids])
    x = torch.zeros(len(ids), max(max(lengths).item(), args.kernel_size), dtype=torch.long)  # >= CNN kernel
    for i, seq in enumerate(ids):
        x[i, : len(seq)] = torch.tensor(seq)
    return x, lengths, torch.tensor(labels)


train_dl = DataLoader(TextDataset(train_tok, train_df["LLM_family"]), batch_size=args.batch_size,
                      shuffle=True, collate_fn=collate)
val_dl = DataLoader(TextDataset(val_tok, val_df["LLM_family"]), batch_size=256, collate_fn=collate)
test_dl = DataLoader(TextDataset(test_tok, test_df["LLM_family"]), batch_size=256, collate_fn=collate)

# ---------------------------------------------------------------- model
if args.model == "cnn":
    model = TextCNN(len(itos), len(classes), num_filters=args.num_filters,
                    kernel_size=args.kernel_size).to(device)
else:
    model = LSTMClassifier(len(itos), len(classes)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[{run_name}] parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def evaluate(dl):
    model.eval()
    preds, golds, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, lengths, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            total_loss += loss_fn(logits, y).item() * len(y)
            preds += logits.argmax(1).tolist()
            golds += y.tolist()
    return total_loss / len(golds), accuracy_score(golds, preds), f1_score(golds, preds, average="macro"), preds, golds


# ---------------------------------------------------------------- training (early stopping on val macro-F1)
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
best_f1, best_state, bad_epochs = -1.0, None, 0
start = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, lengths, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)

    val_loss, val_acc, val_f1, _, _ = evaluate(val_dl)
    history["train_loss"].append(total_loss / n)
    history["train_acc"].append(correct / n)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    print(f"[{run_name}] epoch {epoch:2d}  train_loss {total_loss / n:.4f}  train_acc {correct / n:.4f}  "
          f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  val_f1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1, bad_epochs = val_f1, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"[{run_name}] early stopping at epoch {epoch}")
            break

train_time = time.time() - start

# ---------------------------------------------------------------- test with the best checkpoint
model.load_state_dict(best_state)
_, test_acc, test_f1, preds, golds = evaluate(test_dl)
report = classification_report(golds, preds, target_names=classes, digits=4, output_dict=True)
cm = confusion_matrix(golds, preds)

print(f"\n[{run_name}] TEST accuracy {test_acc:.4f}  macro-F1 {test_f1:.4f}")
print(classification_report(golds, preds, target_names=classes, digits=4))

json.dump({
    "model": args.model, "mode": args.mode, "hyperparameters": vars(args),
    "vocab_size": len(itos), "parameters": n_params, "epochs_run": len(history["val_f1"]),
    "best_val_f1": best_f1, "test_accuracy": test_acc, "test_macro_f1": test_f1,
    "per_class": report, "confusion_matrix": cm.tolist(), "classes": classes,
    "history": history, "train_time_sec": train_time,
}, open(os.path.join("results", f"{run_name}.json"), "w"), indent=2)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"{args.model.upper()} ({args.mode}) — acc {test_acc:.3f}")
fig.tight_layout()
fig.savefig(os.path.join("results", f"{run_name}_confusion.png"), dpi=150)

# Training curves
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
ep = range(1, len(history["train_loss"]) + 1)
a1.plot(ep, history["train_loss"], label="train")
a1.plot(ep, history["val_loss"], label="val")
a1.set_title("Loss")
a1.set_xlabel("epoch")
a1.legend()
a2.plot(ep, history["train_acc"], label="train acc")
a2.plot(ep, history["val_acc"], label="val acc")
a2.plot(ep, history["val_f1"], label="val macro-F1")
a2.set_title("Accuracy / F1")
a2.set_xlabel("epoch")
a2.legend()
fig.suptitle(f"{args.model.upper()} ({args.mode})")
fig.tight_layout()
fig.savefig(os.path.join("results", f"{run_name}_curves.png"), dpi=150)
