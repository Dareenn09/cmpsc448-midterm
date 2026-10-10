# Identifying the LLM Family Behind a Chatbot Response

CMPSC 448 midterm project.

Team members: Zihao Zhou

Report: [report.pdf](report.pdf)

We train a CNN and an RNN (LSTM) to predict which LLM family (Claude, GPT, Gemini or Qwen) wrote a
response from Chatbot Arena. The project covers RQ1 (output only), RQ2 (input only, output only,
input + output) and RQ4 (which characteristics distinguish the families).

## Data

We use the LMSYS Chatbot Arena human preference data (57,477 battles), available from the Kaggle
competition ["LMSYS - Chatbot Arena Human Preference Predictions"](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data)
(`train.csv`) or from Hugging Face as
[`lmsys/lmsys-arena-human-preference-55k`](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k). Download it and save it in the project folder as
`arena_raw.csv`. The file is about 184 MB, so it is not included in this repository.

`prepare_data.py` builds a balanced dataset of 8,116 responses (2,029 per model family).
`split_data.py` makes a stratified 70 / 15 / 15 split into train, validation and test sets.
Their outputs are already included, so you can skip these two steps.

## Setup

Python 3.13 was used. Install the packages with

```
pip install -r requirements.txt
```

## Running the experiments

All commands are run from the project folder.

1. Prepare and split the data (optional, outputs are included):

   ```
   python prepare_data.py
   python split_data.py
   ```

2. TF-IDF + logistic regression baseline:

   ```
   python baseline.py
   ```

3. RQ1 and RQ2: CNN and LSTM on each input type, three seeds (42, 1, 2), 18 runs in total:

   ```
   python train.py --model cnn --mode output --seed 42 --tag s42
   ```

   Repeat for `--model cnn/lstm`, `--mode input/output/both` and `--seed 1/2` (with `--tag s1/s2`).

4. CNN filter width (output only), widths 2 to 5, three seeds:

   ```
   python train.py --model cnn --mode output --kernel_size 3 --seed 42 --tag k3_s42
   ```

5. RQ4 ablations (CNN, output only), three seeds:

   ```
   python train.py --model cnn --mode output --ablation no_format --seed 42 --tag no_format_s42
   python train.py --model cnn --mode output --ablation structure_only --seed 42 --tag structure_only_s42
   python train.py --model cnn --mode output --ablation first50 --seed 42 --tag first50_s42
   ```

6. RQ4 style features and n-gram analysis:

   ```
   python rq4_features.py
   ```

7. Collect all runs into tables and figures:

   ```
   python summarize.py
   ```

Each `train.py` run writes a JSON file with the hyperparameters, test scores, per-class results and
training history, plus a confusion matrix and training curves, to `results/`. The summary tables are
written to `results/summary/summary.md` and `results/summary/rq4_features.md`.

Default settings: 128-dim embeddings, 128 CNN filters of width 5 or 128 LSTM hidden units, dropout
0.5, Adam with learning rate 0.001, batch size 64, at most 400 tokens per example (in input + output
mode, up to 100 prompt tokens), at most 20 epochs with early stopping on validation macro-F1
(patience 4). The test set is evaluated once, with the checkpoint that had the best validation
macro-F1.

## Main results

Test accuracy, mean over three seeds (chance = 0.25):

| Model | Input only | Output only | Input + Output |
|---|---|---|---|
| CNN | 0.281 | 0.654 | 0.655 |
| LSTM | 0.271 | 0.567 | 0.525 |
| TF-IDF + LR | 0.286 | 0.592 | 0.568 |

The full tables, with standard deviations and macro-F1, are in `results/summary/` and the report.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Build the balanced four-family dataset from `arena_raw.csv` |
| `split_data.py` | Stratified train / validation / test split |
| `baseline.py` | TF-IDF + logistic regression baseline |
| `models.py` | CNN and LSTM models |
| `train.py` | Train and evaluate one model |
| `rq4_features.py` | RQ4 style features, feature-group classifiers and top n-grams |
| `summarize.py` | Tables and figures for the report |
| `results/` | Results of every run, and `summary/` with the report tables and figures |
| `dataset.csv`, `splits/` | The balanced dataset and its train / validation / test split |
| `report.pdf` | Project report |
