# Results

Mean (std) over seeds [42, 1, 2]. Test set: 1,218 examples, 4 classes.

Table 1. Test accuracy and macro-F1.

| Model | Input | Accuracy | Macro-F1 |
|---|---|---|---|
| CNN | Input only | 0.281 (0.015) | 0.277 (0.018) |
| CNN | Output only | 0.654 (0.012) | 0.658 (0.013) |
| CNN | Input + Output | 0.655 (0.003) | 0.654 (0.006) |
| LSTM | Input only | 0.271 (0.002) | 0.270 (0.002) |
| LSTM | Output only | 0.567 (0.022) | 0.564 (0.027) |
| LSTM | Input + Output | 0.525 (0.017) | 0.525 (0.025) |
| TF-IDF + LR | Input only | 0.286 | 0.285 |
| TF-IDF + LR | Output only | 0.592 | 0.592 |
| TF-IDF + LR | Input + Output | 0.568 | 0.568 |

Table 2. Per-class test accuracy (recall), output only: share of each family's responses classified correctly.

| Model | Claude | GPT | Gemini | Qwen |
|---|---|---|---|---|
| CNN | 0.823 (0.035) | 0.534 (0.051) | 0.684 (0.027) | 0.576 (0.004) |
| LSTM | 0.684 (0.028) | 0.296 (0.081) | 0.592 (0.028) | 0.694 (0.030) |

Table 3. CNN filter width (output only).

| Width | Val F1 | Test acc | Test F1 |
|---|---|---|---|
| 2 | 0.615 (0.001) | 0.617 (0.024) | 0.620 (0.023) |
| 3 | 0.649 (0.006) | 0.646 (0.017) | 0.650 (0.015) |
| 4 | 0.654 (0.005) | 0.659 (0.009) | 0.661 (0.010) |
| 5 | 0.657 (0.012) | 0.654 (0.012) | 0.658 (0.013) |

Table 4. CNN test accuracy when one kind of signal is removed from the response.

| Input to CNN | Accuracy | Claude | GPT | Gemini | Qwen |
|---|---|---|---|---|---|
| Full response | 0.654 (0.012) | 0.823 | 0.534 | 0.684 | 0.576 |
| Formatting removed | 0.576 (0.023) | 0.800 | 0.423 | 0.591 | 0.490 |
| Words hidden (structure only) | 0.591 (0.011) | 0.742 | 0.480 | 0.565 | 0.576 |
| First 50 tokens only | 0.601 (0.010) | 0.814 | 0.508 | 0.597 | 0.487 |

Family columns: per-class accuracy (recall), mean over seeds.

