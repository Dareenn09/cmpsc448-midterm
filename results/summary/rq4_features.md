# RQ4: Characteristics of each LLM family

Table 6. Style features per family (all 8,116 responses). Medians for length and newline counts, means otherwise.

| Feature | Claude | GPT | Gemini | Qwen |
|---|---|---|---|---|
| Median words | 165 | 145 | 185 | 147 |
| Words per sentence | 16.29 | 17.33 | 14.67 | 19.45 |
| Lexical diversity (MATTR) | 0.80 | 0.79 | 0.78 | 0.77 |
| Newlines per 100 words | 6.09 | 3.93 | 5.58 | 2.64 |
| % with bullet list | 41 | 10 | 31 | 7 |
| % with numbered list | 16 | 30 | 31 | 26 |
| % with bold | 0 | 10 | 35 | 6 |
| % with headers | 3 | 4 | 4 | 3 |
| % opening Sure/Certainly | 0 | 8 | 1 | 6 |
| % apology / AI disclaimer | 10 | 8 | 3 | 10 |
| % containing Chinese characters | 1 | 1 | 1 | 8 |

Table 7. Test accuracy of logistic regression trained only on hand-crafted features.

| Features used | # features | Test accuracy |
|---|---|---|
| Length | 2 | 0.263 |
| Lexical | 2 | 0.295 |
| Formatting | 7 | 0.443 |
| Phrasing | 5 | 0.319 |
| All hand-crafted | 16 | 0.522 |

Chance = 0.25. For reference, the CNN on the full response reaches 0.654.

Table 8. The 12 n-grams with the largest TF-IDF + LR weight for each family (<nl> = line break).

| Family | Top n-grams |
|---|---|
| Claude | `<nl> <nl>`, `Here is`, `Here`, `So`, `I do`, `<nl> •`, `•`, `<nl> So`, `So in`, `summary`, `: <nl>`, `!` |
| GPT | `* *`, `- *`, `# #`, `various`, `assist`, `️`, `might`, `AI ,`, `! Here`, `It '`, `Certainly`, `1995` |
| Gemini | `. <nl>`, `: *`, `<nl> <nl>`, `* *`, `<nl> -`, `` ``, `, a`, `. *`, `$ $`, `be used`, `it is`, `*` |
| Qwen | `，`, `* :`, `, and`, `a`, `:`, `" The`, `it`, `, which`, `it '`, `is :`, `? <nl>`, `I help` |

