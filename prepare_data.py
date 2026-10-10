import json
import pandas as pd

df = pd.read_csv("arena_raw.csv")

def first_turn(x):
    try:
        lst = json.loads(x)
        return lst[0] if lst and lst[0] else None
    except Exception:
        return None

def family(name):
    n = name.lower()
    for key, fam in [("gpt", "GPT"), ("claude", "Claude"), ("llama", "Llama"),
                     ("mixtral", "Mistral"), ("mistral", "Mistral"),
                     ("gemini", "Gemini"), ("qwen", "Qwen"), ("vicuna", "Vicuna"),
                     ("yi-", "Yi"), ("zephyr", "Zephyr")]:
        if key in n:
            return fam
    return "Other"

rows = []
for _, r in df.iterrows():
    prompt = first_turn(r["prompt"])
    for side in ["a", "b"]:
        rows.append({
            "LLM_name": r[f"model_{side}"],
            "LLM_family": family(r[f"model_{side}"]),
            "LLM_Input": prompt,
            "LLM_output": first_turn(r[f"response_{side}"]),
        })

data = pd.DataFrame(rows).dropna()
print("Examples per family:")
print(data["LLM_family"].value_counts())

# Keep the biggest families and balance them (same number from each)
keep = ["GPT", "Claude", "Qwen", "Gemini"]
data = data[data["LLM_family"].isin(keep)]
n = min(5000, data["LLM_family"].value_counts().min())
balanced = data.groupby("LLM_family").sample(n=n, random_state=42)

balanced.to_csv("dataset.csv", index=False)
print(f"\nSaved {len(balanced)} rows ({n} per family) to dataset.csv")
