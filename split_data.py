"""Split dataset.csv into train / val / test (70 / 15 / 15), stratified by LLM family."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv("dataset.csv")

train, temp = train_test_split(df, test_size=0.30, stratify=df["LLM_family"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["LLM_family"], random_state=SEED)

os.makedirs("splits", exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(f"splits/{name}.csv", index=False)
    print(f"{name}: {len(part)} rows  ", dict(part["LLM_family"].value_counts()))
