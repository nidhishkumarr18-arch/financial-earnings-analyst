"""
Preprocess the sentiment dataset for fine-tuning.
1. Split into train/val/test
2. Oversample minority classes in training set
3. Format into Alpaca instruction format
4. Save as JSONL files (one JSON object per line)
"""

import pandas as pd
import json
import os
import random

# Set random seed for reproducibility
# (so you get the same split every time you run this)
random.seed(42)

# ─── Step 1: Load the raw data ────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df = pd.read_csv("data/sentiment_test.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nLabel distribution:")
print(df["sentiment"].value_counts())

# ─── Step 2: Stratified Split (70% train, 15% val, 15% test) ─────

print("\n" + "=" * 60)
print("STEP 2: Splitting into train / validation / test")
print("=" * 60)

# "Stratified" means each split keeps the same ratio of labels
# Without this, all 66 negative examples might end up in one split

# Shuffle the data first
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split by each sentiment class to maintain proportions
train_dfs = []
val_dfs = []
test_dfs = []

for sentiment_label in ["neutral", "positive", "negative"]:
    # Get all rows with this label
    subset = df[df["sentiment"] == sentiment_label]
    n = len(subset)
    
    # Calculate split points
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Split
    train_part = subset.iloc[:train_end]
    val_part = subset.iloc[train_end:val_end]
    test_part = subset.iloc[val_end:]
    
    train_dfs.append(train_part)
    val_dfs.append(val_part)
    test_dfs.append(test_part)
    
    print(f"  {sentiment_label}: {len(train_part)} train, {len(val_part)} val, {len(test_part)} test")

# Combine all classes back together
train_df = pd.concat(train_dfs, ignore_index=True)
val_df = pd.concat(val_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\n  Total: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

# ─── Step 3: Oversample minority classes in training set ──────────

print("\n" + "=" * 60)
print("STEP 3: Balancing the training data (oversampling)")
print("=" * 60)

print(f"\nBefore balancing:")
print(train_df["sentiment"].value_counts())

# Find the majority class count (neutral will be the largest)
max_count = train_df["sentiment"].value_counts().max()
print(f"\nTarget count for each class: {max_count}")

# Oversample: duplicate rows from minority classes until they match the majority
balanced_dfs = []
for sentiment_label in ["neutral", "positive", "negative"]:
    subset = train_df[train_df["sentiment"] == sentiment_label]
    current_count = len(subset)
    
    if current_count < max_count:
        # How many extra copies do we need?
        extra_needed = max_count - current_count
        
        # Randomly sample (with replacement) from existing rows
        extra_rows = subset.sample(n=extra_needed, replace=True, random_state=42)
        
        # Combine original + extra copies
        balanced_subset = pd.concat([subset, extra_rows], ignore_index=True)
    else:
        balanced_subset = subset
    
    balanced_dfs.append(balanced_subset)
    print(f"  {sentiment_label}: {current_count} → {len(balanced_subset)}")

train_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle the balanced training data
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balancing:")
print(train_balanced["sentiment"].value_counts())
print(f"Total training examples: {len(train_balanced)}")

# ─── Step 4: Format into Alpaca instruction template ──────────────

print("\n" + "=" * 60)
print("STEP 4: Formatting into instruction template")
print("=" * 60)

INSTRUCTION = (
    "You are a financial sentiment analyst. "
    "Classify the sentiment of the following earnings call transcript segment. "
    "Respond with exactly one word: positive, negative, or neutral."
)

def format_example(row):
    """
    Convert one row of data into the Alpaca instruction format.
    This is what the LLM will see during training.
    """
    formatted = {
        "instruction": INSTRUCTION,
        "input": row["transcript"],
        "output": row["sentiment"],
        # The 'text' field is the complete formatted string the model trains on
        "text": (
            f"### Instruction:\n{INSTRUCTION}\n\n"
            f"### Input:\n{row['transcript']}\n\n"
            f"### Response:\n{row['sentiment']}"
        )
    }
    return formatted

def save_as_jsonl(df, filename):
    """
    Save DataFrame as JSONL (JSON Lines) format.
    Each line is one JSON object = one training example.
    """
    filepath = os.path.join("data", filename)
    examples = []
    for _, row in df.iterrows():
        examples.append(format_example(row))
    
    with open(filepath, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"  ✅ Saved {len(examples)} examples to data/{filename}")
    return examples

# Format and save each split
print("\nFormatting and saving...")
train_examples = save_as_jsonl(train_balanced, "train.jsonl")
val_examples = save_as_jsonl(val_df, "val.jsonl")
test_examples = save_as_jsonl(test_df, "test.jsonl")

# ─── Step 5: Show a sample of what the formatted data looks like ──

print("\n" + "=" * 60)
print("STEP 5: Sample formatted example")
print("=" * 60)

print("\nThis is what ONE training example looks like to the model:\n")
print("─" * 50)
print(train_examples[0]["text"])
print("─" * 50)

# ─── Step 6: Final summary ────────────────────────────────────────

print("\n" + "=" * 60)
print("✅ PREPROCESSING COMPLETE — Summary")
print("=" * 60)
print(f"""
Files created:
  📄 data/train.jsonl  — {len(train_examples)} examples (balanced, for fine-tuning)
  📄 data/val.jsonl    — {len(val_examples)} examples (for validation during training)
  📄 data/test.jsonl   — {len(test_examples)} examples (for final evaluation)

Label distribution in train set:
  neutral:  ~{len(train_balanced[train_balanced['sentiment']=='neutral'])} examples
  positive: ~{len(train_balanced[train_balanced['sentiment']=='positive'])} examples
  negative: ~{len(train_balanced[train_balanced['sentiment']=='negative'])} examples

Next step: Upload train.jsonl and val.jsonl to Google Colab for fine-tuning.
""")
