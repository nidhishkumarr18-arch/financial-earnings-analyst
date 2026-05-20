"""
Script to download the financial earnings datasets from HuggingFace.
We download two datasets:
1. Sentiment dataset - for fine-tuning (small, labeled data)
2. Transcripts dataset - for the agent pipeline (full transcripts)
"""

from datasets import load_dataset
import pandas as pd
import os

# ─── Step 1: Download the Sentiment Dataset ───────────────────────
# This dataset has earnings call segments labeled as positive/negative/neutral
# We'll use this to fine-tune our model

print("=" * 60)
print("Downloading Sentiment Dataset...")
print("This may take a few minutes on first run.")
print("=" * 60)

sentiment_dataset = load_dataset("Aiera/aiera-transcript-sentiment")

# Let's see what's inside
print("\n📊 Sentiment Dataset Structure:")
print(sentiment_dataset)

# Convert to pandas DataFrame for easier exploration
# The dataset might have train/test splits - let's check
for split_name in sentiment_dataset:
    print(f"\n📂 Split: '{split_name}' — {len(sentiment_dataset[split_name])} rows")
    
    # Save each split as a CSV file
    df = sentiment_dataset[split_name].to_pandas()
    filename = f"sentiment_{split_name}.csv"
    df.to_csv(os.path.join("data", filename), index=False)
    print(f"   ✅ Saved to data/{filename}")

# ─── Step 2: Download a Sample of the Transcripts Dataset ─────────
# This dataset is HUGE (33K+ transcripts), so we only download a small sample
# We'll use this for testing our agent pipeline

print("\n" + "=" * 60)
print("Downloading Transcripts Dataset (sample)...")
print("=" * 60)

# stream=True means we don't download the entire dataset at once
# We'll just grab the first 100 transcripts to work with
transcripts_dataset = load_dataset(
    "Bose345/sp500_earnings_transcripts",
    split="train",
    streaming=True  # Downloads on-the-fly, doesn't eat up disk space
)

# Collect first 100 transcripts
transcripts_list = []
count = 0
for item in transcripts_dataset:
    transcripts_list.append(item)
    count += 1
    if count >= 100:
        break
    if count % 10 == 0:
        print(f"   Downloaded {count}/100 transcripts...")

# Convert to DataFrame and save
transcripts_df = pd.DataFrame(transcripts_list)
transcripts_df.to_csv(os.path.join("data", "transcripts_sample.csv"), index=False)
print(f"   ✅ Saved 100 transcripts to data/transcripts_sample.csv")

print("\n" + "=" * 60)
print("✅ All downloads complete!")
print("=" * 60)
