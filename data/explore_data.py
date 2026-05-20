"""
Script to explore and understand our datasets.
Run this AFTER download_data.py has completed.
"""

import pandas as pd
import os

# ─── PART 1: Explore the Sentiment Dataset ────────────────────────

print("=" * 70)
print("📊 EXPLORING THE SENTIMENT DATASET")
print("=" * 70)

# Find all sentiment CSV files we downloaded
sentiment_files = [f for f in os.listdir("data") if f.startswith("sentiment_")]
print(f"\nFound files: {sentiment_files}")

for filename in sentiment_files:
    filepath = os.path.join("data", filename)
    df = pd.read_csv(filepath)
    
    print(f"\n{'─' * 50}")
    print(f"📂 File: {filename}")
    print(f"{'─' * 50}")
    
    # Basic info
    print(f"\n🔢 Number of rows: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Show first 3 rows
    print(f"\n📝 First 3 rows:")
    for i in range(min(3, len(df))):
        print(f"\n  --- Row {i+1} ---")
        for col in df.columns:
            value = str(df.iloc[i][col])
            # Truncate long text for readability
            if len(value) > 200:
                value = value[:200] + "..."
            print(f"  {col}: {value}")
    
    # If there's a label/sentiment column, show distribution
    # Let's check all columns for possible label columns
    print(f"\n📊 Column details:")
    for col in df.columns:
        print(f"\n  Column: '{col}'")
        print(f"    Data type: {df[col].dtype}")
        print(f"    Non-null count: {df[col].notna().sum()}/{len(df)}")
        
        # If it looks like a category column (few unique values), show distribution
        nunique = df[col].nunique()
        if nunique <= 20:  # Likely a categorical column
            print(f"    Unique values ({nunique}):")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"      '{val}': {count} ({pct:.1f}%)")

# ─── PART 2: Explore the Transcripts Dataset ──────────────────────

print("\n\n" + "=" * 70)
print("📊 EXPLORING THE TRANSCRIPTS DATASET")
print("=" * 70)

transcripts_path = os.path.join("data", "transcripts_sample.csv")
if os.path.exists(transcripts_path):
    df_t = pd.read_csv(transcripts_path)
    
    print(f"\n🔢 Number of transcripts: {len(df_t)}")
    print(f"📋 Columns: {list(df_t.columns)}")
    
    # Show column details
    print(f"\n📊 Column details:")
    for col in df_t.columns:
        print(f"\n  Column: '{col}'")
        print(f"    Data type: {df_t[col].dtype}")
        print(f"    Non-null count: {df_t[col].notna().sum()}/{len(df_t)}")
        
        # Show sample value
        sample_val = str(df_t[col].iloc[0])
        if len(sample_val) > 300:
            sample_val = sample_val[:300] + "..."
        print(f"    Sample value: {sample_val}")
    
    # Show one full transcript (first one, truncated)
    print(f"\n{'─' * 50}")
    print("📝 SAMPLE TRANSCRIPT (first one, truncated to 1000 chars):")
    print(f"{'─' * 50}")
    
    # Find the column that contains the actual transcript text
    # It might be called 'text', 'transcript', 'content', etc.
    text_cols = [col for col in df_t.columns if df_t[col].dtype == 'object']
    for col in text_cols:
        sample = str(df_t[col].iloc[0])
        if len(sample) > 500:  # Likely the transcript column
            print(f"\n  Column '{col}' (first 1000 chars):")
            print(f"  {sample[:1000]}")
            print(f"  ... [{len(sample)} total characters]")
            break
else:
    print("❌ transcripts_sample.csv not found. Run download_data.py first.")

print("\n" + "=" * 70)
print("✅ Exploration complete!")
print("=" * 70)
