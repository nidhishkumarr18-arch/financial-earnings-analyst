"""
Main entry point for the Financial Earnings Call Analyst.
Loads a transcript and runs it through the full agent pipeline.
"""

import sys
import os
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import build_pipeline


def load_sample_transcript():
    """Load one transcript from our downloaded data to test with."""
    
    csv_path = os.path.join("data", "transcripts_sample.csv")
    
    if not os.path.exists(csv_path):
        print("❌ transcripts_sample.csv not found!")
        print("   Run: python data/download_data.py first")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Pick the first transcript
    row = df.iloc[0]
    
    print(f"📂 Loaded transcript:")
    print(f"   Company:  {row['company_name']}")
    print(f"   Quarter:  Q{row['quarter']} {row['year']}")
    print(f"   Date:     {row['date']}")
    print(f"   Length:   {len(str(row['content']))} characters")
    
    return {
        "transcript": str(row["content"]),
        "company_name": str(row["company_name"]),
        "quarter": str(row["quarter"]),
        "year": str(row["year"]),
    }


def main():
    print("=" * 60)
    print("🏦 FINANCIAL EARNINGS CALL ANALYST")
    print("=" * 60)
    
    # Step 1: Load a transcript
    print("\n📥 Loading transcript...")
    input_data = load_sample_transcript()
    if not input_data:
        return
    
    # Step 2: Build the pipeline
    print("\n🔧 Building agent pipeline...")
    pipeline = build_pipeline()
    
    # Step 3: Run the pipeline
    print("\n🚀 Running analysis pipeline...")
    print("   (Each agent will process and pass results to the next)\n")
    
    result = pipeline.invoke(input_data)
    
    # Step 4: Display the report
    print("\n" + "=" * 60)
    print("📋 FINAL ANALYSIS REPORT")
    print("=" * 60)
    print(result["report"])
    
    # Step 5: Save the report
    report_path = os.path.join("evaluation", "sample_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Earnings Analysis: {result['company_name']}\n")
        f.write(f"## Q{result['quarter']} {result['year']}\n\n")
        f.write(result["report"])
    
    print(f"\n💾 Report saved to: {report_path}")
    
    # Also save the raw results
    import json
    raw_path = os.path.join("evaluation", "pipeline_output.json")
    
    # Remove non-serializable items
    saveable = {
        "company_name": result["company_name"],
        "quarter": result["quarter"],
        "year": result["year"],
        "num_chunks": result["num_chunks"],
        "sentiment_result": result["sentiment_result"],
        "metrics_result": result["metrics_result"],
    }
    
    with open(raw_path, "w") as f:
        json.dump(saveable, f, indent=2)
    
    print(f"💾 Raw results saved to: {raw_path}")


if __name__ == "__main__":
    main()
