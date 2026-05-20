"""
Prompt Engineering Experiment — Compare 3 strategies for sentiment classification.
Tests: zero-shot, few-shot, chain-of-thought against fine-tuned baseline (77.4%).
"""

import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import llm

# ─── Data Loading ─────────────────────────────────────────────────

def load_test_data():
    with open("data/test.jsonl", "r") as f:
        return [json.loads(line) for line in f]

def load_few_shot_examples():
    """Get one example per class from training data for few-shot prompting."""
    found = {}
    with open("data/train.jsonl", "r") as f:
        for line in f:
            ex = json.loads(line)
            label = ex["output"]
            if label not in found:
                found[label] = ex["input"][:300]  # Truncate for prompt length
            if len(found) == 3:
                break
    return found

# ─── Three Prompt Strategies ─────────────────────────────────────

def build_zero_shot_prompt(transcript):
    return f"""Classify the sentiment of this earnings call transcript segment.
Respond with exactly one word: positive, negative, or neutral.

Transcript: {transcript}

Sentiment:"""

def build_few_shot_prompt(transcript, examples):
    ex_text = ""
    for label, text in examples.items():
        ex_text += f"\nTranscript: {text}\nSentiment: {label}\n"
    
    return f"""Classify the sentiment of earnings call transcript segments.
Respond with exactly one word: positive, negative, or neutral.

Here are some examples:
{ex_text}
Now classify this one:

Transcript: {transcript}

Sentiment:"""

def build_cot_prompt(transcript):
    return f"""You are a financial sentiment analyst. Analyze this earnings call 
transcript segment step by step:

1. Identify key words/phrases that indicate sentiment
2. Consider the financial context (growth, decline, guidance)
3. Weigh positive vs negative indicators
4. Make your final classification

Transcript: {transcript}

Think step by step, then on the LAST line write ONLY one word: positive, negative, or neutral."""

# ─── Run Experiment ───────────────────────────────────────────────

def extract_label(response_text):
    """Pull out the sentiment label from model response."""
    text = response_text.strip().lower()
    # Check the last line first (for CoT responses)
    last_line = text.split("\n")[-1].strip()
    for label in ["positive", "negative", "neutral"]:
        if label in last_line:
            return label
    # Fallback: check entire response
    for label in ["positive", "negative", "neutral"]:
        if label in text:
            return label
    return "unknown"

def run_strategy(name, prompt_builder, test_data, few_shot_ex=None):
    """Run one prompting strategy on all test examples."""
    print(f"\n{'─'*50}")
    print(f"🧪 Running: {name}")
    print(f"{'─'*50}")
    
    predictions = []
    actuals = []
    
    for i, example in enumerate(test_data):
        # Build the prompt based on strategy
        if few_shot_ex:
            prompt = prompt_builder(example["input"], few_shot_ex)
        else:
            prompt = prompt_builder(example["input"])
        
        # Call Gemini API
        try:
            response = llm.invoke(prompt)
            predicted = extract_label(response.content)
        except Exception as e:
            print(f"   ⚠️ API error on example {i}: {e}")
            predicted = "neutral"  # Default fallback
            time.sleep(2)  # Wait before retry
        
        actual = example["output"]
        predictions.append(predicted)
        actuals.append(actual)
        
        # Progress update every 20 examples
        if (i + 1) % 20 == 0:
            current_acc = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
            print(f"   Progress: {i+1}/{len(test_data)} — Running accuracy: {current_acc:.1%}")
        
        # Small delay to avoid API rate limits
        time.sleep(4)
    
    # Calculate results
    correct = sum(p == a for p, a in zip(predictions, actuals))
    accuracy = correct / len(actuals)
    
    # Per-class breakdown
    class_results = {}
    for label in ["positive", "negative", "neutral"]:
        total = sum(1 for a in actuals if a == label)
        right = sum(1 for p, a in zip(predictions, actuals) if a == label and p == label)
        class_results[label] = {"correct": right, "total": total, "accuracy": right/total if total > 0 else 0}
    
    print(f"\n   ✅ {name} Results:")
    print(f"      Overall accuracy: {correct}/{len(actuals)} = {accuracy:.1%}")
    for label, res in class_results.items():
        print(f"      {label:>10}: {res['correct']}/{res['total']} = {res['accuracy']:.1%}")
    
    return {
        "strategy": name,
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "total": len(actuals),
        "per_class": class_results,
        "predictions": [{"actual": a, "predicted": p} for a, p in zip(actuals, predictions)],
    }

# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🔬 PROMPT ENGINEERING EXPERIMENT")
    print("=" * 60)
    
    # test_data = load_test_data()
    test_data = load_test_data()[:5]  # Use 30 examples instead of 106

    few_shot_examples = load_few_shot_examples()
    
    # Run all 3 strategies
    results = []
    results.append(run_strategy("Zero-Shot", build_zero_shot_prompt, test_data))
    results.append(run_strategy("Few-Shot (3 examples)", build_few_shot_prompt, test_data, few_shot_examples))
    results.append(run_strategy("Chain-of-Thought", build_cot_prompt, test_data))
    
    # ─── Final Comparison Table ───────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON")
    print("=" * 60)
    
    # Add fine-tuned baseline
    print(f"\n  {'Strategy':<30} {'Accuracy':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Fine-Tuned (QLoRA) baseline':<30} {'77.4%':>10}")
    for r in results:
        print(f"  {r['strategy']:<30} {r['accuracy']:>9.1f}%")
    
    # Determine winner
    all_results = [{"strategy": "Fine-Tuned (QLoRA)", "accuracy": 77.4}] + results
    winner = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n  🏆 Best strategy: {winner['strategy']} ({winner['accuracy']:.1f}%)")
    
    # Save all results
    output_path = "evaluation/prompt_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Results saved to {output_path}")

if __name__ == "__main__":
    main()
