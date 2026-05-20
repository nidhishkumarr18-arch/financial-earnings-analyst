# 🏦 Financial Earnings Call Analyst

An AI-powered system that automatically analyzes quarterly earnings call transcripts. This project demonstrates three key AI skills through two integrated components:

1. **LLM Fine-Tuning (QLoRA/PEFT)** — Training a custom sentiment classifier on financial data
2. **Agentic Frameworks (LangGraph)** — Multi-agent pipeline for end-to-end transcript analysis
3. **Prompt Engineering** — Structured prompts for financial metric extraction and report generation

## 🎯 What It Does

Feed in a raw earnings call transcript (60+ pages of financial discussion) and get back:
- **Sentiment classification** (positive / negative / neutral)
- **Key financial metrics** (revenue, EPS, margins, growth rates)
- **Forward guidance** extraction
- **Risk factor** identification
- **Professional analyst report** — auto-generated

## 🏗️ Architecture

This project has **two components** that demonstrate different skills:

### Component 1: Fine-Tuned Sentiment Model (Colab — GPU required)
Trains a custom sentiment classifier using QLoRA on financial data. The model is trained and evaluated on Google Colab (requires T4 GPU). This component demonstrates **LLM fine-tuning, data preprocessing, and model evaluation** skills.

```
Training Data (Aiera Dataset)
        │
        ▼
┌────────────────────────────┐
│  Data Preprocessing        │  Split → Balance → Format (Alpaca template)
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  QLoRA Fine-Tuning (Colab) │  Qwen2.5-7B, 4-bit quantization, LoRA adapters
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  Evaluation                │  77.4% accuracy on 106 test examples
└────────────────────────────┘
```

### Component 2: Multi-Agent Analysis Pipeline (Local — API-based)
A LangGraph pipeline that processes full transcripts using Google Gemini API. Runs locally on any machine — no GPU needed. This component demonstrates **agentic frameworks and prompt engineering** skills.

```
User inputs earnings call transcript
                │
                ▼
┌──────────────────────────────────┐
│     LangGraph Orchestrator       │
└──────────┬───────────────────────┘
           │
     ┌─────┼──────────┬────────────────┐
     ▼     ▼          ▼                ▼
 Ingest  Sentiment   Metric        Report
 Agent   Analyzer   Extractor      Writer
        (Gemini)    (Gemini)      (Gemini)
```

### Why are there two separate components?

The fine-tuned model (7B parameters) requires a GPU to run inference, which isn't available on most local machines. So:
- **Fine-tuning + evaluation** runs on Google Colab (demonstrates QLoRA/PEFT skills)
- **Agent pipeline** runs locally using Gemini API (demonstrates LangGraph + prompt engineering skills)

**Future improvement:** Deploy the fine-tuned model as an API (via HuggingFace Inference Endpoints) and plug it into the LangGraph pipeline to replace the Gemini sentiment agent — giving you a fully custom, end-to-end system.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Fine-tuning | QLoRA / PEFT on Qwen2.5-7B |
| Quantization | BitsAndBytes 4-bit (NF4) |
| Agent Framework | LangGraph (from LangChain) |
| LLM API | Google Gemini 2.0 Flash |
| Training | HuggingFace TRL + SFTTrainer |
| Evaluation | scikit-learn |
| Language | Python 3.x |

## 📊 Results

### Fine-Tuned Sentiment Model (Component 1)
- **Model:** Qwen/Qwen2.5-7B + QLoRA adapter
- **Training data:** ~900 balanced examples (oversampled from Aiera dataset)
- **Test accuracy:** 77.4% on 106 unseen examples
- **Trainable parameters:** 0.55% of total (42M / 7.6B)
- **Training time:** ~45 minutes on a free Google Colab T4 GPU

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Positive | 1.000 | 0.484 | 0.652 |
| Negative | 0.625 | 0.500 | 0.556 |
| Neutral | 0.747 | 0.954 | 0.838 |

### Agent Pipeline (Component 2)
- Processes 56,000+ character transcripts in ~30 seconds
- Extracts 15-20 financial metrics per transcript
- Generates professional analyst reports with actual financial data

## 📁 Project Structure

```
financial-earnings-analyst/
├── data/                          # Dataset scripts and processed data
│   ├── download_data.py           # Download from HuggingFace
│   ├── explore_data.py            # Data exploration
│   ├── preprocess_data.py         # Split, balance, format for training
│   ├── train.jsonl                # Balanced training set
│   ├── val.jsonl                  # Validation set
│   └── test.jsonl                 # Test set
├── src/                           # Source code (agent pipeline)
│   ├── config.py                  # API configuration
│   ├── pipeline.py                # LangGraph pipeline definition
│   ├── main.py                    # Entry point
│   └── agents/
│       └── agent_functions.py     # All 4 agent implementations
├── models/                        # Fine-tuned LoRA adapter (not in repo)
├── evaluation/                    # Results and reports
│   ├── evaluation_results.json    # Fine-tuned model metrics
│   ├── sample_report.md           # Generated analyst report
│   └── prompt_experiment.py       # Prompt strategy comparison
└── notebooks/                     # Colab fine-tuning notebook
```

## 🚀 Quick Start

### Run the Agent Pipeline (no GPU needed)

```bash
# 1. Clone and setup
git clone https://github.com/nidhishkumarr18-arch/financial-earnings-analyst.git
cd financial-earnings-analyst
python3 -m venv venv
source venv/bin/activate
pip install pandas datasets transformers langchain langgraph langchain-google-genai scikit-learn matplotlib python-dotenv tqdm

# 2. Set your Gemini API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Download data
python data/download_data.py
python data/preprocess_data.py

# 4. Run the analyst
python -m src.main
```

### Train the Fine-Tuned Model (GPU required)

The fine-tuned LoRA adapter (~154MB) is **not included in this repo** due to GitHub's file size limits.

1. Open `notebooks/financial_sentiment_finetuning.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Upload `data/train.jsonl` and `data/val.jsonl` to the Colab session
4. Run all cells — training takes ~45 minutes on a free T4
5. Evaluation runs automatically after training (expect ~77% accuracy)

## 🔑 Key Skills Demonstrated

1. **LLM Fine-Tuning (QLoRA/PEFT)** — Fine-tuned a 7B parameter model using 4-bit quantization, training only 0.55% of parameters on a free Google Colab T4 GPU. Handled data preprocessing (stratified splitting, oversampling for class imbalance, Alpaca format conversion) and model evaluation (accuracy, F1, confusion matrix).

2. **Agentic Frameworks (LangGraph)** — Built a multi-agent pipeline with 4 specialized agents (Ingestion, Sentiment Analysis, Metric Extraction, Report Generation) orchestrated through a LangGraph StateGraph with shared typed state.

3. **Prompt Engineering** — Designed structured prompts for financial metric extraction (with JSON output formatting), sentiment classification, and professional report generation. Implemented a prompt comparison framework for zero-shot, few-shot, and chain-of-thought strategies.

## 📚 Datasets Used

- [Aiera Transcript Sentiment](https://huggingface.co/datasets/Aiera/aiera-transcript-sentiment) — Labeled earnings call sentiment data
- [S&P 500 Earnings Transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) — Full transcript corpus
