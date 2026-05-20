# 🏦 Financial Earnings Call Analyst

An AI-powered system that automatically analyzes quarterly earnings call transcripts using **LLM fine-tuning**, **multi-agent orchestration**, and **prompt engineering**.

## 🎯 What It Does

Feed in a raw earnings call transcript (60+ pages of financial discussion) and get back:
- **Sentiment classification** (positive / negative / neutral)
- **Key financial metrics** (revenue, EPS, margins, growth rates)
- **Forward guidance** extraction
- **Risk factor** identification
- **Professional analyst report** — auto-generated

## 🏗️ Architecture

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
         (Fine-     (Gemini)      (Gemini)
          tuned)
```

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

### Fine-Tuned Sentiment Model
- **Model:** Qwen/Qwen2.5-7B + QLoRA adapter
- **Training data:** ~900 balanced examples (oversampled from Aiera dataset)
- **Test accuracy:** 77.4% on 106 unseen examples
- **Trainable parameters:** 0.55% of total (42M / 7.6B)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Positive | 1.000 | 0.484 | 0.652 |
| Negative | 0.625 | 0.500 | 0.556 |
| Neutral | 0.747 | 0.954 | 0.838 |

### Agent Pipeline
- Processes 56,000+ character transcripts in ~30 seconds
- Extracts 15-20 financial metrics per transcript
- Generates professional analyst reports with real data

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
├── src/                           # Source code
│   ├── config.py                  # API configuration
│   ├── pipeline.py                # LangGraph pipeline definition
│   ├── main.py                    # Entry point
│   └── agents/
│       └── agent_functions.py     # All 4 agent implementations
├── models/                        # Fine-tuned LoRA adapter
├── evaluation/                    # Results and reports
│   ├── evaluation_results.json    # Fine-tuned model metrics
│   ├── sample_report.md           # Generated analyst report
│   └── prompt_experiment.py       # Prompt strategy comparison
└── notebooks/                     # Colab fine-tuning notebook

```

## 🚀 Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/financial-earnings-analyst.git
cd financial-earnings-analyst
python3 -m venv venv
source venv/bin/activate
pip install pandas datasets transformers langchain langgraph langchain-google-genai scikit-learn matplotlib python-dotenv tqdm
```

### 2. Set API key
```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 3. Download data
```bash
python data/download_data.py
python data/preprocess_data.py
```

### 4. Get the fine-tuned model

The fine-tuned LoRA adapter (~154MB) is **not included in this repo** due to GitHub's file size limits. You have two options:

**Option A: Train it yourself (recommended for learning)**
1. Open `notebooks/financial_sentiment_finetuning.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Upload `data/train.jsonl` and `data/val.jsonl` to the Colab session
4. Run all cells — training takes ~45 minutes on a free T4
5. Download the adapter zip and extract to `models/financial-sentiment-adapter/`

**Option B: Skip fine-tuned model**
The agent pipeline (`python -m src.main`) works without the fine-tuned model — it uses the Gemini API for sentiment analysis. The fine-tuned model is used for comparison/evaluation only.

### 5. Run the analyst
```bash
python -m src.main
```

## 🔑 Key Skills Demonstrated

1. **LLM Fine-Tuning (QLoRA/PEFT)** — Fine-tuned a 7B parameter model using 4-bit quantization, training only 0.55% of parameters on a free Google Colab T4 GPU.

2. **Agentic Frameworks (LangGraph)** — Built a multi-agent pipeline with 4 specialized agents orchestrated through a state graph.

3. **Prompt Engineering** — Designed structured prompts for financial metric extraction and report generation with JSON-formatted outputs.

## 📚 Datasets Used

- [Aiera Transcript Sentiment](https://huggingface.co/datasets/Aiera/aiera-transcript-sentiment) — Labeled earnings call sentiment data
- [S&P 500 Earnings Transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) — Full transcript corpus
