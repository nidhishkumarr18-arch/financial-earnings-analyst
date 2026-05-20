"""
Agent functions for the earnings call analysis pipeline.
Each function = one node in the LangGraph pipeline.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import llm


def ingest_transcript(state: dict) -> dict:
    """
    AGENT 1: Parse and chunk the transcript into manageable pieces.
    
    Why chunk? Transcripts are ~56,000 characters (~14K tokens).
    Gemini can handle this, but smaller chunks give better analysis.
    We split by speaker turns for natural boundaries.
    """
    transcript = state["transcript"]
    
    # Split into chunks of ~2000 characters each
    chunk_size = 2000
    chunks = []
    for i in range(0, len(transcript), chunk_size):
        chunk = transcript[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    
    print(f"  📄 Ingested transcript: {len(transcript)} chars → {len(chunks)} chunks")
    
    return {
        **state,
        "chunks": chunks,
        "num_chunks": len(chunks),
    }


def analyze_sentiment(state: dict) -> dict:
    """
    AGENT 2: Analyze sentiment of the transcript using Gemini API.
    
    We send the full transcript (or a summary of chunks) and ask
    for an overall sentiment + per-section breakdown.
    """
    chunks = state["chunks"]
    
    # Take first 10 chunks (enough for sentiment, saves API calls)
    sample_text = "\n---\n".join(chunks[:10])
    
    prompt = f"""You are a financial sentiment analyst. Analyze the following 
earnings call transcript segments and provide:

1. overall_sentiment: exactly one of "positive", "negative", or "neutral"
2. confidence: a number between 0 and 1
3. key_phrases: list of 3-5 phrases that indicate the sentiment
4. reasoning: one sentence explaining your classification

Respond ONLY with valid JSON, no other text.

Transcript segments:
{sample_text}"""
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Clean markdown code fences if present
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rsplit("```", 1)[0]
    
    try:
        sentiment_result = json.loads(response_text)
    except json.JSONDecodeError:
        sentiment_result = {
            "overall_sentiment": "neutral",
            "confidence": 0.5,
            "key_phrases": [],
            "reasoning": "Could not parse model response",
            "raw_response": response_text
        }
    
    print(f"  🎭 Sentiment: {sentiment_result.get('overall_sentiment', 'unknown')} "
          f"(confidence: {sentiment_result.get('confidence', 'N/A')})")
    
    return {
        **state,
        "sentiment_result": sentiment_result,
    }


def extract_metrics(state: dict) -> dict:
    """
    AGENT 3: Extract key financial metrics from the transcript.
    
    Looks for: revenue, profit, growth rates, EPS, guidance, etc.
    """
    chunks = state["chunks"]
    
    # Use more chunks for metrics (numbers can appear anywhere)
    sample_text = "\n---\n".join(chunks[:15])
    
    prompt = f"""You are a financial data extraction specialist. Extract all 
key financial metrics from this earnings call transcript.

For each metric found, provide:
- metric_name: what it is (e.g., "Revenue", "Net Income", "EPS")
- value: the number or percentage mentioned
- period: the time period it refers to (e.g., "Q4 2023", "FY 2023")
- context: brief quote from the transcript

Also provide:
- guidance: any forward-looking statements about next quarter/year
- risks_mentioned: any risks or challenges discussed

Respond ONLY with valid JSON in this format:
{{
    "metrics": [
        {{"metric_name": "...", "value": "...", "period": "...", "context": "..."}}
    ],
    "guidance": "...",
    "risks_mentioned": ["..."]
}}

Transcript:
{sample_text}"""
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rsplit("```", 1)[0]
    
    try:
        metrics_result = json.loads(response_text)
    except json.JSONDecodeError:
        metrics_result = {
            "metrics": [],
            "guidance": "Could not parse",
            "risks_mentioned": [],
            "raw_response": response_text
        }
    
    num_metrics = len(metrics_result.get("metrics", []))
    print(f"  📊 Extracted {num_metrics} financial metrics")
    
    return {
        **state,
        "metrics_result": metrics_result,
    }


def generate_report(state: dict) -> dict:
    """
    AGENT 4: Generate a final analysis report combining all findings.
    """
    company = state.get("company_name", "Unknown Company")
    quarter = state.get("quarter", "N/A")
    year = state.get("year", "N/A")
    sentiment = state.get("sentiment_result", {})
    metrics = state.get("metrics_result", {})
    
    prompt = f"""You are a senior financial analyst. Write a professional 
earnings call analysis report based on these findings:

COMPANY: {company}
PERIOD: Q{quarter} {year}

SENTIMENT ANALYSIS:
{json.dumps(sentiment, indent=2)}

EXTRACTED METRICS:
{json.dumps(metrics, indent=2)}

Write a report with these sections:
1. Executive Summary (2-3 sentences)
2. Key Financial Highlights (bullet points)
3. Sentiment Assessment (with evidence)
4. Forward Outlook (based on guidance)
5. Risk Factors

Keep it professional, concise, and data-driven. Use the actual numbers 
from the metrics. Format in markdown."""
    
    response = llm.invoke(prompt)
    report = response.content.strip()
    
    print(f"  📝 Generated report ({len(report)} characters)")
    
    return {
        **state,
        "report": report,
    }
