"""
LangGraph pipeline that orchestrates all agents.

The flow:
  START → ingest_transcript → analyze_sentiment → extract_metrics → generate_report → END

This is a sequential pipeline. LangGraph also supports:
- Conditional edges (if sentiment is negative, add extra analysis)
- Cycles (loop back if quality is low)
We keep it sequential for now and add conditions in Step 8.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any

# Import our agent functions
from src.agents.agent_functions import (
    ingest_transcript,
    analyze_sentiment,
    extract_metrics,
    generate_report,
)


# ─── Define the State ─────────────────────────────────────────────
# This is the "shared memory" that flows through the pipeline.
# Each agent reads from it and writes back to it.

class AnalysisState(TypedDict, total=False):
    # Input fields (provided by the user)
    transcript: str
    company_name: str
    quarter: str
    year: str
    
    # Fields populated by agents
    chunks: list
    num_chunks: int
    sentiment_result: dict
    metrics_result: dict
    report: str


# ─── Build the Graph ──────────────────────────────────────────────

def build_pipeline():
    """
    Creates and compiles the LangGraph pipeline.
    
    Think of it like an assembly line:
    Raw transcript → Chunked → Sentiment analyzed → Metrics extracted → Report generated
    """
    
    # Create the graph with our state schema
    graph = StateGraph(AnalysisState)
    
    # Add nodes (each node = one agent function)
    graph.add_node("ingest", ingest_transcript)
    graph.add_node("sentiment", analyze_sentiment)
    graph.add_node("metrics", extract_metrics)
    graph.add_node("report", generate_report)
    
    # Add edges (define the flow between nodes)
    graph.add_edge(START, "ingest")        # Start → Ingest
    graph.add_edge("ingest", "sentiment")  # Ingest → Sentiment
    graph.add_edge("sentiment", "metrics") # Sentiment → Metrics
    graph.add_edge("metrics", "report")    # Metrics → Report
    graph.add_edge("report", END)          # Report → End
    
    # Compile the graph into a runnable pipeline
    pipeline = graph.compile()
    
    print("✅ LangGraph pipeline compiled!")
    return pipeline
