# comparable-companies-analysis

## Overview
This tool automates the identification of public comparable companies ("comps") for private valuation targets. It utilizes an **Agentic Architecture** to reason about business models, fetch real-time financial data, and validate peer relevance using semantic analysis.

## Key Features
* **Reasoning Engine:** Uses `gpt-4o` to brainstorm strategic peers based on deep semantic understanding of the target's business description.
* **Semantic Validation:** Implements a **Cosine Similarity check (Threshold > 0.3)** using OpenAI Embeddings to mathematically verify that the candidate's business model aligns with the target.
* **Financial Sizing:** Dynamically fetches **Market Cap** to allow analysts to distinguish between strategic peers (competitors) and financial peers (valuation comps).
* **Robustness:** Includes exponential backoff (via `tenacity`) to handle API rate limits and `yfinance` connectivity issues.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
