# comparable-companies-analysis

## Overview
This tool automates the identification of public comparable companies ("comps") for private valuation targets. It utilizes an Agentic Architecture to reason about business models, fetch real-time financial data, and validate peer relevance using semantic analysis.

## Key Features
* **Reasoning Engine:** Uses `gpt-4o` to brainstorm strategic peers based on deep semantic understanding of the target's business description.
* **Semantic Validation:** Implements a **Cosine Similarity check (Threshold > 0.3)** using OpenAI Embeddings to mathematically verify that the candidate's business model aligns with the target.
* **Financial Sizing:** Dynamically fetches **Market Cap** to allow analysts to distinguish between strategic peers (competitors) and financial peers (valuation comps).
* **Robustness:** Includes exponential backoff (via `tenacity`) to handle API rate limits and `yfinance` connectivity issues.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key:
  ```bash
   export OPENAI_API_KEY='your-key-here'
  ```
3. Run the script:
  ```bash
  python main.py
```

## Design Decisions

### Free & Compliance-Friendly Data Sources
`yfinance` is used instead of paid market data APIs. Industry and sector fields are mapped as proxies for SIC-style classification.

### Why Semantic Similarity (Not Rules)
Fashion, luxury, and consumer brands often differ in wording despite similar economics. Embedding-based similarity captures this nuance more effectively than keyword-based filters.

### Threshold Selection (0.30)
The similarity threshold was empirically chosen to:
- Exclude structurally unrelated firms (e.g., retailers or manufacturers without brands)
- Retain differentiated but legitimate peers across lifestyle and luxury segments

### Human-in-the-Loop Philosophy
The system is designed to support analyst judgment, not replace it. Scale outliers and global peers are intentionally retained with transparent context to enable informed decision-making.

---

## Intended Use Cases

- Comparable company screening for valuation
- Peer benchmarking in strategy and corporate finance
- Automated first-pass comps for analyst workflows
- Demonstrations of agentic + deterministic hybrid architectures
