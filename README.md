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
* **Data Sources:** Used `yfinance` to comply with the "no paid APIs" constraint. Mapped `Sector` and Industry fields to approximate SIC classifications.
* **Validation:** Handled ticker collisions (e.g., TechnipFMC vs. FTI Consulting) by strictly enforcing the semantic similarity score between business descriptions.
