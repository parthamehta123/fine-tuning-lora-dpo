"""Prepare training data for SFT and DPO fine-tuning.

Supports two tasks:
1. Function-calling: Model selects the right function and populates correct parameters
2. Financial entity extraction: Structured JSON extraction from messy financial text

Data sources:
- Glaive function-calling dataset (HuggingFace: glaiveai/glaive-function-calling-v2)
- Custom financial NER examples
"""

import json
from pathlib import Path

# ── Function-Calling Training Examples (Glaive-style) ──

FUNCTION_DEFINITIONS = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
                },
                "exchange": {
                    "type": "string",
                    "enum": ["NYSE", "NASDAQ", "LSE"],
                    "description": "Stock exchange",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financial_report",
        "description": "Retrieve a company's financial report for a specific period",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name or ticker"},
                "report_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q", "8-K", "earnings"],
                    "description": "Type of report",
                },
                "year": {"type": "integer", "description": "Fiscal year"},
                "quarter": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4],
                    "description": "Fiscal quarter (for quarterly reports)",
                },
            },
            "required": ["company", "report_type", "year"],
        },
    },
    {
        "name": "calculate_ratio",
        "description": "Calculate a financial ratio given the input values",
        "parameters": {
            "type": "object",
            "properties": {
                "ratio_type": {
                    "type": "string",
                    "enum": [
                        "pe_ratio",
                        "debt_to_equity",
                        "current_ratio",
                        "roe",
                        "roa",
                        "profit_margin",
                    ],
                    "description": "Type of financial ratio to calculate",
                },
                "numerator": {"type": "number", "description": "Numerator value"},
                "denominator": {"type": "number", "description": "Denominator value"},
            },
            "required": ["ratio_type", "numerator", "denominator"],
        },
    },
    {
        "name": "search_sec_filings",
        "description": "Search SEC EDGAR for company filings",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "company_cik": {"type": "string", "description": "Company CIK number"},
                "filing_type": {
                    "type": "string",
                    "description": "Filing type filter (e.g., 10-K, 10-Q)",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD, EUR)",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]


def create_function_calling_sft_examples() -> list[dict]:
    """Create SFT training data for function-calling task."""
    examples = [
        # ── get_stock_price ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "What's the current price of Apple stock?",
            "output": json.dumps(
                {
                    "function": "get_stock_price",
                    "arguments": {"ticker": "AAPL", "exchange": "NASDAQ"},
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Check Microsoft's stock price on NASDAQ",
            "output": json.dumps(
                {
                    "function": "get_stock_price",
                    "arguments": {"ticker": "MSFT", "exchange": "NASDAQ"},
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "How much is Tesla trading for right now?",
            "output": json.dumps(
                {
                    "function": "get_stock_price",
                    "arguments": {"ticker": "TSLA", "exchange": "NASDAQ"},
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Get me the stock price for JPMorgan on NYSE",
            "output": json.dumps(
                {
                    "function": "get_stock_price",
                    "arguments": {"ticker": "JPM", "exchange": "NYSE"},
                }
            ),
        },
        # ── get_financial_report ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Pull up Apple's annual report for 2024",
            "output": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "AAPL",
                        "report_type": "10-K",
                        "year": 2024,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "I need Tesla's Q3 2024 quarterly filing",
            "output": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "TSLA",
                        "report_type": "10-Q",
                        "year": 2024,
                        "quarter": 3,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Show me Goldman Sachs' latest earnings report for fiscal year 2025",
            "output": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "GS",
                        "report_type": "earnings",
                        "year": 2025,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Get the 8-K filing for Microsoft from 2025",
            "output": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "MSFT",
                        "report_type": "8-K",
                        "year": 2025,
                    },
                }
            ),
        },
        # ── calculate_ratio ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Calculate the P/E ratio with earnings of $6.50 per share and stock price of $195",
            "output": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "pe_ratio",
                        "numerator": 195,
                        "denominator": 6.50,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "What's the debt-to-equity ratio if total debt is $120B and equity is $80B?",
            "output": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "debt_to_equity",
                        "numerator": 120,
                        "denominator": 80,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Compute the current ratio with current assets of $150M and current liabilities of $95M",
            "output": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "current_ratio",
                        "numerator": 150,
                        "denominator": 95,
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Find the return on equity given net income of $25B and shareholder equity of $200B",
            "output": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "roe",
                        "numerator": 25,
                        "denominator": 200,
                    },
                }
            ),
        },
        # ── search_sec_filings ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Search for Tesla's risk factor disclosures in their 10-K filings",
            "output": json.dumps(
                {
                    "function": "search_sec_filings",
                    "arguments": {
                        "query": "risk factors",
                        "company_cik": "0001318605",
                        "filing_type": "10-K",
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Find all SEC filings mentioning AI investments from 2024",
            "output": json.dumps(
                {
                    "function": "search_sec_filings",
                    "arguments": {
                        "query": "artificial intelligence investments",
                        "date_from": "2024-01-01",
                        "date_to": "2024-12-31",
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Look up JPMorgan's cybersecurity disclosures in their annual filings",
            "output": json.dumps(
                {
                    "function": "search_sec_filings",
                    "arguments": {
                        "query": "cybersecurity",
                        "company_cik": "0000019617",
                        "filing_type": "10-K",
                    },
                }
            ),
        },
        # ── convert_currency ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Convert 50000 USD to EUR",
            "output": json.dumps(
                {
                    "function": "convert_currency",
                    "arguments": {
                        "amount": 50000,
                        "from_currency": "USD",
                        "to_currency": "EUR",
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "How much is 1 million Japanese yen in US dollars?",
            "output": json.dumps(
                {
                    "function": "convert_currency",
                    "arguments": {
                        "amount": 1000000,
                        "from_currency": "JPY",
                        "to_currency": "USD",
                    },
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Convert 250000 GBP to USD",
            "output": json.dumps(
                {
                    "function": "convert_currency",
                    "arguments": {
                        "amount": 250000,
                        "from_currency": "GBP",
                        "to_currency": "USD",
                    },
                }
            ),
        },
        # ── Refusal examples (no matching function) ──
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Send an email to the CFO about the quarterly results",
            "output": json.dumps(
                {
                    "function": "NONE",
                    "reason": "No available function can send emails. Available functions are limited to stock prices, financial reports, ratio calculations, SEC filing searches, and currency conversion.",
                }
            ),
        },
        {
            "instruction": "You have access to the following functions. Call the appropriate function based on the user's request.",
            "input": "Schedule a meeting with the investment team for next Tuesday",
            "output": json.dumps(
                {
                    "function": "NONE",
                    "reason": "No available function can schedule meetings. Available functions are limited to financial data retrieval and calculations.",
                }
            ),
        },
    ]
    return examples


def create_function_calling_dpo_examples() -> list[dict]:
    """Create DPO preference pairs for function-calling (chosen vs rejected outputs)."""
    examples = [
        {
            "prompt": "You have access to financial tools. User says: 'What is Apple's stock price?'",
            "chosen": json.dumps(
                {
                    "function": "get_stock_price",
                    "arguments": {"ticker": "AAPL", "exchange": "NASDAQ"},
                }
            ),
            "rejected": json.dumps(
                {"function": "get_stock_price", "arguments": {"ticker": "Apple"}}
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'Get Tesla's 10-K for last year'",
            "chosen": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "TSLA",
                        "report_type": "10-K",
                        "year": 2025,
                    },
                }
            ),
            "rejected": json.dumps(
                {"function": "search_sec_filings", "arguments": {"query": "Tesla 10-K"}}
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'Calculate profit margin: revenue $50B, net income $12B'",
            "chosen": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "profit_margin",
                        "numerator": 12,
                        "denominator": 50,
                    },
                }
            ),
            "rejected": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "profit_margin",
                        "numerator": 50,
                        "denominator": 12,
                    },
                }
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'Book a flight to the investor conference'",
            "chosen": json.dumps(
                {
                    "function": "NONE",
                    "reason": "No available function can book flights.",
                }
            ),
            "rejected": json.dumps(
                {
                    "function": "search_sec_filings",
                    "arguments": {"query": "investor conference"},
                }
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'How much is 100k euros in dollars?'",
            "chosen": json.dumps(
                {
                    "function": "convert_currency",
                    "arguments": {
                        "amount": 100000,
                        "from_currency": "EUR",
                        "to_currency": "USD",
                    },
                }
            ),
            "rejected": json.dumps(
                {
                    "function": "convert_currency",
                    "arguments": {
                        "amount": 100,
                        "from_currency": "EUR",
                        "to_currency": "USD",
                    },
                }
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'Find Goldman's annual report for 2024'",
            "chosen": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {"company": "GS", "report_type": "10-K", "year": 2024},
                }
            ),
            "rejected": json.dumps(
                {
                    "function": "get_financial_report",
                    "arguments": {
                        "company": "Goldman Sachs",
                        "report_type": "earnings",
                        "year": 2024,
                    },
                }
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'What's MSFT's debt to equity if they have $60B debt and $120B equity?'",
            "chosen": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "debt_to_equity",
                        "numerator": 60,
                        "denominator": 120,
                    },
                }
            ),
            "rejected": json.dumps(
                {
                    "function": "calculate_ratio",
                    "arguments": {
                        "ratio_type": "current_ratio",
                        "numerator": 60,
                        "denominator": 120,
                    },
                }
            ),
        },
        {
            "prompt": "You have access to financial tools. User says: 'Search SEC for climate risk disclosures filed in 2024'",
            "chosen": json.dumps(
                {
                    "function": "search_sec_filings",
                    "arguments": {
                        "query": "climate risk disclosures",
                        "date_from": "2024-01-01",
                        "date_to": "2024-12-31",
                    },
                }
            ),
            "rejected": json.dumps(
                {"function": "search_sec_filings", "arguments": {"query": "climate"}}
            ),
        },
    ]
    return examples


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} examples to {path}")


def main():
    # Save function definitions for reference
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/function_definitions.json", "w") as f:
        json.dump(FUNCTION_DEFINITIONS, f, indent=2)
    print("Saved function definitions to data/function_definitions.json")

    # SFT data (80/20 split)
    sft_data = create_function_calling_sft_examples()
    split = int(len(sft_data) * 0.8)
    save_jsonl(sft_data[:split], "data/sft_train.jsonl")
    save_jsonl(sft_data[split:], "data/sft_eval.jsonl")

    # DPO data
    dpo_data = create_function_calling_dpo_examples()
    split = int(len(dpo_data) * 0.75)
    save_jsonl(dpo_data[:split], "data/dpo_train.jsonl")
    save_jsonl(dpo_data[split:], "data/dpo_eval.jsonl")

    print(
        f"\nTotal SFT examples: {len(sft_data)} (train: {int(len(sft_data) * 0.8)}, eval: {len(sft_data) - int(len(sft_data) * 0.8)})"
    )
    print(
        f"Total DPO examples: {len(dpo_data)} (train: {int(len(dpo_data) * 0.75)}, eval: {len(dpo_data) - int(len(dpo_data) * 0.75)})"
    )
    print("\nNote: For production, augment with the full Glaive dataset:")
    print("  huggingface-cli download glaiveai/glaive-function-calling-v2")


if __name__ == "__main__":
    main()
