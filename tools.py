from llama_index.core import PromptTemplate
import yfinance as yf
from pandas import DataFrame
from typing import Annotated, Callable, Any, Optional


from llms import get_groq_lm


llm = get_groq_lm()


def save_output(
    data: DataFrame, description: str, save_path: Optional[str] = None
) -> None:
    return
    if save_path:
        data.to_csv(save_path)
        print(f"{description} saved to {save_path}")


def get_stock_data(
    symbol: str, start_date: str, end_date: str, save_path: Optional[str] = None
) -> str:
    # Fetch stock data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(start=start_date, end=end_date).describe()
    save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
    return stock_data.to_string()


def get_balance_sheet(
    symbol: str,
) -> str:
    # Fetch balance sheet from Yahoo Finance
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet

    # Prompt template for summarizing financial data
    template_financial_summary = """You are a financial analyst. Your task is to provide a brief summary of the following financial data concisley in less than 50 words.\n\n{balance_sheet}"""

    prompt_balance_sheet = PromptTemplate(template=template_financial_summary)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(
        prompt_balance_sheet.format(balance_sheet=balance_sheet)
    )

    return result_summary.text


def get_stock_info(
    symbol: str,
) -> str:
    # Fetch stock information from Yahoo Finance
    ticker = yf.Ticker(symbol)
    info = ticker.info

    stock_info = str(info)

    # Prompt template for summarizing financial data
    template_stock_prices = """You are a financial analyst. Your task is to provide a brief summary of insights based based on stock information in maximum 3 lines.\n\n{stock_info}"""

    prompt_stock_proces = PromptTemplate(template=template_stock_prices)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_stock_proces.format(stock_info=stock_info))

    return result_summary.text


def get_company_info(
    symbol: str,
    save_path: Optional[str] = None,
) -> str:
    # Fetch company information from Yahoo Finance
    ticker = yf.Ticker(symbol)
    info = ticker.info
    company_info = {
        "Company Name": info.get("shortName", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Country": info.get("country", "N/A"),
        "Website": info.get("website", "N/A"),
    }
    company_info_df = DataFrame([company_info])
    if save_path:
        company_info_df.to_csv(save_path)
        print(f"Company info for {ticker.ticker} saved to {save_path}")

    company_info = str(company_info_df)

    # Prompt template for summarizing financial data
    template_company_info = """You are a financial analyst. Your task is to summarize the key insights you can derive based on content.\n\n{company_info}"""

    prompt_company_info = PromptTemplate(template=template_company_info)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_company_info.format(company_info=company_info))

    return result_summary.text


def get_income_stmt(
    symbol: str,
) -> str:
    # Fetch income statement from Yahoo Finance
    ticker = yf.Ticker(symbol)
    income_stmt = ticker.financials

    income_stmt = income_stmt.to_string()

    # Prompt template for summarizing financial data
    template_income_stmt = """You are a financial analyst. Your task is to summarize the key insights you can derive based on the following content in 3 lines.\n\n{income_stmt}"""
    prompt_income_stmt = PromptTemplate(template=template_income_stmt)
    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_income_stmt.format(income_stmt=income_stmt))
    return result_summary.text


def get_cash_flow(
    symbol: str,
) -> str:
    # Fetch cash flow from Yahoo Finance
    ticker = yf.Ticker(symbol)
    cash_flows = ticker.cashflow
    cash_flows = cash_flows.to_string()

    # Prompt template for summarizing financial data
    template_cash_flows_template = """You are a financial analyst. Your task is to summarize the key insights you can derive based on the following content.\n\n{cash_flows}"""

    prompt_cash_flows = PromptTemplate(template=template_cash_flows_template)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_cash_flows.format(cash_flows=cash_flows))

    return result_summary.text


def get_stock_dividends(
    symbol: str,
    save_path: Optional[str] = None,
) -> str:
    # Fetch stock dividends from Yahoo Finance
    ticker = yf.Ticker(symbol)
    dividends = ticker.dividends
    if save_path:
        dividends.to_csv(save_path)
        print(f"Dividends for {ticker.ticker} saved to {save_path}")

    dividends_info = dividends.to_string()

    # Prompt template for summarizing financial data
    template_dividends_info = """You are a financial analyst. Your task is to summarize the key trend you can derive based on following content in concise manner.\n\n{dividends_info}"""

    prompt_dividends_info = PromptTemplate(template=template_dividends_info)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(
        prompt_dividends_info.format(dividends_info=dividends_info)
    )

    return result_summary.text


def get_analyst_recommendations(symbol: str) -> str:
    # Fetch analyst recommendations from Yahoo Finance
    ticker = yf.Ticker(symbol)
    recommendations = ticker.recommendations
    if recommendations.empty:
        return "No recommendations available"

    recommendation_counts = recommendations[
        ["strongBuy", "buy", "hold", "sell", "strongSell"]
    ].sum()
    most_common_recommendation = recommendation_counts.idxmax()
    count = recommendation_counts.max()

    return f"Most common recommendation: {most_common_recommendation} ({count} votes)"
