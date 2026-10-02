"""Quarterly financial analysis, cash flow evaluation, debt metrics, and health scoring for NSE companies."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import pandas as pd
import yfinance as yf

from tools.company_tool import _clean_number, _clean_str
from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol

logger = logging.getLogger(__name__)


def is_financial_company(sector: Optional[str], industry: Optional[str]) -> bool:
    """Detect whether a company is a Bank, NBFC, or Financial Institution.

    Financial institutions have structurally different business models where customer
    deposits and wholesale borrowings are raw materials rather than corporate debt.

    Args:
        sector: Sector classification string.
        industry: Industry classification string.

    Returns:
        True if the company is a bank, NBFC, or financial entity, else False.
    """
    sec_str = (sector or "").strip().lower()
    ind_str = (industry or "").strip().lower()

    if "financial" in sec_str or "banking" in sec_str:
        return True

    financial_keywords = [
        "bank",
        "nbfc",
        "insurance",
        "housing finance",
        "asset management",
        "capital markets",
        "credit services",
        "microfinance",
        "lending",
    ]
    return any(k in ind_str for k in financial_keywords)


def evaluate_pe_valuation(
    pe: Optional[float],
    sector: Optional[str] = None,
    earnings_growth: Optional[float] = None,
    forward_pe: Optional[float] = None,
    peg_ratio: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate stock P/E valuation contextually rather than using simplistic thresholds.

    Compares current valuation relative to earnings growth, sector context, and forward
    expectations. A fast-growing firm may reasonably trade at a higher multiple.

    Args:
        pe: Trailing P/E ratio.
        sector: Company sector name.
        earnings_growth: YoY or QoQ profit growth percentage.
        forward_pe: Forward P/E ratio.
        peg_ratio: Price/Earnings to Growth ratio.

    Returns:
        Dictionary with 'assessment' ('Very Good', 'Good', 'Average', 'Bad', 'Very Bad',
        or 'Insufficient Data') and 'reason'.
    """
    if pe is None:
        return {
            "assessment": "Insufficient Data",
            "reason": "Current P/E ratio is unavailable or the company has reported negative earnings.",
        }

    if pe <= 0:
        return {
            "assessment": "Very Bad",
            "reason": f"Company is currently operating at a net loss (negative P/E of {pe:.2f}x).",
        }

    # Contextual sector benchmarks
    sec_lower = (sector or "").lower()
    if "tech" in sec_lower or "information" in sec_lower:
        bench_low, bench_high = 22.0, 28.0
    elif "financial" in sec_lower or "bank" in sec_lower:
        bench_low, bench_high = 12.0, 18.0
    elif "consumer" in sec_lower or "fmcg" in sec_lower:
        bench_low, bench_high = 35.0, 50.0
    elif "auto" in sec_lower:
        bench_low, bench_high = 18.0, 24.0
    elif "pharma" in sec_lower or "health" in sec_lower:
        bench_low, bench_high = 25.0, 35.0
    else:
        bench_low, bench_high = 20.0, 26.0

    growth = earnings_growth if earnings_growth is not None else 0.0
    f_pe_str = f" (forward P/E: {forward_pe:.1f}x)" if forward_pe and forward_pe > 0 else ""

    # PEG evaluation if available
    if peg_ratio and 0 < peg_ratio <= 1.0 and growth > 10:
        return {
            "assessment": "Very Good",
            "reason": (
                f"Trading at {pe:.1f}x earnings{f_pe_str} with an attractive PEG of {peg_ratio:.2f}. "
                f"Strong earnings growth of {growth:+.1f}% comfortably supports the current multiple."
            ),
        }

    # Undervalued relative to sector with positive growth
    if pe < bench_low and growth >= 0:
        return {
            "assessment": "Very Good" if growth > 15 else "Good",
            "reason": (
                f"Trading at {pe:.1f}x earnings{f_pe_str}, below typical sector benchmark "
                f"({bench_low:.0f}-{bench_high:.0f}x), supported by earnings expansion."
            ),
        }

    # In line with sector benchmark
    if bench_low <= pe <= bench_high:
        return {
            "assessment": "Good" if growth > 12 else "Average",
            "reason": (
                f"Trading at {pe:.1f}x earnings{f_pe_str}, broadly in line with sector historical norms "
                f"({bench_low:.0f}-{bench_high:.0f}x)."
            ),
        }

    # Above benchmark but supported by rapid earnings growth
    if pe > bench_high:
        if growth >= 25:
            return {
                "assessment": "Good",
                "reason": (
                    f"P/E is premium at {pe:.1f}x{f_pe_str}, but justifiable given rapid earnings growth "
                    f"of {growth:+.1f}%."
                ),
            }
        elif growth >= 10:
            return {
                "assessment": "Average",
                "reason": (
                    f"Valuation is elevated at {pe:.1f}x{f_pe_str} compared to sector benchmark "
                    f"({bench_low:.0f}-{bench_high:.0f}x) with moderate earnings growth ({growth:+.1f}%)."
                ),
            }
        elif growth > 0:
            return {
                "assessment": "Bad",
                "reason": (
                    f"P/E of {pe:.1f}x{f_pe_str} reflects a significant premium over sector benchmarks "
                    f"with modest earnings expansion ({growth:+.1f}%)."
                ),
            }
        else:
            return {
                "assessment": "Very Bad",
                "reason": (
                    f"Expensive valuation of {pe:.1f}x{f_pe_str} while earnings contracted ({growth:+.1f}%), "
                    "indicating earnings vulnerability."
                ),
            }

    return {
        "assessment": "Average",
        "reason": f"Currently trades at {pe:.1f}x earnings relative to historical peers.",
    }


def evaluate_roce(
    roce: Optional[float], is_financial: bool = False
) -> dict[str, Any]:
    """Evaluate Return on Capital Employed (ROCE) using deterministic rules and industry context.

    Args:
        roce: Current ROCE percentage (e.g. 52.0 for 52%).
        is_financial: Whether the entity is a banking/financial institution.

    Returns:
        Dictionary with 'assessment' and 'reason'.
    """
    if is_financial:
        return {
            "assessment": "Not Applicable",
            "reason": (
                "ROCE is not a standard KPI for banking and financial companies because deposits and debt "
                "form their core operational working capital. Return on Assets (ROA) and Return on Equity (ROE) "
                "are assessed instead."
            ),
        }

    if roce is None:
        return {
            "assessment": "Insufficient Data",
            "reason": "ROCE data is currently not available from reported financial statements.",
        }

    if roce >= 30.0:
        return {
            "assessment": "Very Good",
            "reason": f"Outstanding ROCE of {roce:.1f}%, indicating superior capital allocation and asset efficiency.",
        }
    elif roce >= 20.0:
        return {
            "assessment": "Good",
            "reason": f"Healthy ROCE of {roce:.1f}%, well above standard Indian corporate cost of capital (~12%).",
        }
    elif roce >= 12.0:
        return {
            "assessment": "Average",
            "reason": f"Moderate ROCE of {roce:.1f}%, roughly matching baseline cost of capital.",
        }
    elif roce >= 8.0:
        return {
            "assessment": "Bad",
            "reason": f"Sub-par ROCE of {roce:.1f}%, indicating inefficient returns on deployed capital.",
        }
    else:
        return {
            "assessment": "Very Bad",
            "reason": f"Poor ROCE of {roce:.1f}%, destroying shareholder value below the risk-free rate.",
        }


def evaluate_roe(
    roe: Optional[float],
    debt_to_equity: Optional[float] = None,
    is_financial: bool = False,
) -> dict[str, Any]:
    """Evaluate Return on Equity (ROE), checking whether high ROE is leveraged by debt.

    Args:
        roe: Current ROE percentage (e.g. 43.0 for 43%).
        debt_to_equity: Debt to equity ratio.
        is_financial: Whether the entity is a banking institution.

    Returns:
        Dictionary with 'assessment' and 'reason'.
    """
    if roe is None:
        return {
            "assessment": "Insufficient Data",
            "reason": "ROE data is not currently available from financial disclosures.",
        }

    # Bank-specific ROE norms
    if is_financial:
        if roe >= 16.0:
            return {
                "assessment": "Very Good",
                "reason": f"Top-tier ROE of {roe:.1f}% for a financial institution, demonstrating strong franchise profitability.",
            }
        elif roe >= 12.0:
            return {
                "assessment": "Good",
                "reason": f"Solid banking ROE of {roe:.1f}%, indicating sustainable equity accretion.",
            }
        elif roe >= 8.0:
            return {
                "assessment": "Average",
                "reason": f"Moderate banking ROE of {roe:.1f}%.",
            }
        else:
            return {
                "assessment": "Bad" if roe >= 0 else "Very Bad",
                "reason": f"Subdued banking ROE of {roe:.1f}%, lagging banking sector averages.",
            }

    # Non-financial companies: inspect leverage
    leverage_warning = ""
    if debt_to_equity and debt_to_equity > 2.0 and roe > 25.0:
        leverage_warning = f" Note: Elevated financial leverage (Debt/Equity: {debt_to_equity:.2f}x) amplifies this return."

    if roe >= 25.0:
        assessment = "Average" if (debt_to_equity and debt_to_equity > 2.5) else "Very Good"
        return {
            "assessment": assessment,
            "reason": f"Robust ROE of {roe:.1f}%, reflecting strong shareholder equity compounding.{leverage_warning}",
        }
    elif roe >= 15.0:
        return {
            "assessment": "Good",
            "reason": f"Healthy ROE of {roe:.1f}%, beating standard market hurdle rates.{leverage_warning}",
        }
    elif roe >= 10.0:
        return {
            "assessment": "Average",
            "reason": f"Acceptable ROE of {roe:.1f}%.",
        }
    elif roe >= 0:
        return {
            "assessment": "Bad",
            "reason": f"Low ROE of {roe:.1f}%, indicating weak equity utilization.",
        }
    else:
        return {
            "assessment": "Very Bad",
            "reason": f"Negative ROE of {roe:.1f}% due to reported net losses.",
        }


def get_quarterly_financials(symbol: str) -> dict[str, Any]:
    """Retrieve and parse the latest four reported quarters of financial performance for an NSE equity.

    Extracts Revenue/Sales, Expenses, Operating Profit, Operating Margin (OPM), Net Profit, and EPS.
    Computes QoQ and YoY growth rates without fabricating unavailable periods.

    Args:
        symbol: NSE stock symbol (e.g., 'TCS', 'INFY', 'RELIANCE').

    Returns:
        Dictionary containing:
            - symbol: Clean NSE ticker symbol
            - quarters: List of up to 4 quarter dicts (Revenue, Expenses, Operating Profit, OPM, Net Profit, EPS)
            - qoq_revenue_growth: QoQ percentage revenue growth
            - yoy_revenue_growth: YoY percentage revenue growth where comparable quarter exists
            - qoq_profit_growth: QoQ percentage net profit growth
            - yoy_profit_growth: YoY percentage net profit growth
            - margin_trend: Qualitative margin direction ('Expanding', 'Contracting', 'Stable', or 'N/A')
    """
    empty_result: dict[str, Any] = {
        "symbol": None,
        "quarters": [],
        "qoq_revenue_growth": None,
        "yoy_revenue_growth": None,
        "qoq_profit_growth": None,
        "yoy_profit_growth": None,
        "margin_trend": "N/A",
    }

    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
        yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
    except Exception as exc:
        logger.warning("Symbol normalization error in get_quarterly_financials: %s", exc)
        return empty_result

    empty_result["symbol"] = clean_symbol

    try:
        ticker = yf.Ticker(yahoo_symbol)
        inc_stmt = getattr(ticker, "quarterly_income_stmt", None)
        if inc_stmt is None or getattr(inc_stmt, "empty", True):
            inc_stmt = getattr(ticker, "quarterly_financials", None)
    except Exception as exc:
        logger.warning("Failed to fetch quarterly financials for '%s': %s", yahoo_symbol, exc)
        return empty_result

    if inc_stmt is None or getattr(inc_stmt, "empty", True):
        return empty_result

    try:
        # Columns in yfinance can be ordered ascending or descending.
        # Sort chronologically (oldest -> newest) to ensure consistent indexing.
        cols = list(inc_stmt.columns)
        if not cols:
            return empty_result

        try:
            sorted_cols = sorted(cols, key=lambda c: pd.to_datetime(c))
        except Exception:
            sorted_cols = sorted(cols)

        # Take up to 5 quarters so we have comparable YoY for the latest quarter if available
        chronological_cols = sorted_cols[-5:]

        def _get_val(df: pd.DataFrame, keys: list[str], col: Any) -> Optional[float]:
            for k in keys:
                if k in df.index:
                    v = df.loc[k, col]
                    cleaned = _clean_number(v, round_digits=2)
                    if cleaned is not None:
                        return float(cleaned)
            return None

        rev_keys = ["Total Revenue", "Operating Revenue", "Gross Revenue"]
        op_keys = ["Operating Income", "EBIT", "Net Interest Income"]
        net_keys = [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Continuous Operations",
            "Net Income From Continuing And Discontinued Operation",
        ]
        eps_keys = ["Basic EPS", "Diluted EPS"]
        exp_keys = ["Total Expenses", "Operating Expense"]

        quarters: list[dict[str, Any]] = []
        for col in chronological_cols:
            date_label = col.strftime("%b %Y") if hasattr(col, "strftime") else str(col)[:10]

            rev = _get_val(inc_stmt, rev_keys, col)
            op = _get_val(inc_stmt, op_keys, col)
            net_prof = _get_val(inc_stmt, net_keys, col)
            eps_val = _get_val(inc_stmt, eps_keys, col)
            exp_val = _get_val(inc_stmt, exp_keys, col)

            # Deduce expenses if missing: Expenses = Revenue - Operating Profit
            if exp_val is None and rev is not None and op is not None:
                exp_val = round(rev - op, 2)

            opm = round((op / rev) * 100, 2) if (op is not None and rev is not None and rev > 0) else None

            quarters.append(
                {
                    "date": date_label,
                    "revenue": rev,
                    "expenses": exp_val,
                    "operating_profit": op,
                    "operating_margin": opm,
                    "net_profit": net_prof,
                    "eps": eps_val,
                }
            )

        if not quarters:
            return empty_result

        # Calculate QoQ growth (latest quarter vs previous quarter)
        qoq_rev_growth = None
        qoq_profit_growth = None
        if len(quarters) >= 2:
            latest = quarters[-1]
            prev = quarters[-2]
            if latest["revenue"] and prev["revenue"] and prev["revenue"] > 0:
                qoq_rev_growth = round(((latest["revenue"] - prev["revenue"]) / prev["revenue"]) * 100, 2)
            if latest["net_profit"] is not None and prev["net_profit"] is not None and abs(prev["net_profit"]) > 0:
                qoq_profit_growth = round(
                    ((latest["net_profit"] - prev["net_profit"]) / abs(prev["net_profit"])) * 100, 2
                )

        # Calculate YoY growth if we have 5 quarters (latest vs 4 quarters ago)
        yoy_rev_growth = None
        yoy_profit_growth = None
        if len(quarters) >= 5:
            latest = quarters[-1]
            yoy_prev = quarters[-5]
            if latest["revenue"] and yoy_prev["revenue"] and yoy_prev["revenue"] > 0:
                yoy_rev_growth = round(((latest["revenue"] - yoy_prev["revenue"]) / yoy_prev["revenue"]) * 100, 2)
            if latest["net_profit"] is not None and yoy_prev["net_profit"] is not None and abs(yoy_prev["net_profit"]) > 0:
                yoy_profit_growth = round(
                    ((latest["net_profit"] - yoy_prev["net_profit"]) / abs(yoy_prev["net_profit"])) * 100, 2
                )

        # Margin trend
        margin_trend = "N/A"
        if len(quarters) >= 2:
            m_latest = quarters[-1].get("operating_margin")
            m_prev = quarters[-2].get("operating_margin")
            if m_latest is not None and m_prev is not None:
                diff = m_latest - m_prev
                if diff > 0.5:
                    margin_trend = "Expanding"
                elif diff < -0.5:
                    margin_trend = "Contracting"
                else:
                    margin_trend = "Stable"

        # Return latest 4 quarters
        display_quarters = quarters[-4:]

        return {
            "symbol": clean_symbol,
            "quarters": display_quarters,
            "qoq_revenue_growth": qoq_rev_growth,
            "yoy_revenue_growth": yoy_rev_growth,
            "qoq_profit_growth": qoq_profit_growth,
            "yoy_profit_growth": yoy_profit_growth,
            "margin_trend": margin_trend,
        }

    except Exception as exc:
        logger.warning("Error parsing quarterly financials for '%s': %s", symbol, exc)
        return empty_result


def get_cash_flow(symbol: str) -> dict[str, Any]:
    """Retrieve Cash Flow from Operations (OCF) and evaluate cash conversion vs net profit.

    Handles explicit reporting periods (annual vs quarterly) without manufacturing numbers.

    Args:
        symbol: NSE stock symbol.

    Returns:
        Dictionary containing:
            - symbol: Clean NSE symbol
            - quarterly_reported: True if quarterly cash flow is disclosed
            - latest_quarterly_ocf: Value or None
            - annual_ocf_history: List of annual OCF values with periods
            - cash_conversion_ratio: OCF / Net Profit
            - cash_conversion_assessment: Quality explanation
    """
    empty_result: dict[str, Any] = {
        "symbol": None,
        "quarterly_reported": False,
        "latest_quarterly_ocf": None,
        "annual_ocf_history": [],
        "cash_conversion_ratio": None,
        "cash_conversion_assessment": "Cash flow data unavailable.",
    }

    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
        yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
    except Exception as exc:
        logger.warning("Symbol normalization error in get_cash_flow: %s", exc)
        return empty_result

    empty_result["symbol"] = clean_symbol

    try:
        ticker = yf.Ticker(yahoo_symbol)
        cf_q = getattr(ticker, "quarterly_cashflow", None)
        cf_a = getattr(ticker, "cashflow", None)
    except Exception as exc:
        logger.warning("Failed to fetch cash flow for '%s': %s", yahoo_symbol, exc)
        return empty_result

    ocf_keys = [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ]

    def _extract_ocf(df: pd.DataFrame) -> list[tuple[str, float]]:
        if df is None or getattr(df, "empty", True):
            return []
        items = []
        try:
            sorted_cols = sorted(df.columns, key=lambda c: pd.to_datetime(c), reverse=True)
        except Exception:
            sorted_cols = list(df.columns)
        for col in sorted_cols[:4]:
            date_str = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
            for k in ocf_keys:
                if k in df.index:
                    val = _clean_number(df.loc[k, col], round_digits=2)
                    if val is not None:
                        items.append((date_str, float(val)))
                        break
        return items

    annual_items = _extract_ocf(cf_a)
    quarterly_items = _extract_ocf(cf_q)

    quarterly_reported = bool(quarterly_items)
    latest_quarterly_ocf = quarterly_items[0][1] if quarterly_items else None

    # Calculate Cash Conversion Ratio
    # Prefer Annual OCF / Annual Net Income if annual exists, else quarterly
    conversion_ratio = None
    assessment = "Cash flow data is not separately reported for this period."

    if annual_items:
        latest_year, ocf_val = annual_items[0]
        # Attempt to retrieve comparable net income from annual income statement
        try:
            inc_a = getattr(ticker, "income_stmt", None)
            if inc_a is None or getattr(inc_a, "empty", True):
                inc_a = getattr(ticker, "financials", None)
            net_inc = None
            if inc_a is not None and not inc_a.empty:
                try:
                    sorted_inc = sorted(inc_a.columns, key=lambda c: pd.to_datetime(c), reverse=True)
                except Exception:
                    sorted_inc = list(inc_a.columns)
                col_latest = sorted_inc[0]
                for nk in ["Net Income", "Net Income Common Stockholders"]:
                    if nk in inc_a.index:
                        net_val = _clean_number(inc_a.loc[nk, col_latest], round_digits=2)
                        if net_val is not None:
                            net_inc = float(net_val)
                            break
            if net_inc and net_inc > 0 and ocf_val:
                conversion_ratio = round(ocf_val / net_inc, 2)
        except Exception:
            pass

        if conversion_ratio is not None:
            if conversion_ratio >= 1.0:
                assessment = (
                    f"Healthy Cash Conversion ({conversion_ratio:.2f}x): "
                    "Accounting profits are fully converting into operating cash flow."
                )
            elif conversion_ratio >= 0.75:
                assessment = (
                    f"Moderate Cash Conversion ({conversion_ratio:.2f}x): "
                    "Satisfactory conversion of net profit into operational cash."
                )
            elif conversion_ratio > 0:
                assessment = (
                    f"Subdued Cash Conversion ({conversion_ratio:.2f}x): "
                    "Working capital absorption or receivables may be dampening cash realization."
                )
            else:
                assessment = "Negative Cash Conversion: Operating cash generation is lagging reported net profits."
        elif ocf_val and ocf_val > 0:
            assessment = "Positive Operating Cash Flow generated during the fiscal year."

    return {
        "symbol": clean_symbol,
        "quarterly_reported": quarterly_reported,
        "latest_quarterly_ocf": latest_quarterly_ocf,
        "annual_ocf_history": [{"period": p, "ocf": v} for p, v in annual_items],
        "cash_conversion_ratio": conversion_ratio,
        "cash_conversion_assessment": assessment,
    }


def get_debt_metrics(
    symbol: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve debt, cash holdings, net debt, and leverage metrics for an NSE stock.

    Identifies banks and NBFCs and avoids applying industrial corporate leverage metrics.

    Args:
        symbol: NSE stock ticker symbol.
        sector: Optional sector override string.
        industry: Optional industry override string.

    Returns:
        Dictionary containing debt metrics and leverage assessment.
    """
    empty_result: dict[str, Any] = {
        "symbol": None,
        "is_financial": False,
        "total_debt": None,
        "cash_and_equivalents": None,
        "net_debt": None,
        "debt_to_equity": None,
        "debt_trend": [],
        "assessment": "Debt metrics unavailable.",
    }

    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
        yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
    except Exception as exc:
        logger.warning("Symbol normalization error in get_debt_metrics: %s", exc)
        return empty_result

    empty_result["symbol"] = clean_symbol

    try:
        ticker = yf.Ticker(yahoo_symbol)
        info_data = getattr(ticker, "info", None)
        info = info_data if isinstance(info_data, dict) else {}
        bs = getattr(ticker, "quarterly_balance_sheet", None)
        if bs is None or getattr(bs, "empty", True):
            bs = getattr(ticker, "balance_sheet", None)
    except Exception as exc:
        logger.warning("Failed to fetch debt metrics for '%s': %s", yahoo_symbol, exc)
        return empty_result

    effective_sector = sector or info.get("sector")
    effective_industry = industry or info.get("industry")
    is_fin = is_financial_company(effective_sector, effective_industry)
    empty_result["is_financial"] = is_fin

    if is_fin:
        empty_result["assessment"] = (
            "Not Applicable: As a banking/financial institution, standard corporate debt metrics "
            "(Total Debt / Debt-to-Equity) are not applicable because customer deposits and wholesale "
            "borrowings are core operating inputs rather than leverage."
        )
        return empty_result

    # Non-financial companies: parse debt and cash
    total_debt = _clean_number(info.get("totalDebt"), round_digits=2)
    cash_val = _clean_number(info.get("totalCash"), round_digits=2)
    d_to_e = _clean_number(info.get("debtToEquity"), round_digits=2)

    # Balance sheet fallback if info fields are absent
    if bs is not None and not bs.empty:
        try:
            sorted_bs_cols = sorted(bs.columns, key=lambda c: pd.to_datetime(c), reverse=True)
        except Exception:
            sorted_bs_cols = list(bs.columns)
        col0 = sorted_bs_cols[0]
        if total_debt is None:
            for dk in ["Total Debt", "Long Term Debt", "Current Debt"]:
                if dk in bs.index:
                    td = _clean_number(bs.loc[dk, col0], round_digits=2)
                    if td is not None:
                        total_debt = float(td)
                        break
        if cash_val is None:
            for ck in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"]:
                if ck in bs.index:
                    cv = _clean_number(bs.loc[ck, col0], round_digits=2)
                    if cv is not None:
                        cash_val = float(cv)
                        break
        if d_to_e is None and total_debt is not None:
            for ek in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if ek in bs.index:
                    eq_val = _clean_number(bs.loc[ek, col0], round_digits=2)
                    if eq_val and eq_val > 0:
                        d_to_e = round(total_debt / eq_val, 2)
                        break

    # If yfinance info debtToEquity is in percentage (e.g. 10.2%), normalize
    if d_to_e is not None and d_to_e > 10.0 and total_debt and cash_val:
        # e.g., if totalDebt is 11,000 cr and equity is 100,000 cr, d_to_e is 0.11 or 11%
        pass

    # Extract debt trend from balance sheet
    debt_history: list[dict[str, Any]] = []
    if bs is not None and not bs.empty:
        debt_keys = ["Total Debt", "Long Term Debt", "Current Debt"]
        for col in bs.columns[:4]:
            period_label = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
            for dk in debt_keys:
                if dk in bs.index:
                    dv = _clean_number(bs.loc[dk, col], round_digits=2)
                    if dv is not None:
                        debt_history.append({"period": period_label, "debt": float(dv)})
                        break

    net_debt = None
    if total_debt is not None and cash_val is not None:
        net_debt = round(total_debt - cash_val, 2)

    assessment = "Debt profile is in line with standard capitalization."
    if net_debt is not None:
        if net_debt <= 0:
            assessment = "Net Cash Positive: Cash and liquid investments comfortably exceed total debt obligations."
        elif d_to_e is not None and d_to_e < 0.5:
            assessment = f"Low Leverage (Debt/Equity: {d_to_e:.2f}x): Strong balance sheet with minimal debt risk."
        elif d_to_e is not None and d_to_e > 1.5:
            assessment = f"High Leverage (Debt/Equity: {d_to_e:.2f}x): Significant debt load requires close coverage monitoring."

    return {
        "symbol": clean_symbol,
        "is_financial": False,
        "total_debt": total_debt,
        "cash_and_equivalents": cash_val,
        "net_debt": net_debt,
        "debt_to_equity": d_to_e,
        "debt_trend": debt_history,
        "assessment": assessment,
    }


def calculate_financial_health(
    qoq_revenue_growth: Optional[float] = None,
    yoy_revenue_growth: Optional[float] = None,
    qoq_profit_growth: Optional[float] = None,
    yoy_profit_growth: Optional[float] = None,
    operating_margin: Optional[float] = None,
    margin_trend: Optional[str] = None,
    cash_conversion_ratio: Optional[float] = None,
    net_debt: Optional[float] = None,
    debt_to_equity: Optional[float] = None,
    roce: Optional[float] = None,
    roe: Optional[float] = None,
    is_financial: bool = False,
) -> dict[str, Any]:
    """Compute an explainable, deterministic financial health score from 0.0 to 10.0.

    Applies separate scoring logic for financial institutions vs non-financial companies.

    Args:
        qoq_revenue_growth: QoQ revenue growth percentage.
        yoy_revenue_growth: YoY revenue growth percentage.
        qoq_profit_growth: QoQ net profit growth percentage.
        yoy_profit_growth: YoY net profit growth percentage.
        operating_margin: Latest operating margin percentage.
        margin_trend: 'Expanding', 'Contracting', or 'Stable'.
        cash_conversion_ratio: OCF / Net Profit.
        net_debt: Net debt (Total Debt - Cash).
        debt_to_equity: Debt to equity ratio.
        roce: Return on Capital Employed percentage.
        roe: Return on Equity percentage.
        is_financial: True if bank or NBFC.

    Returns:
        Dictionary containing:
            - rating: 'Very Good', 'Good', 'Average', 'Bad', or 'Very Bad'
            - score: Numeric score (0.0 - 10.0)
            - positive_factors: List of positive financial signals
            - concerns: List of potential risk factors
    """
    positives: list[str] = []
    concerns: list[str] = []
    base_score = 5.0

    # 1. Revenue Growth
    rev_growth = yoy_revenue_growth if yoy_revenue_growth is not None else qoq_revenue_growth
    if rev_growth is not None:
        if rev_growth > 15.0:
            base_score += 1.5
            positives.append(f"Strong revenue expansion ({rev_growth:+.1f}%) demonstrates market leadership.")
        elif rev_growth > 5.0:
            base_score += 0.8
            positives.append(f"Steady top-line growth ({rev_growth:+.1f}%).")
        elif rev_growth < -5.0:
            base_score -= 1.2
            concerns.append(f"Revenue contraction ({rev_growth:+.1f}%) reflects demand headwinds.")
        elif rev_growth < 0:
            base_score -= 0.5
            concerns.append(f"Slight top-line decline ({rev_growth:+.1f}%).")

    # 2. Net Profit Growth
    profit_growth = yoy_profit_growth if yoy_profit_growth is not None else qoq_profit_growth
    if profit_growth is not None:
        if profit_growth > 20.0:
            base_score += 1.5
            positives.append(f"Robust net earnings growth ({profit_growth:+.1f}%).")
        elif profit_growth > 8.0:
            base_score += 0.8
            positives.append(f"Healthy net profit expansion ({profit_growth:+.1f}%).")
        elif profit_growth < -15.0:
            base_score -= 1.5
            concerns.append(f"Sharp bottom-line contraction ({profit_growth:+.1f}%).")
        elif profit_growth < 0:
            base_score -= 0.7
            concerns.append(f"Net profit slipped ({profit_growth:+.1f}%).")

    # 3. Margins
    if margin_trend == "Expanding":
        base_score += 0.8
        positives.append("Operating margins are expanding QoQ.")
    elif margin_trend == "Contracting":
        base_score -= 0.8
        concerns.append("Operating margins contracted relative to the preceding quarter.")

    if operating_margin is not None:
        if operating_margin >= 20.0:
            base_score += 0.7
            positives.append(f"Premium operating margin of {operating_margin:.1f}%.")
        elif operating_margin < 8.0 and not is_financial:
            base_score -= 0.7
            concerns.append(f"Thin operating margin of {operating_margin:.1f}%.")

    # 4. Solvency / Debt (Non-Financials vs Financials)
    if is_financial:
        # Financial companies are scored on ROE & franchise stability
        if roe is not None:
            if roe >= 15.0:
                base_score += 1.2
                positives.append(f"Strong banking return on equity of {roe:.1f}%.")
            elif roe < 8.0:
                base_score -= 1.0
                concerns.append(f"Banking ROE of {roe:.1f}% lags industry benchmarks.")
    else:
        # Non-financial companies
        if net_debt is not None:
            if net_debt <= 0:
                base_score += 1.2
                positives.append("Net cash positive balance sheet with zero debt overhang.")
            elif debt_to_equity is not None and debt_to_equity > 1.8:
                base_score -= 1.2
                concerns.append(f"High balance-sheet leverage (Debt/Equity: {debt_to_equity:.2f}x).")
            elif debt_to_equity is not None and debt_to_equity < 0.5:
                base_score += 0.6
                positives.append(f"Prudent low leverage (Debt/Equity: {debt_to_equity:.2f}x).")

    # 5. Capital Efficiency (ROCE / ROE)
    if not is_financial and roce is not None:
        if roce >= 25.0:
            base_score += 1.0
            positives.append(f"High Return on Capital Employed (ROCE: {roce:.1f}%).")
        elif roce < 10.0:
            base_score -= 1.0
            concerns.append(f"Sub-par capital efficiency (ROCE: {roce:.1f}%).")

    # 6. Cash Flow Conversion
    if cash_conversion_ratio is not None:
        if cash_conversion_ratio >= 1.0:
            base_score += 0.8
            positives.append(f"Excellent cash conversion ({cash_conversion_ratio:.2f}x OCF/PAT).")
        elif cash_conversion_ratio < 0.6:
            base_score -= 0.8
            concerns.append(f"Weak cash conversion ({cash_conversion_ratio:.2f}x OCF/PAT).")

    # Clamp score to [1.0, 10.0]
    final_score = round(max(1.0, min(10.0, base_score)), 1)

    if final_score >= 8.0:
        rating = "Very Good"
    elif final_score >= 6.5:
        rating = "Good"
    elif final_score >= 5.0:
        rating = "Average"
    elif final_score >= 3.5:
        rating = "Bad"
    else:
        rating = "Very Bad"

    return {
        "rating": rating,
        "score": final_score,
        "positive_factors": positives,
        "concerns": concerns,
    }
