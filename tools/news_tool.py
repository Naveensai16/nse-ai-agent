"""Market and company news retrieval tool using RSS feeds and structured news providers."""

import html
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import yfinance as yf

from tools.market_tool import resolve_index
from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol

logger = logging.getLogger(__name__)

DEFAULT_NEWS_LIMIT = 10


def parse_news_date(date_val: Any) -> Optional[str]:
    """Parse various news date formats (RFC 2822, Unix timestamp, ISO 8601) to standard UTC string.

    Args:
        date_val: Date representation as string, int, float, or None.

    Returns:
        Formatted date string 'YYYY-MM-DD HH:MM:SS UTC' or None if unparseable.
    """
    if date_val is None:
        return None

    # 1. Unix timestamp (int or float)
    if isinstance(date_val, (int, float)):
        try:
            dt = datetime.fromtimestamp(date_val, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return None

    if not isinstance(date_val, str):
        return None

    date_str = date_val.strip()
    if not date_str:
        return None

    # 2. String representation of a timestamp
    if date_str.isdigit():
        try:
            dt = datetime.fromtimestamp(int(date_str), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass

    # 3. RFC 2822 / RSS pubDate format (e.g., 'Wed, 30 Sep 2026 14:30:00 GMT')
    try:
        dt = parsedate_to_datetime(date_str)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass

    # 4. ISO 8601 format (e.g., '2026-09-30T14:30:00Z')
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass

    return None


def deduplicate_stories(stories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate news stories by normalized URL or title similarity.

    Args:
        stories: List of story dictionaries.

    Returns:
        Deduplicated list preserving original chronological or input order.
    """
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for story in stories:
        url = (story.get("url") or "").strip().lower()
        title = (story.get("title") or "").strip()

        # Normalize title: strip source suffix (e.g., ' - Economic Times') and special chars
        clean_title = re.sub(r"\s*-\s*[^-]+$", "", title)
        norm_title = re.sub(r"[^\w\s]", "", clean_title.lower()).strip()

        # Check duplicate by URL
        if url and url in seen_urls:
            continue

        # Check duplicate by normalized title
        if norm_title and norm_title in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if norm_title:
            seen_titles.add(norm_title)

        deduped.append(story)

    return deduped


def normalize_news_query(company_or_symbol: Any) -> Optional[str]:
    """Normalize input company name or ticker into an effective search query string.

    Args:
        company_or_symbol: Stock ticker, index name, or company title.

    Returns:
        Clean query string for news search, or None if input is invalid.
    """
    if not isinstance(company_or_symbol, str):
        return None

    cleaned = company_or_symbol.strip()
    if not cleaned:
        return None

    # Reject strings with no alphanumeric characters (e.g. '$$$', '***')
    if not re.search(r"[a-zA-Z0-9]", cleaned):
        return None

    # 1. Check if it matches an index alias (e.g., 'NIFTY', 'BANK NIFTY')
    idx = resolve_index(cleaned)
    if idx:
        return idx["name"]

    # 2. Check if it's a valid NSE equity ticker (e.g., 'TCS', 'reliance', 'INFY.NS')
    if is_valid_nse_symbol(cleaned):
        norm_symbol = normalize_nse_symbol(cleaned, target_format="clean")
        return f"{norm_symbol} NSE"

    # 3. Otherwise treat as raw company name
    return cleaned


# Mapping of common financial domain names to recognizable publisher names
KNOWN_PUBLISHERS: dict[str, str] = {
    "livemint.com": "Livemint",
    "economictimes.indiatimes.com": "The Economic Times",
    "moneycontrol.com": "Moneycontrol",
    "cnbctv18.com": "CNBC-TV18",
    "business-standard.com": "Business Standard",
    "financialexpress.com": "Financial Express",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "ndtvprofit.com": "NDTV Profit",
    "ndtv.com": "NDTV",
    "timesofindia.indiatimes.com": "Times of India",
    "thehindubusinessline.com": "The Hindu BusinessLine",
    "zeebiz.com": "Zee Business",
    "etnownews.com": "ET Now",
    "kalkineindia.com": "Kalkine India",
    "upstox.com": "Upstox",
    "groww.in": "Groww",
    "zerodha.com": "Zerodha",
    "latestly.com": "LatestLY",
    "msn.com": "MSN",
    "bqprime.com": "BQ Prime",
}


def extract_publisher_from_url(url: Optional[str]) -> Optional[str]:
    """Extract clean news publisher name from an article URL domain.

    Args:
        url: Direct or canonical web article URL.

    Returns:
        Publisher name string or None if unidentifiable.
    """
    if not url:
        return None
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for known_domain, name in KNOWN_PUBLISHERS.items():
            if domain == known_domain or domain.endswith("." + known_domain):
                return name
        parts = domain.split(".")
        if parts:
            name = parts[0].replace("-", " ").title()
            if name and name.lower() not in ("news", "m", "article", "stories"):
                return name
    except Exception:
        pass
    return None


def parse_rss_feed(xml_content: str) -> list[dict[str, Any]]:
    """Parse RSS 2.0 XML string into normalized story dictionaries.

    Extracts direct external publisher links (unwrapping click-tracking redirects),
    namespaced and standard publisher sources, and multi-sentence article summaries.

    Args:
        xml_content: Raw XML response body.

    Returns:
        List of story dictionaries.
    """
    stories: list[dict[str, Any]] = []
    if not xml_content or not xml_content.strip():
        return stories

    try:
        root = ET.fromstring(xml_content)
    except Exception as exc:
        logger.warning("Failed to parse RSS XML: %s", exc)
        return stories

    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else root.findall(".//item")

    for item in items:
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        desc_el = item.find("description")
        source_el = item.find("source")

        title = (
            html.unescape(title_el.text).strip()
            if title_el is not None and title_el.text
            else None
        )
        url = link_el.text.strip() if link_el is not None and link_el.text else None
        pub_date = (
            pubdate_el.text.strip()
            if pubdate_el is not None and pubdate_el.text
            else None
        )

        # Direct external link resolution: unwrap click-tracking redirects (e.g. Bing apiclick.aspx?url=...)
        if url and ("apiclick.aspx" in url or "url=" in url):
            try:
                parsed_u = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed_u.query)
                if "url" in qs and qs["url"]:
                    url = qs["url"][0]
            except Exception:
                pass

        # Source resolution: check <source>, namespaced source elements, title separators, or domain
        source = None
        if source_el is not None and source_el.text:
            source = html.unescape(source_el.text).strip()
        else:
            # Check namespaced child tags (e.g. <ns0:Source>LatestLY</ns0:Source>)
            for child in item:
                if child.tag.lower().endswith("source") and child.text:
                    source = html.unescape(child.text).strip()
                    break

        if not source and title:
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                source = parts[1].strip()
            elif " | " in title:
                parts = title.rsplit(" | ", 1)
                source = parts[1].strip()

        if not source and url:
            source = extract_publisher_from_url(url)

        # Summary resolution: strip HTML markup and clean whitespace
        summary = None
        if desc_el is not None and desc_el.text:
            cleaned_desc = re.sub(r"<[^>]+>", "", desc_el.text)
            cleaned_desc = html.unescape(cleaned_desc).strip()
            cleaned_desc = re.sub(r"\s+", " ", cleaned_desc).strip()
            if cleaned_desc:
                summary = cleaned_desc

        if title:
            stories.append(
                {
                    "title": title,
                    "source": source,
                    "published_at": parse_news_date(pub_date),
                    "url": url,
                    "summary": summary,
                }
            )

    return stories


def fetch_news_feed(query: str, timeout: int = 10) -> str:
    """Fetch RSS XML news feed from news provider with multi-source fallback.

    Prioritizes Bing News RSS for rich multi-sentence paragraph summaries and direct
    external links, falling back to Google News RSS if unavailable.

    Args:
        query: News search query string.
        timeout: Network timeout in seconds.

    Returns:
        Raw XML feed string.
    """
    encoded_query = urllib.parse.quote(query)

    # 1. Try Bing News RSS first for rich multi-sentence descriptions and direct publisher links
    bing_url = f"https://www.bing.com/news/search?q={encoded_query}&format=rss"
    req_bing = urllib.request.Request(
        bing_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req_bing, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if "<item>" in content or "<item " in content:
                return content
    except Exception as exc:
        logger.debug("Bing News RSS fetch failed for '%s': %s", query, exc)

    # 2. Fallback to Google News RSS
    google_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    req_google = urllib.request.Request(
        google_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NSEStockAgent/1.0"
        },
    )
    with urllib.request.urlopen(req_google, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_market_news(
    company_or_symbol: str, limit: int = DEFAULT_NEWS_LIMIT
) -> list[dict[str, Any]]:
    """Fetch recent market and company news for an NSE stock, index, or company name.

    Uses an RSS feed approach over fragile HTML scraping. Deduplicates stories,
    sorts recent stories first, and handles network or provider errors gracefully.

    Args:
        company_or_symbol: Stock ticker (e.g. 'TCS', 'RELIANCE'), index ('NIFTY 50'),
            or company name.
        limit: Maximum number of stories to return. Defaults to 10.

    Returns:
        List of dictionaries with keys:
            - title: Headline string
            - source: News publisher name or None
            - published_at: UTC timestamp string 'YYYY-MM-DD HH:MM:SS UTC' or None
            - url: Link to article or None
            - summary: Brief article summary or None
    """
    # 1. Normalize query
    query = normalize_news_query(company_or_symbol)
    if not query:
        logger.warning(
            "Invalid company or symbol provided for news: '%s'", company_or_symbol
        )
        return []

    stories: list[dict[str, Any]] = []

    # 2. Fetch news from RSS provider
    try:
        xml_content = fetch_news_feed(query)
        stories = parse_rss_feed(xml_content)
    except Exception as exc:
        logger.warning("RSS news fetch failed for '%s': %s", query, exc)

    # 3. Fallback to yfinance ticker news if RSS yielded no stories and input is a ticker or company name
    resolved_ticker = None
    if isinstance(company_or_symbol, str):
        try:
            from services.symbol_resolver import resolve_nse_symbol
            res = resolve_nse_symbol(company_or_symbol)
            if res.get("symbol"):
                resolved_ticker = res["symbol"]
        except Exception:
            pass

    if not stories and (resolved_ticker or (isinstance(company_or_symbol, str) and is_valid_nse_symbol(company_or_symbol))):
        try:
            target_sym = resolved_ticker or company_or_symbol
            yahoo_symbol = normalize_nse_symbol(target_sym, target_format="yahoo")
            yf_ticker = yf.Ticker(yahoo_symbol)
            yf_news = getattr(yf_ticker, "news", None)
            if yf_news and isinstance(yf_news, list):
                for item in yf_news:
                    title = item.get("title")
                    if title:
                        stories.append(
                            {
                                "title": title,
                                "source": item.get("publisher"),
                                "published_at": parse_news_date(
                                    item.get("providerPublishTime")
                                ),
                                "url": item.get("link"),
                                "summary": None,
                            }
                        )
        except Exception as exc:
            logger.debug("yfinance news fallback failed: %s", exc)

    if not stories:
        return []

    # 4. Deduplicate stories
    deduped = deduplicate_stories(stories)

    # 5. Sort recent stories first (missing dates sort to end)
    def _sort_key(s: dict[str, Any]) -> str:
        return s.get("published_at") or ""

    deduped.sort(key=_sort_key, reverse=True)

    # 6. Limit output
    return deduped[:limit]
