"""Unit tests for the market and company news tool."""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.news_tool import (
    deduplicate_stories,
    extract_publisher_from_url,
    get_market_news,
    normalize_news_query,
    parse_news_date,
    parse_rss_feed,
)

SAMPLE_NORMAL_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News - TCS NSE</title>
    <item>
      <title>TCS signs multi-year digital transformation deal - The Economic Times</title>
      <link>https://economictimes.indiatimes.com/tech/tcs-deal/1001</link>
      <pubDate>Thu, 01 Oct 2026 10:30:00 GMT</pubDate>
      <description>&lt;p&gt;Tata Consultancy Services secures a large strategic engagement.&lt;/p&gt;</description>
      <source url="https://economictimes.indiatimes.com">The Economic Times</source>
    </item>
    <item>
      <title>TCS quarterly revenue rises 7% YoY - Livemint</title>
      <link>https://livemint.com/companies/tcs-revenue/1002</link>
      <pubDate>Wed, 30 Sep 2026 15:00:00 GMT</pubDate>
      <description>IT major beats analyst estimates across major BFSI verticals.</description>
      <source url="https://livemint.com">Livemint</source>
    </item>
  </channel>
</rss>
"""

SAMPLE_DUPLICATE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>TCS signs multi-year digital transformation deal - The Economic Times</title>
      <link>https://economictimes.indiatimes.com/tech/tcs-deal/1001</link>
      <pubDate>Thu, 01 Oct 2026 10:30:00 GMT</pubDate>
      <description>Tata Consultancy Services deal.</description>
      <source>The Economic Times</source>
    </item>
    <item>
      <title>TCS signs multi-year digital transformation deal - Moneycontrol</title>
      <link>https://moneycontrol.com/news/tcs-deal/9999</link>
      <pubDate>Thu, 01 Oct 2026 11:00:00 GMT</pubDate>
      <description>Duplicate headline from another syndicate.</description>
      <source>Moneycontrol</source>
    </item>
    <item>
      <title>Distinct Headline for Reliance - Reuters</title>
      <link>https://reuters.com/markets/reliance/2001</link>
      <pubDate>Thu, 01 Oct 2026 09:00:00 GMT</pubDate>
      <description>Reliance news story.</description>
      <source>Reuters</source>
    </item>
  </channel>
</rss>
"""

SAMPLE_MISSING_FIELDS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Headline with missing summary - Business Standard</title>
      <link>https://business-standard.com/markets/story/3001</link>
      <pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Headline with missing date - Financial Express</title>
      <link>https://financialexpress.com/markets/story/3002</link>
      <description>This story has no pubDate tag.</description>
    </item>
  </channel>
</rss>
"""


class TestNewsHelpers:
    """Test suite for helper functions: date parsing, deduplication, and query normalization."""

    @pytest.mark.parametrize(
        ("input_date", "expected_pattern"),
        [
            ("Thu, 01 Oct 2026 10:30:00 GMT", "2026-10-01 10:30:00 UTC"),
            ("30 Sep 2026 14:00:00 +0000", "2026-09-30 14:00:00 UTC"),
            ("2026-10-01T08:15:00Z", "2026-10-01 08:15:00 UTC"),
            (1759314600, "2025-10-01"),  # Unix timestamp
            (None, None),
            ("", None),
            ("   ", None),
            ("not a date string", None),
        ],
    )
    def test_parse_news_date(self, input_date: object, expected_pattern: Optional[str]) -> None:
        result = parse_news_date(input_date)
        if expected_pattern is None:
            assert result is None
        else:
            assert result is not None
            assert expected_pattern in result

    def test_deduplicate_stories_by_url(self) -> None:
        stories = [
            {"title": "Story 1", "url": "https://example.com/story1", "summary": "Desc 1"},
            {"title": "Story 1 Copy", "url": "https://example.com/story1", "summary": "Desc 2"},
            {"title": "Story 2", "url": "https://example.com/story2", "summary": "Desc 3"},
        ]
        deduped = deduplicate_stories(stories)
        assert len(deduped) == 2
        assert deduped[0]["title"] == "Story 1"
        assert deduped[1]["title"] == "Story 2"

    def test_deduplicate_stories_by_normalized_title(self) -> None:
        stories = [
            {
                "title": "TCS signs $1B cloud deal - Economic Times",
                "url": "https://et.com/1",
            },
            {
                "title": "TCS signs $1B cloud deal - Livemint",
                "url": "https://livemint.com/2",
            },
            {
                "title": "Infosys quarterly numbers - Reuters",
                "url": "https://reuters.com/3",
            },
        ]
        deduped = deduplicate_stories(stories)
        assert len(deduped) == 2
        assert deduped[0]["url"] == "https://et.com/1"
        assert deduped[1]["url"] == "https://reuters.com/3"

    @pytest.mark.parametrize(
        ("input_query", "expected_query"),
        [
            ("TCS", "TCS NSE"),
            ("reliance", "RELIANCE NSE"),
            ("INFY.NS", "INFY NSE"),
            ("HDFCBANK-EQ", "HDFCBANK NSE"),
            ("NIFTY", "NIFTY 50"),
            ("BANKNIFTY", "BANK NIFTY"),
            ("Tata Motors Limited", "Tata Motors Limited"),
            ("", None),
            ("   ", None),
            ("$$$", None),
            (None, None),
            (12345, None),
        ],
    )
    def test_normalize_news_query(
        self, input_query: object, expected_query: Optional[str]
    ) -> None:
        assert normalize_news_query(input_query) == expected_query


class TestGetMarketNews:
    """Test suite for get_market_news tool function with mocked provider responses."""

    def test_normal_feed(self) -> None:
        """Test parsing of a standard news feed with all fields present."""
        with patch("tools.news_tool.fetch_news_feed", return_value=SAMPLE_NORMAL_RSS):
            news = get_market_news("TCS")

            assert len(news) == 2
            first = news[0]
            assert "TCS signs multi-year digital transformation deal" in first["title"]
            assert first["source"] == "The Economic Times"
            assert "2026-10-01 10:30:00 UTC" in first["published_at"]
            assert first["url"] == "https://economictimes.indiatimes.com/tech/tcs-deal/1001"
            assert (
                first["summary"]
                == "Tata Consultancy Services secures a large strategic engagement."
            )

    def test_duplicate_stories_removal(self) -> None:
        """Test that syndicated duplicate headlines are automatically pruned."""
        with patch("tools.news_tool.fetch_news_feed", return_value=SAMPLE_DUPLICATE_RSS):
            news = get_market_news("TCS")

            assert len(news) == 2
            titles = [n["title"] for n in news]
            assert any("TCS signs multi-year digital transformation deal" in t for t in titles)
            assert any("Distinct Headline for Reliance" in t for t in titles)

    def test_missing_summary(self) -> None:
        """Test stories with omitted description tag return summary as None."""
        with patch(
            "tools.news_tool.fetch_news_feed", return_value=SAMPLE_MISSING_FIELDS_RSS
        ):
            news = get_market_news("TCS")

            assert len(news) == 2
            no_summary_story = next(
                n for n in news if "missing summary" in n["title"]
            )
            assert no_summary_story["summary"] is None
            assert no_summary_story["source"] == "Business Standard"

    def test_missing_date(self) -> None:
        """Test stories with omitted pubDate tag return published_at as None."""
        with patch(
            "tools.news_tool.fetch_news_feed", return_value=SAMPLE_MISSING_FIELDS_RSS
        ):
            news = get_market_news("TCS")

            no_date_story = next(n for n in news if "missing date" in n["title"])
            assert no_date_story["published_at"] is None
            assert no_date_story["summary"] == "This story has no pubDate tag."

    def test_empty_feed(self) -> None:
        """Test empty XML feed returns an empty list gracefully."""
        empty_rss = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        with patch("tools.news_tool.fetch_news_feed", return_value=empty_rss):
            news = get_market_news("TCS")
            assert news == []

    def test_provider_failure(self) -> None:
        """Test graceful error handling when news provider raises network/timeout exceptions."""
        with patch(
            "tools.news_tool.fetch_news_feed",
            side_effect=Exception("Connection timed out"),
        ):
            # Also mock yfinance fallback to verify clean [] return
            with patch("tools.news_tool.yf.Ticker") as mock_ticker_cls:
                mock_inst = MagicMock()
                mock_inst.news = []
                mock_ticker_cls.return_value = mock_inst

                news = get_market_news("TCS")
                assert news == []

    @pytest.mark.parametrize(
        "invalid_input",
        ["", "   ", "$$$", None, 12345],
    )
    def test_invalid_company(self, invalid_input: object) -> None:
        """Test invalid company names or ticker strings return empty list."""
        news = get_market_news(invalid_input)  # type: ignore[arg-type]
        assert news == []

    def test_recent_stories_first_order(self) -> None:
        """Test sorting places newer articles before older ones."""
        with patch("tools.news_tool.fetch_news_feed", return_value=SAMPLE_NORMAL_RSS):
            news = get_market_news("TCS")
            assert len(news) == 2
            # 2026-10-01 should precede 2026-09-30
            assert news[0]["published_at"] > news[1]["published_at"]

    def test_limit_parameter(self) -> None:
        """Test that the limit parameter restricts the number of returned stories."""
        with patch("tools.news_tool.fetch_news_feed", return_value=SAMPLE_NORMAL_RSS):
            news = get_market_news("TCS", limit=1)
            assert len(news) == 1

    def test_unwrap_direct_url_and_namespaced_source(self) -> None:
        """Test unwrapping direct external publisher URLs from redirect links and extracting namespaced source."""
        bing_sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item xmlns:ns0="https://www.bing.com/news">
              <title>HDFC Bank CEO transition: Nomura sees 32% upside</title>
              <link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=1&amp;url=https%3a%2f%2fwww.livemint.com%2fmarket%2fhdfc-bank-ceo-123.html&amp;c=1</link>
              <pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate>
              <description>Anup Bagchi appointed as MD and CEO. Nomura sets target price with substantial upside potential.</description>
              <ns0:Source>Livemint</ns0:Source>
            </item>
          </channel>
        </rss>
        """
        stories = parse_rss_feed(bing_sample_rss)
        assert len(stories) == 1
        story = stories[0]
        assert story["title"] == "HDFC Bank CEO transition: Nomura sees 32% upside"
        assert story["url"] == "https://www.livemint.com/market/hdfc-bank-ceo-123.html"
        assert story["source"] == "Livemint"
        assert "Anup Bagchi appointed as MD and CEO" in story["summary"]

    def test_extract_publisher_from_url(self) -> None:
        """Test extracting publisher names from standard financial domains."""
        assert extract_publisher_from_url("https://www.livemint.com/news/1") == "Livemint"
        assert extract_publisher_from_url("https://economictimes.indiatimes.com/tech/2") == "The Economic Times"
        assert extract_publisher_from_url("https://www.moneycontrol.com/stocks/3") == "Moneycontrol"
        assert extract_publisher_from_url("https://unknownfinanceblog.com/post") == "Unknownfinanceblog"
        assert extract_publisher_from_url(None) is None
