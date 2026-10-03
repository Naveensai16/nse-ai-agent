import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from services.market_assistant_service import run_market_assistant

TEST_QUERIES = [
    "Anything interesting happening in Tata Power today?",
    "Why are people talking about Tata Motors right now?",
    "Give me the quick version on Reliance.",
    "Is YES Bank having a good day or a bad one?",
    "What changed in Infosys since the previous trading session?",
    "Has TCS been strong lately?",
    "What should I look at before judging HDFC Bank?",
    "Show me the important numbers for ICICI Bank, not the full story.",
    "How far is Tata Steel from the best price it has seen this year?",
    "Is ITC closer to its yearly high or yearly low?",
    "Which stocks are almost breaking their yearly highs?",
    "Find shares that are nowhere near their yearly highs.",
    "Which names are recovering from their yearly lows?",
    "Show stocks sitting within 3% of their 52-week high.",
    "Show stocks sitting within 3% of their 52-week low.",
    "Which shares made fresh highs but are red today?",
    "Which shares are near yearly lows but green today?",
    "Show me strong stocks that have not yet made a 52-week high.",
    "Which NSE stocks are closest to a breakout?",
    "What companies are trading near their yearly extremes?",
    "Who is outperforming the market today?",
    "Who is underperforming the market today?",
    "Which stocks are moving unusually fast today?",
    "Show me stocks with the biggest intraday percentage move.",
    "Which stocks opened weak but recovered?",
    "Which stocks opened strong but gave up gains?",
    "Show me the biggest positive movers excluding banks.",
    "Show me the biggest losers excluding IT stocks.",
    "Which large-cap stocks are leading today?",
    "Which large-cap stocks are lagging today?",
    "What is driving Nifty today?",
    "Is Bank Nifty stronger than Nifty today?",
    "Which index is performing better, Nifty or Bank Nifty?",
    "Did Nifty have a strong opening?",
    "Is the broader market stronger than Nifty?",
    "Are more NSE stocks rising than falling?",
    "Give me a one-minute summary of today's market.",
    "What are the three biggest things happening in the Indian market today?",
    "Which sectors are carrying the market today?",
    "Which sectors are dragging the market today?",
    "Is IT stronger than banking today?",
    "Compare the auto sector with pharma today.",
    "Which sector improved the most this week?",
    "Tata Power or NTPC — which looks cheaper on valuation?",
    "TCS or Infosys — which is trading closer to its 52-week high?",
    "HDFC Bank versus ICICI Bank — which moved more today?",
    "Are the markets open right now?",
    "How fresh is the data you are showing me?",
]

def main():
    print(f"Executing {len(TEST_QUERIES)} conversational queries...\n" + "="*80)
    passed = 0
    start_all = time.time()
    for idx, q in enumerate(TEST_QUERIES, 1):
        t0 = time.time()
        res = run_market_assistant(q, skip_synthesis=True, skip_ollama_llm=True)
        dur = time.time() - t0
        intent = res.get("intent", "UNKNOWN")
        response = res.get("response", "")
        # Check that response directly answers with meaningful content
        has_content = len(response.strip()) > 30 and ("###" in response or "•" in response or "*" in response)
        status = "PASSED" if has_content else "FAILED"
        if has_content:
            passed += 1
        first_line = response.strip().split("\n")[0] if response else "EMPTY"
        print(f"[{idx:02d}/{len(TEST_QUERIES):02d}] {status} ({dur:.2f}s) | Intent: {intent:<25} | Q: '{q[:40]}...'")
        print(f"     -> {first_line[:75]}")

    total_dur = time.time() - start_all
    print("="*80)
    print(f"Summary: {passed}/{len(TEST_QUERIES)} Passed in {total_dur:.2f}s ({passed/len(TEST_QUERIES)*100:.1f}%)")

if __name__ == "__main__":
    main()
