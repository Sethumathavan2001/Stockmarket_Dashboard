"""
Company.code -> Yahoo Finance ticker mapping.

Company.code is not guaranteed to match the Yahoo ticker (e.g. NSE tickers
need a ".NS" suffix, some codes differ entirely). Configure exceptions in
Django settings:

    YFINANCE_TICKER_MAP = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "MYWEIRDCODE": "ACTUALTICKER.NS",
    }
    YFINANCE_DEFAULT_SUFFIX = ".NS"   # applied when code has no explicit mapping

If a code isn't in the map, it's assumed to already need the default suffix
appended (skipped if it's already present).
"""
from django.conf import settings

DEFAULT_SUFFIX = ".NS"


def get_yahoo_ticker(company_code: str) -> str:
    mapping = getattr(settings, "YFINANCE_TICKER_MAP", {})
    if company_code in mapping:
        return mapping[company_code]

    suffix = getattr(settings, "YFINANCE_DEFAULT_SUFFIX", DEFAULT_SUFFIX)
    if company_code.endswith(suffix) or "." in company_code:
        return company_code
    return f"{company_code}{suffix}"
