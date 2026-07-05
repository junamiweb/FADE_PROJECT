"""Accumulating news archive downloader (GDELT) — past data only, honest timestamps.

FADE stays rule-based and offline at runtime. News is treated exactly like any
other atom source: a historical time series with REAL publication timestamps,
aggregated per day, that we can align to price bars and put through the same
strict holdout test. If it does not survive out-of-sample, it gets thrown out.

Source: GDELT DOC 2.0 API (https://api.gdeltproject.org/api/v2/doc/doc).
  * mode=timelinetone -> average article tone per day for a query (sentiment).
  * mode=timelinevol  -> volume of matching coverage per day (attention).
Both carry the article's own publication date, so there is no look-ahead: the
value at day D reflects only what was published on day D.

Anti-leakage note: tone is GDELT's fixed lexicon score computed at ingest time,
not a model trained on future outcomes.

Accumulation: results append into news_btc.csv, de-duplicated by date. Re-run
any time to extend the archive forward — "give the tool the past, slowly".

Usage:
    python download_news.py                      # full history, default query
    python download_news.py --query "bitcoin OR cryptocurrency"
    python download_news.py --start 2017-01-01 --out news_btc.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = "bitcoin"
# GDELT asks for <= 1 request / 5s. We stay well clear (10s) to avoid the 429
# throttle entirely, and on any 429 we cool down HARD and long — getting the
# client temporarily blocked is far worse than being slow.
MIN_INTERVAL_S = 10.0
MAX_RETRIES = 8
BACKOFF_START_S = 30.0    # first cooldown after a 429
BACKOFF_MAX_S = 300.0     # cap a single cooldown at 5 minutes


def _get(params: dict) -> dict:
    """GET with a very polite, long backoff on the 429 rate limiter."""
    wait = BACKOFF_START_S
    resp = None
    for attempt in range(MAX_RETRIES):
        resp = requests.get(GDELT, params=params, timeout=90)
        if resp.status_code == 200 and resp.text.strip().startswith("{"):
            return resp.json()
        # 429 or transient: cool down long before retrying.
        print(f"    (status {resp.status_code}; cooldown {wait:.0f}s, "
              f"retry {attempt + 1}/{MAX_RETRIES})", flush=True)
        time.sleep(wait)
        wait = min(wait * 1.6, BACKOFF_MAX_S)
    raise RuntimeError(f"GDELT failed after {MAX_RETRIES} retries "
                       f"(last status {resp.status_code if resp else '?'}): "
                       f"{resp.text[:200] if resp else ''}")


def _timeline(query: str, mode: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one timeline (tone or volume) for a [start, end] date window."""
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d%H%M%S"),
    }
    data = _get(params)
    series = data.get("timeline", [])
    if not series:
        return pd.DataFrame(columns=["date", mode])
    points = series[0].get("data", [])
    rows = [(p["date"], p["value"]) for p in points]
    df = pd.DataFrame(rows, columns=["date", mode])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _fetch_chunk(query: str, lo: str, hi: str) -> pd.DataFrame:
    """One year chunk: tone + volume merged into a daily frame."""
    tone = _timeline(query, "timelinetone", lo, hi)
    time.sleep(MIN_INTERVAL_S)
    vol = _timeline(query, "timelinevol", lo, hi)
    time.sleep(MIN_INTERVAL_S)
    if tone.empty:
        return pd.DataFrame(columns=["date", "news_tone", "news_volume"])
    out = tone.rename(columns={"timelinetone": "news_tone"})
    if not vol.empty:
        out = out.merge(vol.rename(columns={"timelinevol": "news_volume"}),
                        on="date", how="outer")
    else:
        out["news_volume"] = float("nan")
    out = out.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def fetch_news(query: str, start: str, end: str | None = None,
               out_path: str | None = None) -> pd.DataFrame:
    """Build a daily news frame (tone + volume), saving after each year chunk.

    GDELT returns daily resolution for long windows; we page year-by-year to
    keep each response dense and stay within API limits. Each chunk is written
    to ``out_path`` immediately (incremental accumulation), and a chunk that
    fails after all retries is skipped rather than aborting the whole run — so
    partial progress is never lost and a re-run resumes cleanly.
    """
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    cursor = pd.Timestamp(start, tz="UTC")

    combined = pd.DataFrame()
    while cursor < end_ts:
        chunk_end = min(cursor + pd.DateOffset(years=1), end_ts)
        print(f"  {cursor.date()} -> {chunk_end.date()} ...", flush=True)
        try:
            chunk = _fetch_chunk(query, cursor.isoformat(), chunk_end.isoformat())
        except RuntimeError as exc:
            print(f"    ! chunk failed, skipping: {exc}", flush=True)
            cursor = chunk_end
            continue
        print(f"    +{len(chunk)} days", flush=True)
        if out_path and not chunk.empty:
            combined = accumulate(chunk, out_path)
        cursor = chunk_end

    return combined


def accumulate(new: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Merge freshly fetched rows into the existing archive (dedup by date)."""
    path = Path(out_path)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    combined = (combined.drop_duplicates("date", keep="last")
                .sort_values("date").reset_index(drop=True))
    combined.to_csv(path, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Accumulating GDELT news archive")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="news_btc.csv")
    args = parser.parse_args()

    print(f"GDELT news archive: query={args.query!r} from {args.start}")
    combined = fetch_news(args.query, args.start, args.end, out_path=args.out)
    span = (f"{combined['date'].min().date()} -> {combined['date'].max().date()}"
            if not combined.empty else "empty")
    print(f"  archive {args.out}: {len(combined):,} rows   [{span}]")


if __name__ == "__main__":
    main()
