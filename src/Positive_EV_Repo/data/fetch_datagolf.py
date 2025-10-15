# src/Positive_EV_Repo/data/fetch_datagolf.py
# src/Positive_EV_Repo/data/fetch_datagolf.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# ===================== Config (edit these) =====================
YEARS: List[int] = [2023, 2024]      # safer default; expand later
TOUR: str = "pga"
BOOKS: List[str] = ["draftkings", "fanduel"]
MARKETS: List[str] = ["win", "top_10"]   # DataGolf uses underscores, e.g., top_10
ODDS_FORMAT: str = "american"            # or "percent", "decimal", "fraction"
SLEEP_SHORT: float = 0.25
SLEEP_LONG: float = 0.6
# ===============================================================

# Load DATAGOLF_API_KEY from .env at project root
load_dotenv()
DATAGOLF_KEY = os.getenv("DATAGOLF_API_KEY")

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR = Path("data/interim"); INTERIM_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://feeds.datagolf.com"

# --- Endpoints from DataGolf docs ---
ENDPOINTS: Dict[str, str] = {
    # General Use
    "player_list": f"{BASE}/get-player-list",
    "schedule": f"{BASE}/get-schedule",
    "field_updates": f"{BASE}/field-updates",

    # Predictions
    "pre_tournament": f"{BASE}/preds/pre-tournament",
    "pre_tournament_archive": f"{BASE}/preds/pre-tournament-archive",
    "live_tournament_stats": f"{BASE}/preds/live-tournament-stats",

    # Historical Raw Data
    "raw_event_list": f"{BASE}/historical-raw-data/event-list",
    "raw_rounds": f"{BASE}/historical-raw-data/rounds",

    # Historical Odds
    "hist_odds_event_list": f"{BASE}/historical-odds/event-list",
    "hist_outrights": f"{BASE}/historical-odds/outrights",
}
# ------------------------------------


def _get(url: str, params: Optional[Dict[str, Any]] = None,
         retries: int = 3, pause: float = 0.75) -> Dict[str, Any]:
    """
    GET wrapper with retries. Automatically appends key and file_format=json by default.
    """
    if params is None:
        params = {}

    params.setdefault("file_format", "json")

    if DATAGOLF_KEY:
        params.setdefault("key", DATAGOLF_KEY)
    else:
        raise RuntimeError(
            "DATAGOLF_API_KEY not found. Create a .env file at project root with DATAGOLF_API_KEY=YOUR_KEY"
        )

    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, timeout=45)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as e:
                raise RuntimeError(f"JSON parse error: {e}\nText: {r.text[:500]}")
        last_err = f"({r.status_code}) {r.text[:300]}"
        time.sleep(pause * attempt)  # backoff
    raise RuntimeError(f"Request failed after retries: {last_err or 'unknown error'}")


def _save_raw(name: str, payload: Any) -> Path:
    ts = int(time.time())
    p = RAW_DIR / f"{name}_{ts}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return p


def _normalize(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for k in ("data", "events", "tournaments", "players", "odds", "results"):
            if k in payload and isinstance(payload[k], list):
                return pd.json_normalize(payload[k])
        return pd.json_normalize(payload)
    return pd.DataFrame()


def _save_table(df: pd.DataFrame, path: Path) -> Tuple[str, Path]:
    """
    Save as parquet if an engine is present; otherwise save CSV.
    Returns (format, path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # try parquet
        import pyarrow  # noqa: F401
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return ("parquet", path.with_suffix(".parquet"))
    except Exception:
        # fallback to CSV
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return ("csv", csv_path)


def fetch_endpoint(name: str, **params) -> pd.DataFrame:
    """
    Generic fetch → saves raw JSON → returns a DataFrame.
    """
    if name not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint '{name}'. Known: {list(ENDPOINTS)}")

    payload = _get(ENDPOINTS[name], params)
    _save_raw(name, payload)
    return _normalize(payload)


# ----------------------- Bulk helpers -----------------------

def fetch_events(years: List[int], tour: str = "pga") -> pd.DataFrame:
    """
    Pull historical event list for the given years.
    If a yearly call fails (500), fall back to calling without year and filtering.
    Dedupes on ['event_id','year'] and saves to data/interim/events.(parquet|csv)
    """
    frames: List[pd.DataFrame] = []
    for yr in years:
        try:
            ev = fetch_endpoint("raw_event_list", tour=tour, year=yr)
            if not ev.empty:
                ev["tour"] = tour
                frames.append(ev)
            print(f"[events] year={yr} -> {len(ev)} rows")
            time.sleep(SLEEP_SHORT)
        except Exception as e:
            # fallback: try without year, then filter
            print(f"[events] year={yr} ERROR: {e}; trying without year...")
            try:
                all_ev = fetch_endpoint("raw_event_list", tour=tour)
                if not all_ev.empty and "year" in all_ev.columns:
                    evf = all_ev[all_ev["year"].astype(int) == int(yr)].copy()
                    evf["tour"] = tour
                    frames.append(evf)
                    print(f"[events] fallback year={yr} -> {len(evf)} rows")
                time.sleep(SLEEP_LONG)
            except Exception as e2:
                print(f"[events] fallback year={yr} ERROR: {e2}")

    if not frames:
        print("[events] No rows returned.")
        return pd.DataFrame()

    events = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["event_id", "year"])
    fmt, outp = _save_table(events, INTERIM_DIR / "events")
    print(f"✅ saved events -> {outp} ({events.shape}, {fmt})")
    return events


def fetch_rounds_for_years(years: List[int], tour: str = "pga") -> pd.DataFrame:
    """
    For each event in the given years, fetch strokes-gained rounds and save to data/interim/rounds.(parquet|csv).
    """
    events = fetch_events(years, tour=tour)
    if events.empty:
        print("[rounds] No events found.")
        return pd.DataFrame()

    if not {"event_id", "year"}.issubset(events.columns):
        raise RuntimeError("Event list missing required columns 'event_id' and/or 'year'.")

    frames: List[pd.DataFrame] = []
    for _, row in events.iterrows():
        eid = int(row["event_id"])
        yr = int(row["year"])
        try:
            rd = fetch_endpoint("raw_rounds", tour=tour, event_id=eid, year=yr)
            if not rd.empty:
                rd["event_id"] = eid
                rd["year"] = yr
                rd["tour"] = tour
                frames.append(rd)
            print(f"[rounds] event_id={eid} year={yr} -> {len(rd)} rows")
            time.sleep(SLEEP_SHORT)
        except Exception as e:
            print(f"[rounds] event_id={eid} year={yr} ERROR: {e}")

    if not frames:
        print("[rounds] No rounds retrieved.")
        return pd.DataFrame()

    rounds = pd.concat(frames, ignore_index=True)
    fmt, outp = _save_table(rounds, INTERIM_DIR / "rounds")
    print(f"✅ saved rounds -> {outp} ({rounds.shape}, {fmt})")
    return rounds


def fetch_hist_outrights_for_years(
    years: List[int],
    tour: str = "pga",
    books: List[str] = ("draftkings",),
    markets: List[str] = ("win",),
    odds_format: str = "american",
) -> pd.DataFrame:
    """
    Pull historical outrights for multiple years/books/markets and save to data/interim/hist_outrights.(parquet|csv)
    """
    frames: List[pd.DataFrame] = []
    for yr in years:
        for book in books:
            for market in markets:
                try:
                    df = fetch_endpoint(
                        "hist_outrights",
                        tour=tour,
                        year=yr,
                        book=book,
                        market=market,          # NOTE: must be e.g. 'top_10', not 'top10'
                        odds_format=odds_format,
                    )
                    if not df.empty:
                        df["tour"] = tour
                        df["year"] = yr
                        df["book"] = book
                        df["market"] = market
                        frames.append(df)
                    print(f"[hist_outrights] year={yr} book={book} market={market} -> {len(df)} rows")
                    time.sleep(SLEEP_SHORT)
                except Exception as e:
                    print(f"[hist_outrights] year={yr} book={book} market={market} ERROR: {e}")

    if not frames:
        print("[hist_outrights] No data retrieved.")
        return pd.DataFrame()

    odds = pd.concat(frames, ignore_index=True)
    fmt, outp = _save_table(odds, INTERIM_DIR / "hist_outrights")
    print(f"✅ saved hist_outrights -> {outp} ({odds.shape}, {fmt})")
    return odds


# ------------------ simple sanity checker -------------------

def example_fetch_all() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    dfs["player_list"] = fetch_endpoint("player_list")
    dfs["schedule"] = fetch_endpoint("schedule", tour=TOUR)
    dfs["pre_tournament"] = fetch_endpoint("pre_tournament", tour=TOUR, odds_format="percent")
    dfs["hist_outrights_sample"] = fetch_endpoint(
        "hist_outrights", tour=TOUR, market="win", book="draftkings", year=YEARS[-1], odds_format=ODDS_FORMAT
    )
    return dfs


# --------------- Save results (labels) helper ---------------

def save_results(df: pd.DataFrame, path: str = "data/raw/results.parquet") -> None:
    pathp = Path(path)
    pathp.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(pathp, index=False)
        print(f"✅ Saved results to {pathp} (parquet)")
    except Exception:
        csvp = pathp.with_suffix(".csv")
        df.to_csv(csvp, index=False)
        print(f"✅ Saved results to {csvp} (csv)")


# -------------------------- main ----------------------------

if __name__ == "__main__":
    print("=== sanity checks ===")
    sanity = example_fetch_all()
    for k, v in sanity.items():
        print(f"{k}: {v.shape}")

    print("\n=== bulk: events & rounds ===")
    events_df = fetch_events(YEARS, tour=TOUR)
    print(f"events: {events_df.shape}")

    rounds_df = fetch_rounds_for_years(YEARS, tour=TOUR)
    print(f"rounds: {rounds_df.shape}")

    print("\n=== bulk: historical outrights ===")
    odds_df = fetch_hist_outrights_for_years(YEARS, tour=TOUR, books=BOOKS, markets=MARKETS, odds_format=ODDS_FORMAT)
    print(f"hist_outrights: {odds_df.shape}")
