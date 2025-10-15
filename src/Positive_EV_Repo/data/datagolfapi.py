
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
DATAGOLF_KEY = os.getenv("DATAGOLF_API_KEY")

# Output directories
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# API base URL
BASE_URL = "https://feeds.datagolf.com"

# Rate limiting
SLEEP_SHORT = 0.3
SLEEP_LONG = 1.0

# ------------- Helper Functions ----------------

def get(endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Dict[str, Any]:
    """GET request with retries."""
    if params is None:
        params = {}
    params.setdefault("file_format", "json")

    if not DATAGOLF_KEY:
        raise RuntimeError("Missing DATAGOLF_API_KEY")

    params["key"] = DATAGOLF_KEY
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"[API] Rate limit hit. Sleeping...")
                time.sleep(SLEEP_LONG * 2)
            else:
                last_err = f"Error {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)

        print(f"[API] Retry {attempt}/{retries} failed. Sleeping...")
        time.sleep(SLEEP_SHORT * attempt)

    raise RuntimeError(f"Request failed: {last_err}")


def normalize_payload(payload: Any) -> pd.DataFrame:
    """Normalize JSON payload into a DataFrame."""
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for key in ["data", "events", "players", "results", "odds", "rounds", "baseline", "tournaments"]:
            if key in payload:
                return pd.json_normalize(payload[key])
        return pd.json_normalize([payload])
    return pd.DataFrame()


# ------------- Data Fetching Functions ----------------

def fetch_schedule(tour: str = "pga", year: int = 2023) -> pd.DataFrame:
    """Fetch raw event list that includes event_id (needed for raw_rounds)."""
    print(f"\n[FETCH] Raw Event List for {tour.upper()} {year}")
    payload = get("historical-raw-data/event-list", {"tour": tour, "year": year})
    df = normalize_payload(payload)
    df["year"] = year
    df["tour"] = tour 
    return df



def fetch_raw_rounds(events_df: pd.DataFrame, id_col: str = "event_id") -> pd.DataFrame:
    """Fetch raw round-level strokes-gained data for all events."""
    print(f"\n[FETCH] Raw Rounds for {len(events_df)} events")
    rounds_data = []

    for idx, row in events_df.iterrows():
        event_id = row.get(id_col)
        year = row.get("year")
        event_name = row.get("event_name", f"Event {idx}")
        print(f"  > [{idx+1}/{len(events_df)}] {event_name} ({event_id})")

        try:
            payload = get("historical-raw-data/rounds", {"event_id": event_id, "year": year})
            df = normalize_payload(payload)
            if not df.empty:
                df["event_id"] = event_id
                df["year"] = year
                rounds_data.append(df)
                print(f"    ✓ {len(df)} rows")
            else:
                print(f"    ✗ No data")
            time.sleep(SLEEP_SHORT)
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            time.sleep(SLEEP_LONG)

    if rounds_data:
        full_rounds = pd.concat(rounds_data, ignore_index=True)
        print(f"\n✅ Total rounds fetched: {full_rounds.shape}")
        return full_rounds

    print("\n⚠️ No round data retrieved.")
    return pd.DataFrame()


def fetch_player_list() -> pd.DataFrame:
    """Fetch the full list of players and metadata."""
    print("\n[FETCH] Player List")
    payload = get("get-player-list")
    df = normalize_payload(payload)
    print(f"✅ Retrieved {len(df)} players")
    return df


# ------------- Main Execution Pipeline ----------------

def run_full_pipeline(years: List[int], tour: str = "pga") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full DataGolf pipeline: schedule, rounds, players."""
    print(f"\n=== Running DataGolf Pipeline for {tour.upper()} {years} ===")

    # Get events
    all_events = pd.concat([fetch_schedule(tour, yr) for yr in years], ignore_index=True)
    print(f"\n✅ Retrieved {len(all_events)} events")

    # Get rounds
    if "event_id" not in all_events.columns:
        raise ValueError("Missing 'event_id' in schedule data.")
    rounds = fetch_raw_rounds(all_events)

    # Get player list
    players = fetch_player_list()

    # Optionally save to files
    all_events.to_parquet(INTERIM_DIR / "schedule.parquet", index=False)
    rounds.to_parquet(INTERIM_DIR / "rounds.parquet", index=False)
    players.to_parquet(INTERIM_DIR / "players.parquet", index=False)

    print("\n✅ All data saved to data/interim/")
    return all_events, rounds, players


if __name__ == "__main__":
    events_df, rounds_df, players_df = run_full_pipeline([2023, 2024], tour="pga")
