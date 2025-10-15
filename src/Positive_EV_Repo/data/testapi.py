"""
FINAL WORKING DataGolf Fetcher - Get ALL 2024-2025 PGA Events
"""
import requests
import pandas as pd
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DATAGOLF_API_KEY")

print("="*70)
print("FETCHING ALL PGA OUTRIGHTS FROM DATAGOLF")
print("="*70)

BASE_URL = "https://feeds.datagolf.com"

# ============================================================================
# STEP 1: Get schedule (all available events)
# ============================================================================
print("\n[STEP 1] Getting PGA Tour schedule...")

schedule_url = f"{BASE_URL}/get-schedule"
params = {"tour": "pga", "file_format": "json", "key": API_KEY}

r = requests.get(schedule_url, params=params, timeout=30)

if r.status_code != 200:
    print(f"❌ Failed to get schedule: {r.status_code}")
    exit(1)

data = r.json()

if isinstance(data, dict) and 'schedule' in data:
    all_events = data['schedule']
elif isinstance(data, list):
    all_events = data
else:
    print(f"❌ Unexpected format: {list(data.keys())}")
    exit(1)

print(f"✓ Got {len(all_events)} events from schedule")

# ============================================================================
# STEP 2: Fetch odds for ALL events
# ============================================================================
print(f"\n[STEP 2] Fetching odds for ALL {len(all_events)} events...")
print("(This will take 5-10 minutes)\n")

odds_url = f"{BASE_URL}/historical-odds/outrights"
all_odds_rows = []

for i, event in enumerate(all_events, 1):
    event_id = event.get('event_id') or event.get('id')
    event_name = event.get('event_name') or event.get('name')
    start_date = event.get('start_date')
    
    if not event_id:
        continue
    
    print(f"  [{i}/{len(all_events)}] {event_name} (ID: {event_id})...", end=" ")
    
    # Fetch for both DraftKings and FanDuel, win market only
    for book in ["draftkings", "fanduel"]:
        
        params = {
            "tour": "pga",
            "event_id": event_id,
            "market": "win",
            "book": book,
            "odds_format": "american",
            "file_format": "json",
            "key": API_KEY
        }
        
        try:
            r = requests.get(odds_url, params=params, timeout=30)
            
            if r.status_code != 200:
                continue
            
            data = r.json()
            
            if isinstance(data, dict) and 'odds' in data:
                odds_list = data['odds']
                
                for odds_row in odds_list:
                    odds_row['event_id'] = event_id
                    odds_row['event_name'] = event_name
                    odds_row['start_date'] = start_date
                    odds_row['tour'] = 'pga'
                    odds_row['book'] = book
                    odds_row['market'] = 'win'
                    all_odds_rows.append(odds_row)
            
            time.sleep(0.3)  # API rate limiting
            
        except Exception as e:
            continue
    
    # Show progress
    if all_odds_rows:
        print(f"✓ {len(all_odds_rows):,} total rows")
    else:
        print("⚠️")

# ============================================================================
# STEP 3: Clean & Save
# ============================================================================
print("\n" + "="*70)
print("PROCESSING DATA")
print("="*70)

if len(all_odds_rows) == 0:
    print("\n❌ NO DATA FETCHED!")
    exit(1)

# Convert to DataFrame
df = pd.DataFrame(all_odds_rows)

print(f"\nInitial rows: {len(df):,}")

# Extract year from close_time timestamp
df['close_time_dt'] = pd.to_datetime(df['close_time'])
df['year'] = df['close_time_dt'].dt.year

print(f"\n✓ Extracted year from timestamps")
print(f"  Year distribution: {df['year'].value_counts().sort_index().to_dict()}")

# Save
output_path = Path("data/interim/final_dataset.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

file_size_mb = output_path.stat().st_size / 1024 / 1024

print(f"\n✅ SUCCESS!")
print(f"  Saved to: {output_path}")
print(f"  File size: {file_size_mb:.2f} MB")
print(f"  Total rows: {len(df):,}")
print(f"  Events: {df['event_name'].nunique()}")
print(f"  Players: {df['player_name'].nunique()}")
print(f"  Books: {df['book'].value_counts().to_dict()}")
print(f"  Years: {df['year'].value_counts().sort_index().to_dict()}")

print(f"\nSample data:")
print(df[['player_name', 'close_odds', 'outcome', 'event_name', 'year', 'book']].head(5))

if len(df) >= 5000:
    print(f"\n✅ EXCELLENT! You have {len(df):,} rows")
    print(f"   This is perfect for your project!")
    print(f"\n🎯 NEXT STEP: Run 01_DataCleaning.ipynb")
elif len(df) >= 1000:
    print(f"\n✓ Good! You have {len(df):,} rows")
    print(f"  This is enough to complete the project")
    print(f"\n🎯 NEXT STEP: Run 01_DataCleaning.ipynb")
else:
    print(f"\n⚠️ Only {len(df):,} rows")
    print(f"   Still usable, but limited data")

print("="*70)