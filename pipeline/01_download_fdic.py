"""
Download FDIC Summary of Deposits (SOD) branch-level data for 1994-2005.

API endpoint : https://api.fdic.gov/banks/sod
Filter       : YEAR:<yyyy>
Output       : data/fdic_deposits_1994_2005.csv  (~1M rows x 8 columns)

Field name mapping (API -> output column):
  YEAR      -> year        CERT      -> CERT
  RSSDHCR   -> RSSDHCR    STALPBR   -> STALPBR
  STCNTYBR  -> STCNTYBR   NAMEBR    -> UNINAME
  UNINUMBR  -> UNINUM      RSSDID    -> RSSDID

Note: STCNTYBR is the 5-digit state-county FIPS code for the branch.
      RSSDID is the Fed RSSD institution ID, NOT a county FIPS code.
"""
from pathlib import Path
import requests
import pandas as pd
import time

ROOT     = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "fdic_deposits_1994_2005.csv"

API_URL   = "https://api.fdic.gov/banks/sod"
YEARS     = range(1994, 2006)
PAGE_SIZE = 10000
FIELD_MAP = {
    "YEAR":     "year",
    "CERT":     "CERT",
    "RSSDHCR":  "RSSDHCR",
    "STALPBR":  "STALPBR",
    "STCNTYBR": "STCNTYBR",
    "NAMEBR":   "UNINAME",
    "UNINUMBR": "UNINUM",
    "RSSDID":   "RSSDID",
}
API_FIELDS  = ",".join(FIELD_MAP)
RETRY_WAIT  = 10   # seconds to pause after a 429
MAX_RETRIES = 5
PAGE_SLEEP  = 0.5  # polite delay between pages
YEAR_SLEEP  = 1.0  # delay between years


def get_page(year, offset):
    params = {
        "filters": f"YEAR:{year}",
        "fields":  API_FIELDS,
        "limit":   PAGE_SIZE,
        "offset":  offset,
    }
    for attempt in range(MAX_RETRIES):
        r = requests.get(API_URL, params=params, timeout=120)
        if r.status_code == 429:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"    429 rate-limited; waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries for year={year} offset={offset}")


def fetch_year(year):
    frames, offset = [], 0
    while True:
        j       = get_page(year, offset)
        records = [item["data"] for item in j["data"]]
        chunk   = pd.DataFrame(records)
        frames.append(chunk)
        print(f"  {year}: offset {offset:>6} -> {len(chunk):>5} rows  "
              f"(total: {j['meta']['total']:,})")
        if offset + len(chunk) >= j["meta"]["total"]:
            break
        offset += PAGE_SIZE
        time.sleep(PAGE_SLEEP)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    all_frames = []
    for yr in YEARS:
        print(f"\nFetching {yr} ...")
        df = fetch_year(yr)
        print(f"  => {len(df):,} rows for {yr}")
        all_frames.append(df)
        time.sleep(YEAR_SLEEP)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.rename(columns=FIELD_MAP)
    cols     = ["year"] + [c for c in FIELD_MAP.values() if c != "year"]
    combined = combined[[c for c in cols if c in combined.columns]]

    print(f"\nShape  : {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
