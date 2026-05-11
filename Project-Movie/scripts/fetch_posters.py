"""
fetch_posters.py
Re-fetches poster URLs from TMDB for every movie in the dataset.
Run once: python fetch_posters.py
"""
from dotenv import load_dotenv
import os

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
import pandas as pd
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────


TMDB_SEARCH   = "https://api.themoviedb.org/3/search/movie"
POSTER_BASE   = "https://image.tmdb.org/t/p/w500"
CSV_INPUT     = "data/final_indian_movies_dataset.csv"
CSV_OUTPUT    = "data/final_indian_movies_dataset.csv"

SLEEP_BETWEEN = 1.5   # gap between every request
MAX_RETRIES   = 5     # retries per request
BACKOFF       = 2.0   # 2s → 4s → 8s → 16s → 32s


# ─────────────────────────────────────────────
# SESSION WITH AUTO-RETRY
# ─────────────────────────────────────────────

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({
        "Connection": "keep-alive",
        "User-Agent": "CineMatchIndia/1.0",
        "Accept":     "application/json",
    })
    return session


SESSION = make_session()


# ─────────────────────────────────────────────
# FETCH SINGLE POSTER
# ─────────────────────────────────────────────

def fetch_poster(title: str, year: int) -> str:
    """Search TMDB, return full poster URL or empty string."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            params = {
                "api_key":  TMDB_API_KEY,
                "query":    title,
                "year":     int(year),
                "language": "en-US",
            }
            resp    = SESSION.get(TMDB_SEARCH, params=params, timeout=10)
            results = resp.json().get("results", [])

            if not results:
                params.pop("year")
                resp    = SESSION.get(TMDB_SEARCH, params=params, timeout=10)
                results = resp.json().get("results", [])

            if results:
                poster_path = results[0].get("poster_path", "")
                if poster_path:
                    return POSTER_BASE + poster_path

            return ""

        except Exception as e:
            wait = BACKOFF ** attempt
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            print(f"  Waiting {wait:.0f}s before retry...")
            time.sleep(wait)

    return ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Loading {CSV_INPUT} ...")
    movies = pd.read_csv(CSV_INPUT)
    total  = len(movies)
    print(f"Total movies: {total}\n")

    new_urls = []
    success  = 0
    failed   = 0

    for i, row in movies.iterrows():
        title = str(row["title"])
        year  = row["year"]

        url = fetch_poster(title, year)

        if url:
            success += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"

        new_urls.append(url)
        print(f"[{i+1}/{total}] {status}  {title} ({year})")

        time.sleep(SLEEP_BETWEEN)

    movies["poster_url"] = new_urls
    movies.to_csv(CSV_OUTPUT, index=False)

    print(f"\n{'─' * 45}")
    print(f"Done.  ✓ {success} fetched  |  ✗ {failed} not found")
    print(f"Saved → {CSV_OUTPUT}")


if __name__ == "__main__":
    main()