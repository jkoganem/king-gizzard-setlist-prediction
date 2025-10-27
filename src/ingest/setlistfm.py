"""Setlist.fm API client with caching and rate limiting."""

import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import requests
from diskcache import Cache
from tqdm import tqdm

from src.utils.config import settings
from src.utils.aliases import normalize_song_title, create_song_id


class SetlistFMClient:
    """Client for setlist.fm API with rate limiting and caching."""

    BASE_URL = "https://api.setlist.fm/rest/1.0"
    # King Gizzard & The Lizard Wizard MBID (correct one from search API)
    KING_GIZZARD_MBID = "f58384a4-2ad2-4f24-89c5-c7b74ae1cce7"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or settings.setlistfm_api_key
        if not self.api_key:
            raise ValueError(
                "Setlist.fm API key required. Set SETLISTFM_API_KEY in .env"
            )

        if cache_dir is None:
            cache_dir = settings.data_raw_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = Cache(str(cache_dir))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": self.api_key,
                "Accept": "application/json",
            }
        )

        self.last_request_time = 0
        self.min_request_interval = 1.0 / settings.setlistfm_requests_per_second

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(
        self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True
    ) -> Dict:
        """Make API request with retry logic."""
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(settings.setlistfm_retry_attempts):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                self.cache[cache_key] = data
                return data

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = settings.setlistfm_retry_backoff ** (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    return {}
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt == settings.setlistfm_retry_attempts - 1:
                    raise
                time.sleep(settings.setlistfm_retry_backoff ** (attempt + 1))

        return {}

    def get_artist_setlists(self, page: int = 1) -> Dict:
        """Get setlists for King Gizzard."""
        return self._request(
            f"artist/{self.KING_GIZZARD_MBID}/setlists",
            params={"p": page},
        )

    def fetch_all_setlists(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Fetch all available setlists for King Gizzard.

        Args:
            max_pages: Maximum number of pages to fetch (None = all)

        Returns:
            List of raw setlist data

        """
        setlists = []
        page = 1

        print("Fetching setlists from setlist.fm...")

        while True:
            if max_pages and page > max_pages:
                break

            data = self.get_artist_setlists(page=page)
            if not data or "setlist" not in data:
                break

            page_setlists = data["setlist"]
            if not page_setlists:
                break

            setlists.extend(page_setlists)
            print(f"  Page {page}: {len(page_setlists)} setlists")

            # Check if there are more pages
            total = data.get("total", 0)
            items_per_page = data.get("itemsPerPage", 20)
            if page * items_per_page >= total:
                break

            page += 1

        print(f"Fetched {len(setlists)} setlists total")
        return setlists

    def normalize_setlist(self, raw_setlist: Dict) -> Optional[Dict]:
        """Normalize a raw setlist from the API into our schema.

        Args:
            raw_setlist: Raw setlist data from API

        Returns:
            Normalized data or None if invalid

        """
        try:
            # Extract basic info
            show_id = raw_setlist.get("id")
            event_date = raw_setlist.get("eventDate")
            if not show_id or not event_date:
                return None

            # Parse date
            try:
                date_parts = event_date.split("-")
                year = int(date_parts[2])
                month = int(date_parts[1])
                day = int(date_parts[0])
                show_date = datetime(year, month, day).date()
            except (ValueError, IndexError):
                return None

            # Venue info
            venue_data = raw_setlist.get("venue", {})
            venue_id = venue_data.get("id", f"venue_{show_id}")
            city_data = venue_data.get("city", {})

            venue = {
                "venue_id": venue_id,
                "name": venue_data.get("name", "Unknown"),
                "city": city_data.get("name", "Unknown"),
                "country": city_data.get("country", {}).get("code", "XX"),
                "lat": city_data.get("coords", {}).get("lat"),
                "lon": city_data.get("coords", {}).get("long"),
            }

            # Tour info
            tour_name = raw_setlist.get("tour", {}).get("name", "Unknown Tour")

            # Extract songs from sets
            sets = raw_setlist.get("sets", {}).get("set", [])
            songs = []
            pos = 0

            is_partial = False
            if not sets or all(not s.get("song") for s in sets):
                is_partial = True

            for set_idx, set_data in enumerate(sets):
                song_list = set_data.get("song", [])
                encore_index = set_data.get("encore", 0)

                for song_data in song_list:
                    title = song_data.get("name")
                    if not title:
                        continue

                    # Check for cover/tease
                    is_cover = song_data.get("cover") is not None
                    is_tease = (
                        song_data.get("tape", False) or "(tease)" in title.lower()
                    )

                    # Normalize title
                    canonical_title = normalize_song_title(title)
                    song_id = create_song_id(canonical_title)

                    # Check for segue
                    segue_next = "->" in title or ">" in title

                    songs.append(
                        {
                            "pos": pos,
                            "song_id": song_id,
                            "title": canonical_title,
                            "is_cover": is_cover,
                            "is_tease": is_tease,
                            "segue_next": segue_next,
                            "encore_index": encore_index,
                        }
                    )
                    pos += 1

            show = {
                "show_id": show_id,
                "date": show_date.isoformat(),
                "tour_name": tour_name,
                "venue_id": venue_id,
                "city": venue["city"],
                "country": venue["country"],
                "is_festival": "festival" in tour_name.lower(),
                "is_partial": is_partial,
                "notes": raw_setlist.get("info", ""),
            }

            return {
                "show": show,
                "venue": venue,
                "songs": songs,
            }

        except Exception as e:
            print(f"Error normalizing setlist {raw_setlist.get('id')}: {e}")
            return None


def fetch_and_save_setlists(
    max_pages: Optional[int] = None, output_dir: Optional[Path] = None
) -> Path:
    """Fetch setlists and save raw JSON.

    Args:
        max_pages: Maximum pages to fetch
        output_dir: Output directory for raw data

    Returns:
        Path to saved JSON file

    """
    if output_dir is None:
        output_dir = settings.data_raw_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = SetlistFMClient()
        setlists = client.fetch_all_setlists(max_pages=max_pages)

        # Save raw data
        output_file = (
            output_dir / f"setlists_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(setlists, f, indent=2)

        print(f"Saved raw setlists to {output_file}")
        return output_file

    except ValueError as e:
        print(f"API key error: {e}")
        print("Using synthetic data instead...")
        return None
