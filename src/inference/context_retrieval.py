"""Modular context retrieval for model inference.

Automatically fetches previous shows from setlist.fm API to provide
historical context for predictions (RAG-style approach).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.ingest.setlistfm import SetlistFMClient
from src.dataio import load_all_data
from src.utils.venue_mapping import VenueCountryMapper


class ShowContextRetriever:
    """Retrieves historical context for a show from setlist.fm API.

    This acts like RAG (Retrieval-Augmented Generation):
    - User provides minimal input (date, venue, tour)
    - System automatically retrieves previous shows' setlists
    - Packages everything for model inference
    """

    def __init__(self, use_cached_data: bool = True):
        """Initialize context retriever.

        Args:
            use_cached_data: If True, use local database. If False, always fetch from API.

        """
        self.use_cached_data = use_cached_data

        # Initialize venue->country mapper
        self.venue_mapper = VenueCountryMapper()

        if use_cached_data:
            # Load local database
            self.shows, self.songs, self.setlists, self.song_tags = load_all_data()
            self.shows["date"] = pd.to_datetime(self.shows["date"])
            self.shows = self.shows.sort_values("date")
        else:
            # Initialize API client
            self.api_client = SetlistFMClient()
            self.shows = None
            self.setlists = None

    def get_previous_shows(
        self, target_date: str, n_shows: int = 5, max_days_back: int = 365
    ) -> List[Dict]:
        """Get N previous shows before target date.

        Args:
            target_date: Target show date (YYYY-MM-DD)
            n_shows: Number of previous shows to retrieve
            max_days_back: Maximum days to look back

        Returns:
            List of show dictionaries with setlists

        """
        target_dt = pd.to_datetime(target_date)

        if self.use_cached_data:
            return self._get_from_cache(target_dt, n_shows, max_days_back)
        else:
            return self._get_from_api(target_dt, n_shows, max_days_back)

    def _get_from_cache(
        self, target_dt: pd.Timestamp, n_shows: int, max_days_back: int
    ) -> List[Dict]:
        """Get previous shows from local cache."""
        # Filter shows before target date
        cutoff_date = target_dt - timedelta(days=max_days_back)
        prev_shows = self.shows[
            (self.shows["date"] < target_dt)
            & (self.shows["date"] >= cutoff_date)
            & ~self.shows["is_partial"]  # Exclude partial setlists
        ].sort_values("date", ascending=False)

        # Get last N shows
        prev_shows = prev_shows.head(n_shows)

        # Attach setlists
        result = []
        for _, show in prev_shows.iterrows():
            show_setlist = self.setlists[
                self.setlists["show_id"] == show["show_id"]
            ].sort_values("pos")

            result.append(
                {
                    "show_id": show["show_id"],
                    "date": show["date"].strftime("%Y-%m-%d"),
                    "venue_id": show["venue_id"],
                    "city": show["city"],
                    "country": show["country"],
                    "tour_name": show["tour_name"],
                    "setlist": show_setlist["song_id"].tolist(),
                }
            )

        # Return in chronological order (oldest to newest)
        return list(reversed(result))

    def _get_from_api(
        self, target_dt: pd.Timestamp, n_shows: int, max_days_back: int
    ) -> List[Dict]:
        """Get previous shows from setlist.fm API.

        This makes real-time API calls to get the most recent data.
        """
        # Fetch recent setlists (API returns newest first)
        # We'll fetch enough pages to likely cover max_days_back
        estimated_pages = max(3, max_days_back // 30)  # ~1 page per month

        raw_setlists = self.api_client.fetch_all_setlists(max_pages=estimated_pages)

        # Parse and filter
        valid_shows = []
        for raw in raw_setlists:
            normalized = self.api_client.normalize_setlist(raw)
            if not normalized or normalized["show"]["is_partial"]:
                continue

            show_date = pd.to_datetime(normalized["show"]["date"])

            # Check if within date range
            if show_date >= target_dt:
                continue
            if show_date < target_dt - timedelta(days=max_days_back):
                continue

            # Extract setlist (song IDs)
            setlist = [song["song_id"] for song in normalized["songs"]]

            valid_shows.append(
                {
                    "show_id": normalized["show"]["show_id"],
                    "date": normalized["show"]["date"],
                    "venue_id": normalized["show"]["venue_id"],
                    "city": normalized["show"]["city"],
                    "country": normalized["show"]["country"],
                    "tour_name": normalized["show"]["tour_name"],
                    "setlist": setlist,
                }
            )

        # Sort by date (oldest first) and take last N
        valid_shows.sort(key=lambda x: x["date"])
        return valid_shows[-n_shows:]

    def get_tour_context(
        self, tour_name: str, before_date: Optional[str] = None
    ) -> Dict:
        """Get context about a specific tour.

        Args:
            tour_name: Tour name
            before_date: Only include shows before this date

        Returns:
            Dict with tour statistics and shows

        """
        if self.use_cached_data:
            tour_shows = self.shows[self.shows["tour_name"] == tour_name]

            if before_date:
                before_dt = pd.to_datetime(before_date)
                tour_shows = tour_shows[tour_shows["date"] < before_dt]

            # Calculate tour stats
            tour_setlists = self.setlists[
                self.setlists["show_id"].isin(tour_shows["show_id"])
            ]

            # Most played songs on tour
            song_counts = tour_setlists["song_id"].value_counts()

            return {
                "tour_name": tour_name,
                "num_shows": len(tour_shows),
                "start_date": tour_shows["date"].min().strftime("%Y-%m-%d"),
                "end_date": tour_shows["date"].max().strftime("%Y-%m-%d"),
                "most_played_songs": song_counts.head(20).to_dict(),
                "show_ids": tour_shows["show_id"].tolist(),
            }
        else:
            # Would need to implement API-based tour search
            raise NotImplementedError("Tour context from API not yet implemented")

    def is_residency(
        self, venue_id: str, target_date: str, window_days: int = 7
    ) -> Tuple[bool, Optional[Dict]]:
        """Check if target date is part of a multi-night residency.

        Args:
            venue_id: Venue ID
            target_date: Show date
            window_days: Days before/after to check for other shows

        Returns:
            (is_residency, residency_info)

        """
        if not self.use_cached_data:
            raise NotImplementedError("Residency detection only works with cached data")

        target_dt = pd.to_datetime(target_date)
        window_start = target_dt - timedelta(days=window_days)
        window_end = target_dt + timedelta(days=window_days)

        # Find shows at same venue within window
        nearby_shows = self.shows[
            (self.shows["venue_id"] == venue_id)
            & (self.shows["date"] >= window_start)
            & (self.shows["date"] <= window_end)
            & ~self.shows["is_partial"]
        ].sort_values("date")

        if len(nearby_shows) <= 1:
            return False, None

        # Check if consecutive nights
        dates = nearby_shows["date"].tolist()
        is_consecutive = all(
            (dates[i + 1] - dates[i]).days <= 1 for i in range(len(dates) - 1)
        )

        if not is_consecutive:
            return False, None

        # Find target show's position in residency
        target_position = None
        for idx, (_, show) in enumerate(nearby_shows.iterrows(), 1):
            if show["date"] == target_dt:
                target_position = idx
                break

        return True, {
            "is_residency": True,
            "total_nights": len(nearby_shows),
            "night_number": target_position,
            "start_date": nearby_shows.iloc[0]["date"].strftime("%Y-%m-%d"),
            "end_date": nearby_shows.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "show_ids": nearby_shows["show_id"].tolist(),
        }


def prepare_inference_input(
    user_input: Dict, retriever: ShowContextRetriever, model_type: str = "xgboost"
) -> Dict:
    """Prepare complete input for model inference from minimal user input.

    This function automatically retrieves all necessary context.

    Args:
        user_input: Minimal user input
            Required: date, is_festival, is_marathon
            Required (one of): venue_id OR (venue_name + city)
            Optional: tour_name, country (auto-derived if venue is known)
        retriever: ShowContextRetriever instance
        model_type: 'xgboost', 'mlp', 'temporal_gnn'

    Returns:
        Complete input dict ready for model inference

    Raises:
        ValueError: If country cannot be determined (new venue without country)

    """
    # Start with user input
    complete_input = user_input.copy()

    # Auto-derive country if venue is known
    if "venue_id" in user_input and "country" not in user_input:
        country = retriever.venue_mapper.get_country(user_input["venue_id"])
        if country:
            complete_input["country"] = country
        else:
            # New venue - country is required
            raise ValueError(
                f"Unknown venue '{user_input['venue_id']}'. "
                "Please provide 'country' parameter for new venues."
            )

    # Get previous shows (needed for temporal features and co-occurrence)
    n_shows = 5 if model_type == "temporal_gnn" else 1
    prev_shows = retriever.get_previous_shows(
        target_date=user_input["date"], n_shows=n_shows
    )

    complete_input["previous_shows"] = prev_shows

    # For temporal GNN, extract just the setlists
    if model_type == "temporal_gnn":
        complete_input["prev_setlists"] = [show["setlist"] for show in prev_shows]

    # Check for residency
    if "venue_id" in user_input:
        is_res, res_info = retriever.is_residency(
            venue_id=user_input["venue_id"], target_date=user_input["date"]
        )
        if is_res:
            complete_input["residency_info"] = res_info

    # Get tour context if available
    if "tour_name" in user_input:
        try:
            tour_context = retriever.get_tour_context(
                tour_name=user_input["tour_name"], before_date=user_input["date"]
            )
            complete_input["tour_context"] = tour_context
        except Exception as e:
            # Tour might not exist in database yet
            pass

    return complete_input


# Example usage
if __name__ == "__main__":
    # Initialize retriever (use cached data for speed)
    retriever = ShowContextRetriever(use_cached_data=True)

    # Minimal user input (what user actually provides)
    user_input = {
        "date": "2025-10-25",
        "venue_id": "red_rocks_co",
        "city": "Morrison",
        "country": "US",
        "tour_name": "World Tour 2025",
        "is_festival": False,
        "is_marathon": False,
    }

    # Automatically retrieve all context
    complete_input = prepare_inference_input(
        user_input=user_input, retriever=retriever, model_type="xgboost"
    )

    print("User provided:")
    print(f"  {len(user_input)} fields")
    print()
    print("System retrieved:")
    print(f"  Previous shows: {len(complete_input['previous_shows'])}")
    if "residency_info" in complete_input:
        print(f"  Residency: {complete_input['residency_info']}")
    if "tour_context" in complete_input:
        print(f"  Tour: {complete_input['tour_context']['num_shows']} shows")
    print()
    print("Ready for model inference!")
