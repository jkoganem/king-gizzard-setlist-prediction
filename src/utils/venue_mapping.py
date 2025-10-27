"""Venue-country mapping utility for automatic country lookup.

This module provides a persistent venue->country mapping that:
1. Auto-derives country for known venues
2. Gracefully falls back to requiring user input for new venues
3. Self-updates when new venue-country pairs are added
"""

import json
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


class VenueCountryMapper:
    """Manages venue->country mappings with persistent storage.

    This allows the system to automatically derive country from venue_id
    for known venues, while gracefully handling new venues.
    """

    def __init__(self, mapping_file: Optional[Path] = None):
        """Initialize the mapper.

        Args:
            mapping_file: Path to JSON file storing venue->country mappings.
                         Defaults to data/venue_country_mapping.json

        """
        if mapping_file is None:
            mapping_file = (
                Path(__file__).parent.parent.parent
                / "data"
                / "venue_country_mapping.json"
            )

        self.mapping_file = Path(mapping_file)
        self.mapping: Dict[str, str] = {}

        # Load existing mappings if available
        if self.mapping_file.exists():
            self._load_mapping()

    def _load_mapping(self):
        """Load venue->country mapping from JSON file."""
        try:
            with open(self.mapping_file, "r") as f:
                self.mapping = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Could not load venue mapping from {self.mapping_file}: {e}"
            )
            self.mapping = {}

    def _save_mapping(self):
        """Save current mapping to JSON file."""
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, "w") as f:
            json.dump(self.mapping, f, indent=2, sort_keys=True)

    def get_country(self, venue_id: str) -> Optional[str]:
        """Get country for a given venue_id.

        Args:
            venue_id: Venue identifier

        Returns:
            Country code if known, None if venue is new

        """
        return self.mapping.get(venue_id)

    def add_venue(self, venue_id: str, country: str, save: bool = True):
        """Add a new venue->country mapping.

        Args:
            venue_id: Venue identifier
            country: Country code (e.g., 'US', 'AU', 'GB')
            save: If True, persist the updated mapping to disk

        """
        self.mapping[venue_id] = country
        if save:
            self._save_mapping()

    def bulk_add_venues(self, venue_country_dict: Dict[str, str], save: bool = True):
        """Add multiple venue->country mappings at once.

        Args:
            venue_country_dict: Dictionary mapping venue_id -> country
            save: If True, persist the updated mapping to disk

        """
        self.mapping.update(venue_country_dict)
        if save:
            self._save_mapping()

    def build_from_shows_data(self, shows_df: pd.DataFrame, save: bool = True):
        """Build the complete mapping from a shows DataFrame.

        Args:
            shows_df: DataFrame with 'venue_id' and 'country' columns
            save: If True, persist the mapping to disk

        """
        # Group by venue_id and take the first country (they should all be the same)
        venue_country_map = shows_df.groupby("venue_id")["country"].first().to_dict()

        self.mapping = venue_country_map

        if save:
            self._save_mapping()

        return len(venue_country_map)

    def __len__(self):
        """Return number of known venues."""
        return len(self.mapping)

    def __contains__(self, venue_id: str):
        """Check if venue is in the mapping."""
        return venue_id in self.mapping


def build_venue_mapping_from_database():
    """Convenience function to build the mapping from the current database.

    This should be run once to initialize the mapping file, and can be
    re-run periodically to update with new venues.
    """
    from src.dataio import load_all_data

    print("Loading shows data...")
    shows, _, _, _ = load_all_data()

    print(f"Building venue->country mapping from {len(shows)} shows...")
    mapper = VenueCountryMapper()
    num_venues = mapper.build_from_shows_data(shows, save=True)

    print(f"[SUCCESS] Created mapping for {num_venues} unique venues")
    print(f"[SUCCESS] Saved to: {mapper.mapping_file}")

    # Show some statistics
    country_counts = pd.Series(list(mapper.mapping.values())).value_counts()
    print(f"\nTop 5 countries by venue count:")
    for country, count in country_counts.head(5).items():
        print(f"  {country}: {count} venues")

    return mapper


# Example usage
if __name__ == "__main__":
    # Build the initial mapping from database
    mapper = build_venue_mapping_from_database()

    # Example lookups
    print("\nExample lookups:")

    # Known venue
    test_venue = list(mapper.mapping.keys())[0]
    country = mapper.get_country(test_venue)
    print(f"  {test_venue} -> {country}")

    # New venue (returns None)
    new_venue = "brand_new_venue_123"
    country = mapper.get_country(new_venue)
    print(f"  {new_venue} -> {country} (new venue, requires user input)")
