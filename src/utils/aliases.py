"""Song title normalization utilities.

This module provides utilities for normalizing song titles from setlist.fm data.
Handles segue markers, tease indicators, and cover annotations.
"""


def normalize_song_title(title: str) -> str:
    """Normalize a song title by removing special characters.

    Handles common setlist.fm formatting including segue markers, tease
    indicators, and cover annotations.

    Args:
        title: Raw song title from source.

    Returns:
        Normalized title.

    Examples:
        >>> normalize_song_title("Robot Stop -> ")
        'Robot Stop'
        >>> normalize_song_title("Wah Wah (Tease)")
        'Wah Wah'
        >>> normalize_song_title("Superstition (cover)")
        'Superstition'

    """
    # Remove common separators and special characters used in setlist.fm
    title = title.replace("->", "").strip()
    title = title.replace(">", "").strip()

    # Handle teases (in parentheses)
    if " (Tease)" in title or " (tease)" in title:
        title = title.replace(" (Tease)", "").replace(" (tease)", "")

    # Handle covers
    if " (cover)" in title.lower():
        title = title.split(" (cover)")[0].strip()

    return title.strip()


def create_song_id(title: str) -> str:
    """Create a stable song ID from a title.

    Converts title to lowercase, replaces spaces with underscores, and removes
    special characters to create a stable identifier.

    Args:
        title: Canonical song title.

    Returns:
        Song ID suitable for use as a database key.

    Example:
        >>> create_song_id("The River")
        'the_river'
        >>> create_song_id("Head On/Pill")
        'head_onpill'

    """
    song_id = title.lower()
    song_id = song_id.replace(" ", "_")
    song_id = song_id.replace("'", "")
    song_id = song_id.replace("/", "_")
    song_id = song_id.replace("-", "_")
    # Remove any remaining special characters
    song_id = "".join(c for c in song_id if c.isalnum() or c == "_")
    return song_id
