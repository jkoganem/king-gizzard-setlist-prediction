"""Basic tests for the setlist predictor."""

import pytest
from src.utils.aliases import normalize_song_title, create_song_id
from src.utils.config import settings


def test_normalize_song_title():
    """Test song title normalization."""
    # Test segue/transition removal
    assert normalize_song_title("Robot Stop -> ") == "Robot Stop"

    # Test tease removal
    assert normalize_song_title("Wah Wah (Tease)") == "Wah Wah"

    # Test cover removal
    assert normalize_song_title("Superstition (cover)") == "Superstition"

    # Test combined markers
    assert normalize_song_title("The River -> (Tease)") == "The River"


def test_create_song_id():
    """Test song ID creation."""
    assert create_song_id("The River") == "the_river"
    assert create_song_id("Head On/Pill") == "head_on_pill"
    assert create_song_id("I'm in Your Mind") == "im_in_your_mind"


def test_settings():
    """Test settings load correctly."""
    assert settings.random_seed == 42
    assert settings.use_synthetic_data in [True, False]
    assert settings.xgb_n_estimators > 0
