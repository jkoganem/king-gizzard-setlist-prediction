"""Configuration management with type-safe dataclasses.

This module provides application configuration using dataclasses and environment
variables. All configuration is immutable and type-checked.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths.

    Attributes:
        data_raw_dir: Directory for raw data files.
        data_curated_dir: Directory for processed data files.
        artifacts_dir: Directory for trained models (now output/models).
        reports_dir: Directory for evaluation reports.
        configs_dir: Directory for configuration files.

    """

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_curated_dir: Path = PROJECT_ROOT / "data" / "curated"
    artifacts_dir: Path = PROJECT_ROOT / "output" / "models"  # Renamed from artifacts
    reports_dir: Path = PROJECT_ROOT / "output" / "reports"
    configs_dir: Path = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    """XGBoost model hyperparameters.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        random_state: Random seed for reproducibility.

    """

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration configuration.

    Attributes:
        method: Calibration method ("isotonic" or "sigmoid").
        split: Fraction of training data to use for calibration.
        conformal_alpha: Significance level for conformal prediction.

    """

    method: str = "isotonic"
    split: float = 0.15
    conformal_alpha: float = 0.15


@dataclass(frozen=True)
class Settings:
    """Main application settings.

    All settings loaded from environment variables with sensible defaults.
    """

    # API keys
    setlistfm_api_key: Optional[str]
    spotify_client_id: Optional[str]
    spotify_client_secret: Optional[str]

    # Database
    database_url: str

    # Mode
    use_synthetic_data: bool
    random_seed: int

    # API server
    api_host: str
    api_port: int
    dashboard_port: int

    # Paths
    paths: PathConfig

    # Model hyperparameters
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float

    # Calibration
    calibration_method: str
    calibration_split: float
    conformal_alpha: float

    # Time decay
    time_decay_halflife_days: int

    # Decoding penalties
    penalty_repeat_prev: float
    penalty_same_venue: float
    penalty_style_runlength: float

    # API rate limits
    setlistfm_requests_per_second: float
    setlistfm_retry_attempts: int
    setlistfm_retry_backoff: float

    # Legacy compatibility properties
    @property
    def data_raw_dir(self) -> Path:
        return self.paths.data_raw_dir

    @property
    def data_curated_dir(self) -> Path:
        return self.paths.data_curated_dir

    @property
    def artifacts_dir(self) -> Path:
        return self.paths.artifacts_dir

    @property
    def reports_dir(self) -> Path:
        return self.paths.reports_dir

    @property
    def configs_dir(self) -> Path:
        return self.paths.configs_dir

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            random_state=self.random_seed,
        )

    @property
    def calibration(self) -> CalibrationConfig:
        """Get calibration configuration."""
        return CalibrationConfig(
            method=self.calibration_method,
            split=self.calibration_split,
            conformal_alpha=self.conformal_alpha,
        )


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        setlistfm_api_key=os.getenv("SETLISTFM_API_KEY"),
        spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        database_url=os.getenv("DATABASE_URL", "duckdb:///data/gizz.db"),
        use_synthetic_data=os.getenv("USE_SYNTHETIC_DATA", "true").lower() == "true",
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8501")),
        paths=PathConfig(),
        xgb_n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "500")),
        xgb_max_depth=int(os.getenv("XGB_MAX_DEPTH", "6")),
        xgb_learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        xgb_subsample=float(os.getenv("XGB_SUBSAMPLE", "0.8")),
        xgb_colsample_bytree=float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8")),
        calibration_method=os.getenv("CALIBRATION_METHOD", "isotonic"),
        calibration_split=float(os.getenv("CALIBRATION_SPLIT", "0.15")),
        conformal_alpha=float(os.getenv("CONFORMAL_ALPHA", "0.15")),
        time_decay_halflife_days=int(os.getenv("TIME_DECAY_HALFLIFE_DAYS", "30")),
        penalty_repeat_prev=float(os.getenv("PENALTY_REPEAT_PREV", "0.8")),
        penalty_same_venue=float(os.getenv("PENALTY_SAME_VENUE", "0.5")),
        penalty_style_runlength=float(os.getenv("PENALTY_STYLE_RUNLENGTH", "0.3")),
        setlistfm_requests_per_second=float(
            os.getenv("SETLISTFM_REQUESTS_PER_SECOND", "2.0")
        ),
        setlistfm_retry_attempts=int(os.getenv("SETLISTFM_RETRY_ATTEMPTS", "3")),
        setlistfm_retry_backoff=float(os.getenv("SETLISTFM_RETRY_BACKOFF", "2.0")),
    )


# Global settings instance
settings = load_settings()


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    settings.paths.data_raw_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.data_curated_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.paths.artifacts_dir / "set_model").mkdir(exist_ok=True)
    (settings.paths.artifacts_dir / "oc").mkdir(exist_ok=True)
