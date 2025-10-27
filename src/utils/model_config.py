"""Model configuration loader from YAML files.

This module provides type-safe access to model hyperparameters defined in
configs/models.yaml. It supports environment variable overrides and provides
a clean API for accessing model-specific configurations.

Usage:
    from src.utils.model_config import model_config

    # Get XGBoost parameters
    xgb_params = model_config.xgboost
    model = XGBoostModel(**xgb_params)

    # Get specific parameters
    learning_rate = model_config.get('xgboost', 'learning_rate')
    random_seed = model_config.random_seed
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for model configuration. "
        "Install it with: pip install pyyaml"
    )

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelConfig:
    """Type-safe access to model configurations from YAML.

    This class loads model hyperparameters from configs/models.yaml and provides
    convenient access methods. Environment variables can override YAML values.

    Attributes:
        _config: Loaded YAML configuration dictionary
        _config_path: Path to the configuration file

    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to YAML config file. If None, uses configs/models.yaml

        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "models.yaml"
        else:
            config_path = Path(config_path)

        self._config_path = config_path
        self._config = self._load_config()
        self._apply_env_overrides()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing all configuration values

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed

        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        with open(self._config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {self._config_path}")

        return config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration.

        Supports common overrides:
        - RANDOM_SEED: Override training random seed
        - XGB_N_ESTIMATORS: Override XGBoost n_estimators
        - XGB_MAX_DEPTH: Override XGBoost max_depth
        - XGB_LEARNING_RATE: Override XGBoost learning_rate
        - BATCH_SIZE: Override default batch size
        - LEARNING_RATE: Override default learning rate
        """
        # Training config overrides
        if "RANDOM_SEED" in os.environ:
            if "training" not in self._config:
                self._config["training"] = {}
            self._config["training"]["random_seed"] = int(os.environ["RANDOM_SEED"])

        # XGBoost overrides
        if "xgboost" in self._config:
            if "XGB_N_ESTIMATORS" in os.environ:
                self._config["xgboost"]["n_estimators"] = int(
                    os.environ["XGB_N_ESTIMATORS"]
                )
            if "XGB_MAX_DEPTH" in os.environ:
                self._config["xgboost"]["max_depth"] = int(os.environ["XGB_MAX_DEPTH"])
            if "XGB_LEARNING_RATE" in os.environ:
                self._config["xgboost"]["learning_rate"] = float(
                    os.environ["XGB_LEARNING_RATE"]
                )

        # Neural network overrides
        for model_name in ["deepfm", "temporal_sets", "dcn", "tabnet"]:
            if model_name in self._config:
                if "BATCH_SIZE" in os.environ:
                    self._config[model_name]["batch_size"] = int(
                        os.environ["BATCH_SIZE"]
                    )
                if "LEARNING_RATE" in os.environ:
                    self._config[model_name]["learning_rate"] = float(
                        os.environ["LEARNING_RATE"]
                    )

    def get(
        self, model_name: str, param_name: Optional[str] = None, default: Any = None
    ) -> Any:
        """Get configuration value for a model.

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'deepfm')
            param_name: Specific parameter name (e.g., 'learning_rate').
                If None, returns all params.
            default: Default value if parameter not found

        Returns:
            Parameter value or dictionary of all parameters

        Example:
            >>> lr = model_config.get('xgboost', 'learning_rate')
            >>> all_params = model_config.get('xgboost')

        """
        if model_name not in self._config:
            return default

        if param_name is None:
            return self._config[model_name]

        return self._config[model_name].get(param_name, default)

    # Convenient property accessors for common models

    @property
    def xgboost(self) -> Dict[str, Any]:
        """Get XGBoost configuration."""
        return self._config.get("xgboost", {})

    @property
    def logistic_regression(self) -> Dict[str, Any]:
        """Get Logistic Regression configuration."""
        return self._config.get("logistic_regression", {})

    @property
    def random_forest(self) -> Dict[str, Any]:
        """Get Random Forest configuration."""
        return self._config.get("random_forest", {})

    @property
    def deepfm(self) -> Dict[str, Any]:
        """Get DeepFM configuration."""
        return self._config.get("deepfm", {})

    @property
    def temporal_sets(self) -> Dict[str, Any]:
        """Get Temporal Sets configuration."""
        return self._config.get("temporal_sets", {})

    @property
    def dcn(self) -> Dict[str, Any]:
        """Get Deep & Cross Network configuration."""
        return self._config.get("dcn", {})

    @property
    def tabnet(self) -> Dict[str, Any]:
        """Get TabNet configuration."""
        return self._config.get("tabnet", {})

    @property
    def sasrec(self) -> Dict[str, Any]:
        """Get SASRec configuration."""
        return self._config.get("sasrec", {})

    @property
    def random_seed(self) -> int:
        """Get global random seed from training config."""
        return self._config.get("training", {}).get("random_seed", 42)

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get("training", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get("evaluation", {})

    @property
    def optuna(self) -> Dict[str, Any]:
        """Get Optuna hyperparameter tuning configuration."""
        return self._config.get("optuna", {})

    def get_optuna_search_space(self, model_name: str) -> Dict[str, Any]:
        """Get Optuna search space for a specific model.

        Args:
            model_name: Name of the model (e.g., 'xgboost')

        Returns:
            Dictionary defining search space bounds

        """
        search_space_key = f"{model_name}_search_space"
        return self._config.get("optuna", {}).get(search_space_key, {})

    def reload(self) -> None:
        """Reload configuration from file.

        Useful for development when config file is modified.
        """
        self._config = self._load_config()
        self._apply_env_overrides()


# Global configuration instance
model_config = ModelConfig()


# For backward compatibility with existing code
def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of model parameters

    Example:
        >>> params = get_model_config('xgboost')
        >>> model = XGBoostModel(**params)

    """
    return model_config.get(model_name, default={})
