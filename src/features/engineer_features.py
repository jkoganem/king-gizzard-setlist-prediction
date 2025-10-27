"""Feature engineering pipeline - Backward compatibility wrapper.

This module maintains backward compatibility by importing and re-exporting
the main feature engineering function from the refactored feature_pipeline module.

The actual implementation has been split into focused modules:
- temporal_features.py: Time-based features (days since last, decay, etc.)
- cooccurrence_features.py: Song co-occurrence patterns (PMI-based)
- context_features.py: Tour/venue/proximity/advanced features
- residency_features.py: Multi-night residency detection and features
- feature_pipeline.py: Main orchestrator that coordinates all modules

This wrapper ensures existing imports continue to work:
    from src.features.engineer_features import engineer_all_features
"""

from src.features.feature_pipeline import engineer_all_features

# Re-export for backward compatibility
__all__ = ["engineer_all_features"]
