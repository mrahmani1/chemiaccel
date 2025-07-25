"""
Pre-trained models for polymer property prediction.

This module contains pre-trained machine learning models for predicting
various polymer properties including glass transition temperature,
Young's modulus, thermal stability, and density.
"""

from pathlib import Path

# Model directory path
MODEL_DIR = Path(__file__).parent

# Available pre-trained models
AVAILABLE_MODELS = [
    "glass_transition_temp",
    "youngs_modulus",
    "thermal_stability",
    "density",
]

# Model metadata
MODEL_METADATA = {
    "glass_transition_temp": {
        "description": "Glass transition temperature predictor",
        "units": "Celsius",
        "range": "(-100, 400)",
        "accuracy": "MAE ~15°C",
        "training_samples": 5000,
    },
    "youngs_modulus": {
        "description": "Young's modulus predictor",
        "units": "GPa",
        "range": "(0.1, 100)",
        "accuracy": "MAE ~0.5 GPa",
        "training_samples": 3000,
    },
    "thermal_stability": {
        "description": "Thermal decomposition temperature predictor",
        "units": "Celsius",
        "range": "(200, 600)",
        "accuracy": "MAE ~20°C",
        "training_samples": 2500,
    },
    "density": {
        "description": "Polymer density predictor",
        "units": "g/cm³",
        "range": "(0.8, 2.5)",
        "accuracy": "MAE ~0.1 g/cm³",
        "training_samples": 4000,
    },
}


def get_model_path(model_name: str) -> Path:
    """Get the path to a specific model file."""
    return MODEL_DIR / f"{model_name}_model.pkl"


def get_scaler_path(model_name: str) -> Path:
    """Get the path to a specific scaler file."""
    return MODEL_DIR / f"{model_name}_scaler.pkl"


def get_metrics_path(model_name: str) -> Path:
    """Get the path to a specific metrics file."""
    return MODEL_DIR / f"{model_name}_metrics.json"
