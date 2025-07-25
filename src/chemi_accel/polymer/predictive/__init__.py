"""
Predictive Polymer Property Engine.

This module provides machine learning-based prediction of polymer properties
including glass transition temperature, Young's modulus, thermal stability,
and density. It includes pre-trained models and fine-tuning capabilities.
"""

from .descriptors import PolymerDescriptorCalculator
from .model_training import PolymerModelTrainer, train_default_models
from .property_engine import (
    ModelMetrics,
    PolymerPropertyEngine,
    PolymerPropertyPrediction,
    predict_glass_transition_temp,
    predict_polymer_properties,
    predict_youngs_modulus,
)

__all__ = [
    # Main property engine
    "PolymerPropertyEngine",
    "PolymerPropertyPrediction",
    "ModelMetrics",
    # Convenience functions
    "predict_polymer_properties",
    "predict_glass_transition_temp",
    "predict_youngs_modulus",
    # Descriptor calculation
    "PolymerDescriptorCalculator",
    # Model training
    "PolymerModelTrainer",
    "train_default_models",
]
