"""
Predictive Polymer Property Engine.

This module provides machine learning-based prediction of polymer properties
including glass transition temperature, Young's modulus, thermal stability,
and density. It includes pre-trained models and fine-tuning capabilities.
"""

from .property_engine import (
    PolymerPropertyEngine,
    PolymerPropertyPrediction,
    ModelMetrics,
    predict_polymer_properties,
    predict_glass_transition_temp,
    predict_youngs_modulus
)

from .descriptors import PolymerDescriptorCalculator

from .model_training import (
    PolymerModelTrainer,
    train_default_models
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
    "train_default_models"
]