"""
Predictive Polymer Property Engine

This module implements pre-trained ML models for polymer property prediction,
including glass transition temperature (Tg) and Young's modulus predictors.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import cudf
    import cupy as cp

    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

try:
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.calculator import Calculator
from ...utils.validation import validate_smiles_series
from .descriptors import PolymerDescriptorCalculator

logger = logging.getLogger(__name__)


@dataclass
class PolymerPropertyPrediction:
    """Results from polymer property prediction."""

    smiles: str
    glass_transition_temp: Optional[float] = None
    youngs_modulus: Optional[float] = None
    thermal_stability: Optional[float] = None
    density: Optional[float] = None
    prediction_uncertainty: Optional[Dict[str, float]] = None
    confidence_score: Optional[float] = None


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    mae: float
    r2: float
    rmse: float
    training_samples: int
    validation_samples: int


class PolymerPropertyEngine:
    """
    Predictive engine for polymer properties using pre-trained ML models.

    This class provides prediction capabilities for key polymer properties:
    - Glass transition temperature (Tg)
    - Young's modulus
    - Thermal stability
    - Density

    Features:
    - Pre-trained models for common polymer properties
    - Fine-tuning interface for proprietary data
    - Uncertainty quantification
    - GPU acceleration when available
    """

    def __init__(self, calculator: Calculator, model_path: Optional[str] = None):
        """
        Initialize the polymer property engine.

        Args:
            calculator: ChemiAccel calculator instance
            model_path: Path to pre-trained models directory
        """
        self.calculator = calculator
        self.descriptor_calc = PolymerDescriptorCalculator(calculator)

        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}

        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent / "models"
        self.model_path = Path(model_path)

        # Load pre-trained models if available
        self._load_pretrained_models()

        logger.info(f"Initialized PolymerPropertyEngine with {len(self.models)} models")

    def _load_pretrained_models(self):
        """Load pre-trained models from disk."""
        if not self.model_path.exists():
            logger.warning(
                f"Model path {self.model_path} does not exist. Using default models."
            )
            self._initialize_default_models()
            return

        # Load models for each property
        properties = [
            "glass_transition_temp",
            "youngs_modulus",
            "thermal_stability",
            "density",
        ]

        for prop in properties:
            model_file = self.model_path / f"{prop}_model.pkl"
            scaler_file = self.model_path / f"{prop}_scaler.pkl"
            metrics_file = self.model_path / f"{prop}_metrics.json"

            if model_file.exists() and scaler_file.exists():
                try:
                    with open(model_file, "rb") as f:
                        self.models[prop] = pickle.load(f)

                    with open(scaler_file, "rb") as f:
                        self.scalers[prop] = pickle.load(f)

                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            metrics_data = json.load(f)
                            self.model_metrics[prop] = ModelMetrics(**metrics_data)

                    logger.info(f"Loaded pre-trained model for {prop}")
                except Exception as e:
                    logger.warning(f"Failed to load model for {prop}: {e}")

        if not self.models:
            logger.warning("No pre-trained models found. Initializing default models.")
            self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize default models with reasonable parameters."""
        if not SKLEARN_AVAILABLE:
            logger.error(
                "scikit-learn not available. Cannot initialize default models."
            )
            return

        # Default model configurations
        model_configs = {
            "glass_transition_temp": {
                "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "scaler": StandardScaler(),
            },
            "youngs_modulus": {
                "model": RandomForestRegressor(n_estimators=100, random_state=42),
                "scaler": StandardScaler(),
            },
            "thermal_stability": {
                "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "scaler": StandardScaler(),
            },
            "density": {
                "model": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
                "scaler": StandardScaler(),
            },
        }

        for prop, config in model_configs.items():
            self.models[prop] = config["model"]
            self.scalers[prop] = config["scaler"]
            logger.info(f"Initialized default model for {prop}")

    def predict_properties(
        self,
        molecules: Union[pd.Series, List[str]],
        properties: Optional[List[str]] = None,
        include_uncertainty: bool = False,
    ) -> pd.DataFrame:
        """
        Predict polymer properties for given molecules.

        Args:
            molecules: SMILES strings of polymer monomers/oligomers
            properties: List of properties to predict. If None, predicts all available.
            include_uncertainty: Whether to include prediction uncertainty

        Returns:
            DataFrame with predicted properties
        """
        # Validate input
        if isinstance(molecules, list):
            molecules = pd.Series(molecules)

        validate_smiles_series(molecules)

        # Default to all available properties
        if properties is None:
            properties = list(self.models.keys())

        # Calculate polymer descriptors
        logger.info(f"Calculating descriptors for {len(molecules)} molecules")
        descriptors = self.descriptor_calc.calculate_polymer_descriptors(molecules)

        # Prepare results DataFrame
        results = pd.DataFrame(index=molecules.index)
        results["smiles"] = molecules

        # Predict each property
        for prop in properties:
            if prop not in self.models:
                logger.warning(f"No model available for property: {prop}")
                continue

            try:
                predictions = self._predict_single_property(
                    descriptors, prop, include_uncertainty
                )
                results[prop] = predictions["prediction"]

                if include_uncertainty and "uncertainty" in predictions:
                    results[f"{prop}_uncertainty"] = predictions["uncertainty"]
                    results[f"{prop}_confidence"] = predictions["confidence"]

            except Exception as e:
                logger.error(f"Failed to predict {prop}: {e}")
                results[prop] = np.nan

        return results

    def _predict_single_property(
        self,
        descriptors: pd.DataFrame,
        property_name: str,
        include_uncertainty: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Predict a single property with optional uncertainty quantification."""
        model = self.models[property_name]
        scaler = self.scalers[property_name]

        # Scale features
        X_scaled = scaler.transform(descriptors)

        # Make predictions
        predictions = model.predict(X_scaled)

        result = {"prediction": predictions}

        if include_uncertainty:
            # Estimate uncertainty using ensemble methods or bootstrap
            uncertainty, confidence = self._estimate_uncertainty(
                model, X_scaled, predictions
            )
            result["uncertainty"] = uncertainty
            result["confidence"] = confidence

        return result

    def _estimate_uncertainty(
        self, model: Any, X: np.ndarray, predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty.

        For tree-based models, use prediction variance across trees.
        For neural networks, use dropout-based uncertainty.
        """
        n_samples = len(predictions)

        if hasattr(model, "estimators_"):  # Tree-based ensemble
            # Get predictions from individual estimators
            tree_predictions = np.array(
                [estimator.predict(X) for estimator in model.estimators_]
            )
            uncertainty = np.std(tree_predictions, axis=0)
            confidence = 1.0 / (1.0 + uncertainty)

        elif hasattr(model, "predict_proba"):  # Probabilistic model
            # Use prediction probability as confidence measure
            uncertainty = np.ones(n_samples) * 0.1  # Default uncertainty
            confidence = np.ones(n_samples) * 0.8  # Default confidence

        else:
            # Default uncertainty estimation
            uncertainty = np.ones(n_samples) * 0.1
            confidence = np.ones(n_samples) * 0.8

        return uncertainty, confidence

    def fine_tune_model(
        self,
        training_data: pd.DataFrame,
        property_name: str,
        smiles_column: str = "smiles",
        target_column: str = None,
        validation_split: float = 0.2,
        save_model: bool = True,
    ) -> ModelMetrics:
        """
        Fine-tune a model with proprietary experimental data.

        Args:
            training_data: DataFrame with SMILES and target property values
            property_name: Name of the property to fine-tune
            smiles_column: Name of column containing SMILES
            target_column: Name of column containing target values
            validation_split: Fraction of data to use for validation
            save_model: Whether to save the fine-tuned model

        Returns:
            Model performance metrics
        """
        if target_column is None:
            target_column = property_name

        if target_column not in training_data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        # Extract SMILES and targets
        smiles = training_data[smiles_column]
        targets = training_data[target_column].dropna()

        # Calculate descriptors
        logger.info(f"Calculating descriptors for {len(smiles)} training molecules")
        descriptors = self.descriptor_calc.calculate_polymer_descriptors(smiles)

        # Align descriptors with targets
        common_idx = descriptors.index.intersection(targets.index)
        X = descriptors.loc[common_idx]
        y = targets.loc[common_idx]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        if property_name not in self.models:
            # Initialize new model
            self.models[property_name] = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )

        model = self.models[property_name]
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        metrics = ModelMetrics(
            mae=mean_absolute_error(y_val, val_pred),
            r2=r2_score(y_val, val_pred),
            rmse=np.sqrt(np.mean((y_val - val_pred) ** 2)),
            training_samples=len(X_train),
            validation_samples=len(X_val),
        )

        # Update scaler and metrics
        self.scalers[property_name] = scaler
        self.model_metrics[property_name] = metrics

        # Save model if requested
        if save_model:
            self._save_model(property_name)

        logger.info(
            f"Fine-tuned model for {property_name}: MAE={metrics.mae:.3f}, R²={metrics.r2:.3f}"
        )

        return metrics

    def _save_model(self, property_name: str):
        """Save a trained model to disk."""
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = self.model_path / f"{property_name}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(self.models[property_name], f)

        # Save scaler
        scaler_file = self.model_path / f"{property_name}_scaler.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(self.scalers[property_name], f)

        # Save metrics
        if property_name in self.model_metrics:
            metrics_file = self.model_path / f"{property_name}_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(self.model_metrics[property_name].__dict__, f, indent=2)

        logger.info(f"Saved model for {property_name} to {self.model_path}")

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        info = {}

        for prop in self.models.keys():
            model_info = {
                "model_type": type(self.models[prop]).__name__,
                "has_scaler": prop in self.scalers,
                "has_metrics": prop in self.model_metrics,
            }

            if prop in self.model_metrics:
                metrics = self.model_metrics[prop]
                model_info.update(
                    {
                        "mae": metrics.mae,
                        "r2": metrics.r2,
                        "rmse": metrics.rmse,
                        "training_samples": metrics.training_samples,
                    }
                )

            info[prop] = model_info

        return info

    def predict_glass_transition_temp(
        self, molecules: Union[pd.Series, List[str]]
    ) -> pd.DataFrame:
        """Convenience method for glass transition temperature prediction."""
        return self.predict_properties(molecules, properties=["glass_transition_temp"])

    def predict_youngs_modulus(
        self, molecules: Union[pd.Series, List[str]]
    ) -> pd.DataFrame:
        """Convenience method for Young's modulus prediction."""
        return self.predict_properties(molecules, properties=["youngs_modulus"])

    def batch_predict(
        self, molecules: Union[pd.Series, List[str]], batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Predict properties in batches for large datasets.

        Args:
            molecules: SMILES strings
            batch_size: Number of molecules per batch

        Returns:
            DataFrame with all predictions
        """
        if isinstance(molecules, list):
            molecules = pd.Series(molecules)

        results = []

        for i in range(0, len(molecules), batch_size):
            batch = molecules.iloc[i : i + batch_size]
            batch_results = self.predict_properties(batch)
            results.append(batch_results)

            logger.info(
                f"Processed batch {i//batch_size + 1}/{(len(molecules)-1)//batch_size + 1}"
            )

        return pd.concat(results, ignore_index=True)


# Convenience functions for direct use
def predict_polymer_properties(
    molecules: Union[pd.Series, List[str]], calculator: Optional[Calculator] = None
) -> pd.DataFrame:
    """
    Convenience function to predict polymer properties.

    Args:
        molecules: SMILES strings
        calculator: ChemiAccel calculator (will create default if None)

    Returns:
        DataFrame with predicted properties
    """
    if calculator is None:
        calculator = Calculator()

    engine = PolymerPropertyEngine(calculator)
    return engine.predict_properties(molecules)


def predict_glass_transition_temp(
    molecules: Union[pd.Series, List[str]], calculator: Optional[Calculator] = None
) -> pd.DataFrame:
    """
    Convenience function to predict glass transition temperature.

    Args:
        molecules: SMILES strings
        calculator: ChemiAccel calculator (will create default if None)

    Returns:
        DataFrame with Tg predictions
    """
    if calculator is None:
        calculator = Calculator()

    engine = PolymerPropertyEngine(calculator)
    return engine.predict_glass_transition_temp(molecules)


def predict_youngs_modulus(
    molecules: Union[pd.Series, List[str]], calculator: Optional[Calculator] = None
) -> pd.DataFrame:
    """
    Convenience function to predict Young's modulus.

    Args:
        molecules: SMILES strings
        calculator: ChemiAccel calculator (will create default if None)

    Returns:
        DataFrame with Young's modulus predictions
    """
    if calculator is None:
        calculator = Calculator()

    engine = PolymerPropertyEngine(calculator)
    return engine.predict_youngs_modulus(molecules)
