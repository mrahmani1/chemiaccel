"""
Model training utilities for polymer property prediction.

This module provides utilities for training and validating polymer property
prediction models using synthetic and experimental data.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.calculator import Calculator
from .descriptors import PolymerDescriptorCalculator

logger = logging.getLogger(__name__)


class PolymerModelTrainer:
    """
    Trainer for polymer property prediction models.

    This class handles the training, validation, and saving of machine learning
    models for polymer property prediction.
    """

    def __init__(self, calculator: Calculator, model_dir: Optional[str] = None):
        """
        Initialize the model trainer.

        Args:
            calculator: ChemiAccel calculator instance
            model_dir: Directory to save trained models
        """
        self.calculator = calculator
        self.descriptor_calc = PolymerDescriptorCalculator(calculator)

        if model_dir is None:
            model_dir = Path(__file__).parent / "models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.model_configs = {
            "glass_transition_temp": {
                "model": GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
                ),
                "scaler": StandardScaler(),
            },
            "youngs_modulus": {
                "model": RandomForestRegressor(
                    n_estimators=150, max_depth=10, min_samples_split=5, random_state=42
                ),
                "scaler": StandardScaler(),
            },
            "thermal_stability": {
                "model": GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
                ),
                "scaler": StandardScaler(),
            },
            "density": {
                "model": MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                ),
                "scaler": StandardScaler(),
            },
        }

    def generate_synthetic_training_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic training data for polymer properties.

        This creates a diverse set of polymer structures with realistic
        property values for initial model training.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            DataFrame with SMILES and property values
        """
        logger.info(f"Generating {n_samples} synthetic training samples")

        # Common polymer building blocks and functional groups
        monomers = [
            "C=C",  # ethylene
            "C=CC",  # propylene
            "C=CC(C)C",  # isobutylene
            "C=Cc1ccccc1",  # styrene
            "C=CC(=O)O",  # acrylic acid
            "C=CC(=O)OC",  # methyl acrylate
            "C=CC(=O)N",  # acrylamide
            "C=CCl",  # vinyl chloride
            "C=CF",  # vinyl fluoride
            "C=CC#N",  # acrylonitrile
            "C1CO1",  # ethylene oxide
            "CC1CO1",  # propylene oxide
            "c1ccc(O)cc1",  # phenol
            "CC(C)(C)c1ccc(O)cc1",  # BPA-like
            "OCc1ccccc1",  # benzyl alcohol
            "NC(=O)c1ccccc1",  # benzamide
        ]

        # Generate random combinations and modifications
        synthetic_data = []

        for i in range(n_samples):
            # Select base monomer
            base_monomer = np.random.choice(monomers)

            # Add random modifications
            modified_smiles = self._modify_monomer(base_monomer)

            # Calculate synthetic properties based on structure
            properties = self._calculate_synthetic_properties(modified_smiles)

            synthetic_data.append(
                {
                    "smiles": modified_smiles,
                    "glass_transition_temp": properties["tg"],
                    "youngs_modulus": properties["modulus"],
                    "thermal_stability": properties["thermal"],
                    "density": properties["density"],
                }
            )

        return pd.DataFrame(synthetic_data)

    def _modify_monomer(self, base_smiles: str) -> str:
        """Apply random modifications to a base monomer structure."""
        modifications = [
            lambda s: s,  # No modification
            lambda s: s + "C",  # Add methyl
            lambda s: s + "CC",  # Add ethyl
            lambda s: s + "c1ccccc1",  # Add phenyl
            lambda s: s + "O",  # Add hydroxyl
            lambda s: s + "N",  # Add amine
            lambda s: s + "F",  # Add fluorine
            lambda s: s + "Cl",  # Add chlorine
        ]

        # Apply random modification
        modifier = np.random.choice(modifications)
        try:
            return modifier(base_smiles)
        except:
            return base_smiles

    def _calculate_synthetic_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate synthetic property values based on molecular structure."""
        # Simple heuristics for generating realistic property values

        # Glass transition temperature (Tg)
        aromatic_content = smiles.count("c") / max(len(smiles), 1)
        flexibility = smiles.count("C") / max(len(smiles), 1)
        tg_base = 100 + aromatic_content * 150 - flexibility * 50
        tg = max(-50, min(300, tg_base + np.random.normal(0, 20)))

        # Young's modulus
        stiffness = aromatic_content + smiles.count("=") / max(len(smiles), 1)
        modulus_base = 0.5 + stiffness * 5
        modulus = max(0.1, min(50, modulus_base + np.random.normal(0, 1)))

        # Thermal stability
        stability_base = 250 + aromatic_content * 100
        thermal = max(200, min(500, stability_base + np.random.normal(0, 30)))

        # Density
        heavy_atoms = (
            smiles.count("O")
            + smiles.count("N")
            + smiles.count("F")
            + smiles.count("Cl")
        )
        density_base = 1.0 + heavy_atoms * 0.1 + aromatic_content * 0.3
        density = max(0.8, min(2.0, density_base + np.random.normal(0, 0.1)))

        return {"tg": tg, "modulus": modulus, "thermal": thermal, "density": density}

    def train_all_models(
        self, training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all polymer property prediction models.

        Args:
            training_data: Training dataset. If None, generates synthetic data.

        Returns:
            Dictionary of model performance metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")

        if training_data is None:
            training_data = self.generate_synthetic_training_data()

        results = {}

        for property_name in self.model_configs.keys():
            logger.info(f"Training model for {property_name}")

            try:
                metrics = self.train_single_model(training_data, property_name)
                results[property_name] = metrics
                logger.info(
                    f"Completed training for {property_name}: MAE={metrics['mae']:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to train model for {property_name}: {e}")
                results[property_name] = {"error": str(e)}

        return results

    def train_single_model(
        self,
        training_data: pd.DataFrame,
        property_name: str,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train a single property prediction model.

        Args:
            training_data: Training dataset
            property_name: Name of property to predict
            validation_split: Fraction for validation

        Returns:
            Model performance metrics
        """
        if property_name not in training_data.columns:
            raise ValueError(f"Property '{property_name}' not found in training data")

        # Extract SMILES and targets
        smiles = training_data["smiles"]
        targets = training_data[property_name].dropna()

        # Calculate descriptors
        logger.info(f"Calculating descriptors for {len(smiles)} molecules")
        descriptors = self.descriptor_calc.calculate_polymer_descriptors(smiles)

        # Align descriptors with targets
        common_idx = descriptors.index.intersection(targets.index)
        X = descriptors.loc[common_idx]
        y = targets.loc[common_idx]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Get model configuration
        config = self.model_configs[property_name]
        model = config["model"]
        scaler = config["scaler"]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_val, val_pred),
            "r2": r2_score(y_val, val_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_r2": r2_score(y_train, train_pred),
        }

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        metrics["cv_mae"] = -cv_scores.mean()
        metrics["cv_mae_std"] = cv_scores.std()

        # Save model
        self._save_trained_model(property_name, model, scaler, metrics)

        return metrics

    def _save_trained_model(
        self, property_name: str, model: Any, scaler: Any, metrics: Dict[str, float]
    ):
        """Save a trained model to disk."""
        # Save model
        model_file = self.model_dir / f"{property_name}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Save scaler
        scaler_file = self.model_dir / f"{property_name}_scaler.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)

        # Save metrics
        metrics_file = self.model_dir / f"{property_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved trained model for {property_name}")

    def validate_against_experimental_data(
        self, experimental_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate trained models against experimental data.

        Args:
            experimental_data: DataFrame with SMILES and experimental property values

        Returns:
            Validation metrics for each property
        """
        from .property_engine import PolymerPropertyEngine

        # Initialize property engine with trained models
        engine = PolymerPropertyEngine(self.calculator, str(self.model_dir))

        results = {}

        for property_name in self.model_configs.keys():
            if property_name not in experimental_data.columns:
                continue

            # Get experimental values
            exp_values = experimental_data[property_name].dropna()
            smiles = experimental_data.loc[exp_values.index, "smiles"]

            # Make predictions
            predictions = engine.predict_properties(smiles, [property_name])
            pred_values = predictions[property_name]

            # Calculate validation metrics
            mae = mean_absolute_error(exp_values, pred_values)
            r2 = r2_score(exp_values, pred_values)
            rmse = np.sqrt(mean_squared_error(exp_values, pred_values))

            results[property_name] = {
                "experimental_mae": mae,
                "experimental_r2": r2,
                "experimental_rmse": rmse,
                "n_samples": len(exp_values),
            }

            logger.info(
                f"Experimental validation for {property_name}: "
                f"MAE={mae:.3f}, R²={r2:.3f}"
            )

        return results


def train_default_models(
    calculator: Optional[Calculator] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to train default polymer property models.

    Args:
        calculator: ChemiAccel calculator (creates default if None)

    Returns:
        Training results for all models
    """
    if calculator is None:
        calculator = Calculator()

    trainer = PolymerModelTrainer(calculator)
    return trainer.train_all_models()


if __name__ == "__main__":
    # Train models when run as script
    logging.basicConfig(level=logging.INFO)

    print("Training default polymer property models...")
    results = train_default_models()

    print("\nTraining Results:")
    for prop, metrics in results.items():
        if "error" in metrics:
            print(f"{prop}: ERROR - {metrics['error']}")
        else:
            print(f"{prop}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

    print("\nTraining completed!")
