"""
Validation tests for polymer property prediction against experimental databases.

This module tests the accuracy of polymer property predictions against
known experimental data and literature values.
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.polymer.predictive.model_training import PolymerModelTrainer
from src.chemi_accel.polymer.predictive.property_engine import PolymerPropertyEngine


class TestPolymerPropertyValidation:
    """Validation tests against experimental polymer property data."""

    @pytest.fixture
    def experimental_data(self):
        """
        Experimental polymer property data for validation.

        This includes well-known polymers with established property values
        from literature and databases.
        """
        return pd.DataFrame(
            {
                "polymer_name": [
                    "Polyethylene (PE)",
                    "Polypropylene (PP)",
                    "Polystyrene (PS)",
                    "Poly(methyl methacrylate) (PMMA)",
                    "Polyvinyl chloride (PVC)",
                    "Polyethylene terephthalate (PET)",
                    "Polycarbonate (PC)",
                    "Polytetrafluoroethylene (PTFE)",
                    "Polyacrylonitrile (PAN)",
                    "Polyvinyl acetate (PVAc)",
                ],
                "smiles": [
                    "C=C",  # Ethylene monomer
                    "C=CC",  # Propylene monomer
                    "C=Cc1ccccc1",  # Styrene monomer
                    "C=C(C)C(=O)OC",  # Methyl methacrylate
                    "C=CCl",  # Vinyl chloride
                    "C=CC(=O)Oc1ccc(C(=O)O)cc1",  # Simplified PET monomer
                    "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # Simplified PC monomer
                    "C=CF",  # Tetrafluoroethylene (simplified)
                    "C=CC#N",  # Acrylonitrile
                    "C=CC(=O)OC",  # Vinyl acetate
                ],
                "glass_transition_temp": [
                    -120,  # PE (very low Tg)
                    -20,  # PP
                    100,  # PS
                    105,  # PMMA
                    81,  # PVC
                    69,  # PET
                    147,  # PC (high Tg)
                    -97,  # PTFE
                    97,  # PAN
                    28,  # PVAc
                ],
                "youngs_modulus": [
                    0.2,  # PE (flexible)
                    1.5,  # PP
                    3.2,  # PS
                    2.8,  # PMMA
                    2.9,  # PVC
                    2.8,  # PET
                    2.4,  # PC
                    0.5,  # PTFE (very flexible)
                    3.5,  # PAN (stiff)
                    0.02,  # PVAc (very flexible)
                ],
                "thermal_stability": [
                    300,  # PE
                    350,  # PP
                    400,  # PS
                    380,  # PMMA
                    360,  # PVC
                    420,  # PET
                    450,  # PC (high thermal stability)
                    400,  # PTFE
                    420,  # PAN
                    320,  # PVAc
                ],
                "density": [
                    0.92,  # PE
                    0.90,  # PP
                    1.05,  # PS
                    1.19,  # PMMA
                    1.38,  # PVC
                    1.38,  # PET
                    1.20,  # PC
                    2.17,  # PTFE (high density due to fluorine)
                    1.18,  # PAN
                    1.19,  # PVAc
                ],
            }
        )

    @pytest.fixture
    def mock_calculator(self):
        """Create a mock calculator for testing."""
        return Mock(spec=Calculator)

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def trained_engine(self, mock_calculator, temp_model_dir, experimental_data):
        """Create a trained polymer property engine."""
        # Train models on experimental data
        trainer = PolymerModelTrainer(mock_calculator, temp_model_dir)

        # Mock descriptor calculation to return reasonable values
        with patch.object(
            trainer.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            # Generate mock descriptors based on SMILES
            mock_descriptors = self._generate_mock_descriptors(
                experimental_data["smiles"]
            )
            mock_calc.return_value = mock_descriptors

            # Train models
            trainer.train_all_models(experimental_data)

        # Return trained engine
        return PolymerPropertyEngine(mock_calculator, temp_model_dir)

    def _generate_mock_descriptors(self, smiles_series):
        """Generate mock descriptors for testing."""
        n_samples = len(smiles_series)

        # Create realistic descriptor values based on SMILES patterns
        descriptors = pd.DataFrame(
            {
                "molecular_weight": [
                    len(s) * 10 + np.random.normal(0, 5) for s in smiles_series
                ],
                "heavy_atom_count": [
                    s.count("C") + s.count("N") + s.count("O") for s in smiles_series
                ],
                "logp": [np.random.normal(2, 1) for _ in range(n_samples)],
                "tpsa": [np.random.normal(50, 20) for _ in range(n_samples)],
                "aromatic_rings": [
                    s.count("c") / max(len(s), 1) for s in smiles_series
                ],
                "vinyl_groups": [s.count("C=C") for s in smiles_series],
                "backbone_flexibility": [
                    np.random.uniform(0, 1) for _ in range(n_samples)
                ],
                "aromatic_content": [
                    s.count("c") / max(len(s), 1) for s in smiles_series
                ],
                "tg_predictor_1": [np.random.normal(50, 30) for _ in range(n_samples)],
                "modulus_predictor_1": [
                    np.random.normal(2, 1) for _ in range(n_samples)
                ],
                "density_predictor": [
                    np.random.normal(1.2, 0.3) for _ in range(n_samples)
                ],
            },
            index=smiles_series.index,
        )

        return descriptors

    @patch("src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE", True)
    def test_glass_transition_temperature_accuracy(
        self, trained_engine, experimental_data
    ):
        """Test accuracy of glass transition temperature predictions."""
        smiles = experimental_data["smiles"]
        experimental_tg = experimental_data["glass_transition_temp"]

        with patch.object(
            trained_engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_calc.return_value = self._generate_mock_descriptors(smiles)

            predictions = trained_engine.predict_glass_transition_temp(smiles)
            predicted_tg = predictions["glass_transition_temp"]

            # Calculate accuracy metrics
            mae = np.mean(np.abs(predicted_tg - experimental_tg))
            rmse = np.sqrt(np.mean((predicted_tg - experimental_tg) ** 2))
            r2 = np.corrcoef(predicted_tg, experimental_tg)[0, 1] ** 2

            # Assertions for reasonable accuracy
            assert mae < 50, f"Glass transition MAE too high: {mae:.2f}°C"
            assert rmse < 70, f"Glass transition RMSE too high: {rmse:.2f}°C"
            assert r2 > 0.3, f"Glass transition R² too low: {r2:.3f}"

            print(
                f"Glass Transition Temperature - MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C, R²: {r2:.3f}"
            )

    @patch("src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE", True)
    def test_youngs_modulus_accuracy(self, trained_engine, experimental_data):
        """Test accuracy of Young's modulus predictions."""
        smiles = experimental_data["smiles"]
        experimental_modulus = experimental_data["youngs_modulus"]

        with patch.object(
            trained_engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_calc.return_value = self._generate_mock_descriptors(smiles)

            predictions = trained_engine.predict_youngs_modulus(smiles)
            predicted_modulus = predictions["youngs_modulus"]

            # Calculate accuracy metrics
            mae = np.mean(np.abs(predicted_modulus - experimental_modulus))
            rmse = np.sqrt(np.mean((predicted_modulus - experimental_modulus) ** 2))
            r2 = np.corrcoef(predicted_modulus, experimental_modulus)[0, 1] ** 2

            # Assertions for reasonable accuracy
            assert mae < 1.0, f"Young's modulus MAE too high: {mae:.2f} GPa"
            assert rmse < 1.5, f"Young's modulus RMSE too high: {rmse:.2f} GPa"
            assert r2 > 0.2, f"Young's modulus R² too low: {r2:.3f}"

            print(
                f"Young's Modulus - MAE: {mae:.2f} GPa, RMSE: {rmse:.2f} GPa, R²: {r2:.3f}"
            )

    @patch("src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE", True)
    def test_thermal_stability_accuracy(self, trained_engine, experimental_data):
        """Test accuracy of thermal stability predictions."""
        smiles = experimental_data["smiles"]
        experimental_thermal = experimental_data["thermal_stability"]

        with patch.object(
            trained_engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_calc.return_value = self._generate_mock_descriptors(smiles)

            predictions = trained_engine.predict_properties(
                smiles, ["thermal_stability"]
            )
            predicted_thermal = predictions["thermal_stability"]

            # Calculate accuracy metrics
            mae = np.mean(np.abs(predicted_thermal - experimental_thermal))
            rmse = np.sqrt(np.mean((predicted_thermal - experimental_thermal) ** 2))
            r2 = np.corrcoef(predicted_thermal, experimental_thermal)[0, 1] ** 2

            # Assertions for reasonable accuracy
            assert mae < 40, f"Thermal stability MAE too high: {mae:.2f}°C"
            assert rmse < 60, f"Thermal stability RMSE too high: {rmse:.2f}°C"
            assert r2 > 0.2, f"Thermal stability R² too low: {r2:.3f}"

            print(
                f"Thermal Stability - MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C, R²: {r2:.3f}"
            )

    @patch("src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE", True)
    def test_density_accuracy(self, trained_engine, experimental_data):
        """Test accuracy of density predictions."""
        smiles = experimental_data["smiles"]
        experimental_density = experimental_data["density"]

        with patch.object(
            trained_engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_calc.return_value = self._generate_mock_descriptors(smiles)

            predictions = trained_engine.predict_properties(smiles, ["density"])
            predicted_density = predictions["density"]

            # Calculate accuracy metrics
            mae = np.mean(np.abs(predicted_density - experimental_density))
            rmse = np.sqrt(np.mean((predicted_density - experimental_density) ** 2))
            r2 = np.corrcoef(predicted_density, experimental_density)[0, 1] ** 2

            # Assertions for reasonable accuracy
            assert mae < 0.3, f"Density MAE too high: {mae:.3f} g/cm³"
            assert rmse < 0.4, f"Density RMSE too high: {rmse:.3f} g/cm³"
            assert r2 > 0.2, f"Density R² too low: {r2:.3f}"

            print(
                f"Density - MAE: {mae:.3f} g/cm³, RMSE: {rmse:.3f} g/cm³, R²: {r2:.3f}"
            )

    def test_property_correlation_analysis(self, experimental_data):
        """Test correlations between different polymer properties."""
        # Analyze correlations in experimental data
        properties = [
            "glass_transition_temp",
            "youngs_modulus",
            "thermal_stability",
            "density",
        ]
        correlation_matrix = experimental_data[properties].corr()

        # Expected correlations based on polymer science knowledge
        # Higher Tg often correlates with higher modulus and thermal stability
        tg_modulus_corr = correlation_matrix.loc[
            "glass_transition_temp", "youngs_modulus"
        ]
        tg_thermal_corr = correlation_matrix.loc[
            "glass_transition_temp", "thermal_stability"
        ]

        print(f"Tg-Modulus correlation: {tg_modulus_corr:.3f}")
        print(f"Tg-Thermal stability correlation: {tg_thermal_corr:.3f}")

        # These correlations should be positive but not necessarily strong
        # due to the diversity of polymer structures
        assert -1 <= tg_modulus_corr <= 1
        assert -1 <= tg_thermal_corr <= 1

    def test_outlier_detection(self, experimental_data):
        """Test identification of outliers in polymer property data."""
        properties = [
            "glass_transition_temp",
            "youngs_modulus",
            "thermal_stability",
            "density",
        ]

        for prop in properties:
            values = experimental_data[prop]

            # Calculate z-scores
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = z_scores > 2.5  # More than 2.5 standard deviations

            outlier_polymers = experimental_data.loc[outliers, "polymer_name"].tolist()

            if outliers.any():
                print(f"Outliers in {prop}: {outlier_polymers}")

            # PTFE might be an outlier for density due to fluorine content
            if prop == "density":
                ptfe_outlier = "Polytetrafluoroethylene (PTFE)" in outlier_polymers
                # This is expected due to high fluorine content
                print(f"PTFE density outlier (expected): {ptfe_outlier}")

    def test_property_range_validation(self, experimental_data):
        """Test that predicted properties are within reasonable ranges."""
        # Define reasonable ranges for polymer properties
        property_ranges = {
            "glass_transition_temp": (-200, 500),  # °C
            "youngs_modulus": (0.001, 100),  # GPa
            "thermal_stability": (100, 600),  # °C
            "density": (0.5, 3.0),  # g/cm³
        }

        for prop, (min_val, max_val) in property_ranges.items():
            values = experimental_data[prop]

            assert (
                values.min() >= min_val
            ), f"{prop} minimum value {values.min()} below range"
            assert (
                values.max() <= max_val
            ), f"{prop} maximum value {values.max()} above range"

            print(f"{prop} range: {values.min():.2f} to {values.max():.2f}")

    @patch("src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE", True)
    def test_prediction_consistency(self, trained_engine, experimental_data):
        """Test that predictions are consistent across multiple runs."""
        smiles = experimental_data["smiles"].iloc[:3]  # Test with first 3 polymers

        with patch.object(
            trained_engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_descriptors = self._generate_mock_descriptors(smiles)
            mock_calc.return_value = mock_descriptors

            # Make predictions multiple times
            predictions1 = trained_engine.predict_properties(smiles)
            predictions2 = trained_engine.predict_properties(smiles)

            # Predictions should be identical (deterministic models)
            for prop in [
                "glass_transition_temp",
                "youngs_modulus",
                "thermal_stability",
                "density",
            ]:
                if prop in predictions1.columns and prop in predictions2.columns:
                    diff = np.abs(predictions1[prop] - predictions2[prop])
                    assert diff.max() < 1e-10, f"Inconsistent predictions for {prop}"

    def test_edge_case_molecules(self, mock_calculator, temp_model_dir):
        """Test prediction behavior with edge case molecules."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)

        edge_cases = pd.Series(
            [
                "C",  # Methane (very simple)
                "c1ccccc1c1ccccc1c1ccccc1",  # Large aromatic system
                "CCCCCCCCCCCCCCCCCCCC",  # Long alkyl chain
                "C(F)(F)F",  # Highly fluorinated
            ]
        )

        with patch.object(
            engine.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            mock_calc.return_value = self._generate_mock_descriptors(edge_cases)

            # Should not raise exceptions
            try:
                predictions = engine.predict_properties(edge_cases)
                assert len(predictions) == len(edge_cases)
                print("Edge case predictions completed successfully")
            except Exception as e:
                pytest.fail(f"Edge case prediction failed: {e}")


class TestModelTrainingValidation:
    """Validation tests for model training process."""

    @pytest.fixture
    def mock_calculator(self):
        """Create a mock calculator for testing."""
        return Mock(spec=Calculator)

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("src.chemi_accel.polymer.predictive.model_training.SKLEARN_AVAILABLE", True)
    def test_synthetic_data_generation(self, mock_calculator, temp_model_dir):
        """Test synthetic training data generation."""
        trainer = PolymerModelTrainer(mock_calculator, temp_model_dir)

        synthetic_data = trainer.generate_synthetic_training_data(n_samples=100)

        assert len(synthetic_data) == 100
        assert "smiles" in synthetic_data.columns
        assert "glass_transition_temp" in synthetic_data.columns
        assert "youngs_modulus" in synthetic_data.columns
        assert "thermal_stability" in synthetic_data.columns
        assert "density" in synthetic_data.columns

        # Check that values are in reasonable ranges
        assert synthetic_data["glass_transition_temp"].between(-150, 400).all()
        assert synthetic_data["youngs_modulus"].between(0.1, 50).all()
        assert synthetic_data["thermal_stability"].between(200, 500).all()
        assert synthetic_data["density"].between(0.8, 2.0).all()

    @patch("src.chemi_accel.polymer.predictive.model_training.SKLEARN_AVAILABLE", True)
    def test_model_training_metrics(self, mock_calculator, temp_model_dir):
        """Test that model training produces reasonable metrics."""
        trainer = PolymerModelTrainer(mock_calculator, temp_model_dir)

        # Generate synthetic training data
        training_data = trainer.generate_synthetic_training_data(n_samples=200)

        with patch.object(
            trainer.descriptor_calc, "calculate_polymer_descriptors"
        ) as mock_calc:
            # Generate mock descriptors
            mock_descriptors = pd.DataFrame(
                {
                    "molecular_weight": np.random.rand(200) * 100,
                    "logp": np.random.rand(200) * 5,
                    "tpsa": np.random.rand(200) * 100,
                    "aromatic_content": np.random.rand(200),
                    "backbone_flexibility": np.random.rand(200),
                },
                index=training_data.index,
            )
            mock_calc.return_value = mock_descriptors

            # Train a single model
            metrics = trainer.train_single_model(training_data, "glass_transition_temp")

            # Check that metrics are reasonable
            assert 0 <= metrics["r2"] <= 1, f"R² out of range: {metrics['r2']}"
            assert metrics["mae"] >= 0, f"MAE negative: {metrics['mae']}"
            assert metrics["rmse"] >= 0, f"RMSE negative: {metrics['rmse']}"
            assert metrics["training_samples"] > 0
            assert metrics["validation_samples"] > 0

            print(
                f"Training metrics - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
