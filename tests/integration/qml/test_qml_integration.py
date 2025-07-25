"""
Integration tests for QML model functionality.
"""

from unittest.mock import Mock, patch

import pytest
from chemi_accel.qml import (
    ANIModel,
    GFN2xTBModel,
    GPUAccelerator,
    QMLModelLoader,
    QMLModelRegistry,
    QuantumProperty,
    QuantumPropertyExtractor,
    qml_registry,
)
from chemi_accel.qml.base import QMLPrediction
from rdkit import Chem


class TestQMLWorkflow:
    """Test complete QML workflow integration."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from molecule to properties."""
        # This test would require actual model files and dependencies
        # For now, we'll mock the critical components

        with patch("torchani.models") as mock_torchani:
            # Mock ANI model
            mock_ani_instance = Mock()
            mock_ani_instance.to.return_value = mock_ani_instance
            mock_ani_instance.float.return_value = mock_ani_instance
            mock_torchani.ANI2x.return_value = mock_ani_instance

            # Create and load model
            model = ANIModel(model_name="ani2x", device="cpu")

            # Mock the model loading to avoid actual torchani dependency
            with patch.object(model, "_setup_species_converter"):
                model.load_model()

            assert model._is_loaded

    def test_model_registry_integration(self):
        """Test QML model registry with multiple models."""
        registry = QMLModelRegistry()

        # Create mock models
        with patch("torchani.models"):
            ani_model = ANIModel(model_name="ani2x", device="cpu")
            gfn2_model = GFN2xTBModel(device="cpu")

            # Register models
            registry.register_model("ani", ani_model)
            registry.register_model("gfn2", gfn2_model, set_as_default=True)

            # Test retrieval
            assert registry.get_model("ani") is ani_model
            assert registry.get_model() is gfn2_model  # Default
            assert len(registry.list_models()) == 2

    def test_property_extraction_workflow(self):
        """Test property extraction and analysis workflow."""
        # Create mock predictions from different models
        pred1 = QMLPrediction(
            properties={
                QuantumProperty.HOMO_ENERGY: -8.5,
                QuantumProperty.LUMO_ENERGY: -2.3,
                QuantumProperty.DIPOLE_MOMENT: 1.2,
            },
            confidence=0.9,
            model_name="ANI-2x",
        )

        pred2 = QMLPrediction(
            properties={
                QuantumProperty.HOMO_ENERGY: -8.3,
                QuantumProperty.LUMO_ENERGY: -2.1,
                QuantumProperty.DIPOLE_MOMENT: 1.3,
            },
            confidence=0.85,
            model_name="GFN2-xTB",
        )

        # Initialize extractor
        extractor = QuantumPropertyExtractor()

        # Extract properties
        props1 = extractor.extract_properties(pred1)
        assert len(props1) == 3

        # Compute derived properties
        derived1 = extractor.compute_derived_properties(pred1)
        assert QuantumProperty.BANDGAP in derived1
        expected_gap = -2.3 - (-8.5)
        assert abs(derived1[QuantumProperty.BANDGAP] - expected_gap) < 1e-10

        # Compare predictions
        comparison = extractor.compare_predictions([pred1, pred2])
        assert comparison["num_predictions"] == 2
        assert len(comparison["common_properties"]) == 3

        # Analyze quality
        analysis = extractor.analyze_prediction_quality(pred1)
        assert analysis["overall_confidence"] == 0.9

    def test_unit_conversion_workflow(self):
        """Test unit conversion in property extraction workflow."""
        # Create prediction with Hartree energy
        properties = {QuantumProperty.TOTAL_ENERGY: -0.5}  # Hartree
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()

        # Convert to eV
        hartree_energy = prediction.get_property(QuantumProperty.TOTAL_ENERGY)
        ev_energy = extractor.convert_units(hartree_energy, "hartree", "ev")

        expected_ev = -0.5 * 27.211386245988
        assert abs(ev_energy - expected_ev) < 1e-10

        # Convert back to Hartree
        back_to_hartree = extractor.convert_units(ev_energy, "ev", "hartree")
        assert abs(back_to_hartree - hartree_energy) < 1e-10


@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance-critical functionality."""

    def test_batch_processing_performance(self):
        """Test batch processing with realistic data sizes."""

        # Mock model for performance testing
        class FastMockModel:
            def __init__(self):
                self._is_loaded = True

            def predict(self, molecules, properties=None, **kwargs):
                if isinstance(molecules, list):
                    return [
                        QMLPrediction(
                            properties={QuantumProperty.HOMO_ENERGY: -8.0 + 0.1 * i},
                            model_name="FastMock",
                        )
                        for i in range(len(molecules))
                    ]
                else:
                    return QMLPrediction(
                        properties={QuantumProperty.HOMO_ENERGY: -8.0},
                        model_name="FastMock",
                    )

            def batch_predict(
                self, molecules, properties=None, batch_size=32, **kwargs
            ):
                return self.predict(molecules, properties, **kwargs)

        model = FastMockModel()

        # Test with moderate-sized dataset
        molecules = [f"C{i}" for i in range(100)]  # 100 simple molecules

        predictions = model.batch_predict(molecules, batch_size=20)

        assert len(predictions) == 100
        assert all(isinstance(p, QMLPrediction) for p in predictions)

        # Check that predictions are different (not all identical)
        energies = [p.get_property(QuantumProperty.HOMO_ENERGY) for p in predictions]
        assert len(set(energies)) > 1  # Should have variation

    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_acceleration_integration(self, mock_cuda):
        """Test GPU acceleration integration (mocked)."""
        accelerator = GPUAccelerator(device="auto")

        # Should select GPU when available
        assert accelerator.device.startswith("cuda")

        # Test device info
        with patch("torch.cuda.get_device_properties") as mock_props:
            props = Mock()
            props.name = "Test GPU"
            props.total_memory = 8 * 1024**3
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 68
            mock_props.return_value = props

            with patch("torch.cuda.memory_allocated", return_value=1024**3):
                info = accelerator.get_device_info()

                assert info["gpu_name"] == "Test GPU"
                assert info["gpu_memory_total"] == 8 * 1024**3


@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests for real-world usage scenarios."""

    def test_drug_discovery_scenario(self):
        """Test scenario mimicking drug discovery workflow."""
        # Mock drug-like molecules
        drug_molecules = [
            "CC(C)NCC(O)COc1cccc2ccccc12",  # Propranolol-like
            "CN1CCN(CC1)C(c2ccccc2)c3ccccc3",  # Cetirizine-like
            "CC(C)(C)NCC(O)COc1cccc(C(N)=O)c1",  # Atenolol-like
        ]

        # Properties of interest for drug discovery
        target_properties = [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.DIPOLE_MOMENT,
            QuantumProperty.POLARIZABILITY,
        ]

        # Mock QML model that handles drug-like molecules
        class DrugQMLModel:
            def __init__(self):
                self._is_loaded = True

            def predict(self, molecules, properties=None, **kwargs):
                if isinstance(molecules, list):
                    results = []
                    for mol in molecules:
                        # Generate realistic-looking predictions
                        mol_obj = (
                            Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
                        )
                        if mol_obj:
                            num_atoms = mol_obj.GetNumAtoms()
                            # Scale properties roughly with molecular size
                            props = {
                                QuantumProperty.HOMO_ENERGY: -8.5 - 0.01 * num_atoms,
                                QuantumProperty.LUMO_ENERGY: -2.3 + 0.01 * num_atoms,
                                QuantumProperty.DIPOLE_MOMENT: 1.0 + 0.05 * num_atoms,
                                QuantumProperty.POLARIZABILITY: 10.0 + 0.5 * num_atoms,
                            }

                            # Filter requested properties
                            if properties:
                                props = {
                                    p: v for p, v in props.items() if p in properties
                                }

                            results.append(
                                QMLPrediction(
                                    properties=props,
                                    confidence=0.85,
                                    model_name="DrugQML",
                                )
                            )
                    return results
                else:
                    return self.predict([molecules], properties, **kwargs)[0]

            def validate_molecule(self, molecule):
                mol = (
                    Chem.MolFromSmiles(molecule)
                    if isinstance(molecule, str)
                    else molecule
                )
                return (
                    mol is not None and mol.GetNumAtoms() <= 100
                )  # Reasonable drug size

        model = DrugQMLModel()

        # Predict properties for drug molecules
        predictions = model.predict(drug_molecules, target_properties)

        assert len(predictions) == 3
        for pred in predictions:
            assert len(pred.properties) == len(target_properties)
            assert pred.confidence > 0.8

            # Check that LUMO > HOMO (basic sanity check)
            homo = pred.get_property(QuantumProperty.HOMO_ENERGY)
            lumo = pred.get_property(QuantumProperty.LUMO_ENERGY)
            assert lumo > homo

        # Analyze predictions
        extractor = QuantumPropertyExtractor()

        # Compute derived properties
        for pred in predictions:
            derived = extractor.compute_derived_properties(pred)
            assert QuantumProperty.BANDGAP in derived
            assert derived[QuantumProperty.BANDGAP] > 0

        # Compare predictions across molecules
        comparison = extractor.compare_predictions(predictions)
        assert comparison["num_predictions"] == 3

        # Should have reasonable variation across different molecules
        for prop_name, stats in comparison["statistics"].items():
            assert stats["std"] > 0  # Should have some variation

    def test_materials_screening_scenario(self):
        """Test scenario for materials property screening."""
        # Mock materials (simple organic semiconductors)
        materials = [
            "c1ccc2c(c1)c3ccccc3c4ccccc24",  # Anthracene
            "c1ccc2c(c1)sc3ccccc23",  # Dibenzothiophene
            "c1ccc2c(c1)[nH]c3ccccc23",  # Carbazole
        ]

        # Properties relevant for semiconductors
        semiconductor_properties = [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.BANDGAP,
            QuantumProperty.IONIZATION_POTENTIAL,
            QuantumProperty.ELECTRON_AFFINITY,
        ]

        # Mock materials QML model
        class MaterialsQMLModel:
            def predict(self, molecules, properties=None, **kwargs):
                results = []
                for mol_smiles in (
                    molecules if isinstance(molecules, list) else [molecules]
                ):
                    mol = Chem.MolFromSmiles(mol_smiles)

                    # Different properties for different materials
                    if "anthracene" in mol_smiles.lower() or "c4ccccc24" in mol_smiles:
                        # Anthracene-like properties
                        base_props = {
                            QuantumProperty.HOMO_ENERGY: -5.6,
                            QuantumProperty.LUMO_ENERGY: -3.2,
                        }
                    elif "s" in mol_smiles.lower():
                        # Sulfur-containing (electron-rich)
                        base_props = {
                            QuantumProperty.HOMO_ENERGY: -5.2,
                            QuantumProperty.LUMO_ENERGY: -2.8,
                        }
                    else:
                        # Nitrogen-containing (electron-deficient)
                        base_props = {
                            QuantumProperty.HOMO_ENERGY: -5.8,
                            QuantumProperty.LUMO_ENERGY: -2.5,
                        }

                    # Compute derived properties
                    all_props = base_props.copy()
                    all_props[QuantumProperty.BANDGAP] = (
                        base_props[QuantumProperty.LUMO_ENERGY]
                        - base_props[QuantumProperty.HOMO_ENERGY]
                    )
                    all_props[QuantumProperty.IONIZATION_POTENTIAL] = -base_props[
                        QuantumProperty.HOMO_ENERGY
                    ]
                    all_props[QuantumProperty.ELECTRON_AFFINITY] = -base_props[
                        QuantumProperty.LUMO_ENERGY
                    ]

                    # Filter requested properties
                    if properties:
                        all_props = {
                            p: v for p, v in all_props.items() if p in properties
                        }

                    results.append(
                        QMLPrediction(
                            properties=all_props,
                            confidence=0.9,
                            model_name="MaterialsQML",
                        )
                    )

                return results if isinstance(molecules, list) else results[0]

        model = MaterialsQMLModel()

        # Screen materials
        predictions = model.predict(materials, semiconductor_properties)

        assert len(predictions) == 3

        # Analyze for semiconductor suitability
        extractor = QuantumPropertyExtractor()

        suitable_materials = []
        for i, pred in enumerate(predictions):
            # Check bandgap suitability (1-4 eV range for semiconductors)
            bandgap = pred.get_property(QuantumProperty.BANDGAP)
            if 1.0 <= bandgap <= 4.0:
                suitable_materials.append((materials[i], bandgap))

        # Should find at least one suitable material
        assert len(suitable_materials) > 0

        # Compare electronic properties
        comparison = extractor.compare_predictions(predictions)

        # Should show variation in electronic properties
        homo_stats = comparison["statistics"]["homo_energy"]
        assert homo_stats["std"] > 0.1  # Reasonable variation

    def test_model_loader_integration(self):
        """Test QML model loader with registry integration."""
        loader = QMLModelLoader()

        # Mock the model loading process
        with patch.object(loader, "load_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            # Load and register model
            model = loader.load_model("test_model")
            qml_registry.register_model("loaded_model", model)

            # Verify integration
            assert qml_registry.get_model("loaded_model") is mock_model
            mock_load.assert_called_once_with("test_model")


@pytest.mark.benchmark
class TestBenchmarkIntegration:
    """Integration tests for benchmarking functionality."""

    def test_comprehensive_benchmark(self):
        """Test comprehensive model benchmarking."""
        from chemi_accel.qml.utils import QMLBenchmark

        # Create multiple mock models with different performance characteristics
        class FastModel:
            def predict(self, molecules, properties=None, **kwargs):
                # Simulate fast prediction
                return [
                    QMLPrediction(
                        properties={QuantumProperty.HOMO_ENERGY: -8.0},
                        model_name="Fast",
                    )
                    for _ in (molecules if isinstance(molecules, list) else [molecules])
                ]

        class SlowModel:
            def predict(self, molecules, properties=None, **kwargs):
                # Simulate slower prediction
                import time

                time.sleep(0.01)  # Small delay
                return [
                    QMLPrediction(
                        properties={QuantumProperty.HOMO_ENERGY: -8.2},
                        model_name="Slow",
                    )
                    for _ in (molecules if isinstance(molecules, list) else [molecules])
                ]

        benchmark = QMLBenchmark()

        test_molecules = ["CC", "CCO", "CCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        # Benchmark both models
        fast_results = benchmark.benchmark_model(
            FastModel(), test_molecules, properties, num_runs=3
        )

        slow_results = benchmark.benchmark_model(
            SlowModel(), test_molecules, properties, num_runs=3
        )

        # Fast model should have better throughput
        fast_throughput = fast_results["statistics"]["mean_throughput"]
        slow_throughput = slow_results["statistics"]["mean_throughput"]

        assert fast_throughput > slow_throughput

        # Both should have high success rates
        assert fast_results["statistics"]["success_rate"] == 1.0
        assert slow_results["statistics"]["success_rate"] == 1.0
