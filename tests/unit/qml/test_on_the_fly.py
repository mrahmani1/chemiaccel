"""
Tests for on-the-fly QML prediction system.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.qml.base import (
    BaseQMLModel,
    QMLModelError,
    QMLPrediction,
    QuantumProperty,
)
from chemi_accel.qml.on_the_fly import (
    BatchPredictionRequest,
    BatchPredictionResult,
    ConformerGenerator,
    ConformerStrategy,
    GeometryInput,
    GPUBatchProcessor,
    OnTheFlyPredictor,
)


class MockQMLModel(BaseQMLModel):
    """Mock QML model for testing."""

    def load_model(self):
        self._model = Mock()
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        if isinstance(molecules, list):
            return [
                QMLPrediction(
                    properties={QuantumProperty.HOMO_ENERGY: -8.0 - i * 0.1},
                    model_name="MockModel",
                )
                for i in range(len(molecules))
            ]
        else:
            return QMLPrediction(
                properties={QuantumProperty.HOMO_ENERGY: -8.0}, model_name="MockModel"
            )

    def get_supported_properties(self):
        return [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

    def validate_molecule(self, molecule):
        return True


class TestGeometryInput:
    """Test GeometryInput class."""

    def test_initialization(self):
        """Test geometry input initialization."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atomic_nums = np.array([6, 1])

        geom = GeometryInput(
            coordinates=coords, atomic_numbers=atomic_nums, charge=0, multiplicity=1
        )

        assert np.array_equal(geom.coordinates, coords)
        assert np.array_equal(geom.atomic_numbers, atomic_nums)
        assert geom.charge == 0
        assert geom.multiplicity == 1

    @patch("torch.tensor")
    def test_to_tensor(self, mock_tensor):
        """Test conversion to tensor."""
        coords = np.array([[0.0, 0.0, 0.0]])
        atomic_nums = np.array([6])

        geom = GeometryInput(coordinates=coords, atomic_numbers=atomic_nums)

        mock_tensor.return_value = Mock()

        with patch("chemi_accel.qml.on_the_fly.TORCH_AVAILABLE", True):
            tensors = geom.to_tensor()

            assert "coordinates" in tensors
            assert "atomic_numbers" in tensors
            assert "charge" in tensors
            assert "multiplicity" in tensors


class TestBatchPredictionRequest:
    """Test BatchPredictionRequest class."""

    def test_initialization(self):
        """Test request initialization."""
        molecules = ["CCO", "CC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        request = BatchPredictionRequest(
            molecules=molecules, properties=properties, batch_size=16
        )

        assert request.molecules == molecules
        assert request.properties == properties
        assert request.batch_size == 16
        assert request.conformer_strategy == ConformerStrategy.SINGLE

    def test_default_values(self):
        """Test default values."""
        request = BatchPredictionRequest(
            molecules=["CCO"], properties=[QuantumProperty.HOMO_ENERGY]
        )

        assert request.batch_size == 32
        assert request.max_conformers == 10
        assert request.use_cache == True


class TestConformerGenerator:
    """Test ConformerGenerator class."""

    @patch("chemi_accel.qml.on_the_fly.RDKIT_AVAILABLE", False)
    def test_initialization_no_rdkit(self):
        """Test initialization without RDKit."""
        with pytest.raises(QMLModelError, match="RDKit required"):
            ConformerGenerator()

    @patch("chemi_accel.qml.on_the_fly.RDKIT_AVAILABLE", True)
    def test_initialization_with_rdkit(self):
        """Test initialization with RDKit."""
        generator = ConformerGenerator(
            max_conformers=20, energy_threshold=15.0, rms_threshold=0.3
        )

        assert generator.max_conformers == 20
        assert generator.energy_threshold == 15.0
        assert generator.rms_threshold == 0.3

    @patch("chemi_accel.qml.on_the_fly.RDKIT_AVAILABLE", True)
    @patch("chemi_accel.qml.on_the_fly.Chem")
    @patch("chemi_accel.qml.on_the_fly.AllChem")
    def test_generate_conformers_success(self, mock_allchem, mock_chem):
        """Test successful conformer generation."""
        generator = ConformerGenerator()

        # Mock RDKit components
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetAtoms.return_value = [
            Mock(GetAtomicNum=lambda: 6) for _ in range(3)
        ]

        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.rdmolops.GetFormalCharge.return_value = 0

        # Mock conformer generation
        mock_allchem.EmbedMultipleConfs.return_value = [0, 1]
        mock_allchem.UFFOptimizeMolecule.return_value = 0
        mock_allchem.UFFGetMoleculeForceField.return_value.CalcEnergy.return_value = (
            -100.0
        )

        # Mock conformer
        mock_conf = Mock()
        mock_conf.GetAtomPosition.return_value.x = 0.0
        mock_conf.GetAtomPosition.return_value.y = 0.0
        mock_conf.GetAtomPosition.return_value.z = 0.0
        mock_mol.GetConformer.return_value = mock_conf

        geometries = generator.generate_conformers("CCO")

        assert len(geometries) == 2
        assert all(isinstance(g, GeometryInput) for g in geometries)

    @patch("chemi_accel.qml.on_the_fly.RDKIT_AVAILABLE", True)
    @patch("chemi_accel.qml.on_the_fly.Chem")
    def test_generate_conformers_invalid_smiles(self, mock_chem):
        """Test conformer generation with invalid SMILES."""
        generator = ConformerGenerator()
        mock_chem.MolFromSmiles.return_value = None

        with pytest.raises(QMLModelError, match="Invalid SMILES"):
            generator.generate_conformers("INVALID")


class TestGPUBatchProcessor:
    """Test GPUBatchProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = GPUBatchProcessor(
            device="cpu", max_batch_size=64, memory_fraction=0.9
        )

        assert processor.device == "cpu"
        assert processor.max_batch_size == 64
        assert processor.memory_fraction == 0.9

    def test_process_batch_empty(self):
        """Test processing empty batch."""
        processor = GPUBatchProcessor()
        model = MockQMLModel()
        model.load_model()

        results = processor.process_batch([], model, [QuantumProperty.HOMO_ENERGY])
        assert results == []

    def test_process_batch_single_geometry(self):
        """Test processing single geometry."""
        processor = GPUBatchProcessor()
        model = MockQMLModel()
        model.load_model()

        geometry = GeometryInput(
            coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
        )

        results = processor.process_batch(
            [geometry], model, [QuantumProperty.HOMO_ENERGY]
        )

        assert len(results) == 1
        assert isinstance(results[0], QMLPrediction)

    @patch("chemi_accel.qml.on_the_fly.TORCH_AVAILABLE", True)
    @patch("chemi_accel.qml.on_the_fly.torch")
    def test_prepare_batch_tensors(self, mock_torch):
        """Test batch tensor preparation."""
        processor = GPUBatchProcessor()

        # Mock torch functions
        mock_torch.zeros.return_value = Mock()
        mock_torch.tensor.return_value = Mock()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6, 1]),
            )
        ]

        batch_data = processor._prepare_batch_tensors(geometries)

        assert "coordinates" in batch_data
        assert "atomic_numbers" in batch_data
        assert "atom_mask" in batch_data
        assert "charges" in batch_data
        assert batch_data["batch_size"] == 1
        assert batch_data["max_atoms"] == 2

    def test_get_memory_info_cpu(self):
        """Test memory info for CPU."""
        processor = GPUBatchProcessor(device="cpu")
        info = processor.get_memory_info()

        assert info["device"] == "cpu"
        assert info["memory_used"] == 0
        assert info["memory_total"] == 0


class TestOnTheFlyPredictor:
    """Test OnTheFlyPredictor class."""

    def test_initialization(self):
        """Test predictor initialization."""
        model = MockQMLModel()

        predictor = OnTheFlyPredictor(
            model=model, device="cpu", max_batch_size=32, cache_size=500
        )

        assert predictor.model is model
        assert predictor.cache_size == 500
        assert predictor.conformer_strategy == ConformerStrategy.SINGLE

    def test_predict_batch_simple(self):
        """Test simple batch prediction."""
        model = MockQMLModel()
        model.load_model()

        predictor = OnTheFlyPredictor(
            model=model, device="cpu", conformer_strategy=ConformerStrategy.SINGLE
        )

        request = BatchPredictionRequest(
            molecules=["CCO", "CC"],
            properties=[QuantumProperty.HOMO_ENERGY],
            batch_size=2,
        )

        with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
            # Mock return geometries for each molecule
            mock_prepare.return_value = [
                [
                    GeometryInput(
                        coordinates=np.array([[0.0, 0.0, 0.0]]),
                        atomic_numbers=np.array([6]),
                    )
                ],
                [
                    GeometryInput(
                        coordinates=np.array([[1.0, 0.0, 0.0]]),
                        atomic_numbers=np.array([6]),
                    )
                ],
            ]

            result = predictor.predict_batch(request)

            assert isinstance(result, BatchPredictionResult)
            assert len(result.predictions) == 2
            assert result.request is request
            assert result.execution_time > 0

    def test_predict_batch_with_errors(self):
        """Test batch prediction with errors."""
        model = MockQMLModel()
        model.load_model()

        # Make model fail
        model.predict = Mock(side_effect=Exception("Model error"))

        predictor = OnTheFlyPredictor(model=model)

        request = BatchPredictionRequest(
            molecules=["CCO"], properties=[QuantumProperty.HOMO_ENERGY]
        )

        with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
            mock_prepare.return_value = [
                [
                    GeometryInput(
                        coordinates=np.array([[0.0, 0.0, 0.0]]),
                        atomic_numbers=np.array([6]),
                    )
                ]
            ]

            result = predictor.predict_batch(request)

            assert len(result.errors) > 0
            assert result.statistics["failed_predictions"] > 0

    def test_predict_streaming(self):
        """Test streaming prediction interface."""
        model = MockQMLModel()
        model.load_model()

        predictor = OnTheFlyPredictor(model=model)

        molecules = ["CCO", "CC", "CCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        with patch.object(predictor, "predict_batch") as mock_batch:
            mock_result = BatchPredictionResult(
                predictions=[
                    QMLPrediction(
                        properties={QuantumProperty.HOMO_ENERGY: -8.0},
                        model_name="MockModel",
                    )
                ],
                request=Mock(),
                execution_time=0.1,
                memory_usage={},
                statistics={},
            )
            mock_batch.return_value = mock_result

            predictions = list(
                predictor.predict_streaming(iter(molecules), properties, buffer_size=2)
            )

            assert len(predictions) >= len(molecules)

    def test_conformer_strategy_application(self):
        """Test conformer strategy application."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model)

        # Test single conformer strategy
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-90.0,
            ),
        ]

        predictions = [
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0}),
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -7.5}),
        ]

        result = predictor._apply_conformer_strategy(
            predictions, ConformerStrategy.SINGLE, geometries
        )

        assert isinstance(result, QMLPrediction)
        assert (
            result.properties[QuantumProperty.HOMO_ENERGY] == -8.0
        )  # First prediction

    def test_average_predictions(self):
        """Test averaging predictions across conformers."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model)

        predictions = [
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0}),
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -7.0}),
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -9.0}),
        ]

        averaged = predictor._average_predictions(predictions)

        assert isinstance(averaged, QMLPrediction)
        expected_avg = (-8.0 + -7.0 + -9.0) / 3
        assert (
            abs(averaged.properties[QuantumProperty.HOMO_ENERGY] - expected_avg) < 1e-10
        )
        assert averaged.metadata["conformer_average"] == True

    def test_weighted_average_predictions(self):
        """Test weighted average with Boltzmann factors."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model)

        predictions = [
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0}),
            QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -7.0}),
        ]

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-95.0,
            ),
        ]

        weighted = predictor._weighted_average_predictions(predictions, geometries)

        assert isinstance(weighted, QMLPrediction)
        assert weighted.metadata["boltzmann_weighted"] == True
        assert "weights" in weighted.metadata

    def test_cache_functionality(self):
        """Test prediction caching."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model, cache_size=10)

        # Test cache key generation
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            )
        ]
        properties = [QuantumProperty.HOMO_ENERGY]

        cache_key = predictor._generate_cache_key(geometries, properties)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length

        # Test caching and retrieval
        prediction = QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0})
        predictor._cache_prediction(cache_key, prediction)

        cached = predictor._get_cached_prediction(cache_key)
        assert cached is prediction

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model)

        # Update statistics
        predictor._update_statistics(
            num_molecules=10, execution_time=2.0, cache_hits=3, cache_misses=7
        )

        stats = predictor.get_statistics()

        assert stats["total_predictions"] == 10
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 7
        assert stats["total_execution_time"] == 2.0
        assert stats["cache_hit_rate"] == 0.3

    def test_clear_cache(self):
        """Test cache clearing."""
        model = MockQMLModel()
        predictor = OnTheFlyPredictor(model=model)

        # Add something to cache
        predictor._prediction_cache["test"] = Mock()
        assert len(predictor._prediction_cache) == 1

        predictor.clear_cache()
        assert len(predictor._prediction_cache) == 0
