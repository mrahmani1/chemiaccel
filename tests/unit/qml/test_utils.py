"""
Tests for QML utility functions and classes.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from chemi_accel.qml.base import (
    BaseQMLModel,
    QMLModelError,
    QMLPrediction,
    QuantumProperty,
)
from chemi_accel.qml.utils import (
    GPUAccelerator,
    QMLBenchmark,
    QMLModelLoader,
    create_molecular_hash,
    validate_qml_environment,
)
from rdkit import Chem


class MockQMLModel(BaseQMLModel):
    """Mock QML model for testing."""

    def load_model(self):
        self._model = Mock()
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        return QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -8.0}, model_name="MockModel"
        )

    def get_supported_properties(self):
        return [QuantumProperty.HOMO_ENERGY]

    def validate_molecule(self, molecule):
        return True


class TestQMLModelLoader:
    """Test QMLModelLoader class."""

    def test_initialization_default_cache(self):
        """Test loader initialization with default cache directory."""
        loader = QMLModelLoader()

        assert loader.cache_dir.exists()
        assert loader.cache_dir.name == "qml_models"
        assert loader._model_cache == {}

    def test_initialization_custom_cache(self):
        """Test loader initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "custom_cache"
            loader = QMLModelLoader(cache_dir=cache_path)

            assert loader.cache_dir == cache_path
            assert loader.cache_dir.exists()

    @patch("chemi_accel.qml.utils.ANIModel")
    def test_load_ani_model(self, mock_ani_class):
        """Test loading ANI model."""
        mock_model = Mock()
        mock_ani_class.return_value = mock_model

        loader = QMLModelLoader()
        result = loader.load_model("ani2x")

        mock_ani_class.assert_called_once()
        mock_model.load_model.assert_called_once()
        assert result == mock_model

    @patch("chemi_accel.qml.utils.GFN2xTBModel")
    def test_load_gfn2_model(self, mock_gfn2_class):
        """Test loading GFN2-xTB model."""
        mock_model = Mock()
        mock_gfn2_class.return_value = mock_model

        loader = QMLModelLoader()
        result = loader.load_model("gfn2_xtb")

        mock_gfn2_class.assert_called_once()
        mock_model.load_model.assert_called_once()
        assert result == mock_model

    def test_load_unknown_model(self):
        """Test loading unknown model raises error."""
        loader = QMLModelLoader()

        with pytest.raises(QMLModelError):
            loader.load_model("unknown_model")

    def test_model_caching(self):
        """Test that models are cached after loading."""
        with patch("chemi_accel.qml.utils.ANIModel") as mock_ani_class:
            mock_model = Mock()
            mock_ani_class.return_value = mock_model

            loader = QMLModelLoader()

            # Load model twice
            result1 = loader.load_model("ani2x")
            result2 = loader.load_model("ani2x")

            # Should use cached version second time
            assert result1 == result2
            assert result1 == mock_model
            mock_ani_class.assert_called_once()  # Only called once due to caching

    def test_download_model_existing(self):
        """Test download behavior with existing model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            loader = QMLModelLoader(cache_dir=cache_dir)

            # Create existing model directory
            model_dir = cache_dir / "test_model"
            model_dir.mkdir(parents=True)

            result = loader.download_model("test_model")
            assert result == model_dir

    def test_download_model_force(self):
        """Test forced download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            loader = QMLModelLoader(cache_dir=cache_dir)

            result = loader.download_model("test_model", force_download=True)
            assert result.exists()
            assert result.name == "test_model"

    def test_validate_model_files(self):
        """Test model file validation."""
        loader = QMLModelLoader()

        # Non-existent path
        assert not loader.validate_model_files(Path("/nonexistent"))

        # Existing path
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            assert loader.validate_model_files(model_path)

    def test_list_available_models(self):
        """Test listing available models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            loader = QMLModelLoader(cache_dir=cache_dir)

            # Create some model directories
            (cache_dir / "model1").mkdir(parents=True)
            (cache_dir / "model2").mkdir(parents=True)
            (cache_dir / "not_a_dir.txt").touch()

            models = loader.list_available_models()
            assert len(models) == 2
            assert "model1" in models
            assert "model2" in models
            assert "not_a_dir.txt" not in models

    def test_clear_cache(self):
        """Test cache clearing."""
        loader = QMLModelLoader()
        loader._model_cache["test"] = Mock()

        assert len(loader._model_cache) == 1
        loader.clear_cache()
        assert len(loader._model_cache) == 0


class TestGPUAccelerator:
    """Test GPUAccelerator class."""

    def test_initialization_cpu(self):
        """Test accelerator initialization for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            accelerator = GPUAccelerator(device="auto")
            assert accelerator.device == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    def test_initialization_gpu(self, mock_props, mock_count, mock_available):
        """Test accelerator initialization for GPU."""
        # Mock GPU properties
        props1 = Mock()
        props1.total_memory = 8 * 1024**3  # 8 GB
        props2 = Mock()
        props2.total_memory = 16 * 1024**3  # 16 GB

        mock_props.side_effect = [props1, props2]

        accelerator = GPUAccelerator(device="auto")
        assert accelerator.device == "cuda:1"  # Should select GPU with more memory

    def test_initialization_explicit_device(self):
        """Test accelerator with explicit device."""
        accelerator = GPUAccelerator(device="cuda:0")
        assert accelerator.device == "cuda:0"

    def test_optimize_batch_size_cpu(self):
        """Test batch size optimization for CPU."""
        accelerator = GPUAccelerator(device="cpu")

        batch_size = accelerator.optimize_batch_size(
            model=Mock(), sample_input=Mock(), max_batch_size=1000
        )

        assert batch_size == 32  # Conservative CPU batch size

    @patch("torch.cuda.is_available", return_value=True)
    def test_optimize_batch_size_gpu(self, mock_available):
        """Test batch size optimization for GPU."""
        accelerator = GPUAccelerator(device="cuda:0")

        with patch("torch.cuda.empty_cache"):
            batch_size = accelerator.optimize_batch_size(
                model=Mock(), sample_input=Mock(), max_batch_size=128
            )

            assert 1 <= batch_size <= 128

    def test_get_device_info_cpu(self):
        """Test device info for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            accelerator = GPUAccelerator(device="cpu")
            info = accelerator.get_device_info()

            assert info["device"] == "cpu"
            assert info["device_type"] == "cpu"
            assert "gpu_name" not in info

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=1024**3)
    def test_get_device_info_gpu(self, mock_memory, mock_props, mock_available):
        """Test device info for GPU."""
        # Mock GPU properties
        props = Mock()
        props.name = "Test GPU"
        props.total_memory = 8 * 1024**3
        props.major = 8
        props.minor = 6
        props.multi_processor_count = 68
        mock_props.return_value = props

        accelerator = GPUAccelerator(device="cuda:0")
        info = accelerator.get_device_info()

        assert info["device"] == "cuda:0"
        assert info["device_type"] == "cuda"
        assert info["gpu_name"] == "Test GPU"
        assert info["gpu_memory_total"] == 8 * 1024**3
        assert info["gpu_compute_capability"] == "8.6"


class TestQMLBenchmark:
    """Test QMLBenchmark class."""

    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = QMLBenchmark()
        assert benchmark.results == {}

    def test_benchmark_model_success(self):
        """Test successful model benchmarking."""
        model = MockQMLModel()
        model.load_model()

        benchmark = QMLBenchmark()
        test_molecules = ["CCO", "CC", "CCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        with patch("time.time", side_effect=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
            results = benchmark.benchmark_model(
                model=model,
                test_molecules=test_molecules,
                properties=properties,
                num_runs=3,
            )

        assert results["model_name"] == "MockQMLModel"
        assert results["num_molecules"] == 3
        assert results["num_properties"] == 1
        assert len(results["runs"]) == 3
        assert all(run["success"] for run in results["runs"])
        assert "statistics" in results

    def test_benchmark_model_failure(self):
        """Test model benchmarking with failures."""
        model = MockQMLModel()
        model.predict = Mock(side_effect=Exception("Test error"))
        model.load_model()

        benchmark = QMLBenchmark()

        results = benchmark.benchmark_model(
            model=model,
            test_molecules=["CCO"],
            properties=[QuantumProperty.HOMO_ENERGY],
            num_runs=2,
        )

        assert len(results["runs"]) == 2
        assert all(not run["success"] for run in results["runs"])
        assert all("error" in run for run in results["runs"])

    def test_compare_models(self):
        """Test model comparison."""
        model1 = MockQMLModel()
        model2 = MockQMLModel()

        benchmark = QMLBenchmark()
        results = benchmark.compare_models([model1, model2], test_data=Mock())

        assert "models" in results
        assert len(results["models"]) == 2
        assert "MockQMLModel" in results["models"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_molecular_hash_smiles(self):
        """Test molecular hash creation from SMILES."""
        smiles = "CCO"
        hash1 = create_molecular_hash(smiles)
        hash2 = create_molecular_hash(smiles)

        assert hash1 == hash2  # Same molecule should give same hash
        assert len(hash1) == 32  # MD5 hash length

        # Different molecules should give different hashes
        hash3 = create_molecular_hash("CC")
        assert hash1 != hash3

    def test_create_molecular_hash_rdkit_mol(self):
        """Test molecular hash creation from RDKit molecule."""
        mol = Chem.MolFromSmiles("CCO")
        hash1 = create_molecular_hash(mol)
        hash2 = create_molecular_hash("CCO")  # Same molecule as SMILES

        # Should produce same hash regardless of input format
        assert hash1 == hash2

    @patch("importlib.import_module")
    def test_validate_qml_environment_all_available(self, mock_import):
        """Test environment validation with all dependencies available."""
        # Mock successful imports
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True

        def mock_import_func(module_name):
            if module_name == "torch":
                return mock_torch
            else:
                return Mock()  # Other modules

        mock_import.side_effect = mock_import_func

        with patch("torch.cuda.is_available", return_value=True):
            validation = validate_qml_environment()

            assert validation["torch_available"] == True
            assert validation["cuda_available"] == True

    def test_validate_qml_environment_missing_deps(self):
        """Test environment validation with missing dependencies."""
        with patch("importlib.import_module", side_effect=ImportError):
            validation = validate_qml_environment()

            # All should be False due to import errors
            for key, value in validation.items():
                if key != "xtb_available":  # XTB is checked differently
                    assert value == False

    @patch("subprocess.run")
    def test_validate_qml_environment_xtb_available(self, mock_run):
        """Test XTB binary availability check."""
        mock_run.return_value.returncode = 0

        validation = validate_qml_environment()
        assert validation["xtb_available"] == True

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_validate_qml_environment_xtb_missing(self, mock_run):
        """Test XTB binary missing."""
        validation = validate_qml_environment()
        assert validation["xtb_available"] == False

    def test_molecular_hash_consistency(self):
        """Test that molecular hashes are consistent across runs."""
        molecules = ["CCO", "CC", "CCC", "CCCC"]

        # Generate hashes multiple times
        hashes1 = [create_molecular_hash(mol) for mol in molecules]
        hashes2 = [create_molecular_hash(mol) for mol in molecules]

        assert hashes1 == hashes2

        # All hashes should be different
        assert len(set(hashes1)) == len(hashes1)
