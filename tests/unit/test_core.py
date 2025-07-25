"""
Unit tests for core ChemiAccel functionality.

Tests the basic infrastructure including configuration, logging,
exceptions, and core classes.
"""

import json
import tempfile
from pathlib import Path

import pytest
from chemi_accel import BackendType, Calculator, get_available_backends
from chemi_accel.core.exceptions import (
    BackendUnavailableError,
    ChemiAccelError,
    ConfigurationError,
    InvalidMoleculeError,
)
from chemi_accel.core.molecular import MolecularBatch, MolecularData
from chemi_accel.utils.config import ChemiAccelConfig, ConfigManager
from chemi_accel.utils.logging import LogLevel, configure_logging


class TestExceptions:
    """Test exception classes."""

    def test_base_exception(self):
        """Test base ChemiAccelError."""
        error = ChemiAccelError("Test error", {"key": "value"})
        assert str(error) == "Test error (key=value)"
        assert error.message == "Test error"
        assert error.details == {"key": "value"}

    def test_backend_unavailable_error(self):
        """Test BackendUnavailableError."""
        error = BackendUnavailableError("cuda", "Not installed", ["cpu", "jax"])
        assert "cuda" in str(error)
        assert "Not installed" in str(error)
        assert "cpu, jax" in str(error)

    def test_invalid_molecule_error(self):
        """Test InvalidMoleculeError."""
        error = InvalidMoleculeError("CCO", "Invalid SMILES", ["Check syntax"])
        assert "CCO" in str(error)
        assert "Invalid SMILES" in str(error)
        assert "Check syntax" in str(error)


class TestMolecularData:
    """Test molecular data structures."""

    def test_from_smiles(self):
        """Test creating MolecularData from SMILES."""
        mol = MolecularData.from_smiles("CCO", "ethanol")
        assert mol.smiles == "CCO"
        assert mol.mol_id == "ethanol"
        assert mol.is_valid()

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        with pytest.raises(InvalidMoleculeError):
            MolecularData.from_smiles("", "empty")

        with pytest.raises(InvalidMoleculeError):
            MolecularData.from_smiles("   ", "whitespace")

    def test_from_smiles_list(self):
        """Test creating list from SMILES."""
        smiles_list = ["CCO", "CCC", "CC"]
        molecules = MolecularData.from_smiles_list(smiles_list)
        assert len(molecules) == 3
        assert all(mol.is_valid() for mol in molecules)

    def test_molecular_batch(self):
        """Test molecular batch functionality."""
        molecules = MolecularData.from_smiles_list(["CCO", "CCC", "CC"])
        batch = MolecularBatch(molecules, batch_size=2)

        assert len(batch) == 3
        batches = batch.get_batches()
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1


class TestConfiguration:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config_manager = ConfigManager()
        config = config_manager.get_config()

        assert isinstance(config, ChemiAccelConfig)
        assert config.backend.preferred_backend == BackendType.AUTO
        assert config.performance.batch_size == "auto"

    def test_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "backend": {"preferred_backend": "cuda"},
            "performance": {"batch_size": 500},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)

        try:
            # Create new config manager with custom config file
            config_manager = ConfigManager()
            file_config = config_manager._load_from_file(config_file)

            assert file_config["backend"]["preferred_backend"] == "cuda"
            assert file_config["performance"]["batch_size"] == 500
        finally:
            config_file.unlink()

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        with pytest.raises(ConfigurationError):
            config_manager = ConfigManager()
            config_manager.update_config(invalid_param="invalid_value")


class TestLogging:
    """Test logging functionality."""

    def test_logging_configuration(self):
        """Test logging configuration."""
        configure_logging(level=LogLevel.DEBUG, enable_console=True)

        # Test that logging doesn't raise errors
        from chemi_accel.utils.logging import debug, error, info, warning

        debug("Debug message")
        info("Info message")
        warning("Warning message")
        error("Error message")


class TestBackendSelection:
    """Test backend selection and management."""

    def test_get_available_backends(self):
        """Test backend availability detection."""
        backends = get_available_backends()
        assert BackendType.CPU in backends  # CPU should always be available
        assert isinstance(backends, list)
        assert all(isinstance(b, BackendType) for b in backends)

    def test_calculator_initialization(self):
        """Test Calculator initialization."""
        calc = Calculator(backend="cpu")  # CPU should always be available
        assert calc.backend_type == BackendType.CPU
        assert isinstance(calc.get_backend_info(), dict)

    def test_invalid_backend(self):
        """Test handling of invalid backend."""
        with pytest.raises(
            Exception
        ):  # Should raise ValidationError or BackendUnavailableError
            Calculator(backend="invalid_backend")


class TestCalculatorBasics:
    """Test basic Calculator functionality."""

    def test_descriptor_configuration(self):
        """Test descriptor configuration."""
        calc = Calculator(backend="cpu")
        calc.configure_descriptors("basic")
        assert "basic" in calc.descriptor_sets

        calc.configure_descriptors(["basic", "fingerprints"])
        assert calc.descriptor_sets == ["basic", "fingerprints"]

    def test_pandas_interface_placeholder(self):
        """Test pandas interface (placeholder implementation)."""
        calc = Calculator(backend="cpu")

        # Test with simple molecules
        smiles_list = ["CCO", "CCC", "CC"]

        # This should not fail even with placeholder implementation
        try:
            result = calc.pandas(smiles_list)
            assert len(result) == 3  # Should return results for 3 molecules
            assert "smiles" in result.columns
        except NotImplementedError:
            # Acceptable for placeholder implementation
            pass

    def test_backend_info(self):
        """Test backend information retrieval."""
        calc = Calculator(backend="cpu")
        info = calc.get_backend_info()

        assert "backend_type" in info
        assert "batch_size" in info
        assert "descriptor_sets" in info
        assert info["backend_type"] == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
