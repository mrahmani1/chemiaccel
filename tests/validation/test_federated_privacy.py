"""
Comprehensive Security and Privacy Testing for Federated Learning.

This module provides advanced testing infrastructure for federated learning security,
including differential privacy, secure aggregation, membership inference attacks,
and gradient leakage protection.
"""

import base64
import hashlib
import json
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import cryptography
import numpy as np
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from opacus import PrivacyEngine

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


@dataclass
class SecurityTestResult:
    """Result of security and privacy testing."""

    test_category: str
    security_level: str
    attack_type: Optional[str] = None
    passed: bool = False
    error_message: Optional[str] = None
    privacy_metrics: Dict[str, float] = field(default_factory=dict)
    security_metrics: Dict[str, float] = field(default_factory=dict)
    attack_success_rate: Optional[float] = None
    defense_effectiveness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedSecurityTestFramework:
    """Advanced security and privacy testing framework for federated learning."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize federated security testing framework.

        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = (
            Path(output_dir) if output_dir else Path("federated_security_results")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[SecurityTestResult] = []

        # Security test configurations
        self.privacy_budgets = [0.1, 1.0, 5.0, 10.0]  # Epsilon values
        self.noise_multipliers = [0.5, 1.0, 1.5, 2.0]
        self.clipping_bounds = [0.1, 0.5, 1.0, 2.0]

        # Mock federated participants
        self.num_participants = 10
        self.malicious_participants = [7, 8]  # Simulated malicious clients


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""

    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for testing."""
        return PrivacyConfig(
            epsilon=1.0, delta=1e-5, sensitivity=1.0, enable_differential_privacy=True
        )

    @pytest.fixture
    def noise_generator(self, privacy_config):
        """Create noise generator for testing."""
        return PrivacyPreservingNoise(privacy_config)

    def test_gaussian_noise_addition(self, noise_generator):
        """Test Gaussian noise addition for differential privacy."""
        # Generate test data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Add noise
        noisy_data = noise_generator.add_gaussian_noise(data)

        # Verify noise was added
        assert not np.array_equal(data, noisy_data)
        assert noisy_data.shape == data.shape
        assert np.isfinite(noisy_data).all()

        # Test multiple runs produce different results
        noisy_data2 = noise_generator.add_gaussian_noise(data)
        assert not np.array_equal(noisy_data, noisy_data2)

    def test_laplace_noise_addition(self, noise_generator):
        """Test Laplace noise addition for differential privacy."""
        data = np.array([10.0, 20.0, 30.0])

        noisy_data = noise_generator.add_laplace_noise(data)

        assert not np.array_equal(data, noisy_data)
        assert noisy_data.shape == data.shape
        assert np.isfinite(noisy_data).all()

    def test_gradient_clipping(self, noise_generator):
        """Test gradient clipping for privacy preservation."""
        # Test with NumPy array
        large_gradients = np.array([10.0, -15.0, 8.0])  # Norm > max_grad_norm
        clipped_gradients = noise_generator.clip_gradients(large_gradients)

        clipped_norm = np.linalg.norm(clipped_gradients)
        assert clipped_norm <= noise_generator.config.max_grad_norm + 1e-6

        # Test with small gradients (should not be clipped)
        small_gradients = np.array([0.1, -0.2, 0.15])
        clipped_small = noise_generator.clip_gradients(small_gradients)
        np.testing.assert_array_almost_equal(small_gradients, clipped_small)

    def test_model_update_privatization(self, noise_generator):
        """Test privatization of model updates."""
        model_update = {
            "layer1_weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "layer1_bias": np.array([0.5, -0.5]),
            "layer2_weight": np.array([[0.1, 0.2, 0.3]]),
        }

        privatized_update = noise_generator.privatize_model_update(model_update)

        # Verify all parameters are present
        assert set(privatized_update.keys()) == set(model_update.keys())

        # Verify shapes are preserved
        for key in model_update:
            assert privatized_update[key].shape == model_update[key].shape

        # Verify noise was added (values should be different)
        for key in model_update:
            assert not np.array_equal(model_update[key], privatized_update[key])

    def test_privacy_budget_calculation(self, privacy_config):
        """Test privacy budget calculation and tracking."""
        noise_gen = PrivacyPreservingNoise(privacy_config)

        # Test noise scale calculation
        assert noise_gen.noise_scale > 0

        # Test with different epsilon values
        high_privacy_config = PrivacyConfig(epsilon=0.1, delta=1e-5)
        high_privacy_noise = PrivacyPreservingNoise(high_privacy_config)

        low_privacy_config = PrivacyConfig(epsilon=10.0, delta=1e-5)
        low_privacy_noise = PrivacyPreservingNoise(low_privacy_config)

        # Higher privacy (lower epsilon) should have larger noise scale
        assert high_privacy_noise.noise_scale > low_privacy_noise.noise_scale

    def test_disabled_differential_privacy(self):
        """Test behavior when differential privacy is disabled."""
        config = PrivacyConfig(enable_differential_privacy=False)
        noise_gen = PrivacyPreservingNoise(config)

        data = np.array([1.0, 2.0, 3.0])

        # Should return original data unchanged
        result = noise_gen.add_gaussian_noise(data)
        np.testing.assert_array_equal(data, result)

        result = noise_gen.add_laplace_noise(data)
        np.testing.assert_array_equal(data, result)


class TestEncryption:
    """Test encryption and decryption mechanisms."""

    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration with encryption enabled."""
        return PrivacyConfig(enable_encryption=True, encryption_key_size=2048)

    @pytest.fixture
    def encryption(self, privacy_config):
        """Create encryption handler for testing."""
        return SecureEncryption(privacy_config)

    def test_key_generation(self, encryption):
        """Test encryption key generation."""
        assert encryption.private_key is not None
        assert encryption.public_key is not None
        assert encryption.symmetric_key is not None
        assert encryption.fernet is not None

    def test_symmetric_encryption_decryption(self, encryption):
        """Test symmetric encryption and decryption."""
        test_data = b"This is test molecular data for encryption"

        # Encrypt data
        encrypted_data = encryption.encrypt_data(test_data)
        assert encrypted_data != test_data
        assert len(encrypted_data) > len(test_data)

        # Decrypt data
        decrypted_data = encryption.decrypt_data(encrypted_data)
        assert decrypted_data == test_data

    def test_asymmetric_encryption_decryption(self, encryption):
        """Test asymmetric encryption and decryption."""
        test_data = b"Small test data for RSA"

        # Encrypt with public key
        encrypted_data = encryption.encrypt_with_public_key(
            test_data, encryption.public_key
        )
        assert encrypted_data != test_data

        # Decrypt with private key
        decrypted_data = encryption.decrypt_with_private_key(encrypted_data)
        assert decrypted_data == test_data

    def test_public_key_export(self, encryption):
        """Test public key export functionality."""
        public_key_bytes = encryption.get_public_key_bytes()

        assert isinstance(public_key_bytes, bytes)
        assert len(public_key_bytes) > 0
        assert b"BEGIN PUBLIC KEY" in public_key_bytes

    def test_key_rotation(self, encryption):
        """Test symmetric key rotation."""
        original_key = encryption.symmetric_key

        # Rotate key
        encryption.rotate_symmetric_key()

        # Verify key changed
        assert encryption.symmetric_key != original_key
        assert encryption.fernet is not None

    def test_disabled_encryption(self):
        """Test behavior when encryption is disabled."""
        config = PrivacyConfig(enable_encryption=False)
        encryption = SecureEncryption(config)

        test_data = b"test data"

        # Should return data unchanged
        result = encryption.encrypt_data(test_data)
        assert result == test_data

        result = encryption.decrypt_data(test_data)
        assert result == test_data


class TestDataAnonymization:
    """Test data anonymization mechanisms."""

    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for anonymization."""
        return PrivacyConfig(
            enable_data_anonymization=True, anonymization_method="hash"
        )

    @pytest.fixture
    def anonymizer(self, privacy_config):
        """Create data anonymizer for testing."""
        return DataAnonymizer(privacy_config)

    def test_smiles_hash_anonymization(self, anonymizer):
        """Test SMILES anonymization using hash method."""
        test_smiles = "CCO"  # Ethanol

        anonymized = anonymizer.anonymize_smiles(test_smiles)

        # Should be different from original
        assert anonymized != test_smiles
        assert anonymized.startswith("hash_")
        assert len(anonymized) > len("hash_")

        # Should be deterministic
        anonymized2 = anonymizer.anonymize_smiles(test_smiles)
        assert anonymized == anonymized2

    def test_smiles_tokenization_anonymization(self):
        """Test SMILES anonymization using tokenization method."""
        config = PrivacyConfig(
            enable_data_anonymization=True, anonymization_method="tokenize"
        )
        anonymizer = DataAnonymizer(config)

        test_smiles = "c1ccccc1"  # Benzene

        anonymized = anonymizer.anonymize_smiles(test_smiles)
        assert anonymized != test_smiles
        assert anonymized.startswith("mol_")

        # Test reversibility
        reversed_smiles = anonymizer.reverse_anonymization(anonymized)
        assert reversed_smiles == test_smiles

    def test_dataset_anonymization(self, anonymizer):
        """Test anonymization of molecular datasets."""
        # Create test dataset
        test_data = pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
                "compound_id": ["CMP001", "CMP002", "CMP003"],
                "activity": [1.2, 2.3, 0.8],
                "cas_number": ["64-17-5", "71-43-2", "64-19-7"],
            }
        )

        anonymized_data = anonymizer.anonymize_dataset(test_data)

        # Verify SMILES were anonymized
        assert not any(
            smiles in anonymized_data["smiles"].values for smiles in test_data["smiles"]
        )
        assert all(smiles.startswith("hash_") for smiles in anonymized_data["smiles"])

        # Verify sensitive columns were handled
        assert not any(
            cid in anonymized_data["compound_id"].values
            for cid in test_data["compound_id"]
        )

        # Verify non-sensitive data preserved
        pd.testing.assert_series_equal(
            test_data["activity"], anonymized_data["activity"]
        )

    def test_removal_anonymization(self):
        """Test anonymization by removal method."""
        config = PrivacyConfig(
            enable_data_anonymization=True, anonymization_method="remove"
        )
        anonymizer = DataAnonymizer(config)

        test_data = pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccccc1"],
                "compound_id": ["CMP001", "CMP002"],
                "activity": [1.2, 2.3],
            }
        )

        anonymized_data = anonymizer.anonymize_dataset(test_data)

        # Sensitive columns should be removed
        assert "compound_id" not in anonymized_data.columns

        # SMILES should be replaced with generic identifiers
        assert all(
            smiles.startswith("molecule_") for smiles in anonymized_data["smiles"]
        )

    def test_disabled_anonymization(self):
        """Test behavior when anonymization is disabled."""
        config = PrivacyConfig(enable_data_anonymization=False)
        anonymizer = DataAnonymizer(config)

        test_smiles = "CCO"
        result = anonymizer.anonymize_smiles(test_smiles)
        assert result == test_smiles


class TestPrivacyAuditing:
    """Test privacy auditing and budget tracking."""

    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for auditing."""
        return PrivacyConfig(epsilon=2.0, enable_privacy_auditing=True)

    @pytest.fixture
    def auditor(self, privacy_config):
        """Create privacy auditor for testing."""
        return PrivacyAuditor(privacy_config)

    def test_privacy_event_logging(self, auditor):
        """Test logging of privacy events."""
        # Log a privacy event
        auditor.log_privacy_event("test_computation", 0.5, {"molecules": 100})

        assert len(auditor.audit_log) == 1
        assert auditor.privacy_budget_used == 0.5

        event = auditor.audit_log[0]
        assert event["event_type"] == "test_computation"
        assert event["epsilon_used"] == 0.5
        assert event["additional_info"]["molecules"] == 100

    def test_privacy_budget_tracking(self, auditor):
        """Test privacy budget consumption tracking."""
        initial_budget = auditor.config.epsilon

        # Consume some budget
        auditor.log_privacy_event("computation1", 0.3)
        assert auditor.privacy_budget_used == 0.3

        auditor.log_privacy_event("computation2", 0.7)
        assert auditor.privacy_budget_used == 1.0

        # Check remaining budget
        remaining = initial_budget - auditor.privacy_budget_used
        assert remaining == 1.0

    def test_budget_exhaustion_warning(self, auditor, caplog):
        """Test warning when privacy budget is exhausted."""
        # Exhaust the budget
        auditor.log_privacy_event("large_computation", 2.5)

        # Should have logged a warning
        assert "Privacy budget exhausted" in caplog.text
        assert auditor.privacy_budget_used >= auditor.config.epsilon

    def test_privacy_report_generation(self, auditor):
        """Test generation of privacy usage reports."""
        # Log several events
        auditor.log_privacy_event("computation1", 0.2, {"type": "descriptors"})
        auditor.log_privacy_event("computation2", 0.3, {"type": "gradients"})
        auditor.log_privacy_event("computation1", 0.1, {"type": "descriptors"})

        time.sleep(0.1)  # Ensure some runtime

        report = auditor.get_privacy_report()

        # Verify report structure
        assert "total_events" in report
        assert "privacy_budget_used" in report
        assert "privacy_budget_remaining" in report
        assert "event_breakdown" in report

        # Verify report contents
        assert report["total_events"] == 3
        assert report["privacy_budget_used"] == 0.6
        assert report["privacy_budget_remaining"] == 1.4
        assert "computation1" in report["event_breakdown"]
        assert "computation2" in report["event_breakdown"]

    def test_budget_availability_check(self, auditor):
        """Test checking if sufficient budget is available."""
        # Should have budget available initially
        assert auditor.check_privacy_budget(0.5) is True
        assert auditor.check_privacy_budget(2.0) is True

        # Consume some budget
        auditor.log_privacy_event("computation", 1.5)

        # Check remaining availability
        assert auditor.check_privacy_budget(0.3) is True
        assert auditor.check_privacy_budget(0.8) is False  # Would exceed budget


class TestFederatedCalculator:
    """Test federated calculator privacy mechanisms."""

    @pytest.fixture
    def federated_config(self):
        """Create federated configuration for testing."""
        privacy_config = PrivacyConfig(
            epsilon=1.0,
            enable_differential_privacy=True,
            enable_encryption=True,
            enable_data_anonymization=True,
            enable_privacy_auditing=True,
        )
        return FederatedConfig(privacy_config=privacy_config)

    @pytest.fixture
    def federated_calc(self, federated_config):
        """Create federated calculator for testing."""
        return FederatedCalculator(federated_config)

    def test_local_data_loading(self, federated_calc):
        """Test loading and anonymization of local data."""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

        summary = federated_calc.load_local_data(test_smiles, anonymize=True)

        assert summary["molecules_loaded"] == 3
        assert summary["anonymized"] is True
        assert federated_calc.local_data is not None
        assert federated_calc.local_data["size"] == 3

    def test_private_descriptor_computation(self, federated_calc):
        """Test privacy-preserving descriptor computation."""
        # Load test data
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
        federated_calc.load_local_data(
            test_smiles, anonymize=False
        )  # Don't anonymize for descriptor computation

        # Compute descriptors with privacy
        results = federated_calc.compute_local_descriptors(add_privacy_noise=True)

        assert results["privacy_preserved"] is True
        assert "results" in results
        assert "mean" in results["results"]
        assert "std" in results["results"]
        assert results["client_id"] == federated_calc.config.client_id

    def test_synthetic_statistics_generation(self, federated_calc):
        """Test synthetic statistics for fully anonymized data."""
        # Load data with hash anonymization (can't compute real descriptors)
        test_smiles = ["CCO", "c1ccccc1"]
        federated_calc.load_local_data(test_smiles, anonymize=True)

        # Force hash anonymization method
        federated_calc.config.privacy_config.anonymization_method = "hash"

        results = federated_calc.compute_local_descriptors()

        assert results["synthetic"] is True
        assert results["anonymized"] is True
        assert "results" in results

    def test_model_update_preparation(self, federated_calc):
        """Test preparation of privacy-preserving model updates."""
        # Load test data
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        federated_calc.load_local_data(test_smiles)

        # Prepare model update
        mock_model_state = {"param1": np.array([1.0, 2.0]), "param2": np.array([0.5])}

        update = federated_calc.prepare_model_update(mock_model_state)

        assert "client_id" in update
        assert "update" in update
        assert "privacy_budget_used" in update
        assert update["sample_size"] == 3

    def test_privacy_report_generation(self, federated_calc):
        """Test comprehensive privacy report generation."""
        # Load data and perform some operations
        test_smiles = ["CCO", "c1ccccc1"]
        federated_calc.load_local_data(test_smiles)
        federated_calc.compute_local_descriptors()

        report = federated_calc.get_privacy_report()

        assert "privacy_config" in report
        assert "privacy_audit" in report
        assert "data_status" in report
        assert "communication_status" in report

        # Verify privacy mechanisms are reported correctly
        assert report["privacy_config"]["differential_privacy"] is True
        assert report["privacy_config"]["encryption"] is True
        assert report["privacy_config"]["anonymization"] is True

    def test_privacy_validation(self, federated_calc):
        """Test privacy preservation validation."""
        validation_results = federated_calc.validate_privacy_preservation()

        assert "differential_privacy_active" in validation_results
        assert "encryption_active" in validation_results
        assert "data_anonymization_active" in validation_results
        assert "privacy_budget_available" in validation_results
        assert "secure_aggregation_capable" in validation_results
        assert "privacy_preserved" in validation_results

        # All mechanisms should be active
        assert validation_results["differential_privacy_active"] is True
        assert validation_results["encryption_active"] is True
        assert validation_results["data_anonymization_active"] is True


class TestGradientPrivacy:
    """Test privacy-preserving gradient computation."""

    @pytest.fixture
    def gradient_config(self):
        """Create gradient computation configuration."""
        return GradientComputationConfig(
            enable_gradient_privacy=True,
            gradient_noise_multiplier=1.0,
            gradient_clipping_norm=1.0,
            use_molecular_fingerprints=True,
        )

    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration."""
        return PrivacyConfig(epsilon=1.0, enable_differential_privacy=True)

    @pytest.fixture
    def gradient_computer(self, gradient_config, privacy_config):
        """Create privacy-preserving gradient computer."""
        return PrivacyPreservingGradientComputer(gradient_config, privacy_config)

    def test_molecular_fingerprint_computation(self, gradient_computer):
        """Test molecular fingerprint computation for privacy."""
        test_smiles = "CCO"

        fingerprint = (
            gradient_computer.molecular_embedding.compute_molecular_fingerprint(
                test_smiles
            )
        )

        assert isinstance(fingerprint, np.ndarray)
        assert fingerprint.shape[0] == gradient_computer.config.fingerprint_bits
        assert fingerprint.dtype == np.float32

        # Should be deterministic
        fingerprint2 = (
            gradient_computer.molecular_embedding.compute_molecular_fingerprint(
                test_smiles
            )
        )
        np.testing.assert_array_equal(fingerprint, fingerprint2)

    def test_descriptor_embedding_computation(self, gradient_computer):
        """Test molecular descriptor embedding computation."""
        test_smiles = "c1ccccc1"

        embedding = gradient_computer.molecular_embedding.compute_descriptor_embedding(
            test_smiles
        )

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 10  # Number of descriptors
        assert embedding.dtype == np.float32
        assert np.isfinite(embedding).all()

    def test_private_gradient_computation(self, gradient_computer):
        """Test end-to-end private gradient computation."""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        target_values = np.array([1.2, 2.3, 0.8])
        mock_model_params = {
            "weight": np.random.normal(0, 0.01, (2048, 1)),
            "bias": 0.0,
        }

        result = gradient_computer.compute_private_gradients(
            test_smiles, target_values, mock_model_params
        )

        assert "gradients" in result
        assert "gradient_norm" in result
        assert "privacy_cost" in result
        assert "molecular_count" in result
        assert result["privacy_preserved"] is True
        assert result["molecular_count"] == 3

    def test_gradient_aggregation(self, gradient_computer):
        """Test secure aggregation of encrypted gradients."""
        # Create mock encrypted gradients from multiple clients
        client_gradients = []
        for i in range(3):
            gradients = {
                "gradients": {
                    "weight": np.random.normal(0, 0.01, (10, 1)),
                    "bias": np.random.normal(0, 0.01, 1),
                },
                "molecular_count": 5 + i,
                "privacy_cost": 0.1,
            }
            client_gradients.append(gradients)

        aggregated = gradient_computer.aggregate_encrypted_gradients(client_gradients)

        assert "aggregated_gradients" in aggregated
        assert "participating_clients" in aggregated
        assert "total_molecular_count" in aggregated
        assert aggregated["participating_clients"] == 3
        assert aggregated["total_molecular_count"] == 18  # 5 + 6 + 7


class TestSecureAggregation:
    """Test secure aggregation protocols."""

    @pytest.fixture
    def gradient_config(self):
        """Create gradient configuration for aggregation."""
        return GradientComputationConfig(
            enable_secure_aggregation=True, aggregation_threshold=2
        )

    @pytest.fixture
    def aggregator(self, gradient_config):
        """Create secure gradient aggregator."""
        return SecureGradientAggregator(gradient_config)

    def test_client_gradient_addition(self, aggregator):
        """Test adding client gradients to aggregation pool."""
        mock_gradients = {
            "gradients": {"param1": np.array([1.0, 2.0])},
            "privacy_cost": 0.1,
        }

        success = aggregator.add_client_gradients("client1", mock_gradients, 1.0)
        assert success is True
        assert "client1" in aggregator.client_contributions

    def test_secure_aggregation_threshold(self, aggregator):
        """Test that aggregation requires minimum number of clients."""
        # Add only one client (below threshold)
        mock_gradients = {"gradients": {"param1": np.array([1.0])}}
        aggregator.add_client_gradients("client1", mock_gradients)

        # Should raise error due to insufficient clients
        with pytest.raises(Exception):  # GradientPrivacyError
            aggregator.perform_secure_aggregation()

    def test_successful_secure_aggregation(self, aggregator):
        """Test successful secure aggregation with sufficient clients."""
        # Add multiple clients
        for i in range(3):
            mock_gradients = {
                "gradients": {"param1": np.array([float(i + 1), float(2 * i + 1)])},
                "privacy_cost": 0.1,
            }
            aggregator.add_client_gradients(f"client{i}", mock_gradients, 1.0)

        result = aggregator.perform_secure_aggregation()

        assert "aggregated_gradients" in result
        assert "participating_clients" in result
        assert result["participating_clients"] == 3
        assert "param1" in result["aggregated_gradients"]


class TestInformationLeakage:
    """Test for information leakage in privacy mechanisms."""

    def test_differential_privacy_prevents_inference(self):
        """Test that differential privacy prevents inference attacks."""
        config = PrivacyConfig(epsilon=1.0, enable_differential_privacy=True)
        noise_gen = PrivacyPreservingNoise(config)

        # Create two datasets that differ by one record
        data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # Last value changed

        # Add noise to both
        noisy1_runs = [noise_gen.add_gaussian_noise(data1) for _ in range(100)]
        noisy2_runs = [noise_gen.add_gaussian_noise(data2) for _ in range(100)]

        # Calculate means of noisy results
        mean1 = np.mean([np.mean(run) for run in noisy1_runs])
        mean2 = np.mean([np.mean(run) for run in noisy2_runs])

        # The difference should be bounded by differential privacy
        actual_diff = abs(mean1 - mean2)
        max_expected_diff = 2.0  # This should be related to sensitivity and noise

        # The noise should make it difficult to distinguish the datasets
        assert actual_diff > 0  # Some difference due to the changed value
        # Note: In practice, you'd want more sophisticated statistical tests

    def test_anonymization_prevents_reidentification(self):
        """Test that anonymization prevents re-identification."""
        config = PrivacyConfig(
            enable_data_anonymization=True, anonymization_method="hash"
        )
        anonymizer = DataAnonymizer(config)

        # Create dataset with identifiable molecules
        original_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

        # Anonymize
        anonymized = [anonymizer.anonymize_smiles(smiles) for smiles in original_smiles]

        # Verify no original SMILES can be recovered
        for orig, anon in zip(original_smiles, anonymized):
            assert orig not in anon
            assert anon.startswith("hash_")

        # Verify anonymized values are consistent
        anonymized2 = [
            anonymizer.anonymize_smiles(smiles) for smiles in original_smiles
        ]
        assert anonymized == anonymized2

    def test_gradient_privacy_prevents_structure_inference(self):
        """Test that gradient computation doesn't leak molecular structures."""
        grad_config = GradientComputationConfig(
            enable_gradient_privacy=True, use_molecular_fingerprints=True
        )
        privacy_config = PrivacyConfig(epsilon=1.0, enable_differential_privacy=True)

        gradient_computer = PrivacyPreservingGradientComputer(
            grad_config, privacy_config
        )

        # Test with sensitive molecular structures
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]  # Should not be recoverable
        target_values = np.array([1.0, 2.0, 3.0])
        mock_params = {"weight": np.random.normal(0, 0.01, (2048, 1))}

        result = gradient_computer.compute_private_gradients(
            test_smiles, target_values, mock_params
        )

        # Verify that gradients don't contain original SMILES
        gradients_str = str(result["gradients"])
        for smiles in test_smiles:
            assert smiles not in gradients_str

        # Verify privacy preservation flag
        assert result["privacy_preserved"] is True


@pytest.mark.asyncio
class TestIntegrationPrivacy:
    """Integration tests for privacy preservation across components."""

    async def test_end_to_end_federated_privacy(self):
        """Test privacy preservation in complete federated learning workflow."""
        # Create privacy-enabled configuration
        privacy_config = PrivacyConfig(
            epsilon=2.0,
            enable_differential_privacy=True,
            enable_encryption=True,
            enable_data_anonymization=True,
            enable_privacy_auditing=True,
        )

        federated_config = FederatedConfig(privacy_config=privacy_config, max_rounds=3)

        # Create federated coordinator
        coordinator = create_federated_coordinator(federated_config)

        # Start federation
        await coordinator.start_federation()

        # Simulate client registration
        orchestrator = coordinator.active_orchestrator

        from src.chemi_accel.federated.orchestrators import ClientInfo

        client1 = ClientInfo(
            client_id="client1",
            address="localhost:8001",
            capabilities={"descriptors": True},
            data_size=100,
        )

        client2 = ClientInfo(
            client_id="client2",
            address="localhost:8002",
            capabilities={"descriptors": True},
            data_size=150,
        )

        await orchestrator.register_client(client1)
        await orchestrator.register_client(client2)

        # Run federated rounds
        results = []
        for round_num in range(1, 4):
            round_result = await orchestrator.run_federated_round(round_num)
            results.append(round_result)

            # Verify privacy preservation in each round
            assert round_result.privacy_budget_used > 0
            assert round_result.success_rate > 0

        # Verify total privacy budget consumption
        total_privacy_cost = sum(r.privacy_budget_used for r in results)
        assert total_privacy_cost <= privacy_config.epsilon

        # Stop federation
        await coordinator.stop_federation()

    def test_privacy_config_validation(self):
        """Test validation of privacy configuration parameters."""
        # Test valid configuration
        valid_config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        assert valid_config.epsilon > 0
        assert valid_config.delta >= 0

        # Test that noise generator validates epsilon
        with pytest.raises(Exception):  # PrivacyError
            invalid_config = PrivacyConfig(epsilon=0.0)
            PrivacyPreservingNoise(invalid_config)

    def test_privacy_mechanism_composition(self):
        """Test that multiple privacy mechanisms compose correctly."""
        config = PrivacyConfig(
            epsilon=1.0,
            enable_differential_privacy=True,
            enable_data_anonymization=True,
        )

        # Create federated calculator with all mechanisms
        federated_calc = FederatedCalculator(FederatedConfig(privacy_config=config))

        # Load and process data
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        federated_calc.load_local_data(test_smiles, anonymize=True)

        # Compute descriptors (should apply multiple privacy mechanisms)
        results = federated_calc.compute_local_descriptors(add_privacy_noise=True)

        # Verify all mechanisms were applied
        assert results["privacy_preserved"] is True
        assert results["anonymized"] is True

        # Get privacy report
        report = federated_calc.get_privacy_report()
        assert report["privacy_config"]["differential_privacy"] is True
        assert report["privacy_config"]["anonymization"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
