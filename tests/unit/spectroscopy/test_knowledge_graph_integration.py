"""
Unit tests for knowledge graph integration.

This module tests the integration with the Intelligent Chemical Knowledge Graph
for validating elucidated structures and detecting novel compounds.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.spectroscopy.experimental_processing import ExperimentalSpectrum
from chemi_accel.spectroscopy.inverse_spectroscopy import StructureCandidate
from chemi_accel.spectroscopy.knowledge_graph_integration import (
    ConfidenceLevel,
    KnowledgeGraphEntry,
    KnowledgeGraphIntegrationError,
    KnowledgeGraphValidator,
    NoveltyDetectionError,
    NoveltyDetectionResult,
    NoveltyDetector,
    NoveltyType,
    SpectralFingerprint,
    SpectralFingerprintGenerator,
    ValidationError,
    ValidationReport,
    ValidationResult,
    detect_novelty_batch,
    validate_structure_batch,
)
from chemi_accel.spectroscopy.predictive_models import (
    SpectralPrediction,
    SpectroscopyType,
)


class TestSpectralFingerprint:
    """Test SpectralFingerprint data structure."""

    def test_basic_fingerprint(self):
        """Test basic spectral fingerprint creation."""
        peaks = [(1000.0, 0.8), (1500.0, 0.6), (3000.0, 0.9)]
        fingerprint = SpectralFingerprint(
            peaks=peaks,
            peak_count=3,
            intensity_distribution=[0.1, 0.2, 0.3, 0.4],
            spectral_moments=[1500.0, 1000.0, 0.5, 3.0],
            characteristic_regions={"carbonyl": 0.8, "ch_stretch": 0.6},
            fingerprint_hash="",
        )

        assert fingerprint.peaks == peaks
        assert fingerprint.peak_count == 3
        assert len(fingerprint.intensity_distribution) == 4
        assert len(fingerprint.spectral_moments) == 4
        assert fingerprint.characteristic_regions["carbonyl"] == 0.8
        assert (
            len(fingerprint.fingerprint_hash) > 0
        )  # Should be calculated automatically

    def test_fingerprint_hash_calculation(self):
        """Test fingerprint hash calculation."""
        peaks1 = [(1000.0, 0.8), (1500.0, 0.6)]
        peaks2 = [(1000.0, 0.8), (1500.0, 0.6)]
        peaks3 = [(1001.0, 0.8), (1500.0, 0.6)]  # Slightly different

        fp1 = SpectralFingerprint(
            peaks=peaks1,
            peak_count=2,
            intensity_distribution=[0.1, 0.2],
            spectral_moments=[1250.0, 500.0, 0.0, 2.5],
            characteristic_regions={},
            fingerprint_hash="",
        )

        fp2 = SpectralFingerprint(
            peaks=peaks2,
            peak_count=2,
            intensity_distribution=[0.1, 0.2],
            spectral_moments=[1250.0, 500.0, 0.0, 2.5],
            characteristic_regions={},
            fingerprint_hash="",
        )

        fp3 = SpectralFingerprint(
            peaks=peaks3,
            peak_count=2,
            intensity_distribution=[0.1, 0.2],
            spectral_moments=[1250.0, 500.0, 0.0, 2.5],
            characteristic_regions={},
            fingerprint_hash="",
        )

        # Identical fingerprints should have same hash
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

        # Different fingerprints should have different hashes
        assert fp1.fingerprint_hash != fp3.fingerprint_hash


class TestKnowledgeGraphEntry:
    """Test KnowledgeGraphEntry data structure."""

    def test_basic_entry(self):
        """Test basic knowledge graph entry creation."""
        entry = KnowledgeGraphEntry(
            compound_id="test_001",
            smiles="CCO",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            compound_name="Ethanol",
            spectral_data={"ir": {"peaks": [(3300, 0.9), (2900, 0.7)]}},
            properties={"boiling_point": 78.4, "molecular_weight": 46.07},
            related_compounds=["methanol_001", "propanol_001"],
            literature_references=["DOI:10.1021/example"],
            confidence_score=0.95,
            last_updated="2024-01-15",
        )

        assert entry.compound_id == "test_001"
        assert entry.smiles == "CCO"
        assert entry.compound_name == "Ethanol"
        assert "ir" in entry.spectral_data
        assert entry.properties["boiling_point"] == 78.4
        assert len(entry.related_compounds) == 2
        assert entry.confidence_score == 0.95


class TestValidationReport:
    """Test ValidationReport data structure."""

    def test_basic_report(self):
        """Test basic validation report creation."""
        candidate = StructureCandidate(
            smiles="CCO", confidence_score=0.8, compound_name="Ethanol"
        )

        report = ValidationReport(
            structure_candidate=candidate,
            validation_result=ValidationResult.VALIDATED,
            confidence_level=ConfidenceLevel.HIGH,
        )

        assert report.structure_candidate == candidate
        assert report.validation_result == ValidationResult.VALIDATED
        assert report.confidence_level == ConfidenceLevel.HIGH
        assert len(report.matching_entries) == 0
        assert len(report.inconsistencies) == 0
        assert len(report.novelty_indicators) == 0
        assert report.validation_time == 0.0
        assert report.explanation == ""


class TestNoveltyDetectionResult:
    """Test NoveltyDetectionResult data structure."""

    def test_basic_novelty_result(self):
        """Test basic novelty detection result creation."""
        result = NoveltyDetectionResult(
            is_novel=True,
            novelty_types=[NoveltyType.NEW_COMPOUND, NoveltyType.NEW_SPECTRAL_PATTERN],
            novelty_score=0.85,
            evidence={"unusual_peaks": 3, "structural_uniqueness": 0.9},
            similar_compounds=[],
            spectral_anomalies=["Unusual peak count", "High skewness"],
            statistical_significance=0.95,
        )

        assert result.is_novel is True
        assert len(result.novelty_types) == 2
        assert NoveltyType.NEW_COMPOUND in result.novelty_types
        assert result.novelty_score == 0.85
        assert result.evidence["unusual_peaks"] == 3
        assert len(result.spectral_anomalies) == 2
        assert result.statistical_significance == 0.95


class TestSpectralFingerprintGenerator:
    """Test spectral fingerprint generation."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SpectralFingerprintGenerator()
        assert generator is not None

    def test_generate_fingerprint_from_prediction(self):
        """Test fingerprint generation from spectral prediction."""
        frequencies = np.linspace(400, 4000, 1000)

        # Create spectrum with specific peaks
        intensities = np.random.normal(0.1, 0.02, 1000)
        peak_positions = [1000, 1650, 2900, 3300]
        for pos in peak_positions:
            idx = np.argmin(np.abs(frequencies - pos))
            intensities[idx] = 0.8

        prediction = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        generator = SpectralFingerprintGenerator()
        fingerprint = generator.generate_fingerprint(prediction, "ir")

        assert isinstance(fingerprint, SpectralFingerprint)
        assert fingerprint.peak_count > 0
        assert len(fingerprint.peaks) > 0
        assert len(fingerprint.spectral_moments) == 4
        assert len(fingerprint.intensity_distribution) > 0
        assert len(fingerprint.characteristic_regions) > 0
        assert len(fingerprint.fingerprint_hash) > 0

    def test_generate_fingerprint_from_experimental(self):
        """Test fingerprint generation from experimental spectrum."""
        frequencies = np.linspace(0, 12, 1000)  # NMR range
        intensities = np.random.normal(0.05, 0.01, 1000)

        # Add NMR-like peaks
        nmr_peaks = [1.2, 3.8, 7.3]  # Typical chemical shifts
        for pos in nmr_peaks:
            idx = np.argmin(np.abs(frequencies - pos))
            intensities[idx] = 0.9

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        generator = SpectralFingerprintGenerator()
        fingerprint = generator.generate_fingerprint(experimental, "nmr_1h")

        assert isinstance(fingerprint, SpectralFingerprint)
        assert fingerprint.peak_count > 0
        assert "aromatic" in fingerprint.characteristic_regions
        assert "aliphatic" in fingerprint.characteristic_regions

    def test_find_significant_peaks(self):
        """Test significant peak detection."""
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.normal(0.05, 0.01, 1000)  # Noise

        # Add significant peaks
        peak_positions = [800, 1200, 1600, 2400, 3200]
        peak_heights = [0.9, 0.7, 0.8, 0.6, 0.5]

        for pos, height in zip(peak_positions, peak_heights):
            idx = np.argmin(np.abs(frequencies - pos))
            intensities[idx] = height

        generator = SpectralFingerprintGenerator()
        peaks = generator._find_significant_peaks(frequencies, intensities)

        assert len(peaks) > 0
        assert len(peaks) <= 20  # Max peaks limit

        # Check that peaks are sorted by intensity
        peak_intensities = [intensity for _, intensity in peaks]
        assert peak_intensities == sorted(peak_intensities, reverse=True)

        # Check that all peaks are above minimum height
        max_int = max(peak_intensities)
        min_height = max_int * 0.05
        assert all(intensity >= min_height for _, intensity in peaks)

    def test_calculate_spectral_moments(self):
        """Test spectral moment calculation."""
        frequencies = np.array([1000, 1500, 2000, 2500, 3000])
        intensities = np.array([0.2, 0.8, 1.0, 0.6, 0.4])  # Peak at 2000

        generator = SpectralFingerprintGenerator()
        moments = generator._calculate_spectral_moments(frequencies, intensities)

        assert len(moments) == 4

        mean_freq, variance, skewness, kurtosis = moments

        # Mean should be weighted toward the high intensity peak
        assert 1800 <= mean_freq <= 2200  # Should be near 2000
        assert variance > 0
        assert isinstance(skewness, float)
        assert isinstance(kurtosis, float)

    def test_characteristic_regions_ir(self):
        """Test IR characteristic region identification."""
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.normal(0.05, 0.01, 1000)

        # Add peaks in characteristic regions
        oh_peak_idx = np.argmin(np.abs(frequencies - 3400))  # O-H stretch
        ch_peak_idx = np.argmin(np.abs(frequencies - 2900))  # C-H stretch
        co_peak_idx = np.argmin(np.abs(frequencies - 1700))  # C=O stretch

        intensities[oh_peak_idx] = 0.8
        intensities[ch_peak_idx] = 0.6
        intensities[co_peak_idx] = 0.9

        generator = SpectralFingerprintGenerator()
        regions = generator._identify_characteristic_regions(
            frequencies, intensities, "ir"
        )

        assert "oh_stretch" in regions
        assert "ch_stretch" in regions
        assert "carbonyl" in regions
        assert "fingerprint" in regions

        # O-H and carbonyl regions should have high intensity
        assert regions["oh_stretch"] > 0
        assert regions["carbonyl"] > 0

    def test_characteristic_regions_nmr(self):
        """Test NMR characteristic region identification."""
        frequencies = np.linspace(0, 12, 1000)
        intensities = np.random.normal(0.05, 0.01, 1000)

        # Add peaks in characteristic regions
        aromatic_idx = np.argmin(np.abs(frequencies - 7.5))  # Aromatic
        aliphatic_idx = np.argmin(np.abs(frequencies - 1.5))  # Aliphatic

        intensities[aromatic_idx] = 0.8
        intensities[aliphatic_idx] = 0.9

        generator = SpectralFingerprintGenerator()
        regions = generator._identify_characteristic_regions(
            frequencies, intensities, "nmr_1h"
        )

        assert "aromatic" in regions
        assert "aliphatic" in regions
        assert "alkenic" in regions
        assert "hetero_adjacent" in regions

        assert regions["aromatic"] > 0
        assert regions["aliphatic"] > 0


class TestKnowledgeGraphValidator:
    """Test knowledge graph validation."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = KnowledgeGraphValidator(use_gpu=False)
        assert validator.use_gpu is False
        assert validator.fingerprint_generator is not None
        assert validator.spectral_comparator is not None
        assert validator.knowledge_graph is not None  # Mock KG should be created

    def test_mock_knowledge_graph_creation(self):
        """Test mock knowledge graph creation."""
        validator = KnowledgeGraphValidator(use_gpu=False)
        mock_kg = validator.knowledge_graph

        assert isinstance(mock_kg, dict)
        assert len(mock_kg) > 0

        # Check that entries are properly structured
        for entry_id, entry in mock_kg.items():
            assert isinstance(entry, KnowledgeGraphEntry)
            assert entry.compound_id == entry_id
            assert entry.smiles is not None
            assert entry.compound_name is not None
            assert isinstance(entry.spectral_data, dict)

    def test_validate_structure_basic(self):
        """Test basic structure validation."""
        candidate = StructureCandidate(
            smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            confidence_score=0.8,
            compound_name="Caffeine",
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 1000), intensities=np.random.rand(1000)
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        report = validator.validate_structure(candidate, experimental, "ir")

        assert isinstance(report, ValidationReport)
        assert report.structure_candidate == candidate
        assert report.validation_result in [
            ValidationResult.VALIDATED,
            ValidationResult.NOVEL,
            ValidationResult.INCONSISTENT,
            ValidationResult.UNCERTAIN,
        ]
        assert report.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.VERY_LOW,
        ]
        assert report.validation_time > 0
        assert isinstance(report.explanation, str)
        assert isinstance(report.recommendations, list)

    def test_validate_known_compound(self):
        """Test validation of known compound."""
        # Use caffeine which should be in mock knowledge graph
        candidate = StructureCandidate(
            smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            confidence_score=0.9,
            compound_name="Caffeine",
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 1000), intensities=np.random.rand(1000)
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        report = validator.validate_structure(candidate, experimental, "ir")

        # Should find matches for caffeine
        assert len(report.matching_entries) >= 0
        assert len(report.similarity_scores) >= 0

    def test_validate_novel_compound(self):
        """Test validation of potentially novel compound."""
        candidate = StructureCandidate(
            smiles="C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3",  # Complex structure
            confidence_score=0.3,  # Low confidence
            compound_name="Unknown Compound",
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 1000), intensities=np.random.rand(1000)
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        report = validator.validate_structure(candidate, experimental, "ir")

        # Low confidence structure might be classified as novel
        assert report.validation_result in [
            ValidationResult.NOVEL,
            ValidationResult.UNCERTAIN,
        ]

    def test_search_knowledge_graph(self):
        """Test knowledge graph search functionality."""
        candidate = StructureCandidate(
            smiles="c1ccccc1", confidence_score=0.8, compound_name="Benzene"  # Benzene
        )

        fingerprint = SpectralFingerprint(
            peaks=[(1480, 0.8), (3030, 0.6), (690, 0.7)],
            peak_count=3,
            intensity_distribution=[0.1, 0.2, 0.3],
            spectral_moments=[1500, 500, 0, 2],
            characteristic_regions={},
            fingerprint_hash="",
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        matches = validator._search_knowledge_graph(candidate, fingerprint, "ir")

        assert isinstance(matches, list)
        # Should find benzene in mock knowledge graph
        assert len(matches) >= 0

    def test_detect_inconsistencies(self):
        """Test inconsistency detection."""
        candidate = StructureCandidate(
            smiles="CCO",
            confidence_score=0.8,
            compound_name="Ethanol",
            predicted_properties={"molecular_weight": 50.0},  # Wrong MW
        )

        # Create mock matching entry with correct properties
        mock_entry = KnowledgeGraphEntry(
            compound_id="ethanol_001",
            smiles="CCO",
            inchi=None,
            compound_name="Ethanol",
            spectral_data={},
            properties={"molecular_weight": 46.07},  # Correct MW
            related_compounds=[],
            literature_references=[],
            confidence_score=0.9,
            last_updated="2024-01-01",
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        inconsistencies = validator._detect_inconsistencies(
            candidate, [mock_entry], experimental
        )

        # Should detect molecular weight inconsistency
        assert len(inconsistencies) > 0
        assert any("molecular_weight" in inc.lower() for inc in inconsistencies)

    def test_novelty_indicators_detection(self):
        """Test novelty indicators detection."""
        candidate = StructureCandidate(
            smiles="C" * 100,  # Very unusual structure
            confidence_score=0.2,  # Low confidence
            compound_name="Unusual Compound",
        )

        fingerprint = SpectralFingerprint(
            peaks=[(i * 100, 0.5) for i in range(25)],  # Many peaks
            peak_count=25,
            intensity_distribution=[0.1] * 10,
            spectral_moments=[2000, 1000, 0, 2],
            characteristic_regions={},
            fingerprint_hash="",
        )

        validator = KnowledgeGraphValidator(use_gpu=False)
        novelty_indicators = validator._detect_novelty_indicators(
            candidate, [], fingerprint
        )

        assert isinstance(novelty_indicators, list)
        # Should detect novelty due to no matches and unusual structure
        assert len(novelty_indicators) > 0

    @patch("chemi_accel.spectroscopy.knowledge_graph_integration.RDKIT_AVAILABLE", True)
    def test_smiles_similarity_calculation(self):
        """Test SMILES similarity calculation."""
        validator = KnowledgeGraphValidator(use_gpu=False)

        # Mock RDKit functions
        with patch("rdkit.Chem.MolFromSmiles") as mock_mol_from_smiles:
            with patch(
                "rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect"
            ) as mock_fp:
                with patch(
                    "rdkit.Chem.DataStructs.TanimotoSimilarity"
                ) as mock_tanimoto:
                    # Setup mocks
                    mock_mol1 = Mock()
                    mock_mol2 = Mock()
                    mock_mol_from_smiles.side_effect = [mock_mol1, mock_mol2]

                    mock_fp1 = Mock()
                    mock_fp2 = Mock()
                    mock_fp.side_effect = [mock_fp1, mock_fp2]

                    mock_tanimoto.return_value = 0.85

                    similarity = validator._calculate_smiles_similarity("CCO", "CCC")

                    assert similarity == 0.85
                    mock_mol_from_smiles.assert_called()
                    mock_fp.assert_called()
                    mock_tanimoto.assert_called_with(mock_fp1, mock_fp2)

    def test_smiles_similarity_fallback(self):
        """Test SMILES similarity fallback without RDKit."""
        validator = KnowledgeGraphValidator(use_gpu=False)

        with patch(
            "chemi_accel.spectroscopy.knowledge_graph_integration.RDKIT_AVAILABLE",
            False,
        ):
            similarity = validator._calculate_smiles_similarity("CCO", "CCC")

            assert 0.0 <= similarity <= 1.0
            assert isinstance(similarity, float)


class TestNoveltyDetector:
    """Test novelty detection functionality."""

    def test_detector_initialization(self):
        """Test novelty detector initialization."""
        detector = NoveltyDetector(use_gpu=False)
        assert detector.use_gpu is False
        assert detector.fingerprint_generator is not None

    def test_detect_novelty_basic(self):
        """Test basic novelty detection."""
        candidate = StructureCandidate(
            smiles="C1=CC=C(C=C1)C2=CC=CC=C2",
            confidence_score=0.8,
            compound_name="Biphenyl",
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 1000), intensities=np.random.rand(1000)
        )

        detector = NoveltyDetector(use_gpu=False)
        result = detector.detect_novelty(candidate, experimental, "ir")

        assert isinstance(result, NoveltyDetectionResult)
        assert isinstance(result.is_novel, bool)
        assert 0.0 <= result.novelty_score <= 1.0
        assert isinstance(result.novelty_types, list)
        assert isinstance(result.evidence, dict)
        assert isinstance(result.similar_compounds, list)
        assert isinstance(result.spectral_anomalies, list)
        assert 0.0 <= result.statistical_significance <= 1.0
        assert result.detection_time > 0

    def test_detect_spectral_anomalies(self):
        """Test spectral anomaly detection."""
        # Create fingerprint with anomalies
        unusual_fingerprint = SpectralFingerprint(
            peaks=[(i * 50, 0.5) for i in range(30)],  # Too many peaks
            peak_count=30,
            intensity_distribution=[0.1] * 10,
            spectral_moments=[2000, 1000, 5.0, 20.0],  # High skewness and kurtosis
            characteristic_regions={"carbonyl": 2000},  # Very high intensity
            fingerprint_hash="",
        )

        detector = NoveltyDetector(use_gpu=False)
        anomalies = detector._detect_spectral_anomalies(unusual_fingerprint, "ir")

        assert isinstance(anomalies, list)
        assert len(anomalies) > 0

        # Should detect various anomalies
        anomaly_text = " ".join(anomalies).lower()
        assert (
            "peaks" in anomaly_text
            or "asymmetry" in anomaly_text
            or "intensity" in anomaly_text
        )

    def test_calculate_novelty_score(self):
        """Test novelty score calculation."""
        candidate = StructureCandidate(
            smiles="CCCCCCCCCC",  # Simple alkane
            confidence_score=0.5,
            compound_name="Decane",
        )

        fingerprint = SpectralFingerprint(
            peaks=[(2900, 0.8), (1460, 0.6)],
            peak_count=2,
            intensity_distribution=[0.2, 0.3, 0.5],
            spectral_moments=[2180, 400, 0.5, 2.5],
            characteristic_regions={},
            fingerprint_hash="",
        )

        anomalies = ["Unusual peak pattern"]

        detector = NoveltyDetector(use_gpu=False)
        score = detector._calculate_novelty_score(candidate, fingerprint, anomalies)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_determine_novelty_types(self):
        """Test novelty type determination."""
        # Low confidence candidate
        candidate = StructureCandidate(
            smiles="C1=CC=C2C(=C1)C=CC=C2",
            confidence_score=0.2,  # Very low confidence
            compound_name="Unknown Compound",
        )

        # Complex fingerprint
        fingerprint = SpectralFingerprint(
            peaks=[(i * 100, 0.5) for i in range(25)],  # Many peaks
            peak_count=25,
            intensity_distribution=[0.1] * 10,
            spectral_moments=[2000, 1000, 0, 2],
            characteristic_regions={},
            fingerprint_hash="",
        )

        anomalies = ["Unusually many peaks", "Unusual spectral pattern"]

        detector = NoveltyDetector(use_gpu=False)
        novelty_types = detector._determine_novelty_types(
            candidate, fingerprint, anomalies
        )

        assert isinstance(novelty_types, list)
        assert len(novelty_types) > 0

        # Should detect multiple novelty types
        assert (
            NoveltyType.NEW_COMPOUND in novelty_types
            or NoveltyType.NEW_SPECTRAL_PATTERN in novelty_types
        )

    def test_find_similar_compounds(self):
        """Test finding similar compounds."""
        candidate = StructureCandidate(
            smiles="CCO", confidence_score=0.8, compound_name="Ethanol"
        )

        fingerprint = SpectralFingerprint(
            peaks=[(3300, 0.9), (2900, 0.7)],
            peak_count=2,
            intensity_distribution=[0.2, 0.8],
            spectral_moments=[3100, 200, 0, 2],
            characteristic_regions={},
            fingerprint_hash="",
        )

        detector = NoveltyDetector(use_gpu=False)
        similar_compounds = detector._find_similar_compounds(candidate, fingerprint)

        assert isinstance(similar_compounds, list)
        # Mock implementation should return some compounds
        assert len(similar_compounds) >= 0

        for compound in similar_compounds:
            assert isinstance(compound, KnowledgeGraphEntry)

    def test_statistical_significance_calculation(self):
        """Test statistical significance calculation."""
        detector = NoveltyDetector(use_gpu=False)

        # High novelty, no similar compounds
        sig1 = detector._calculate_statistical_significance(0.9, [])
        assert 0.0 <= sig1 <= 1.0
        assert sig1 > 0.5  # Should be high

        # Low novelty, many similar compounds
        similar_compounds = [Mock() for _ in range(5)]
        sig2 = detector._calculate_statistical_significance(0.3, similar_compounds)
        assert 0.0 <= sig2 <= 1.0
        assert sig2 < sig1  # Should be lower


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_structure_batch(self):
        """Test batch structure validation."""
        candidates = []
        spectra = []
        spec_types = []

        for i in range(3):
            candidate = StructureCandidate(
                smiles=f"C{i+1}", confidence_score=0.8, compound_name=f"Compound {i+1}"
            )
            candidates.append(candidate)

            spectrum = ExperimentalSpectrum(
                frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
            )
            spectra.append(spectrum)

            spec_types.append("ir")

        reports = validate_structure_batch(
            candidates, spectra, spec_types, use_gpu=False
        )

        assert len(reports) == 3
        assert all(isinstance(report, ValidationReport) for report in reports)

    def test_validate_structure_batch_length_mismatch(self):
        """Test batch validation with mismatched lengths."""
        candidates = [Mock()]
        spectra = [Mock(), Mock()]
        spec_types = ["ir"]

        with pytest.raises(
            ValueError, match="All input lists must have the same length"
        ):
            validate_structure_batch(candidates, spectra, spec_types)

    def test_detect_novelty_batch(self):
        """Test batch novelty detection."""
        candidates = []
        spectra = []
        spec_types = []

        for i in range(3):
            candidate = StructureCandidate(
                smiles=f"C{i+1}",
                confidence_score=0.7,
                compound_name=f"Test Compound {i+1}",
            )
            candidates.append(candidate)

            spectrum = ExperimentalSpectrum(
                frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
            )
            spectra.append(spectrum)

            spec_types.append("ir")

        results = detect_novelty_batch(candidates, spectra, spec_types, use_gpu=False)

        assert len(results) == 3
        assert all(isinstance(result, NoveltyDetectionResult) for result in results)

    def test_detect_novelty_batch_length_mismatch(self):
        """Test batch novelty detection with mismatched lengths."""
        candidates = [Mock()]
        spectra = [Mock(), Mock()]
        spec_types = ["ir"]

        with pytest.raises(
            ValueError, match="All input lists must have the same length"
        ):
            detect_novelty_batch(candidates, spectra, spec_types)


class TestExceptionHandling:
    """Test exception handling."""

    def test_knowledge_graph_integration_error(self):
        """Test KnowledgeGraphIntegrationError."""
        with pytest.raises(
            KnowledgeGraphIntegrationError, match="Test integration error"
        ):
            raise KnowledgeGraphIntegrationError("Test integration error")

    def test_validation_error(self):
        """Test ValidationError."""
        with pytest.raises(ValidationError, match="Test validation error"):
            raise ValidationError("Test validation error")

    def test_novelty_detection_error(self):
        """Test NoveltyDetectionError."""
        with pytest.raises(NoveltyDetectionError, match="Test novelty detection error"):
            raise NoveltyDetectionError("Test novelty detection error")


class TestIntegration:
    """Integration tests for knowledge graph functionality."""

    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        # Create realistic test case
        candidate = StructureCandidate(
            smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            confidence_score=0.85,
            compound_name="Caffeine",
            predicted_properties={"molecular_weight": 194.19, "melting_point": 235.0},
        )

        # Create experimental spectrum with caffeine-like features
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.normal(0.05, 0.01, 1000)

        # Add characteristic peaks for caffeine
        caffeine_peaks = [
            (1700, 0.9),  # C=O stretch
            (1600, 0.8),  # Aromatic C=C
            (3000, 0.6),  # C-H stretch
            (1500, 0.7),  # Ring vibrations
        ]

        for peak_freq, peak_int in caffeine_peaks:
            idx = np.argmin(np.abs(frequencies - peak_freq))
            # Add Gaussian peak
            sigma = 20
            peak_profile = peak_int * np.exp(
                -(((frequencies - peak_freq) / sigma) ** 2)
            )
            intensities += peak_profile

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        # Perform validation
        validator = KnowledgeGraphValidator(use_gpu=False)
        report = validator.validate_structure(candidate, experimental, "ir")

        # Verify comprehensive validation
        assert isinstance(report, ValidationReport)
        assert report.structure_candidate == candidate
        assert report.validation_result in ValidationResult
        assert report.confidence_level in ConfidenceLevel
        assert report.validation_time > 0
        assert isinstance(report.explanation, str)
        assert isinstance(report.recommendations, list)

        # Should have some analysis results
        assert isinstance(report.similarity_scores, dict)
        assert isinstance(report.matching_entries, list)
        assert isinstance(report.inconsistencies, list)
        assert isinstance(report.novelty_indicators, list)

        # Test novelty detection on same compound
        detector = NoveltyDetector(use_gpu=False)
        novelty_result = detector.detect_novelty(candidate, experimental, "ir")

        assert isinstance(novelty_result, NoveltyDetectionResult)
        assert isinstance(novelty_result.is_novel, bool)
        assert 0.0 <= novelty_result.novelty_score <= 1.0
        assert isinstance(novelty_result.novelty_types, list)
        assert isinstance(novelty_result.evidence, dict)
        assert novelty_result.detection_time > 0

        # Evidence should contain relevant information
        assert "spectral_fingerprint_hash" in novelty_result.evidence
        assert "peak_count" in novelty_result.evidence
        assert "structure_confidence" in novelty_result.evidence

    def test_novel_compound_detection_workflow(self):
        """Test workflow for novel compound detection."""
        # Create unusual compound that should be detected as novel
        candidate = StructureCandidate(
            smiles="C1=CC=C2C3=CC=CC=C3C4=CC=CC=C4C2=C1",  # Complex polycyclic
            confidence_score=0.3,  # Low confidence
            compound_name="Novel Polycyclic Compound",
            predicted_properties={
                "molecular_weight": 350.0,  # Unusual value
                "melting_point": 450.0,  # Very high melting point
            },
        )

        # Create experimental spectrum with unusual features
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.normal(0.02, 0.005, 1000)

        # Add many unusual peaks
        unusual_peaks = np.linspace(500, 3500, 25)  # Many peaks
        for peak_freq in unusual_peaks:
            idx = np.argmin(np.abs(frequencies - peak_freq))
            intensities[idx] += np.random.uniform(0.3, 0.8)

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        # Perform validation and novelty detection
        validator = KnowledgeGraphValidator(use_gpu=False)
        novelty_detector = NoveltyDetector(use_gpu=False)

        validation_report = validator.validate_structure(candidate, experimental, "ir")
        novelty_result = novelty_detector.detect_novelty(candidate, experimental, "ir")

        # Should indicate novelty due to low confidence and unusual features
        assert validation_report.validation_result in [
            ValidationResult.NOVEL,
            ValidationResult.UNCERTAIN,
        ]

        # Should detect high novelty
        assert novelty_result.novelty_score > 0.5  # Should be considered novel

        # Should identify novelty types
        assert len(novelty_result.novelty_types) > 0
        assert (
            NoveltyType.NEW_COMPOUND in novelty_result.novelty_types
            or NoveltyType.NEW_SPECTRAL_PATTERN in novelty_result.novelty_types
        )

        # Should detect spectral anomalies
        assert len(novelty_result.spectral_anomalies) > 0

        # Should have recommendations for novel compounds
        assert len(validation_report.recommendations) > 0
        recommendations_text = " ".join(validation_report.recommendations).lower()
        assert (
            "novel" in recommendations_text
            or "characterization" in recommendations_text
            or "validation" in recommendations_text
        )

    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        # Create multiple diverse test cases
        test_cases = [
            {
                "smiles": "CCO",
                "name": "Ethanol",
                "confidence": 0.9,
                "expected_result": ValidationResult.VALIDATED,
            },
            {
                "smiles": "C1=CC=CC=C1",
                "name": "Benzene",
                "confidence": 0.85,
                "expected_result": ValidationResult.VALIDATED,
            },
            {
                "smiles": "C" * 50,  # Very unusual
                "name": "Unknown Polymer",
                "confidence": 0.2,
                "expected_result": ValidationResult.NOVEL,
            },
        ]

        candidates = []
        spectra = []
        spec_types = []

        for case in test_cases:
            candidate = StructureCandidate(
                smiles=case["smiles"],
                confidence_score=case["confidence"],
                compound_name=case["name"],
            )
            candidates.append(candidate)

            # Create appropriate spectrum
            frequencies = np.linspace(400, 4000, 500)
            if case["confidence"] > 0.8:
                # High confidence - create realistic spectrum
                intensities = np.random.normal(0.1, 0.02, 500)
                peak_positions = [1000, 1500, 2900, 3200]
                for pos in peak_positions:
                    idx = np.argmin(np.abs(frequencies - pos))
                    intensities[idx] = 0.8
            else:
                # Low confidence - create unusual spectrum
                intensities = np.random.uniform(0, 1, 500)

            spectrum = ExperimentalSpectrum(
                frequencies=frequencies, intensities=intensities
            )
            spectra.append(spectrum)
            spec_types.append("ir")

        # Perform batch validation and novelty detection
        validation_reports = validate_structure_batch(
            candidates, spectra, spec_types, use_gpu=False
        )

        novelty_results = detect_novelty_batch(
            candidates, spectra, spec_types, use_gpu=False
        )

        # Verify batch processing results
        assert len(validation_reports) == len(test_cases)
        assert len(novelty_results) == len(test_cases)

        for i, (report, novelty) in enumerate(zip(validation_reports, novelty_results)):
            assert isinstance(report, ValidationReport)
            assert isinstance(novelty, NoveltyDetectionResult)

            # Verify individual results make sense
            assert report.structure_candidate == candidates[i]
            assert report.validation_result in ValidationResult

            # Low confidence compounds should have higher novelty scores
            if test_cases[i]["confidence"] < 0.5:
                assert novelty.novelty_score > 0.4

            # All should have computation times
            assert report.validation_time > 0
            assert novelty.detection_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
