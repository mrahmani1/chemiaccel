"""
Knowledge graph integration for spectroscopy.

This module provides integration with the Intelligent Chemical Knowledge Graph (IC-KG)
for validating elucidated structures against known compounds and reactions, and for
detecting novel compounds or inconsistencies using spectroscopic data.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .experimental_processing import ExperimentalSpectrum, SpectralPeak
from .inverse_spectroscopy import InverseResult, StructureCandidate
from .predictive_models import SpectralPrediction
from .spectral_comparison import SpectralComparator, SpectralComparison

try:
    from ..knowledge_graph.entity_resolution import EntityResolver
    from ..knowledge_graph.query_engine import QueryEngine
    from ..knowledge_graph.validation import KnowledgeGraphValidator

    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Results of knowledge graph validation."""

    VALIDATED = "validated"
    INCONSISTENT = "inconsistent"
    NOVEL = "novel"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class NoveltyType(Enum):
    """Types of chemical novelty."""

    NEW_COMPOUND = "new_compound"
    NEW_REACTION = "new_reaction"
    NEW_PROPERTY = "new_property"
    NEW_SPECTRAL_PATTERN = "new_spectral_pattern"
    UNEXPECTED_BEHAVIOR = "unexpected_behavior"


class ConfidenceLevel(Enum):
    """Confidence levels for validation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class SpectralFingerprint:
    """Spectral fingerprint for compound identification."""

    peaks: List[Tuple[float, float]]  # (frequency, intensity) pairs
    peak_count: int
    intensity_distribution: List[float]
    spectral_moments: List[float]  # Statistical moments
    characteristic_regions: Dict[str, float]  # Named spectral regions
    fingerprint_hash: str

    def __post_init__(self):
        """Calculate fingerprint hash after initialization."""
        if not self.fingerprint_hash:
            self.fingerprint_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate unique hash for this fingerprint."""
        # Create string representation of key features
        features = []
        features.extend(
            [f"{freq:.2f}:{int:.3f}" for freq, int in self.peaks[:10]]
        )  # Top 10 peaks
        features.append(f"count:{self.peak_count}")
        features.extend(
            [
                f"moment_{i}:{moment:.4f}"
                for i, moment in enumerate(self.spectral_moments)
            ]
        )

        feature_string = "|".join(sorted(features))
        return hashlib.md5(feature_string.encode()).hexdigest()[:16]


@dataclass
class KnowledgeGraphEntry:
    """Entry in the knowledge graph."""

    compound_id: str
    smiles: Optional[str]
    inchi: Optional[str]
    compound_name: str
    spectral_data: Dict[str, Any]  # Spectroscopy type -> data
    properties: Dict[str, Any]
    related_compounds: List[str]
    literature_references: List[str]
    confidence_score: float
    last_updated: str


@dataclass
class ValidationReport:
    """Report from knowledge graph validation."""

    structure_candidate: StructureCandidate
    validation_result: ValidationResult
    confidence_level: ConfidenceLevel
    matching_entries: List[KnowledgeGraphEntry] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    novelty_indicators: List[NoveltyType] = field(default_factory=list)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    spectral_comparison: Optional[SpectralComparison] = None
    validation_time: float = 0.0
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class NoveltyDetectionResult:
    """Result of novelty detection analysis."""

    is_novel: bool
    novelty_types: List[NoveltyType]
    novelty_score: float  # 0-1 scale
    evidence: Dict[str, Any]
    similar_compounds: List[KnowledgeGraphEntry]
    spectral_anomalies: List[str]
    statistical_significance: float
    detection_time: float = 0.0


class SpectralFingerprintGenerator:
    """Generate spectral fingerprints for compound identification."""

    def __init__(self):
        """Initialize fingerprint generator."""
        logger.info("SpectralFingerprintGenerator initialized")

    def generate_fingerprint(
        self,
        spectrum: Union[SpectralPrediction, ExperimentalSpectrum],
        spectroscopy_type: str = "unknown",
    ) -> SpectralFingerprint:
        """
        Generate spectral fingerprint from spectrum.

        Args:
            spectrum: Input spectrum
            spectroscopy_type: Type of spectroscopy

        Returns:
            Spectral fingerprint
        """
        frequencies = spectrum.frequencies
        intensities = spectrum.intensities

        # Find significant peaks
        peaks = self._find_significant_peaks(frequencies, intensities)

        # Calculate statistical moments
        moments = self._calculate_spectral_moments(frequencies, intensities)

        # Calculate intensity distribution
        intensity_dist = self._calculate_intensity_distribution(intensities)

        # Identify characteristic regions
        char_regions = self._identify_characteristic_regions(
            frequencies, intensities, spectroscopy_type
        )

        fingerprint = SpectralFingerprint(
            peaks=peaks,
            peak_count=len(peaks),
            intensity_distribution=intensity_dist,
            spectral_moments=moments,
            characteristic_regions=char_regions,
            fingerprint_hash="",  # Will be calculated in __post_init__
        )

        return fingerprint

    def _find_significant_peaks(
        self,
        frequencies: np.ndarray,
        intensities: np.ndarray,
        min_height_ratio: float = 0.05,
        max_peaks: int = 20,
    ) -> List[Tuple[float, float]]:
        """Find significant peaks in spectrum."""
        # Simple peak detection
        max_intensity = np.max(intensities)
        min_height = max_intensity * min_height_ratio

        peaks = []
        for i in range(1, len(intensities) - 1):
            if (
                intensities[i] > intensities[i - 1]
                and intensities[i] > intensities[i + 1]
                and intensities[i] > min_height
            ):
                peaks.append((float(frequencies[i]), float(intensities[i])))

        # Sort by intensity and take top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:max_peaks]

    def _calculate_spectral_moments(
        self, frequencies: np.ndarray, intensities: np.ndarray
    ) -> List[float]:
        """Calculate statistical moments of the spectrum."""
        # Normalize intensities
        if np.sum(intensities) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        weights = intensities / np.sum(intensities)

        # First moment (mean frequency)
        mean_freq = float(np.sum(frequencies * weights))

        # Second moment (variance)
        variance = float(np.sum((frequencies - mean_freq) ** 2 * weights))

        # Third moment (skewness)
        if variance > 0:
            skewness = float(
                np.sum((frequencies - mean_freq) ** 3 * weights) / (variance**1.5)
            )
        else:
            skewness = 0.0

        # Fourth moment (kurtosis)
        if variance > 0:
            kurtosis = float(
                np.sum((frequencies - mean_freq) ** 4 * weights) / (variance**2)
            )
        else:
            kurtosis = 0.0

        return [mean_freq, variance, skewness, kurtosis]

    def _calculate_intensity_distribution(
        self, intensities: np.ndarray, n_bins: int = 10
    ) -> List[float]:
        """Calculate intensity distribution histogram."""
        if len(intensities) == 0:
            return [0.0] * n_bins

        hist, _ = np.histogram(intensities, bins=n_bins, density=True)
        return hist.tolist()

    def _identify_characteristic_regions(
        self, frequencies: np.ndarray, intensities: np.ndarray, spectroscopy_type: str
    ) -> Dict[str, float]:
        """Identify characteristic spectral regions."""
        regions = {}

        if spectroscopy_type.lower() in ["ir", "infrared"]:
            # IR characteristic regions
            regions.update(
                {
                    "oh_stretch": self._integrate_region(
                        frequencies, intensities, 3200, 3600
                    ),
                    "ch_stretch": self._integrate_region(
                        frequencies, intensities, 2800, 3000
                    ),
                    "carbonyl": self._integrate_region(
                        frequencies, intensities, 1650, 1750
                    ),
                    "fingerprint": self._integrate_region(
                        frequencies, intensities, 800, 1400
                    ),
                }
            )

        elif spectroscopy_type.lower() in ["nmr_1h", "1h_nmr"]:
            # 1H NMR characteristic regions
            regions.update(
                {
                    "aromatic": self._integrate_region(
                        frequencies, intensities, 7.0, 8.5
                    ),
                    "alkenic": self._integrate_region(
                        frequencies, intensities, 5.0, 6.5
                    ),
                    "aliphatic": self._integrate_region(
                        frequencies, intensities, 0.8, 3.0
                    ),
                    "hetero_adjacent": self._integrate_region(
                        frequencies, intensities, 3.0, 5.0
                    ),
                }
            )

        elif spectroscopy_type.lower() in ["nmr_13c", "13c_nmr"]:
            # 13C NMR characteristic regions
            regions.update(
                {
                    "aromatic_carbon": self._integrate_region(
                        frequencies, intensities, 120, 160
                    ),
                    "carbonyl_carbon": self._integrate_region(
                        frequencies, intensities, 160, 220
                    ),
                    "aliphatic_carbon": self._integrate_region(
                        frequencies, intensities, 10, 60
                    ),
                    "alkenic_carbon": self._integrate_region(
                        frequencies, intensities, 100, 150
                    ),
                }
            )

        elif spectroscopy_type.lower() in ["ms", "mass_spec"]:
            # MS characteristic regions
            regions.update(
                {
                    "molecular_ion": self._integrate_region(
                        frequencies,
                        intensities,
                        np.max(frequencies) * 0.9,
                        np.max(frequencies),
                    ),
                    "base_peak": float(np.max(intensities)),
                    "fragmentation": self._integrate_region(
                        frequencies, intensities, 0, np.max(frequencies) * 0.8
                    ),
                }
            )

        return regions

    def _integrate_region(
        self,
        frequencies: np.ndarray,
        intensities: np.ndarray,
        freq_min: float,
        freq_max: float,
    ) -> float:
        """Integrate intensity over frequency region."""
        mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        if not np.any(mask):
            return 0.0

        region_intensities = intensities[mask]
        return float(np.sum(region_intensities))


class KnowledgeGraphValidator:
    """Validator for spectroscopic data against knowledge graph."""

    def __init__(self, use_gpu: bool = True):
        """Initialize validator."""
        self.use_gpu = use_gpu
        self.fingerprint_generator = SpectralFingerprintGenerator()
        self.spectral_comparator = SpectralComparator(use_gpu=use_gpu)

        # Mock knowledge graph if real one not available
        if not KG_AVAILABLE:
            logger.warning("Knowledge graph not available. Using mock implementation.")
            self.knowledge_graph = self._create_mock_knowledge_graph()
        else:
            self.knowledge_graph = None  # Would be initialized with real KG

        logger.info(f"KnowledgeGraphValidator initialized with GPU: {use_gpu}")

    def _create_mock_knowledge_graph(self) -> Dict[str, KnowledgeGraphEntry]:
        """Create mock knowledge graph for testing."""
        mock_entries = {}

        # Add some example compounds
        compounds = [
            {
                "id": "caffeine_001",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "name": "Caffeine",
                "spectral_data": {
                    "ir": {"peaks": [(1600, 0.8), (1700, 0.9), (3000, 0.6)]},
                    "nmr_1h": {"peaks": [(3.9, 1.0), (7.5, 0.7)]},
                },
            },
            {
                "id": "glucose_001",
                "smiles": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
                "name": "D-Glucose",
                "spectral_data": {
                    "ir": {"peaks": [(3300, 0.9), (1050, 0.8), (2900, 0.6)]},
                    "nmr_1h": {"peaks": [(3.2, 0.8), (3.8, 0.9), (5.2, 0.4)]},
                },
            },
            {
                "id": "benzene_001",
                "smiles": "c1ccccc1",
                "name": "Benzene",
                "spectral_data": {
                    "ir": {"peaks": [(3030, 0.6), (1480, 0.8), (690, 0.7)]},
                    "nmr_1h": {"peaks": [(7.3, 1.0)]},
                },
            },
        ]

        for comp in compounds:
            entry = KnowledgeGraphEntry(
                compound_id=comp["id"],
                smiles=comp["smiles"],
                inchi=None,
                compound_name=comp["name"],
                spectral_data=comp["spectral_data"],
                properties={},
                related_compounds=[],
                literature_references=[],
                confidence_score=0.9,
                last_updated="2024-01-01",
            )
            mock_entries[comp["id"]] = entry

        return mock_entries

    def validate_structure(
        self,
        candidate: StructureCandidate,
        experimental_spectrum: ExperimentalSpectrum,
        spectroscopy_type: str = "ir",
    ) -> ValidationReport:
        """
        Validate structure candidate against knowledge graph.

        Args:
            candidate: Structure candidate to validate
            experimental_spectrum: Experimental spectrum
            spectroscopy_type: Type of spectroscopy

        Returns:
            Validation report
        """
        start_time = time.time()

        # Generate spectral fingerprint
        fingerprint = self.fingerprint_generator.generate_fingerprint(
            experimental_spectrum, spectroscopy_type
        )

        # Search knowledge graph for matching compounds
        matching_entries = self._search_knowledge_graph(
            candidate, fingerprint, spectroscopy_type
        )

        # Analyze matches and determine validation result
        validation_result, confidence_level = self._analyze_matches(
            candidate, matching_entries, fingerprint
        )

        # Detect inconsistencies
        inconsistencies = self._detect_inconsistencies(
            candidate, matching_entries, experimental_spectrum
        )

        # Check for novelty indicators
        novelty_indicators = self._detect_novelty_indicators(
            candidate, matching_entries, fingerprint
        )

        # Calculate similarity scores
        similarity_scores = self._calculate_similarity_scores(
            candidate, matching_entries, fingerprint
        )

        # Generate explanation and recommendations
        explanation = self._generate_explanation(
            validation_result, confidence_level, matching_entries, inconsistencies
        )
        recommendations = self._generate_recommendations(
            validation_result, novelty_indicators, inconsistencies
        )

        validation_time = time.time() - start_time

        report = ValidationReport(
            structure_candidate=candidate,
            validation_result=validation_result,
            confidence_level=confidence_level,
            matching_entries=matching_entries,
            inconsistencies=inconsistencies,
            novelty_indicators=novelty_indicators,
            similarity_scores=similarity_scores,
            validation_time=validation_time,
            explanation=explanation,
            recommendations=recommendations,
        )

        return report

    def _search_knowledge_graph(
        self,
        candidate: StructureCandidate,
        fingerprint: SpectralFingerprint,
        spectroscopy_type: str,
    ) -> List[KnowledgeGraphEntry]:
        """Search knowledge graph for matching compounds."""
        matches = []

        # Simple mock search - in reality would use sophisticated graph queries
        for entry_id, entry in self.knowledge_graph.items():
            # Check SMILES similarity if available
            smiles_match = False
            if candidate.smiles and entry.smiles:
                smiles_match = (
                    self._calculate_smiles_similarity(candidate.smiles, entry.smiles)
                    > 0.7
                )

            # Check spectral similarity
            spectral_match = False
            if spectroscopy_type in entry.spectral_data:
                spectral_match = (
                    self._calculate_spectral_similarity(
                        fingerprint, entry.spectral_data[spectroscopy_type]
                    )
                    > 0.5
                )

            # Check name similarity
            name_match = False
            if candidate.compound_name and entry.compound_name:
                name_match = (
                    candidate.compound_name.lower() in entry.compound_name.lower()
                )

            # Add to matches if any criteria met
            if smiles_match or spectral_match or name_match:
                matches.append(entry)

        # Sort by confidence score
        matches.sort(key=lambda x: x.confidence_score, reverse=True)

        return matches[:10]  # Return top 10 matches

    def _analyze_matches(
        self,
        candidate: StructureCandidate,
        matches: List[KnowledgeGraphEntry],
        fingerprint: SpectralFingerprint,
    ) -> Tuple[ValidationResult, ConfidenceLevel]:
        """Analyze matches to determine validation result."""
        if not matches:
            return ValidationResult.NOVEL, ConfidenceLevel.MEDIUM

        # Check for exact matches
        exact_matches = []
        for match in matches:
            if candidate.smiles and match.smiles:
                if (
                    self._calculate_smiles_similarity(candidate.smiles, match.smiles)
                    > 0.95
                ):
                    exact_matches.append(match)

        if exact_matches:
            # Check for inconsistencies in exact matches
            high_confidence_matches = [
                m for m in exact_matches if m.confidence_score > 0.8
            ]
            if high_confidence_matches:
                return ValidationResult.VALIDATED, ConfidenceLevel.HIGH
            else:
                return ValidationResult.VALIDATED, ConfidenceLevel.MEDIUM

        # Check for similar compounds
        similar_matches = [m for m in matches if m.confidence_score > 0.6]
        if similar_matches:
            # Check for conflicting information
            if self._has_conflicting_information(candidate, similar_matches):
                return ValidationResult.INCONSISTENT, ConfidenceLevel.MEDIUM
            else:
                return ValidationResult.UNCERTAIN, ConfidenceLevel.LOW

        # No good matches found
        return ValidationResult.NOVEL, ConfidenceLevel.MEDIUM

    def _detect_inconsistencies(
        self,
        candidate: StructureCandidate,
        matches: List[KnowledgeGraphEntry],
        experimental_spectrum: ExperimentalSpectrum,
    ) -> List[str]:
        """Detect inconsistencies between candidate and known data."""
        inconsistencies = []

        for match in matches:
            # Check molecular formula conflicts
            if (
                hasattr(candidate, "molecular_formula")
                and "molecular_formula" in match.properties
            ):
                if candidate.molecular_formula != match.properties["molecular_formula"]:
                    inconsistencies.append(
                        f"Molecular formula mismatch: {candidate.molecular_formula} "
                        f"vs {match.properties['molecular_formula']} for {match.compound_name}"
                    )

            # Check spectral data conflicts
            # This would involve detailed spectral comparison

            # Check property conflicts
            if hasattr(candidate, "predicted_properties"):
                for prop, value in candidate.predicted_properties.items():
                    if prop in match.properties:
                        known_value = match.properties[prop]
                        if abs(value - known_value) > abs(
                            known_value * 0.2
                        ):  # 20% threshold
                            inconsistencies.append(
                                f"Property conflict for {prop}: predicted {value:.3f} "
                                f"vs known {known_value:.3f} for {match.compound_name}"
                            )

        return inconsistencies

    def _detect_novelty_indicators(
        self,
        candidate: StructureCandidate,
        matches: List[KnowledgeGraphEntry],
        fingerprint: SpectralFingerprint,
    ) -> List[NoveltyType]:
        """Detect indicators of chemical novelty."""
        novelty_indicators = []

        # Check if no close structural matches found
        if not matches:
            novelty_indicators.append(NoveltyType.NEW_COMPOUND)

        # Check for novel spectral patterns
        if matches:
            spectral_similarities = []
            for match in matches:
                # Calculate spectral similarity (mock implementation)
                similarity = np.random.random()  # Would use real comparison
                spectral_similarities.append(similarity)

            if max(spectral_similarities) < 0.6:
                novelty_indicators.append(NoveltyType.NEW_SPECTRAL_PATTERN)

        # Check for unusual peak patterns
        if len(fingerprint.peaks) > 15:  # Unusually complex spectrum
            novelty_indicators.append(NoveltyType.UNEXPECTED_BEHAVIOR)

        # Check for novel property predictions
        if (
            hasattr(candidate, "predicted_properties")
            and candidate.predicted_properties
        ):
            # Check if properties are outside known ranges
            unusual_properties = 0
            for prop, value in candidate.predicted_properties.items():
                # Mock check - would use real property distributions
                if abs(value) > 100:  # Arbitrary threshold
                    unusual_properties += 1

            if unusual_properties > 0:
                novelty_indicators.append(NoveltyType.NEW_PROPERTY)

        return novelty_indicators

    def _calculate_similarity_scores(
        self,
        candidate: StructureCandidate,
        matches: List[KnowledgeGraphEntry],
        fingerprint: SpectralFingerprint,
    ) -> Dict[str, float]:
        """Calculate various similarity scores."""
        scores = {}

        if not matches:
            return scores

        # Structural similarity (if SMILES available)
        if candidate.smiles:
            structural_similarities = []
            for match in matches:
                if match.smiles:
                    sim = self._calculate_smiles_similarity(
                        candidate.smiles, match.smiles
                    )
                    structural_similarities.append(sim)

            if structural_similarities:
                scores["max_structural_similarity"] = max(structural_similarities)
                scores["avg_structural_similarity"] = np.mean(structural_similarities)

        # Spectral similarity
        spectral_similarities = []
        for match in matches:
            # Mock spectral similarity calculation
            sim = np.random.random() * 0.8 + 0.1  # 0.1-0.9 range
            spectral_similarities.append(sim)

        if spectral_similarities:
            scores["max_spectral_similarity"] = max(spectral_similarities)
            scores["avg_spectral_similarity"] = np.mean(spectral_similarities)

        # Overall similarity
        all_similarities = []
        if "max_structural_similarity" in scores:
            all_similarities.append(scores["max_structural_similarity"])
        if "max_spectral_similarity" in scores:
            all_similarities.append(scores["max_spectral_similarity"])

        if all_similarities:
            scores["overall_similarity"] = np.mean(all_similarities)

        return scores

    def _calculate_smiles_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate SMILES similarity using molecular fingerprints."""
        if not RDKIT_AVAILABLE:
            # Fallback to simple string similarity
            return float(
                len(set(smiles1) & set(smiles2)) / len(set(smiles1) | set(smiles2))
            )

        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            # Calculate Tanimoto similarity using Morgan fingerprints
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

            return float(Chem.DataStructs.TanimotoSimilarity(fp1, fp2))

        except Exception as e:
            logger.warning(f"Error calculating SMILES similarity: {e}")
            return 0.0

    def _calculate_spectral_similarity(
        self, fingerprint: SpectralFingerprint, known_spectral_data: Dict[str, Any]
    ) -> float:
        """Calculate spectral similarity."""
        # Mock implementation - would use sophisticated spectral comparison

        if "peaks" not in known_spectral_data:
            return 0.0

        known_peaks = known_spectral_data["peaks"]

        # Simple peak matching
        matches = 0
        tolerance = 50.0  # Frequency tolerance

        for fp_freq, fp_int in fingerprint.peaks[:5]:  # Check top 5 peaks
            for kp_freq, kp_int in known_peaks:
                if abs(fp_freq - kp_freq) < tolerance:
                    # Weight by intensity similarity
                    int_sim = 1.0 - abs(fp_int - kp_int) / max(fp_int, kp_int)
                    matches += int_sim
                    break

        return min(matches / 5.0, 1.0)  # Normalize to 0-1

    def _has_conflicting_information(
        self, candidate: StructureCandidate, matches: List[KnowledgeGraphEntry]
    ) -> bool:
        """Check if there's conflicting information among matches."""
        # Check for conflicting properties among high-confidence matches
        high_conf_matches = [m for m in matches if m.confidence_score > 0.7]

        if len(high_conf_matches) < 2:
            return False

        # Check for property conflicts
        for prop in ["molecular_weight", "melting_point", "boiling_point"]:
            values = []
            for match in high_conf_matches:
                if prop in match.properties:
                    values.append(match.properties[prop])

            if len(values) > 1:
                # Check coefficient of variation
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                if cv > 0.1:  # 10% variation threshold
                    return True

        return False

    def _generate_explanation(
        self,
        result: ValidationResult,
        confidence: ConfidenceLevel,
        matches: List[KnowledgeGraphEntry],
        inconsistencies: List[str],
    ) -> str:
        """Generate human-readable explanation of validation result."""
        explanations = {
            ValidationResult.VALIDATED: {
                ConfidenceLevel.HIGH: "Structure validated with high confidence based on exact matches in knowledge graph.",
                ConfidenceLevel.MEDIUM: "Structure validated with medium confidence based on good matches in knowledge graph.",
                ConfidenceLevel.LOW: "Structure possibly validated but with low confidence due to limited matching data.",
            },
            ValidationResult.INCONSISTENT: {
                ConfidenceLevel.HIGH: "Structure shows significant inconsistencies with known data.",
                ConfidenceLevel.MEDIUM: "Structure shows some inconsistencies with known data that require investigation.",
                ConfidenceLevel.LOW: "Structure shows minor inconsistencies that may be within experimental error.",
            },
            ValidationResult.NOVEL: {
                ConfidenceLevel.HIGH: "Structure appears to be a novel compound with high confidence.",
                ConfidenceLevel.MEDIUM: "Structure appears to be novel or has limited representation in knowledge graph.",
                ConfidenceLevel.LOW: "Structure may be novel but requires further validation.",
            },
            ValidationResult.UNCERTAIN: {
                ConfidenceLevel.HIGH: "Validation result uncertain despite good data quality.",
                ConfidenceLevel.MEDIUM: "Validation result uncertain due to conflicting or incomplete information.",
                ConfidenceLevel.LOW: "Validation result highly uncertain due to limited data.",
            },
        }

        base_explanation = explanations[result][confidence]

        # Add match information
        if matches:
            base_explanation += f" Found {len(matches)} potentially related compounds in knowledge graph."
        else:
            base_explanation += (
                " No closely related compounds found in knowledge graph."
            )

        # Add inconsistency information
        if inconsistencies:
            base_explanation += f" {len(inconsistencies)} inconsistencies detected."

        return base_explanation

    def _generate_recommendations(
        self,
        result: ValidationResult,
        novelty_indicators: List[NoveltyType],
        inconsistencies: List[str],
    ) -> List[str]:
        """Generate recommendations based on validation result."""
        recommendations = []

        if result == ValidationResult.VALIDATED:
            recommendations.append(
                "Structure appears valid. Consider cross-validation with additional spectroscopy techniques."
            )

        elif result == ValidationResult.INCONSISTENT:
            recommendations.append(
                "Investigate inconsistencies before accepting structure."
            )
            recommendations.append(
                "Consider re-examining experimental conditions and data quality."
            )

        elif result == ValidationResult.NOVEL:
            recommendations.append(
                "Structure appears novel. Recommend comprehensive characterization."
            )
            recommendations.append("Consider synthesis and independent confirmation.")
            recommendations.append("Submit to chemical databases after validation.")

        elif result == ValidationResult.UNCERTAIN:
            recommendations.append("Collect additional experimental evidence.")
            recommendations.append("Try alternative structural elucidation methods.")

        # Add novelty-specific recommendations
        if NoveltyType.NEW_COMPOUND in novelty_indicators:
            recommendations.append(
                "Novel compound detected. Recommend full structural characterization."
            )

        if NoveltyType.NEW_SPECTRAL_PATTERN in novelty_indicators:
            recommendations.append(
                "Unusual spectral pattern detected. Verify experimental conditions."
            )

        # Add inconsistency-specific recommendations
        if inconsistencies:
            recommendations.append("Review and resolve identified inconsistencies.")

        return recommendations


class NoveltyDetector:
    """Detector for novel compounds and spectral patterns."""

    def __init__(self, use_gpu: bool = True):
        """Initialize novelty detector."""
        self.use_gpu = use_gpu
        self.fingerprint_generator = SpectralFingerprintGenerator()
        logger.info(f"NoveltyDetector initialized with GPU: {use_gpu}")

    def detect_novelty(
        self,
        candidate: StructureCandidate,
        experimental_spectrum: ExperimentalSpectrum,
        spectroscopy_type: str = "ir",
        novelty_threshold: float = 0.7,
    ) -> NoveltyDetectionResult:
        """
        Detect novelty in structure and spectroscopic data.

        Args:
            candidate: Structure candidate
            experimental_spectrum: Experimental spectrum
            spectroscopy_type: Type of spectroscopy
            novelty_threshold: Threshold for novelty detection

        Returns:
            Novelty detection result
        """
        start_time = time.time()

        # Generate spectral fingerprint
        fingerprint = self.fingerprint_generator.generate_fingerprint(
            experimental_spectrum, spectroscopy_type
        )

        # Analyze spectral anomalies
        spectral_anomalies = self._detect_spectral_anomalies(
            fingerprint, spectroscopy_type
        )

        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(
            candidate, fingerprint, spectral_anomalies
        )

        # Determine novelty types
        novelty_types = self._determine_novelty_types(
            candidate, fingerprint, spectral_anomalies
        )

        # Find similar compounds for comparison
        similar_compounds = self._find_similar_compounds(candidate, fingerprint)

        # Collect evidence
        evidence = self._collect_novelty_evidence(
            candidate, fingerprint, spectral_anomalies, similar_compounds
        )

        # Calculate statistical significance
        significance = self._calculate_statistical_significance(
            novelty_score, similar_compounds
        )

        detection_time = time.time() - start_time

        result = NoveltyDetectionResult(
            is_novel=novelty_score > novelty_threshold,
            novelty_types=novelty_types,
            novelty_score=novelty_score,
            evidence=evidence,
            similar_compounds=similar_compounds,
            spectral_anomalies=spectral_anomalies,
            statistical_significance=significance,
            detection_time=detection_time,
        )

        return result

    def _detect_spectral_anomalies(
        self, fingerprint: SpectralFingerprint, spectroscopy_type: str
    ) -> List[str]:
        """Detect anomalies in spectral fingerprint."""
        anomalies = []

        # Check for unusual peak count
        expected_peak_ranges = {
            "ir": (5, 20),
            "nmr_1h": (3, 15),
            "nmr_13c": (3, 25),
            "ms": (5, 30),
        }

        if spectroscopy_type in expected_peak_ranges:
            min_peaks, max_peaks = expected_peak_ranges[spectroscopy_type]
            if fingerprint.peak_count < min_peaks:
                anomalies.append(f"Unusually few peaks ({fingerprint.peak_count})")
            elif fingerprint.peak_count > max_peaks:
                anomalies.append(f"Unusually many peaks ({fingerprint.peak_count})")

        # Check spectral moments
        mean_freq, variance, skewness, kurtosis = fingerprint.spectral_moments

        if abs(skewness) > 2.0:
            anomalies.append(f"Unusual spectral asymmetry (skewness: {skewness:.2f})")

        if kurtosis > 10.0:
            anomalies.append(f"Unusual peak sharpness (kurtosis: {kurtosis:.2f})")

        # Check characteristic regions
        for region, intensity in fingerprint.characteristic_regions.items():
            # Mock thresholds - would use statistical analysis of known data
            if intensity > 1000:  # Arbitrary high threshold
                anomalies.append(f"Unusually high intensity in {region} region")

        return anomalies

    def _calculate_novelty_score(
        self,
        candidate: StructureCandidate,
        fingerprint: SpectralFingerprint,
        anomalies: List[str],
    ) -> float:
        """Calculate overall novelty score (0-1)."""
        score_components = []

        # Spectral uniqueness score
        spectral_score = min(len(anomalies) / 5.0, 1.0)  # Normalize to 0-1
        score_components.append(spectral_score * 0.4)  # 40% weight

        # Structural uniqueness score (mock)
        if candidate.smiles:
            # Would use sophisticated structural analysis
            structural_score = np.random.random() * 0.5 + 0.25  # 0.25-0.75 range
        else:
            structural_score = 0.5  # Default moderate score
        score_components.append(structural_score * 0.3)  # 30% weight

        # Property uniqueness score
        if (
            hasattr(candidate, "predicted_properties")
            and candidate.predicted_properties
        ):
            # Mock property analysis
            prop_score = np.random.random() * 0.6 + 0.2  # 0.2-0.8 range
        else:
            prop_score = 0.4  # Default moderate score
        score_components.append(prop_score * 0.3)  # 30% weight

        return sum(score_components)

    def _determine_novelty_types(
        self,
        candidate: StructureCandidate,
        fingerprint: SpectralFingerprint,
        anomalies: List[str],
    ) -> List[NoveltyType]:
        """Determine types of novelty detected."""
        novelty_types = []

        # Check for new compound indicators
        if (
            candidate.confidence_score < 0.3
            or len(anomalies) > 3  # Low structural confidence
        ):  # Many spectral anomalies
            novelty_types.append(NoveltyType.NEW_COMPOUND)

        # Check for new spectral patterns
        if any("unusual" in anomaly.lower() for anomaly in anomalies):
            novelty_types.append(NoveltyType.NEW_SPECTRAL_PATTERN)

        # Check for unexpected behavior
        if len(fingerprint.peaks) > 20:  # Very complex spectrum
            novelty_types.append(NoveltyType.UNEXPECTED_BEHAVIOR)

        # Check for new properties
        if (
            hasattr(candidate, "predicted_properties")
            and candidate.predicted_properties
            and any(abs(v) > 100 for v in candidate.predicted_properties.values())
        ):
            novelty_types.append(NoveltyType.NEW_PROPERTY)

        return novelty_types

    def _find_similar_compounds(
        self, candidate: StructureCandidate, fingerprint: SpectralFingerprint
    ) -> List[KnowledgeGraphEntry]:
        """Find similar compounds for comparison."""
        # Mock implementation - would use real knowledge graph search
        similar_compounds = []

        # Create mock similar compounds
        for i in range(3):
            similar = KnowledgeGraphEntry(
                compound_id=f"similar_{i:03d}",
                smiles=f"C{i}",
                inchi=None,
                compound_name=f"Similar Compound {i+1}",
                spectral_data={},
                properties={},
                related_compounds=[],
                literature_references=[],
                confidence_score=0.8 - i * 0.1,
                last_updated="2024-01-01",
            )
            similar_compounds.append(similar)

        return similar_compounds

    def _collect_novelty_evidence(
        self,
        candidate: StructureCandidate,
        fingerprint: SpectralFingerprint,
        anomalies: List[str],
        similar_compounds: List[KnowledgeGraphEntry],
    ) -> Dict[str, Any]:
        """Collect evidence supporting novelty assessment."""
        evidence = {
            "spectral_fingerprint_hash": fingerprint.fingerprint_hash,
            "spectral_anomalies": anomalies,
            "peak_count": fingerprint.peak_count,
            "spectral_moments": fingerprint.spectral_moments,
            "similar_compound_count": len(similar_compounds),
            "structure_confidence": candidate.confidence_score,
        }

        if hasattr(candidate, "predicted_properties"):
            evidence["predicted_properties"] = candidate.predicted_properties

        if candidate.smiles:
            evidence["smiles"] = candidate.smiles

        return evidence

    def _calculate_statistical_significance(
        self, novelty_score: float, similar_compounds: List[KnowledgeGraphEntry]
    ) -> float:
        """Calculate statistical significance of novelty."""
        # Mock implementation - would use proper statistical analysis

        if not similar_compounds:
            return 0.9  # High significance if no similar compounds

        # Simple significance based on novelty score and similarity
        base_significance = novelty_score

        # Adjust based on number and quality of similar compounds
        adjustment = len(similar_compounds) * 0.1
        significance = max(0.0, base_significance - adjustment)

        return min(significance, 1.0)


# Utility functions
def validate_structure_batch(
    candidates: List[StructureCandidate],
    experimental_spectra: List[ExperimentalSpectrum],
    spectroscopy_types: List[str],
    use_gpu: bool = True,
) -> List[ValidationReport]:
    """
    Validate multiple structure candidates in batch.

    Args:
        candidates: List of structure candidates
        experimental_spectra: List of experimental spectra
        spectroscopy_types: List of spectroscopy types
        use_gpu: Whether to use GPU acceleration

    Returns:
        List of validation reports
    """
    if not (len(candidates) == len(experimental_spectra) == len(spectroscopy_types)):
        raise ValueError("All input lists must have the same length")

    validator = KnowledgeGraphValidator(use_gpu=use_gpu)
    reports = []

    for candidate, spectrum, spec_type in zip(
        candidates, experimental_spectra, spectroscopy_types
    ):
        try:
            report = validator.validate_structure(candidate, spectrum, spec_type)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to validate structure: {e}")
            # Create error report
            error_report = ValidationReport(
                structure_candidate=candidate,
                validation_result=ValidationResult.ERROR,
                confidence_level=ConfidenceLevel.VERY_LOW,
                explanation=f"Validation failed: {str(e)}",
            )
            reports.append(error_report)

    return reports


def detect_novelty_batch(
    candidates: List[StructureCandidate],
    experimental_spectra: List[ExperimentalSpectrum],
    spectroscopy_types: List[str],
    use_gpu: bool = True,
) -> List[NoveltyDetectionResult]:
    """
    Detect novelty for multiple candidates in batch.

    Args:
        candidates: List of structure candidates
        experimental_spectra: List of experimental spectra
        spectroscopy_types: List of spectroscopy types
        use_gpu: Whether to use GPU acceleration

    Returns:
        List of novelty detection results
    """
    if not (len(candidates) == len(experimental_spectra) == len(spectroscopy_types)):
        raise ValueError("All input lists must have the same length")

    detector = NoveltyDetector(use_gpu=use_gpu)
    results = []

    for candidate, spectrum, spec_type in zip(
        candidates, experimental_spectra, spectroscopy_types
    ):
        try:
            result = detector.detect_novelty(candidate, spectrum, spec_type)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to detect novelty: {e}")
            # Create error result
            error_result = NoveltyDetectionResult(
                is_novel=False,
                novelty_types=[],
                novelty_score=0.0,
                evidence={"error": str(e)},
                similar_compounds=[],
                spectral_anomalies=[],
                statistical_significance=0.0,
            )
            results.append(error_result)

    return results


# Exception classes
class KnowledgeGraphIntegrationError(Exception):
    """Base exception for knowledge graph integration errors."""

    pass


class ValidationError(KnowledgeGraphIntegrationError):
    """Error in structure validation."""

    pass


class NoveltyDetectionError(KnowledgeGraphIntegrationError):
    """Error in novelty detection."""

    pass
