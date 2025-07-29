"""
AI-Driven Inverse Spectroscopy for Structure Elucidation.

This module implements a sophisticated framework for inverse spectroscopy that takes
experimental spectra as input and proposes candidate chemical structures. It combines
generative chemistry, spectral matching algorithms, and structure validation to explore
chemical space for optimal structural matches.
"""

import itertools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cosine, euclidean

# Optional imports with fallbacks
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Molecular operations will be limited.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning features will be limited.")

# Import from other modules
from .experimental_processing import ExperimentalSpectrum, SpectralPeak
from .predictive_models import (
    GPUSpectroscopyPredictor,
    ModelArchitecture,
    ModelConfiguration,
    SpectralPrediction,
    SpectroscopyType,
)

# =============================================================================
# Enumerations
# =============================================================================


class MatchingMethod(Enum):
    """Methods for spectral matching."""

    CROSS_CORRELATION = "cross_correlation"
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    PEAK_MATCHING = "peak_matching"
    WEIGHTED_SIMILARITY = "weighted_similarity"
    ML_SIMILARITY = "ml_similarity"


class ScoringFunction(Enum):
    """Scoring functions for structure ranking."""

    SPECTRAL_SIMILARITY = "spectral_similarity"
    CHEMICAL_PLAUSIBILITY = "chemical_plausibility"
    COMBINED_SCORE = "combined_score"
    BAYESIAN_SCORE = "bayesian_score"
    ENSEMBLE_SCORE = "ensemble_score"


class StructureGenerationMethod(Enum):
    """Methods for generating candidate structures."""

    FRAGMENT_ASSEMBLY = "fragment_assembly"
    GENERATIVE_MODEL = "generative_model"
    DATABASE_SEARCH = "database_search"
    EVOLUTIONARY_ALGORITHM = "evolutionary_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID_APPROACH = "hybrid_approach"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StructureCandidate:
    """Container for candidate molecular structures."""

    smiles: str
    mol: Optional[Any] = None  # RDKit Mol object
    confidence_score: float = 0.0
    spectral_match_score: float = 0.0
    chemical_plausibility: float = 0.0
    generation_method: Optional[StructureGenerationMethod] = None
    molecular_properties: Dict[str, float] = field(default_factory=dict)
    predicted_spectrum: Optional[SpectralPrediction] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize RDKit molecule if available."""
        if RDKIT_AVAILABLE and self.mol is None and self.smiles:
            try:
                self.mol = Chem.MolFromSmiles(self.smiles)
                if self.mol is not None:
                    # Calculate basic molecular properties
                    self.molecular_properties.update(
                        {
                            "molecular_weight": Descriptors.MolWt(self.mol),
                            "logp": Descriptors.MolLogP(self.mol),
                            "tpsa": Descriptors.TPSA(self.mol),
                            "rotatable_bonds": Descriptors.NumRotatableBonds(self.mol),
                            "hbd": Descriptors.NumHDonors(self.mol),
                            "hba": Descriptors.NumHAcceptors(self.mol),
                        }
                    )
            except Exception:
                self.mol = None


@dataclass
class SpectralMatch:
    """Container for spectral matching results."""

    experimental_spectrum: ExperimentalSpectrum
    predicted_spectrum: SpectralPrediction
    similarity_score: float
    peak_matches: List[Tuple[SpectralPeak, SpectralPeak]] = field(default_factory=list)
    matching_method: MatchingMethod = MatchingMethod.COSINE_SIMILARITY
    frequency_shift: float = 0.0
    intensity_scaling: float = 1.0
    residual_spectrum: Optional[np.ndarray] = None

    def calculate_residual(self) -> np.ndarray:
        """Calculate residual between experimental and predicted spectra."""
        if self.residual_spectrum is not None:
            return self.residual_spectrum

        # Interpolate predicted spectrum to experimental frequency grid
        exp_freq = self.experimental_spectrum.frequencies
        pred_freq = self.predicted_spectrum.frequencies
        pred_int = self.predicted_spectrum.intensities

        # Interpolate
        from scipy.interpolate import interp1d

        interp_func = interp1d(pred_freq, pred_int, bounds_error=False, fill_value=0)
        aligned_pred = interp_func(exp_freq)

        # Calculate residual
        self.residual_spectrum = (
            self.experimental_spectrum.intensities
            - aligned_pred * self.intensity_scaling
        )
        return self.residual_spectrum


@dataclass
class InverseResult:
    """Container for inverse spectroscopy results."""

    query_spectrum: ExperimentalSpectrum
    candidate_structures: List[StructureCandidate]
    best_match: Optional[StructureCandidate] = None
    search_parameters: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)

    def get_top_candidates(self, n: int = 10) -> List[StructureCandidate]:
        """Get top N candidates by confidence score."""
        sorted_candidates = sorted(
            self.candidate_structures, key=lambda x: x.confidence_score, reverse=True
        )
        return sorted_candidates[:n]

    def get_diversity_filtered_candidates(
        self, max_candidates: int = 20, diversity_threshold: float = 0.8
    ) -> List[StructureCandidate]:
        """Get diverse set of candidates using molecular fingerprints."""
        if not RDKIT_AVAILABLE:
            return self.get_top_candidates(max_candidates)

        selected = []
        candidates = self.get_top_candidates(
            max_candidates * 2
        )  # Start with more candidates

        for candidate in candidates:
            if candidate.mol is None:
                continue

            # Calculate fingerprint
            fp = GetMorganFingerprintAsBitVect(candidate.mol, radius=2, nBits=1024)
            fp_array = np.array([fp.GetBit(i) for i in range(fp.GetNumBits())])

            # Check diversity against selected candidates
            is_diverse = True
            for selected_candidate in selected:
                if selected_candidate.mol is None:
                    continue

                selected_fp = GetMorganFingerprintAsBitVect(
                    selected_candidate.mol, radius=2, nBits=1024
                )
                selected_fp_array = np.array(
                    [selected_fp.GetBit(i) for i in range(selected_fp.GetNumBits())]
                )

                # Calculate Tanimoto similarity
                intersection = np.sum(fp_array & selected_fp_array)
                union = np.sum(fp_array | selected_fp_array)
                tanimoto = intersection / union if union > 0 else 0

                if tanimoto > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)
                if len(selected) >= max_candidates:
                    break

        return selected


@dataclass
class MatchingParameters:
    """Parameters for spectral matching algorithms."""

    matching_method: MatchingMethod = MatchingMethod.WEIGHTED_SIMILARITY
    peak_tolerance: float = 0.1  # Frequency tolerance for peak matching
    intensity_weight: float = 0.7
    frequency_weight: float = 0.3
    normalization: bool = True
    allow_frequency_shift: bool = True
    max_frequency_shift: float = 5.0
    similarity_threshold: float = 0.5

    def __post_init__(self):
        """Validate parameters."""
        if self.intensity_weight + self.frequency_weight != 1.0:
            # Normalize weights
            total = self.intensity_weight + self.frequency_weight
            self.intensity_weight /= total
            self.frequency_weight /= total


# =============================================================================
# Base Classes
# =============================================================================


class BaseSpectralMatcher(ABC):
    """Abstract base class for spectral matching algorithms."""

    @abstractmethod
    def calculate_similarity(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate similarity between experimental and predicted spectra."""
        pass


class BaseStructureGenerator(ABC):
    """Abstract base class for structure generation methods."""

    @abstractmethod
    def generate_candidates(
        self, spectrum: ExperimentalSpectrum, num_candidates: int = 100
    ) -> List[StructureCandidate]:
        """Generate candidate structures from spectrum."""
        pass


# =============================================================================
# Spectral Matching Classes
# =============================================================================


class SpectralMatcher(BaseSpectralMatcher):
    """Main class for spectral matching operations."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    def calculate_similarity(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate similarity between experimental and predicted spectra."""
        if params.matching_method == MatchingMethod.CROSS_CORRELATION:
            return self._cross_correlation_match(experimental, predicted, params)
        elif params.matching_method == MatchingMethod.COSINE_SIMILARITY:
            return self._cosine_similarity_match(experimental, predicted, params)
        elif params.matching_method == MatchingMethod.EUCLIDEAN_DISTANCE:
            return self._euclidean_distance_match(experimental, predicted, params)
        elif params.matching_method == MatchingMethod.PEAK_MATCHING:
            return self._peak_matching(experimental, predicted, params)
        elif params.matching_method == MatchingMethod.WEIGHTED_SIMILARITY:
            return self._weighted_similarity_match(experimental, predicted, params)
        else:
            return self._cosine_similarity_match(experimental, predicted, params)

    def _align_spectra(
        self, experimental: ExperimentalSpectrum, predicted: SpectralPrediction
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align experimental and predicted spectra to common frequency grid."""
        # Find common frequency range
        exp_freq = experimental.frequencies
        pred_freq = predicted.frequencies

        min_freq = max(np.min(exp_freq), np.min(pred_freq))
        max_freq = min(np.max(exp_freq), np.max(pred_freq))

        # Create common frequency grid
        n_points = min(len(exp_freq), len(pred_freq))
        common_freq = np.linspace(min_freq, max_freq, n_points)

        # Interpolate both spectra
        from scipy.interpolate import interp1d

        exp_interp = interp1d(
            exp_freq, experimental.intensities, bounds_error=False, fill_value=0
        )(common_freq)
        pred_interp = interp1d(
            pred_freq, predicted.intensities, bounds_error=False, fill_value=0
        )(common_freq)

        return exp_interp, pred_interp

    def _cross_correlation_match(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate cross-correlation based similarity."""
        exp_int, pred_int = self._align_spectra(experimental, predicted)

        # Normalize if requested
        if params.normalization:
            exp_int = (
                exp_int / np.linalg.norm(exp_int)
                if np.linalg.norm(exp_int) > 0
                else exp_int
            )
            pred_int = (
                pred_int / np.linalg.norm(pred_int)
                if np.linalg.norm(pred_int) > 0
                else pred_int
            )

        # Calculate cross-correlation
        correlation = np.correlate(exp_int, pred_int, mode="full")
        max_corr = np.max(correlation)

        # Normalize correlation coefficient
        similarity = (
            max_corr / (len(exp_int) * np.std(exp_int) * np.std(pred_int))
            if np.std(exp_int) * np.std(pred_int) > 0
            else 0
        )
        similarity = max(0, min(1, similarity))  # Clamp to [0,1]

        return SpectralMatch(
            experimental_spectrum=experimental,
            predicted_spectrum=predicted,
            similarity_score=similarity,
            matching_method=params.matching_method,
        )

    def _cosine_similarity_match(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate cosine similarity."""
        exp_int, pred_int = self._align_spectra(experimental, predicted)

        # Calculate cosine similarity
        similarity = (
            1 - cosine(exp_int, pred_int) if np.any(exp_int) and np.any(pred_int) else 0
        )
        similarity = max(0, min(1, similarity))  # Clamp to [0,1]

        return SpectralMatch(
            experimental_spectrum=experimental,
            predicted_spectrum=predicted,
            similarity_score=similarity,
            matching_method=params.matching_method,
        )

    def _euclidean_distance_match(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate similarity based on Euclidean distance."""
        exp_int, pred_int = self._align_spectra(experimental, predicted)

        # Calculate normalized Euclidean distance
        distance = euclidean(exp_int, pred_int)
        max_distance = np.sqrt(np.sum(exp_int**2) + np.sum(pred_int**2))

        # Convert distance to similarity
        similarity = 1 - (distance / max_distance) if max_distance > 0 else 0
        similarity = max(0, min(1, similarity))  # Clamp to [0,1]

        return SpectralMatch(
            experimental_spectrum=experimental,
            predicted_spectrum=predicted,
            similarity_score=similarity,
            matching_method=params.matching_method,
        )

    def _peak_matching(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Match based on peak positions and intensities."""
        # Extract peaks from experimental spectrum
        exp_peaks = experimental.peaks
        if not exp_peaks:
            # Detect peaks if not already done
            from scipy.signal import find_peaks

            peak_indices, _ = find_peaks(
                experimental.intensities, height=0.1 * np.max(experimental.intensities)
            )
            exp_peaks = []
            for idx in peak_indices:
                peak = SpectralPeak(
                    frequency=experimental.frequencies[idx],
                    intensity=experimental.intensities[idx],
                    width=1.0,
                    area=experimental.intensities[idx],
                )
                exp_peaks.append(peak)

        # Extract peaks from predicted spectrum
        from scipy.signal import find_peaks

        pred_peak_indices, _ = find_peaks(
            predicted.intensities, height=0.1 * np.max(predicted.intensities)
        )
        pred_peaks = []
        for idx in pred_peak_indices:
            peak = SpectralPeak(
                frequency=predicted.frequencies[idx],
                intensity=predicted.intensities[idx],
                width=1.0,
                area=predicted.intensities[idx],
            )
            pred_peaks.append(peak)

        # Match peaks
        matched_peaks = []
        total_exp_intensity = sum(p.intensity for p in exp_peaks)
        matched_intensity = 0

        for exp_peak in exp_peaks:
            best_match = None
            best_score = 0

            for pred_peak in pred_peaks:
                freq_diff = abs(exp_peak.frequency - pred_peak.frequency)
                if freq_diff <= params.peak_tolerance:
                    # Calculate match score
                    freq_score = 1 - (freq_diff / params.peak_tolerance)
                    int_score = min(exp_peak.intensity, pred_peak.intensity) / max(
                        exp_peak.intensity, pred_peak.intensity
                    )

                    match_score = (
                        params.frequency_weight * freq_score
                        + params.intensity_weight * int_score
                    )

                    if match_score > best_score:
                        best_score = match_score
                        best_match = (exp_peak, pred_peak)

            if best_match:
                matched_peaks.append(best_match)
                matched_intensity += exp_peak.intensity

        # Calculate overall similarity
        similarity = (
            matched_intensity / total_exp_intensity if total_exp_intensity > 0 else 0
        )

        return SpectralMatch(
            experimental_spectrum=experimental,
            predicted_spectrum=predicted,
            similarity_score=similarity,
            matching_method=params.matching_method,
            peak_matches=matched_peaks,
        )

    def _weighted_similarity_match(
        self,
        experimental: ExperimentalSpectrum,
        predicted: SpectralPrediction,
        params: MatchingParameters,
    ) -> SpectralMatch:
        """Calculate weighted combination of multiple similarity measures."""
        # Calculate individual similarities
        cosine_match = self._cosine_similarity_match(experimental, predicted, params)
        peak_match = self._peak_matching(experimental, predicted, params)

        # Weighted combination
        weighted_similarity = (
            0.6 * cosine_match.similarity_score + 0.4 * peak_match.similarity_score
        )

        return SpectralMatch(
            experimental_spectrum=experimental,
            predicted_spectrum=predicted,
            similarity_score=weighted_similarity,
            matching_method=params.matching_method,
            peak_matches=peak_match.peak_matches,
        )


# =============================================================================
# Structure Generation Classes
# =============================================================================


class FragmentBasedGenerator(BaseStructureGenerator):
    """Generate structures using fragment assembly approach."""

    def __init__(self, fragment_database: Optional[List[str]] = None):
        self.fragment_database = fragment_database or self._get_default_fragments()

    def _get_default_fragments(self) -> List[str]:
        """Get default molecular fragments."""
        return [
            # Common functional groups
            "C",
            "CC",
            "CCC",
            "CCCC",  # Alkyl chains
            "c1ccccc1",  # Benzene
            "C(=O)",  # Carbonyl
            "C(=O)O",  # Carboxyl
            "N",  # Amino
            "O",  # Hydroxyl
            "S",  # Sulfur
            "P",  # Phosphorus
            "C(=O)N",  # Amide
            "C=C",  # Alkene
            "C#C",  # Alkyne
            "c1ccc2ccccc2c1",  # Naphthalene
            "C1CCCCC1",  # Cyclohexane
            "C1CCCC1",  # Cyclopentane
        ]

    def generate_candidates(
        self, spectrum: ExperimentalSpectrum, num_candidates: int = 100
    ) -> List[StructureCandidate]:
        """Generate candidates by assembling molecular fragments."""
        candidates = []

        if not RDKIT_AVAILABLE:
            # Return empty list if RDKit not available
            return candidates

        # Estimate molecular complexity from spectrum
        num_peaks = (
            len(spectrum.peaks) if spectrum.peaks else len(spectrum.frequencies) // 10
        )
        complexity = min(max(2, num_peaks // 5), 8)  # Limit complexity

        # Generate combinations of fragments
        for size in range(2, complexity + 1):
            for combination in itertools.combinations_with_replacement(
                self.fragment_database, size
            ):
                if len(candidates) >= num_candidates:
                    break

                # Try to assemble fragments
                try:
                    mol = self._assemble_fragments(combination)
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        candidate = StructureCandidate(
                            smiles=smiles,
                            mol=mol,
                            generation_method=StructureGenerationMethod.FRAGMENT_ASSEMBLY,
                        )
                        candidates.append(candidate)

                except Exception:
                    continue

        return candidates[:num_candidates]

    def _assemble_fragments(self, fragments: Tuple[str, ...]) -> Optional[Any]:
        """Assemble molecular fragments into a structure."""
        if not RDKIT_AVAILABLE:
            return None

        try:
            # Simple assembly by string concatenation (simplified approach)
            combined_smiles = ""
            for i, fragment in enumerate(fragments):
                if i == 0:
                    combined_smiles = fragment
                else:
                    # Try different connection patterns
                    if "c1ccccc1" in fragment:  # Aromatic ring
                        combined_smiles += f".{fragment}"
                    else:
                        combined_smiles += fragment

            # Parse and sanitize
            mol = Chem.MolFromSmiles(combined_smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
                return mol

        except Exception:
            pass

        return None


class DatabaseSearchGenerator(BaseStructureGenerator):
    """Generate candidates by searching molecular databases."""

    def __init__(self, database_smiles: Optional[List[str]] = None):
        self.database = database_smiles or self._get_default_database()

    def _get_default_database(self) -> List[str]:
        """Get default molecular database."""
        # Simplified database with common organic molecules
        return [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCc1ccccc1",  # Ethylbenzene
            "Cc1ccccc1",  # Toluene
            "CC(C)O",  # Isopropanol
            "CC(=O)C",  # Acetone
            "CCCC",  # Butane
            "c1ccc2ccccc2c1",  # Naphthalene
            "CC(=O)N",  # Acetamide
            "CCCCCC",  # Hexane
            "c1ccc(cc1)O",  # Phenol
            "CCN",  # Ethylamine
            "CC(=O)CC",  # Butanone
            "c1ccc(cc1)C",  # Methylbenzene
        ]

    def generate_candidates(
        self, spectrum: ExperimentalSpectrum, num_candidates: int = 100
    ) -> List[StructureCandidate]:
        """Generate candidates from database search."""
        candidates = []

        # Simple filtering based on spectrum properties
        # In practice, this would use more sophisticated matching
        for smiles in self.database[:num_candidates]:
            candidate = StructureCandidate(
                smiles=smiles,
                generation_method=StructureGenerationMethod.DATABASE_SEARCH,
            )
            candidates.append(candidate)

        return candidates


# =============================================================================
# Main Inverse Spectroscopy Engine
# =============================================================================


class InverseSpectroscopyEngine:
    """Main engine for inverse spectroscopy structure elucidation."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.spectral_matcher = SpectralMatcher(use_gpu)
        self.predictive_model = None
        self.structure_generators = {
            StructureGenerationMethod.FRAGMENT_ASSEMBLY: FragmentBasedGenerator(),
            StructureGenerationMethod.DATABASE_SEARCH: DatabaseSearchGenerator(),
        }

    def set_predictive_model(self, model: GPUSpectroscopyPredictor) -> None:
        """Set the predictive model for spectrum generation."""
        self.predictive_model = model

    def analyze_spectrum(
        self,
        experimental_spectrum: ExperimentalSpectrum,
        num_candidates: int = 50,
        generation_methods: Optional[List[StructureGenerationMethod]] = None,
        matching_params: Optional[MatchingParameters] = None,
    ) -> InverseResult:
        """Perform inverse spectroscopy analysis."""
        import time

        start_time = time.time()

        if generation_methods is None:
            generation_methods = [
                StructureGenerationMethod.FRAGMENT_ASSEMBLY,
                StructureGenerationMethod.DATABASE_SEARCH,
            ]

        if matching_params is None:
            matching_params = MatchingParameters()

        # Generate candidate structures
        all_candidates = []
        candidates_per_method = num_candidates // len(generation_methods)

        for method in generation_methods:
            if method in self.structure_generators:
                generator = self.structure_generators[method]
                method_candidates = generator.generate_candidates(
                    experimental_spectrum, candidates_per_method
                )
                all_candidates.extend(method_candidates)

        # Score candidates using predictive model
        if self.predictive_model is not None:
            all_candidates = self._score_candidates_with_prediction(
                all_candidates, experimental_spectrum, matching_params
            )
        else:
            # Use chemical plausibility scoring
            all_candidates = self._score_candidates_chemically(all_candidates)

        # Find best match
        best_candidate = None
        if all_candidates:
            best_candidate = max(all_candidates, key=lambda x: x.confidence_score)

        computation_time = time.time() - start_time

        return InverseResult(
            query_spectrum=experimental_spectrum,
            candidate_structures=all_candidates,
            best_match=best_candidate,
            search_parameters={
                "num_candidates": num_candidates,
                "generation_methods": [m.value for m in generation_methods],
                "matching_method": matching_params.matching_method.value,
            },
            computation_time=computation_time,
            convergence_metrics={
                "candidates_generated": len(all_candidates),
                "best_score": best_candidate.confidence_score
                if best_candidate
                else 0.0,
            },
        )

    def _score_candidates_with_prediction(
        self,
        candidates: List[StructureCandidate],
        experimental: ExperimentalSpectrum,
        matching_params: MatchingParameters,
    ) -> List[StructureCandidate]:
        """Score candidates using predictive model."""
        if not RDKIT_AVAILABLE:
            return self._score_candidates_chemically(candidates)

        scored_candidates = []

        for candidate in candidates:
            if candidate.mol is None:
                continue

            try:
                # Extract molecular features (simplified)
                features = self._extract_molecular_features(candidate.mol)

                # Predict spectrum
                predictions = self.predictive_model.predict(features.reshape(1, -1))
                if predictions:
                    candidate.predicted_spectrum = predictions[0]

                    # Calculate spectral match
                    match = self.spectral_matcher.calculate_similarity(
                        experimental, predictions[0], matching_params
                    )

                    candidate.spectral_match_score = match.similarity_score

                    # Calculate chemical plausibility
                    candidate.chemical_plausibility = (
                        self._calculate_chemical_plausibility(candidate.mol)
                    )

                    # Combined confidence score
                    candidate.confidence_score = (
                        0.7 * candidate.spectral_match_score
                        + 0.3 * candidate.chemical_plausibility
                    )

                scored_candidates.append(candidate)

            except Exception as e:
                print(f"Warning: Failed to score candidate {candidate.smiles}: {e}")
                continue

        return scored_candidates

    def _score_candidates_chemically(
        self, candidates: List[StructureCandidate]
    ) -> List[StructureCandidate]:
        """Score candidates based on chemical plausibility only."""
        if not RDKIT_AVAILABLE:
            # Assign random scores if RDKit not available
            for candidate in candidates:
                candidate.chemical_plausibility = np.random.random()
                candidate.confidence_score = candidate.chemical_plausibility
            return candidates

        for candidate in candidates:
            if candidate.mol is not None:
                candidate.chemical_plausibility = self._calculate_chemical_plausibility(
                    candidate.mol
                )
                candidate.confidence_score = candidate.chemical_plausibility

        return candidates

    def _extract_molecular_features(self, mol: Any) -> np.ndarray:
        """Extract molecular features for prediction."""
        if not RDKIT_AVAILABLE:
            return np.random.random(256)  # Dummy features

        # Calculate molecular descriptors
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
        ]

        # Pad or truncate to expected size
        expected_size = 256  # Match predictive model input
        if len(features) < expected_size:
            features.extend([0.0] * (expected_size - len(features)))
        else:
            features = features[:expected_size]

        return np.array(features)

    def _calculate_chemical_plausibility(self, mol: Any) -> float:
        """Calculate chemical plausibility score."""
        if not RDKIT_AVAILABLE:
            return np.random.random()

        score = 1.0

        # Penalize unusual molecular weight
        mw = Descriptors.MolWt(mol)
        if mw < 50 or mw > 800:
            score *= 0.5

        # Penalize extreme LogP values
        logp = Descriptors.MolLogP(mol)
        if logp < -3 or logp > 7:
            score *= 0.7

        # Penalize high rotatable bond count
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if rot_bonds > 10:
            score *= 0.8

        # Check for valid structure
        try:
            Chem.SanitizeMol(mol)
        except:
            score *= 0.3

        return max(0.1, min(1.0, score))


class StructureProposer:
    """High-level interface for structure proposal from spectra."""

    def __init__(self, spectroscopy_type: SpectroscopyType, use_gpu: bool = True):
        self.spectroscopy_type = spectroscopy_type
        self.engine = InverseSpectroscopyEngine(use_gpu)

        # Initialize appropriate predictive model
        self._setup_predictive_model()

    def _setup_predictive_model(self) -> None:
        """Setup predictive model for the spectroscopy type."""
        from .predictive_models import (
            IRPredictor,
            MSPredictor,
            NMRPredictor,
            UVVisPredictor,
        )

        if self.spectroscopy_type == SpectroscopyType.NMR_1H:
            model = NMRPredictor("1H", use_gpu=self.engine.use_gpu)
        elif self.spectroscopy_type == SpectroscopyType.NMR_13C:
            model = NMRPredictor("13C", use_gpu=self.engine.use_gpu)
        elif self.spectroscopy_type == SpectroscopyType.IR:
            model = IRPredictor(use_gpu=self.engine.use_gpu)
        elif self.spectroscopy_type == SpectroscopyType.MS:
            model = MSPredictor(use_gpu=self.engine.use_gpu)
        elif self.spectroscopy_type == SpectroscopyType.UV_VIS:
            model = UVVisPredictor(use_gpu=self.engine.use_gpu)
        else:
            # Default to IR
            model = IRPredictor(use_gpu=self.engine.use_gpu)

        self.engine.set_predictive_model(model)

    def propose_structures(
        self, experimental_spectrum: ExperimentalSpectrum, max_proposals: int = 20
    ) -> InverseResult:
        """Propose structures for experimental spectrum."""
        return self.engine.analyze_spectrum(
            experimental_spectrum,
            num_candidates=max_proposals * 3,  # Generate more, then filter
            matching_params=MatchingParameters(similarity_threshold=0.3),
        )


# =============================================================================
# Utility Functions
# =============================================================================


def propose_structures_from_spectrum(
    spectrum: ExperimentalSpectrum,
    spectroscopy_type: SpectroscopyType,
    max_structures: int = 10,
) -> List[StructureCandidate]:
    """High-level function to propose structures from spectrum."""
    proposer = StructureProposer(spectroscopy_type)
    result = proposer.propose_structures(spectrum, max_structures)
    return result.get_top_candidates(max_structures)


def match_experimental_spectrum(
    experimental: ExperimentalSpectrum,
    predicted: SpectralPrediction,
    method: MatchingMethod = MatchingMethod.WEIGHTED_SIMILARITY,
) -> SpectralMatch:
    """Match experimental spectrum against predicted spectrum."""
    matcher = SpectralMatcher()
    params = MatchingParameters(matching_method=method)
    return matcher.calculate_similarity(experimental, predicted, params)


def validate_structural_proposals(
    candidates: List[StructureCandidate], experimental_spectrum: ExperimentalSpectrum
) -> List[StructureCandidate]:
    """Validate and rank structural proposals."""
    # Additional validation can be added here
    # For now, sort by confidence score
    return sorted(candidates, key=lambda x: x.confidence_score, reverse=True)


# =============================================================================
# Exception Classes
# =============================================================================


class InverseError(Exception):
    """Exception raised during inverse spectroscopy analysis."""

    pass


class MatchingError(Exception):
    """Exception raised during spectral matching."""

    pass


class StructureGenerationError(Exception):
    """Exception raised during structure generation."""

    pass
