"""
Advanced 3D molecular geometry and conformer ensemble handling for QML.

This module provides sophisticated tools for managing 3D molecular geometries,
conformer ensembles, and their efficient processing in QML workflows.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
    from rdkit.Chem.rdMolAlign import AlignMol

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .base import QMLModelError
from .on_the_fly import GeometryInput


class GeometryQuality(Enum):
    """Quality levels for molecular geometries."""

    LOW = "low"  # Fast generation, lower quality
    MEDIUM = "medium"  # Balanced quality/speed
    HIGH = "high"  # High quality, slower generation
    ULTRA = "ultra"  # Maximum quality, very slow


class ConformerMethod(Enum):
    """Methods for conformer generation."""

    ETKDG = "etkdg"  # Enhanced Conformer Generation
    MMFF = "mmff"  # MMFF force field optimization
    UFF = "uff"  # UFF force field optimization
    DFT_OPT = "dft_opt"  # DFT optimization (external)
    MIXED = "mixed"  # Multiple methods combined


class AlignmentMethod(Enum):
    """Methods for conformer alignment."""

    MCS = "mcs"  # Maximum Common Substructure
    PHARMACOPHORE = "pharmacophore"  # Pharmacophore-based
    SHAPE = "shape"  # Shape-based alignment
    RMSD = "rmsd"  # Simple RMSD minimization


@dataclass
class GeometryMetadata:
    """Metadata for molecular geometries."""

    method: ConformerMethod
    quality: GeometryQuality
    optimization_steps: int = 0
    final_energy: Optional[float] = None
    rmsd_to_reference: Optional[float] = None
    generation_time: float = 0.0
    convergence_status: str = "unknown"
    force_field_energy: Optional[float] = None
    strain_energy: Optional[float] = None


@dataclass
class ConformerEnsemble:
    """Container for conformer ensembles."""

    geometries: List[GeometryInput]
    reference_geometry: Optional[GeometryInput] = None
    ensemble_energy_spread: float = 0.0
    boltzmann_weights: Optional[np.ndarray] = None
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.geometries)

    def __getitem__(self, index: int) -> GeometryInput:
        return self.geometries[index]

    def get_weighted_average_geometry(self) -> GeometryInput:
        """Get Boltzmann-weighted average geometry."""
        if not self.boltzmann_weights or len(self.boltzmann_weights) != len(
            self.geometries
        ):
            # Equal weights if no Boltzmann weights
            weights = np.ones(len(self.geometries)) / len(self.geometries)
        else:
            weights = self.boltzmann_weights

        # Weighted average coordinates
        avg_coords = np.zeros_like(self.geometries[0].coordinates)
        for geom, weight in zip(self.geometries, weights):
            avg_coords += weight * geom.coordinates

        return GeometryInput(
            coordinates=avg_coords,
            atomic_numbers=self.geometries[0].atomic_numbers.copy(),
            charge=self.geometries[0].charge,
            multiplicity=self.geometries[0].multiplicity,
        )


class AdvancedConformerGenerator:
    """Advanced conformer generation with multiple methods and quality levels."""

    def __init__(
        self,
        max_conformers: int = 50,
        energy_threshold: float = 25.0,  # kcal/mol
        rms_threshold: float = 0.5,  # Angstrom
        quality: GeometryQuality = GeometryQuality.MEDIUM,
        method: ConformerMethod = ConformerMethod.ETKDG,
        random_seed: int = 42,
        num_threads: int = 4,
    ):
        """
        Initialize advanced conformer generator.

        Args:
            max_conformers: Maximum number of conformers
            energy_threshold: Energy threshold for filtering (kcal/mol)
            rms_threshold: RMSD threshold for filtering (Angstrom)
            quality: Quality level for generation
            method: Conformer generation method
            random_seed: Random seed for reproducibility
            num_threads: Number of threads for parallel processing
        """
        if not RDKIT_AVAILABLE:
            raise QMLModelError("RDKit required for advanced conformer generation")

        self.max_conformers = max_conformers
        self.energy_threshold = energy_threshold
        self.rms_threshold = rms_threshold
        self.quality = quality
        self.method = method
        self.random_seed = random_seed
        self.num_threads = num_threads

        # Method-specific parameters
        self._setup_method_parameters()

    def _setup_method_parameters(self):
        """Setup parameters based on method and quality."""
        # Base parameters for ETKDG
        self.etkdg_params = AllChem.ETKDGv3()
        self.etkdg_params.randomSeed = self.random_seed
        self.etkdg_params.numThreads = self.num_threads

        # Quality-based adjustments
        if self.quality == GeometryQuality.LOW:
            self.etkdg_params.maxAttempts = 100
            self.etkdg_params.pruneRmsThresh = 1.0
        elif self.quality == GeometryQuality.MEDIUM:
            self.etkdg_params.maxAttempts = 500
            self.etkdg_params.pruneRmsThresh = 0.5
        elif self.quality == GeometryQuality.HIGH:
            self.etkdg_params.maxAttempts = 1000
            self.etkdg_params.pruneRmsThresh = 0.25
        elif self.quality == GeometryQuality.ULTRA:
            self.etkdg_params.maxAttempts = 5000
            self.etkdg_params.pruneRmsThresh = 0.1

    def generate_conformer_ensemble(
        self,
        molecule: Union[str, Chem.Mol],
        reference_geometry: Optional[GeometryInput] = None,
    ) -> ConformerEnsemble:
        """
        Generate comprehensive conformer ensemble.

        Args:
            molecule: Input molecule
            reference_geometry: Optional reference geometry

        Returns:
            ConformerEnsemble object
        """
        start_time = time.time()

        # Convert to RDKit molecule
        if isinstance(molecule, str):
            mol = Chem.MolFromSmiles(molecule)
            if mol is None:
                raise QMLModelError(f"Invalid SMILES: {molecule}")
        else:
            mol = Chem.Mol(molecule)  # Copy

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate conformers based on method
        if self.method == ConformerMethod.ETKDG:
            geometries = self._generate_etkdg_conformers(mol)
        elif self.method == ConformerMethod.MMFF:
            geometries = self._generate_mmff_conformers(mol)
        elif self.method == ConformerMethod.UFF:
            geometries = self._generate_uff_conformers(mol)
        elif self.method == ConformerMethod.MIXED:
            geometries = self._generate_mixed_conformers(mol)
        else:
            raise QMLModelError(f"Unsupported conformer method: {self.method}")

        if not geometries:
            raise QMLModelError("Failed to generate any conformers")

        # Post-process ensemble
        geometries = self._post_process_conformers(geometries, mol)

        # Calculate Boltzmann weights
        boltzmann_weights = self._calculate_boltzmann_weights(geometries)

        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(geometries)

        # Energy spread
        energies = [g.energy for g in geometries if g.energy is not None]
        energy_spread = max(energies) - min(energies) if energies else 0.0

        generation_time = time.time() - start_time

        ensemble = ConformerEnsemble(
            geometries=geometries,
            reference_geometry=reference_geometry,
            ensemble_energy_spread=energy_spread,
            boltzmann_weights=boltzmann_weights,
            diversity_metrics=diversity_metrics,
            metadata={
                "method": self.method.value,
                "quality": self.quality.value,
                "generation_time": generation_time,
                "original_molecule": Chem.MolToSmiles(Chem.RemoveHs(mol)),
            },
        )

        return ensemble

    def _generate_etkdg_conformers(self, mol: Chem.Mol) -> List[GeometryInput]:
        """Generate conformers using ETKDG method."""
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=self.max_conformers, params=self.etkdg_params
        )

        if not conformer_ids:
            # Fallback to single conformer
            conf_id = AllChem.EmbedMolecule(mol, params=self.etkdg_params)
            if conf_id == -1:
                raise QMLModelError("Failed to generate ETKDG conformers")
            conformer_ids = [conf_id]

        # Optimize with MMFF94
        geometries = []
        for conf_id in conformer_ids:
            try:
                # MMFF optimization
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(
                        mol, mmff_props, confId=conf_id
                    )
                    if ff is not None:
                        convergence = ff.Minimize()
                        energy = ff.CalcEnergy()
                    else:
                        # Fallback to UFF
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        convergence = ff.Minimize() if ff else 1
                        energy = ff.CalcEnergy() if ff else float("inf")
                else:
                    # Use UFF as fallback
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    convergence = ff.Minimize() if ff else 1
                    energy = ff.CalcEnergy() if ff else float("inf")

                geometry = self._extract_geometry(mol, conf_id, energy)
                geometry.metadata = GeometryMetadata(
                    method=ConformerMethod.ETKDG,
                    quality=self.quality,
                    final_energy=energy,
                    convergence_status="converged" if convergence == 0 else "failed",
                )

                geometries.append(geometry)

            except Exception as e:
                logging.warning(f"Failed to optimize ETKDG conformer {conf_id}: {e}")
                continue

        return geometries

    def _generate_mmff_conformers(self, mol: Chem.Mol) -> List[GeometryInput]:
        """Generate conformers with MMFF optimization."""
        # Start with ETKDG embedding
        initial_geometries = self._generate_etkdg_conformers(mol)

        # Re-optimize with MMFF
        optimized_geometries = []

        for geometry in initial_geometries:
            try:
                # Create new conformer with geometry
                conf_id = mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
                conf = mol.GetConformer(conf_id)

                for i, coord in enumerate(geometry.coordinates):
                    conf.SetAtomPosition(i, coord)

                # MMFF optimization with more iterations
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(
                        mol, mmff_props, confId=conf_id
                    )

                    if ff is not None:
                        ff.Initialize()
                        convergence = ff.Minimize(maxIts=1000)
                        energy = ff.CalcEnergy()

                        optimized_geometry = self._extract_geometry(
                            mol, conf_id, energy
                        )
                        optimized_geometry.metadata = GeometryMetadata(
                            method=ConformerMethod.MMFF,
                            quality=self.quality,
                            final_energy=energy,
                            convergence_status="converged"
                            if convergence == 0
                            else "partial",
                        )

                        optimized_geometries.append(optimized_geometry)

                # Remove temporary conformer
                mol.RemoveConformer(conf_id)

            except Exception as e:
                logging.warning(f"MMFF optimization failed: {e}")
                continue

        return optimized_geometries

    def _generate_uff_conformers(self, mol: Chem.Mol) -> List[GeometryInput]:
        """Generate conformers with UFF optimization."""
        # Similar to MMFF but using UFF
        initial_geometries = self._generate_etkdg_conformers(mol)

        optimized_geometries = []

        for geometry in initial_geometries:
            try:
                conf_id = mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
                conf = mol.GetConformer(conf_id)

                for i, coord in enumerate(geometry.coordinates):
                    conf.SetAtomPosition(i, coord)

                # UFF optimization
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)

                if ff is not None:
                    convergence = ff.Minimize(maxIts=1000)
                    energy = ff.CalcEnergy()

                    optimized_geometry = self._extract_geometry(mol, conf_id, energy)
                    optimized_geometry.metadata = GeometryMetadata(
                        method=ConformerMethod.UFF,
                        quality=self.quality,
                        final_energy=energy,
                        convergence_status="converged"
                        if convergence == 0
                        else "partial",
                    )

                    optimized_geometries.append(optimized_geometry)

                mol.RemoveConformer(conf_id)

            except Exception as e:
                logging.warning(f"UFF optimization failed: {e}")
                continue

        return optimized_geometries

    def _generate_mixed_conformers(self, mol: Chem.Mol) -> List[GeometryInput]:
        """Generate conformers using mixed methods."""
        all_geometries = []

        # Generate with different methods
        methods = [ConformerMethod.ETKDG, ConformerMethod.MMFF]

        original_max = self.max_conformers
        self.max_conformers = self.max_conformers // len(methods)

        for method in methods:
            original_method = self.method
            self.method = method

            try:
                if method == ConformerMethod.ETKDG:
                    geometries = self._generate_etkdg_conformers(mol)
                elif method == ConformerMethod.MMFF:
                    geometries = self._generate_mmff_conformers(mol)

                all_geometries.extend(geometries)

            except Exception as e:
                logging.warning(f"Mixed method {method.value} failed: {e}")

            self.method = original_method

        self.max_conformers = original_max

        # Remove duplicates and select best
        unique_geometries = self._remove_similar_conformers(all_geometries)

        return unique_geometries[: self.max_conformers]

    def _extract_geometry(
        self, mol: Chem.Mol, conf_id: int, energy: float
    ) -> GeometryInput:
        """Extract geometry from RDKit conformer."""
        conf = mol.GetConformer(conf_id)

        coordinates = np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ]
        )

        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        return GeometryInput(
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            charge=Chem.rdmolops.GetFormalCharge(mol),
            conformer_id=conf_id,
            energy=energy,
        )

    def _post_process_conformers(
        self, geometries: List[GeometryInput], mol: Chem.Mol
    ) -> List[GeometryInput]:
        """Post-process conformers with filtering and sorting."""
        if not geometries:
            return geometries

        # Filter by energy threshold
        if len(geometries) > 1:
            min_energy = min(g.energy for g in geometries if g.energy is not None)
            filtered_geometries = []

            for geom in geometries:
                if geom.energy is not None:
                    energy_diff = (
                        geom.energy - min_energy
                    ) * 627.509  # Convert to kcal/mol
                    if energy_diff <= self.energy_threshold:
                        filtered_geometries.append(geom)
                else:
                    filtered_geometries.append(geom)

            geometries = filtered_geometries

        # Remove similar conformers
        geometries = self._remove_similar_conformers(geometries)

        # Sort by energy
        geometries.sort(
            key=lambda x: x.energy if x.energy is not None else float("inf")
        )

        return geometries[: self.max_conformers]

    def _remove_similar_conformers(
        self, geometries: List[GeometryInput]
    ) -> List[GeometryInput]:
        """Remove similar conformers based on RMSD."""
        if len(geometries) <= 1:
            return geometries

        unique_geometries = [geometries[0]]  # Keep first (lowest energy)

        for geom in geometries[1:]:
            is_unique = True

            for unique_geom in unique_geometries:
                rmsd = self._calculate_rmsd(geom.coordinates, unique_geom.coordinates)
                if rmsd < self.rms_threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_geometries.append(geom)

        return unique_geometries

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets."""
        if coords1.shape != coords2.shape:
            return float("inf")

        # Simple RMSD without alignment
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    def _calculate_boltzmann_weights(
        self, geometries: List[GeometryInput]
    ) -> np.ndarray:
        """Calculate Boltzmann weights for conformers."""
        energies = [g.energy for g in geometries if g.energy is not None]

        if not energies or len(energies) != len(geometries):
            # Equal weights if no energies
            return np.ones(len(geometries)) / len(geometries)

        # Boltzmann weighting at 298K
        kT = 0.59616  # kcal/mol
        min_energy = min(energies)

        weights = np.exp(-(np.array(energies) - min_energy) / kT)
        weights /= weights.sum()

        return weights

    def _calculate_diversity_metrics(
        self, geometries: List[GeometryInput]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for conformer ensemble."""
        if len(geometries) <= 1:
            return {"diversity": 0.0, "avg_rmsd": 0.0, "max_rmsd": 0.0}

        # Calculate all pairwise RMSDs
        n = len(geometries)
        rmsds = []

        for i in range(n):
            for j in range(i + 1, n):
                rmsd = self._calculate_rmsd(
                    geometries[i].coordinates, geometries[j].coordinates
                )
                rmsds.append(rmsd)

        if not rmsds:
            return {"diversity": 0.0, "avg_rmsd": 0.0, "max_rmsd": 0.0}

        avg_rmsd = np.mean(rmsds)
        max_rmsd = np.max(rmsds)

        # Diversity score (normalized average RMSD)
        diversity = min(avg_rmsd / 5.0, 1.0)  # Normalize to [0,1]

        return {
            "diversity": diversity,
            "avg_rmsd": avg_rmsd,
            "max_rmsd": max_rmsd,
            "rmsd_std": np.std(rmsds),
        }


class ConformerEnsembleProcessor:
    """Process and analyze conformer ensembles for QML applications."""

    def __init__(self, alignment_method: AlignmentMethod = AlignmentMethod.RMSD):
        """
        Initialize conformer ensemble processor.

        Args:
            alignment_method: Method for conformer alignment
        """
        self.alignment_method = alignment_method

    def align_ensemble(
        self, ensemble: ConformerEnsemble, reference_idx: int = 0
    ) -> ConformerEnsemble:
        """
        Align all conformers in ensemble to a reference.

        Args:
            ensemble: Input conformer ensemble
            reference_idx: Index of reference conformer

        Returns:
            Aligned conformer ensemble
        """
        if len(ensemble) <= 1:
            return ensemble

        reference_geom = ensemble[reference_idx]
        aligned_geometries = [reference_geom]  # Reference stays the same

        for i, geom in enumerate(ensemble.geometries):
            if i == reference_idx:
                continue

            try:
                aligned_coords = self._align_coordinates(
                    geom.coordinates, reference_geom.coordinates, geom.atomic_numbers
                )

                aligned_geom = GeometryInput(
                    coordinates=aligned_coords,
                    atomic_numbers=geom.atomic_numbers.copy(),
                    charge=geom.charge,
                    multiplicity=geom.multiplicity,
                    conformer_id=geom.conformer_id,
                    energy=geom.energy,
                )

                # Calculate RMSD to reference
                rmsd = self._calculate_rmsd(aligned_coords, reference_geom.coordinates)
                if hasattr(aligned_geom, "metadata"):
                    aligned_geom.metadata.rmsd_to_reference = rmsd

                aligned_geometries.append(aligned_geom)

            except Exception as e:
                logging.warning(f"Failed to align conformer {i}: {e}")
                aligned_geometries.append(geom)  # Keep original if alignment fails

        # Create new ensemble with aligned geometries
        aligned_ensemble = ConformerEnsemble(
            geometries=aligned_geometries,
            reference_geometry=reference_geom,
            ensemble_energy_spread=ensemble.ensemble_energy_spread,
            boltzmann_weights=ensemble.boltzmann_weights,
            diversity_metrics=ensemble.diversity_metrics,
            metadata=ensemble.metadata.copy(),
        )

        aligned_ensemble.metadata["aligned"] = True
        aligned_ensemble.metadata["alignment_method"] = self.alignment_method.value

        return aligned_ensemble

    def _align_coordinates(
        self, coords: np.ndarray, ref_coords: np.ndarray, atomic_numbers: np.ndarray
    ) -> np.ndarray:
        """Align coordinates to reference using Kabsch algorithm."""
        if coords.shape != ref_coords.shape:
            raise ValueError("Coordinate arrays must have same shape")

        # Center coordinates
        coords_centered = coords - coords.mean(axis=0)
        ref_centered = ref_coords - ref_coords.mean(axis=0)

        # Kabsch algorithm
        H = coords_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation and translation
        aligned_coords = coords_centered @ R.T + ref_coords.mean(axis=0)

        return aligned_coords

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between aligned coordinates."""
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    def extract_representative_conformers(
        self,
        ensemble: ConformerEnsemble,
        num_representatives: int = 5,
        method: str = "energy_diversity",
    ) -> List[GeometryInput]:
        """
        Extract representative conformers from ensemble.

        Args:
            ensemble: Input ensemble
            num_representatives: Number of representatives to extract
            method: Selection method ("energy_diversity", "clustering", "boltzmann")

        Returns:
            List of representative conformers
        """
        if len(ensemble) <= num_representatives:
            return ensemble.geometries

        if method == "energy_diversity":
            return self._select_energy_diversity(ensemble, num_representatives)
        elif method == "clustering":
            return self._select_clustering(ensemble, num_representatives)
        elif method == "boltzmann":
            return self._select_boltzmann(ensemble, num_representatives)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _select_energy_diversity(
        self, ensemble: ConformerEnsemble, num_representatives: int
    ) -> List[GeometryInput]:
        """Select representatives based on energy and diversity."""
        representatives = []
        candidates = ensemble.geometries.copy()

        # Always include lowest energy conformer
        candidates.sort(
            key=lambda x: x.energy if x.energy is not None else float("inf")
        )
        representatives.append(candidates[0])
        candidates.remove(candidates[0])

        # Select remaining based on diversity
        while len(representatives) < num_representatives and candidates:
            best_candidate = None
            best_diversity = -1

            for candidate in candidates:
                # Calculate minimum RMSD to existing representatives
                min_rmsd = min(
                    self._calculate_rmsd(candidate.coordinates, rep.coordinates)
                    for rep in representatives
                )

                # Diversity score (higher is better)
                diversity_score = min_rmsd

                if diversity_score > best_diversity:
                    best_diversity = diversity_score
                    best_candidate = candidate

            if best_candidate:
                representatives.append(best_candidate)
                candidates.remove(best_candidate)

        return representatives

    def _select_clustering(
        self, ensemble: ConformerEnsemble, num_representatives: int
    ) -> List[GeometryInput]:
        """Select representatives using clustering."""
        try:
            from sklearn.cluster import KMeans

            # Create feature matrix from coordinates
            features = np.array(
                [geom.coordinates.flatten() for geom in ensemble.geometries]
            )

            # Perform clustering
            kmeans = KMeans(n_clusters=num_representatives, random_state=42)
            labels = kmeans.fit_predict(features)

            # Select representative from each cluster (lowest energy)
            representatives = []
            for cluster_id in range(num_representatives):
                cluster_members = [
                    geom
                    for i, geom in enumerate(ensemble.geometries)
                    if labels[i] == cluster_id
                ]

                if cluster_members:
                    # Select lowest energy from cluster
                    cluster_rep = min(
                        cluster_members,
                        key=lambda x: x.energy
                        if x.energy is not None
                        else float("inf"),
                    )
                    representatives.append(cluster_rep)

            return representatives

        except ImportError:
            logging.warning("sklearn not available, falling back to energy_diversity")
            return self._select_energy_diversity(ensemble, num_representatives)

    def _select_boltzmann(
        self, ensemble: ConformerEnsemble, num_representatives: int
    ) -> List[GeometryInput]:
        """Select representatives based on Boltzmann weights."""
        if ensemble.boltzmann_weights is None:
            return self._select_energy_diversity(ensemble, num_representatives)

        # Sort by Boltzmann weight (descending)
        sorted_indices = np.argsort(ensemble.boltzmann_weights)[::-1]
        representatives = [
            ensemble.geometries[i] for i in sorted_indices[:num_representatives]
        ]

        return representatives

    def calculate_ensemble_properties(
        self, ensemble: ConformerEnsemble
    ) -> Dict[str, Any]:
        """Calculate comprehensive properties of conformer ensemble."""
        if not ensemble.geometries:
            return {}

        properties = {}

        # Basic statistics
        properties["num_conformers"] = len(ensemble)
        properties["energy_spread"] = ensemble.ensemble_energy_spread

        # Energy statistics
        energies = [g.energy for g in ensemble.geometries if g.energy is not None]
        if energies:
            properties["energy_stats"] = {
                "min": min(energies),
                "max": max(energies),
                "mean": np.mean(energies),
                "std": np.std(energies),
            }

        # Diversity metrics
        properties["diversity_metrics"] = ensemble.diversity_metrics

        # Boltzmann statistics
        if ensemble.boltzmann_weights is not None:
            properties["boltzmann_stats"] = {
                "entropy": -np.sum(
                    ensemble.boltzmann_weights
                    * np.log(ensemble.boltzmann_weights + 1e-10)
                ),
                "effective_conformers": 1.0 / np.sum(ensemble.boltzmann_weights**2),
                "weight_concentration": np.max(ensemble.boltzmann_weights),
            }

        # Geometric properties
        coords_array = np.array([g.coordinates for g in ensemble.geometries])
        properties["geometric_stats"] = {
            "coordinate_std": np.std(coords_array, axis=0).mean(),
            "center_of_mass_variation": self._calculate_com_variation(
                coords_array, ensemble.geometries[0].atomic_numbers
            ),
            "radius_of_gyration_variation": self._calculate_rog_variation(
                coords_array, ensemble.geometries[0].atomic_numbers
            ),
        }

        return properties

    def _calculate_com_variation(
        self, coords_array: np.ndarray, atomic_numbers: np.ndarray
    ) -> float:
        """Calculate center of mass variation across conformers."""
        masses = np.array([self._get_atomic_mass(num) for num in atomic_numbers])

        coms = []
        for coords in coords_array:
            com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
            coms.append(com)

        coms = np.array(coms)
        return np.std(coms, axis=0).mean()

    def _calculate_rog_variation(
        self, coords_array: np.ndarray, atomic_numbers: np.ndarray
    ) -> float:
        """Calculate radius of gyration variation across conformers."""
        masses = np.array([self._get_atomic_mass(num) for num in atomic_numbers])

        rogs = []
        for coords in coords_array:
            com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
            rog_sq = np.sum(masses * np.sum((coords - com) ** 2, axis=1)) / np.sum(
                masses
            )
            rogs.append(np.sqrt(rog_sq))

        return np.std(rogs)

    def _get_atomic_mass(self, atomic_number: int) -> float:
        """Get atomic mass for given atomic number."""
        # Simplified atomic masses
        masses = {
            1: 1.008,
            6: 12.011,
            7: 14.007,
            8: 15.999,
            9: 18.998,
            15: 30.974,
            16: 32.065,
            17: 35.453,
            35: 79.904,
            53: 126.904,
        }
        return masses.get(atomic_number, 12.011)  # Default to carbon
