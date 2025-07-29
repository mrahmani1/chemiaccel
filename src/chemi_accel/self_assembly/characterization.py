"""
Self-Assembly Characterization Module for ChemiAccel.

This module provides comprehensive tools to analyze simulation trajectories and characterize
emergent self-assembled structures including micelles, vesicles, liquid crystals, and hydrogel networks.
It implements GPU-accelerated algorithms for calculating self-assembly metrics, morphology analysis,
and phase behavior characterization.
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Optional GPU acceleration dependencies
try:
    import cupy as cp
    import cupyx.scipy.spatial as cupy_spatial
    import cupyx.scipy.spatial.distance as cupy_distance

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_spatial = None
    cupy_distance = None
    CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

# Core scientific dependencies
try:
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.ndimage import center_of_mass, label
    from scipy.optimize import curve_fit, minimize
    from scipy.spatial import ConvexHull, Voronoi
    from scipy.spatial.distance import cdist, pdist, squareform

    SCIPY_AVAILABLE = True
except ImportError:
    pdist = None
    squareform = None
    cdist = None
    linkage = None
    fcluster = None
    dendrogram = None
    ConvexHull = None
    Voronoi = None
    label = None
    center_of_mass = None
    minimize = None
    curve_fit = None
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    KMeans = None
    AgglomerativeClustering = None
    PCA = None
    silhouette_score = None
    SKLEARN_AVAILABLE = False

# Visualization dependencies
try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    patches = None
    Axes3D = None
    MATPLOTLIB_AVAILABLE = False

# Import simulation classes
from .coarse_grained_simulation import Particle, ParticleType, TrajectoryFrame

# Setup logging
logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of self-assembled structures."""

    MICELLE = "micelle"
    REVERSE_MICELLE = "reverse_micelle"
    VESICLE = "vesicle"
    BILAYER = "bilayer"
    LIQUID_CRYSTAL = "liquid_crystal"
    HYDROGEL = "hydrogel"
    CLUSTER = "cluster"
    FIBER = "fiber"
    SHEET = "sheet"
    UNKNOWN = "unknown"


class MorphologyType(Enum):
    """Morphology classifications for structures."""

    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    DISK_LIKE = "disk_like"
    WORM_LIKE = "worm_like"
    BRANCHED = "branched"
    LAMELLAR = "lamellar"
    NETWORK = "network"
    IRREGULAR = "irregular"


class PhaseType(Enum):
    """Types of phases in self-assembled systems."""

    ISOTROPIC = "isotropic"
    NEMATIC = "nematic"
    SMECTIC_A = "smectic_a"
    SMECTIC_C = "smectic_c"
    COLUMNAR = "columnar"
    CUBIC = "cubic"
    HEXAGONAL = "hexagonal"
    LAMELLAR = "lamellar"
    GEL = "gel"
    CRYSTAL = "crystal"


class ClusteringMethod(Enum):
    """Methods for particle clustering."""

    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DENSITY_BASED = "density_based"
    GRID_BASED = "grid_based"


@dataclass
class StructureMetrics:
    """Metrics characterizing a self-assembled structure."""

    structure_id: int
    structure_type: StructureType
    morphology: MorphologyType
    aggregation_number: int
    center_of_mass: npt.NDArray[np.float64]
    radius_of_gyration: float
    asphericity: float
    eccentricity: float
    surface_area: float
    volume: float
    compactness: float
    hydrophobic_core_radius: float
    hydrophilic_shell_thickness: float
    order_parameter: float

    def __post_init__(self):
        if self.aggregation_number <= 0:
            raise ValueError("Aggregation number must be positive")
        if len(self.center_of_mass) != 3:
            raise ValueError("Center of mass must be 3D vector")


@dataclass
class MicelleCharacteristics:
    """Specific characteristics for micelle structures."""

    core_radius: float
    shell_thickness: float
    aggregation_number: int
    critical_packing_parameter: float
    core_water_content: float
    interface_width: float
    shape_factor: float


@dataclass
class VesicleCharacteristics:
    """Specific characteristics for vesicle structures."""

    inner_radius: float
    outer_radius: float
    membrane_thickness: float
    enclosed_volume: float
    surface_area_to_volume_ratio: float
    membrane_curvature: float
    asymmetry_factor: float


@dataclass
class LiquidCrystalCharacteristics:
    """Specific characteristics for liquid crystal phases."""

    nematic_order_parameter: float
    director_vector: npt.NDArray[np.float64]
    correlation_length: float
    elastic_constants: Dict[str, float]
    phase_type: PhaseType
    layer_spacing: Optional[float] = None
    tilt_angle: Optional[float] = None


@dataclass
class HydrogelCharacteristics:
    """Specific characteristics for hydrogel networks."""

    crosslink_density: float
    mesh_size: float
    water_content: float
    elasticity_modulus: float
    network_connectivity: float
    pore_size_distribution: npt.NDArray[np.float64]


@dataclass
class PhaseAnalysisResult:
    """Result of phase behavior analysis."""

    phase_type: PhaseType
    order_parameters: Dict[str, float]
    correlation_functions: Dict[str, npt.NDArray[np.float64]]
    phase_transition_temperature: Optional[float] = None
    critical_concentration: Optional[float] = None


@dataclass
class CharacterizationResult:
    """Complete characterization result for a trajectory frame."""

    frame_id: int
    time: float
    structures: List[StructureMetrics]
    micelles: List[MicelleCharacteristics]
    vesicles: List[VesicleCharacteristics]
    liquid_crystals: List[LiquidCrystalCharacteristics]
    hydrogels: List[HydrogelCharacteristics]
    phase_analysis: PhaseAnalysisResult
    global_metrics: Dict[str, float]


class CharacterizationError(Exception):
    """Base exception for characterization errors."""

    pass


class ClusteringError(CharacterizationError):
    """Exception raised during clustering operations."""

    pass


class MorphologyAnalysisError(CharacterizationError):
    """Exception raised during morphology analysis."""

    pass


class StructureClassifier:
    """GPU-accelerated classifier for self-assembled structures."""

    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id

        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            logger.info(f"Using GPU device {device_id} for structure classification")
        else:
            logger.info("Using CPU for structure classification")

    def classify_structure(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        cluster_indices: npt.NDArray[np.int32],
    ) -> StructureType:
        """Classify the type of self-assembled structure."""

        # Analyze geometric properties
        metrics = self._calculate_geometric_metrics(positions, cluster_indices)

        # Apply classification rules
        return self._apply_classification_rules(metrics, particle_types, positions)

    def _calculate_geometric_metrics(
        self, positions: npt.NDArray[np.float64], cluster_indices: npt.NDArray[np.int32]
    ) -> Dict[str, float]:
        """Calculate geometric metrics for structure classification."""
        if self.use_gpu:
            return self._calculate_metrics_gpu(positions, cluster_indices)
        else:
            return self._calculate_metrics_cpu(positions, cluster_indices)

    def _calculate_metrics_gpu(
        self, positions: npt.NDArray[np.float64], cluster_indices: npt.NDArray[np.int32]
    ) -> Dict[str, float]:
        """GPU implementation of geometric metrics calculation."""
        positions_gpu = cp.asarray(positions, dtype=cp.float64)
        indices_gpu = cp.asarray(cluster_indices, dtype=cp.int32)

        # Center of mass
        com = cp.mean(positions_gpu, axis=0)

        # Radius of gyration
        centered_pos = positions_gpu - com
        rg_squared = cp.mean(cp.sum(centered_pos**2, axis=1))
        radius_of_gyration = cp.sqrt(rg_squared)

        # Asphericity and shape parameters
        gyration_tensor = cp.zeros((3, 3), dtype=cp.float64)
        for i in range(3):
            for j in range(3):
                gyration_tensor[i, j] = cp.mean(centered_pos[:, i] * centered_pos[:, j])

        eigenvals = cp.linalg.eigvals(gyration_tensor)
        eigenvals = cp.sort(eigenvals)[::-1]  # Sort in descending order

        # Convert to numpy for return
        asphericity = float(
            cp.asnumpy(eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2]))
        )
        sphericity = float(
            cp.asnumpy(eigenvals[2] / eigenvals[0]) if eigenvals[0] > 0 else 0
        )

        return {
            "radius_of_gyration": float(cp.asnumpy(radius_of_gyration)),
            "asphericity": asphericity,
            "sphericity": sphericity,
            "eigenvals": [float(cp.asnumpy(ev)) for ev in eigenvals],
        }

    def _calculate_metrics_cpu(
        self, positions: npt.NDArray[np.float64], cluster_indices: npt.NDArray[np.int32]
    ) -> Dict[str, float]:
        """CPU implementation of geometric metrics calculation."""
        # Center of mass
        com = np.mean(positions, axis=0)

        # Radius of gyration
        centered_pos = positions - com
        rg_squared = np.mean(np.sum(centered_pos**2, axis=1))
        radius_of_gyration = np.sqrt(rg_squared)

        # Asphericity and shape parameters
        gyration_tensor = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                gyration_tensor[i, j] = np.mean(centered_pos[:, i] * centered_pos[:, j])

        eigenvals, _ = np.linalg.eigh(gyration_tensor)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order

        asphericity = eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])
        sphericity = eigenvals[2] / eigenvals[0] if eigenvals[0] > 0 else 0

        return {
            "radius_of_gyration": radius_of_gyration,
            "asphericity": asphericity,
            "sphericity": sphericity,
            "eigenvals": eigenvals.tolist(),
        }

    def _apply_classification_rules(
        self,
        metrics: Dict[str, float],
        particle_types: List[str],
        positions: npt.NDArray[np.float64],
    ) -> StructureType:
        """Apply heuristic rules to classify structure type."""

        sphericity = metrics["sphericity"]
        asphericity = metrics["asphericity"]
        n_particles = len(positions)

        # Check for amphiphilic content
        has_hydrophobic = any(
            "TAIL" in ptype or "hydrophobic" in ptype.lower()
            for ptype in particle_types
        )
        has_hydrophilic = any(
            "HEAD" in ptype or "hydrophilic" in ptype.lower() or "WATER" in ptype
            for ptype in particle_types
        )

        # Classification logic
        if sphericity > 0.8 and asphericity < 0.1:
            if has_hydrophobic and has_hydrophilic:
                if n_particles < 50:
                    return StructureType.MICELLE
                else:
                    # Check for hollow structure (vesicle)
                    if self._is_hollow_structure(positions):
                        return StructureType.VESICLE
                    else:
                        return StructureType.MICELLE
            else:
                return StructureType.CLUSTER

        elif sphericity < 0.3 and asphericity > 0.5:
            if self._has_extended_structure(positions):
                return StructureType.FIBER
            else:
                return StructureType.SHEET

        elif 0.3 <= sphericity <= 0.8:
            if has_hydrophobic and has_hydrophilic:
                return StructureType.LIQUID_CRYSTAL
            else:
                return StructureType.CLUSTER

        else:
            return StructureType.UNKNOWN

    def _is_hollow_structure(self, positions: npt.NDArray[np.float64]) -> bool:
        """Check if structure is hollow (vesicle-like)."""
        com = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - com, axis=1)

        # Check for bimodal distribution of radial distances
        hist, bin_edges = np.histogram(distances, bins=20)
        peaks = []
        for i in range(1, len(hist) - 1):
            if (
                hist[i] > hist[i - 1]
                and hist[i] > hist[i + 1]
                and hist[i] > 0.1 * np.max(hist)
            ):
                peaks.append(i)

        # If there are two distinct peaks, it might be hollow
        return len(peaks) >= 2

    def _has_extended_structure(self, positions: npt.NDArray[np.float64]) -> bool:
        """Check if structure is extended (fiber-like vs sheet-like)."""
        # Calculate principal axes
        com = np.mean(positions, axis=0)
        centered_pos = positions - com

        covariance_matrix = np.cov(centered_pos.T)
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.sort(eigenvals)[::-1]

        # Check aspect ratios
        aspect_ratio_1 = eigenvals[0] / eigenvals[1] if eigenvals[1] > 0 else np.inf
        aspect_ratio_2 = eigenvals[1] / eigenvals[2] if eigenvals[2] > 0 else np.inf

        # Fiber-like: one dimension much larger
        if aspect_ratio_1 > 5 and aspect_ratio_2 > 2:
            return True

        return False


class ClusterAnalyzer:
    """GPU-accelerated analyzer for particle clustering and aggregation."""

    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id

        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            logger.info(f"Using GPU device {device_id} for cluster analysis")
        else:
            logger.info("Using CPU for cluster analysis")

    def cluster_particles(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        method: ClusteringMethod = ClusteringMethod.DBSCAN,
        cutoff_distance: float = 2.0,
        min_cluster_size: int = 5,
    ) -> npt.NDArray[np.int32]:
        """Cluster particles based on spatial proximity and type."""

        if method == ClusteringMethod.DBSCAN:
            return self._cluster_dbscan(positions, cutoff_distance, min_cluster_size)
        elif method == ClusteringMethod.DENSITY_BASED:
            return self._cluster_density_based(
                positions, particle_types, cutoff_distance
            )
        elif method == ClusteringMethod.HIERARCHICAL:
            return self._cluster_hierarchical(positions, min_cluster_size)
        else:
            raise ClusteringError(f"Clustering method {method} not implemented")

    def _cluster_dbscan(
        self, positions: npt.NDArray[np.float64], eps: float, min_samples: int
    ) -> npt.NDArray[np.int32]:
        """DBSCAN clustering implementation."""
        if not SKLEARN_AVAILABLE:
            raise ClusteringError("scikit-learn not available for DBSCAN clustering")

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(positions)

        return cluster_labels.astype(np.int32)

    def _cluster_density_based(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        cutoff: float,
    ) -> npt.NDArray[np.int32]:
        """Custom density-based clustering considering particle types."""
        n_particles = len(positions)

        if self.use_gpu:
            return self._density_cluster_gpu(positions, particle_types, cutoff)
        else:
            return self._density_cluster_cpu(positions, particle_types, cutoff)

    def _density_cluster_gpu(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        cutoff: float,
    ) -> npt.NDArray[np.int32]:
        """GPU implementation of density-based clustering."""
        n_particles = len(positions)
        positions_gpu = cp.asarray(positions, dtype=cp.float64)

        # Calculate distance matrix
        distance_matrix = cupy_distance.cdist(positions_gpu, positions_gpu)

        # Create adjacency matrix
        adjacency = distance_matrix < cutoff

        # Find connected components
        cluster_labels = (
            cp.zeros(n_particles, dtype=cp.int32) - 1
        )  # Initialize as noise
        current_cluster = 0
        visited = cp.zeros(n_particles, dtype=cp.bool_)

        for i in range(n_particles):
            if not visited[i]:
                # BFS to find connected component
                queue = [i]
                cluster_particles = []

                while queue:
                    current = queue.pop(0)
                    if visited[current]:
                        continue

                    visited[current] = True
                    cluster_particles.append(current)

                    # Find neighbors
                    neighbors = cp.where(adjacency[current])[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            queue.append(int(neighbor))

                # Assign cluster label if cluster is large enough
                if len(cluster_particles) >= 5:  # Minimum cluster size
                    for particle_idx in cluster_particles:
                        cluster_labels[particle_idx] = current_cluster
                    current_cluster += 1

        return cp.asnumpy(cluster_labels)

    def _density_cluster_cpu(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        cutoff: float,
    ) -> npt.NDArray[np.int32]:
        """CPU implementation of density-based clustering."""
        n_particles = len(positions)

        # Calculate distance matrix
        distance_matrix = cdist(positions, positions)

        # Create adjacency matrix
        adjacency = distance_matrix < cutoff

        # Find connected components
        cluster_labels = np.full(n_particles, -1, dtype=np.int32)  # Initialize as noise
        current_cluster = 0
        visited = np.zeros(n_particles, dtype=bool)

        for i in range(n_particles):
            if not visited[i]:
                # BFS to find connected component
                queue = [i]
                cluster_particles = []

                while queue:
                    current = queue.pop(0)
                    if visited[current]:
                        continue

                    visited[current] = True
                    cluster_particles.append(current)

                    # Find neighbors
                    neighbors = np.where(adjacency[current])[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            queue.append(neighbor)

                # Assign cluster label if cluster is large enough
                if len(cluster_particles) >= 5:  # Minimum cluster size
                    for particle_idx in cluster_particles:
                        cluster_labels[particle_idx] = current_cluster
                    current_cluster += 1

        return cluster_labels

    def _cluster_hierarchical(
        self, positions: npt.NDArray[np.float64], min_cluster_size: int
    ) -> npt.NDArray[np.int32]:
        """Hierarchical clustering implementation."""
        if not SCIPY_AVAILABLE:
            raise ClusteringError("SciPy not available for hierarchical clustering")

        # Calculate condensed distance matrix
        distances = pdist(positions)

        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method="ward")

        # Form clusters
        max_clusters = len(positions) // min_cluster_size
        cluster_labels = fcluster(linkage_matrix, max_clusters, criterion="maxclust")

        return cluster_labels.astype(np.int32) - 1  # Convert to 0-based indexing

    def analyze_cluster_morphology(
        self, positions: npt.NDArray[np.float64], cluster_indices: npt.NDArray[np.int32]
    ) -> MorphologyType:
        """Analyze the morphology of a particle cluster."""

        # Calculate shape descriptors
        com = np.mean(positions, axis=0)
        centered_positions = positions - com

        # Covariance matrix and principal axes
        cov_matrix = np.cov(centered_positions.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order

        # Shape ratios
        if eigenvals[1] > 0 and eigenvals[2] > 0:
            ratio_12 = eigenvals[0] / eigenvals[1]
            ratio_23 = eigenvals[1] / eigenvals[2]
            sphericity = eigenvals[2] / eigenvals[0]
        else:
            ratio_12 = ratio_23 = sphericity = 0

        # Classify morphology
        if sphericity > 0.8:
            return MorphologyType.SPHERICAL
        elif ratio_12 > 3 and ratio_23 < 2:
            return MorphologyType.CYLINDRICAL
        elif ratio_12 < 2 and ratio_23 > 3:
            return MorphologyType.DISK_LIKE
        elif ratio_12 > 2 and ratio_23 > 2:
            return MorphologyType.WORM_LIKE
        else:
            return MorphologyType.IRREGULAR


class MicelleAnalyzer:
    """Specialized analyzer for micelle structures."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE

    def analyze_micelle(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        cluster_indices: npt.NDArray[np.int32],
    ) -> MicelleCharacteristics:
        """Analyze characteristics of a micelle structure."""

        # Separate head and tail particles
        head_indices = [i for i, ptype in enumerate(particle_types) if "HEAD" in ptype]
        tail_indices = [i for i, ptype in enumerate(particle_types) if "TAIL" in ptype]

        if not head_indices or not tail_indices:
            raise MorphologyAnalysisError(
                "Micelle analysis requires both head and tail particles"
            )

        head_positions = positions[head_indices]
        tail_positions = positions[tail_indices]

        # Center of mass
        com = np.mean(positions, axis=0)

        # Core radius (average distance of tail particles from COM)
        tail_distances = np.linalg.norm(tail_positions - com, axis=1)
        core_radius = np.mean(tail_distances)

        # Shell thickness (difference between head and tail radial positions)
        head_distances = np.linalg.norm(head_positions - com, axis=1)
        shell_thickness = np.mean(head_distances) - core_radius

        # Aggregation number
        aggregation_number = len(positions)

        # Critical packing parameter (simplified)
        volume_per_molecule = (4 / 3) * np.pi * core_radius**3 / len(tail_indices)
        area_per_head = (
            4 * np.pi * (core_radius + shell_thickness) ** 2 / len(head_indices)
        )
        chain_length = self._estimate_chain_length(tail_positions, com)

        cpp = (
            volume_per_molecule / (area_per_head * chain_length)
            if area_per_head > 0 and chain_length > 0
            else 0
        )

        # Interface width
        interface_width = np.std(head_distances)

        # Shape factor (sphericity)
        shape_factor = self._calculate_shape_factor(positions, com)

        return MicelleCharacteristics(
            core_radius=core_radius,
            shell_thickness=shell_thickness,
            aggregation_number=aggregation_number,
            critical_packing_parameter=cpp,
            core_water_content=0.0,  # Would need water particle analysis
            interface_width=interface_width,
            shape_factor=shape_factor,
        )

    def _estimate_chain_length(
        self, tail_positions: npt.NDArray[np.float64], com: npt.NDArray[np.float64]
    ) -> float:
        """Estimate average chain length of tail particles."""
        # Simplified: use average distance from COM
        distances = np.linalg.norm(tail_positions - com, axis=1)
        return np.mean(distances)

    def _calculate_shape_factor(
        self, positions: npt.NDArray[np.float64], com: npt.NDArray[np.float64]
    ) -> float:
        """Calculate shape factor (measure of sphericity)."""
        # Calculate gyration tensor
        centered_pos = positions - com
        gyration_tensor = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                gyration_tensor[i, j] = np.mean(centered_pos[:, i] * centered_pos[:, j])

        eigenvals, _ = np.linalg.eigh(gyration_tensor)
        eigenvals = np.sort(eigenvals)

        # Shape factor: ratio of smallest to largest eigenvalue
        return eigenvals[0] / eigenvals[2] if eigenvals[2] > 0 else 0


class VesicleAnalyzer:
    """Specialized analyzer for vesicle structures."""

    def analyze_vesicle(
        self, positions: npt.NDArray[np.float64], particle_types: List[str]
    ) -> VesicleCharacteristics:
        """Analyze characteristics of a vesicle structure."""

        com = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - com, axis=1)

        # Find inner and outer radii by analyzing radial distribution
        hist, bin_edges = np.histogram(distances, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in radial distribution
        peaks = []
        for i in range(1, len(hist) - 1):
            if (
                hist[i] > hist[i - 1]
                and hist[i] > hist[i + 1]
                and hist[i] > 0.1 * np.max(hist)
            ):
                peaks.append(bin_centers[i])

        if len(peaks) >= 2:
            inner_radius = min(peaks)
            outer_radius = max(peaks)
        else:
            # Fallback: assume single shell
            inner_radius = np.min(distances)
            outer_radius = np.max(distances)

        membrane_thickness = outer_radius - inner_radius

        # Enclosed volume and surface area
        enclosed_volume = (4 / 3) * np.pi * inner_radius**3
        outer_surface_area = 4 * np.pi * outer_radius**2
        sa_vol_ratio = (
            outer_surface_area / enclosed_volume if enclosed_volume > 0 else 0
        )

        # Membrane curvature (simplified)
        membrane_curvature = (
            1 / ((inner_radius + outer_radius) / 2)
            if (inner_radius + outer_radius) > 0
            else 0
        )

        # Asymmetry factor
        asymmetry_factor = (
            abs(outer_radius - 2 * inner_radius) / outer_radius
            if outer_radius > 0
            else 0
        )

        return VesicleCharacteristics(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            membrane_thickness=membrane_thickness,
            enclosed_volume=enclosed_volume,
            surface_area_to_volume_ratio=sa_vol_ratio,
            membrane_curvature=membrane_curvature,
            asymmetry_factor=asymmetry_factor,
        )


class LiquidCrystalAnalyzer:
    """Specialized analyzer for liquid crystal phases."""

    def analyze_liquid_crystal(
        self,
        positions: npt.NDArray[np.float64],
        orientations: Optional[npt.NDArray[np.float64]] = None,
    ) -> LiquidCrystalCharacteristics:
        """Analyze liquid crystal phase characteristics."""

        # If orientations not provided, estimate from positions
        if orientations is None:
            orientations = self._estimate_orientations(positions)

        # Calculate nematic order parameter
        nematic_order = self._calculate_nematic_order_parameter(orientations)

        # Calculate director vector
        director = self._calculate_director_vector(orientations)

        # Correlation length
        correlation_length = self._calculate_correlation_length(positions, orientations)

        # Classify phase type
        phase_type = self._classify_lc_phase(nematic_order, positions)

        # Elastic constants (simplified)
        elastic_constants = {
            "K11": 1.0,  # Splay
            "K22": 1.0,  # Twist
            "K33": 1.0,  # Bend
        }

        return LiquidCrystalCharacteristics(
            nematic_order_parameter=nematic_order,
            director_vector=director,
            correlation_length=correlation_length,
            elastic_constants=elastic_constants,
            phase_type=phase_type,
        )

    def _estimate_orientations(
        self, positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Estimate molecular orientations from positions."""
        # Simplified: use nearest neighbor directions
        n_particles = len(positions)
        orientations = np.zeros((n_particles, 3))

        for i in range(n_particles):
            # Find nearest neighbors
            distances = np.linalg.norm(positions - positions[i], axis=1)
            nearest_idx = np.argsort(distances)[1:4]  # Skip self

            # Average direction to nearest neighbors
            direction = np.mean(positions[nearest_idx] - positions[i], axis=0)
            norm = np.linalg.norm(direction)
            if norm > 0:
                orientations[i] = direction / norm
            else:
                orientations[i] = np.array([1, 0, 0])  # Default orientation

        return orientations

    def _calculate_nematic_order_parameter(
        self, orientations: npt.NDArray[np.float64]
    ) -> float:
        """Calculate nematic order parameter S."""
        n_particles = len(orientations)

        # Calculate orientation tensor
        Q_tensor = np.zeros((3, 3))
        for i in range(n_particles):
            u = orientations[i]
            Q_tensor += np.outer(u, u) - np.eye(3) / 3

        Q_tensor /= n_particles

        # Order parameter is largest eigenvalue
        eigenvals = np.linalg.eigvals(Q_tensor)
        return float(np.max(eigenvals))

    def _calculate_director_vector(
        self, orientations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate director vector (average orientation)."""
        # Calculate orientation tensor and find eigenvector with largest eigenvalue
        n_particles = len(orientations)
        Q_tensor = np.zeros((3, 3))

        for i in range(n_particles):
            u = orientations[i]
            Q_tensor += np.outer(u, u)

        Q_tensor /= n_particles

        eigenvals, eigenvecs = np.linalg.eigh(Q_tensor)
        director_idx = np.argmax(eigenvals)
        return eigenvecs[:, director_idx]

    def _calculate_correlation_length(
        self, positions: npt.NDArray[np.float64], orientations: npt.NDArray[np.float64]
    ) -> float:
        """Calculate orientational correlation length."""
        # Simplified calculation
        distances = cdist(positions, positions)

        correlation_sum = 0.0
        count = 0

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r = distances[i, j]
                correlation = np.dot(orientations[i], orientations[j])
                correlation_sum += correlation * r
                count += 1

        return correlation_sum / count if count > 0 else 0.0

    def _classify_lc_phase(
        self, order_parameter: float, positions: npt.NDArray[np.float64]
    ) -> PhaseType:
        """Classify liquid crystal phase type."""
        if order_parameter < 0.3:
            return PhaseType.ISOTROPIC
        elif order_parameter < 0.7:
            return PhaseType.NEMATIC
        else:
            # Check for layer structure (smectic)
            if self._has_layer_structure(positions):
                return PhaseType.SMECTIC_A
            else:
                return PhaseType.NEMATIC

    def _has_layer_structure(self, positions: npt.NDArray[np.float64]) -> bool:
        """Check for layered structure characteristic of smectic phases."""
        # Simplified: look for periodicity in one direction
        z_positions = positions[:, 2]  # Check z-direction
        hist, _ = np.histogram(z_positions, bins=20)

        # Look for multiple peaks
        peaks = 0
        for i in range(1, len(hist) - 1):
            if (
                hist[i] > hist[i - 1]
                and hist[i] > hist[i + 1]
                and hist[i] > 0.1 * np.max(hist)
            ):
                peaks += 1

        return peaks >= 3


class SelfAssemblyCharacterizer:
    """Main characterization engine for self-assembled systems."""

    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id

        # Initialize analyzers
        self.structure_classifier = StructureClassifier(use_gpu, device_id)
        self.cluster_analyzer = ClusterAnalyzer(use_gpu, device_id)
        self.micelle_analyzer = MicelleAnalyzer(use_gpu)
        self.vesicle_analyzer = VesicleAnalyzer()
        self.lc_analyzer = LiquidCrystalAnalyzer()

        logger.info("Initialized self-assembly characterization engine")

    def characterize_frame(
        self,
        frame: TrajectoryFrame,
        particle_types: List[str],
        clustering_method: ClusteringMethod = ClusteringMethod.DBSCAN,
        cutoff_distance: float = 2.0,
    ) -> CharacterizationResult:
        """Characterize a single trajectory frame."""

        positions = frame.positions

        # Cluster particles
        cluster_labels = self.cluster_analyzer.cluster_particles(
            positions, particle_types, clustering_method, cutoff_distance
        )

        # Analyze each cluster
        structures = []
        micelles = []
        vesicles = []
        liquid_crystals = []
        hydrogels = []

        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise (-1)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_types = [
                particle_types[i] for i in range(len(particle_types)) if cluster_mask[i]
            ]

            if len(cluster_positions) < 5:  # Skip small clusters
                continue

            # Classify structure type
            structure_type = self.structure_classifier.classify_structure(
                cluster_positions, cluster_types, cluster_labels[cluster_mask]
            )

            # Calculate basic metrics
            structure_metrics = self._calculate_structure_metrics(
                cluster_id, structure_type, cluster_positions, cluster_types
            )
            structures.append(structure_metrics)

            # Specialized analysis based on structure type
            try:
                if structure_type == StructureType.MICELLE:
                    micelle_char = self.micelle_analyzer.analyze_micelle(
                        cluster_positions, cluster_types, cluster_labels[cluster_mask]
                    )
                    micelles.append(micelle_char)

                elif structure_type == StructureType.VESICLE:
                    vesicle_char = self.vesicle_analyzer.analyze_vesicle(
                        cluster_positions, cluster_types
                    )
                    vesicles.append(vesicle_char)

                elif structure_type == StructureType.LIQUID_CRYSTAL:
                    lc_char = self.lc_analyzer.analyze_liquid_crystal(cluster_positions)
                    liquid_crystals.append(lc_char)

            except Exception as e:
                logger.warning(
                    f"Failed specialized analysis for cluster {cluster_id}: {e}"
                )

        # Global phase analysis
        phase_analysis = self._analyze_global_phase(positions, particle_types)

        # Global metrics
        global_metrics = self._calculate_global_metrics(positions, cluster_labels)

        return CharacterizationResult(
            frame_id=frame.step,
            time=frame.time,
            structures=structures,
            micelles=micelles,
            vesicles=vesicles,
            liquid_crystals=liquid_crystals,
            hydrogels=hydrogels,
            phase_analysis=phase_analysis,
            global_metrics=global_metrics,
        )

    def _calculate_structure_metrics(
        self,
        structure_id: int,
        structure_type: StructureType,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
    ) -> StructureMetrics:
        """Calculate basic metrics for a structure."""

        # Center of mass
        com = np.mean(positions, axis=0)

        # Radius of gyration
        centered_pos = positions - com
        rg_squared = np.mean(np.sum(centered_pos**2, axis=1))
        radius_of_gyration = np.sqrt(rg_squared)

        # Shape parameters
        gyration_tensor = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                gyration_tensor[i, j] = np.mean(centered_pos[:, i] * centered_pos[:, j])

        eigenvals, _ = np.linalg.eigh(gyration_tensor)
        eigenvals = np.sort(eigenvals)[::-1]

        asphericity = eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])
        eccentricity = (
            np.sqrt(1 - eigenvals[1] / eigenvals[0]) if eigenvals[0] > 0 else 0
        )

        # Determine morphology
        morphology = self.cluster_analyzer.analyze_cluster_morphology(
            positions, np.zeros(len(positions), dtype=np.int32)
        )

        # Simplified surface area and volume estimates
        if SCIPY_AVAILABLE:
            try:
                hull = ConvexHull(positions)
                surface_area = hull.area
                volume = hull.volume
            except Exception:
                surface_area = 4 * np.pi * radius_of_gyration**2
                volume = (4 / 3) * np.pi * radius_of_gyration**3
        else:
            surface_area = 4 * np.pi * radius_of_gyration**2
            volume = (4 / 3) * np.pi * radius_of_gyration**3

        compactness = (
            (36 * np.pi * volume**2) ** (1 / 3) / surface_area
            if surface_area > 0
            else 0
        )

        # Hydrophobic/hydrophilic analysis
        hydrophobic_particles = [
            i
            for i, ptype in enumerate(particle_types)
            if "TAIL" in ptype or "hydrophobic" in ptype.lower()
        ]
        if hydrophobic_particles:
            hydrophobic_positions = positions[hydrophobic_particles]
            hydrophobic_distances = np.linalg.norm(hydrophobic_positions - com, axis=1)
            hydrophobic_core_radius = np.mean(hydrophobic_distances)

            hydrophilic_particles = [
                i
                for i, ptype in enumerate(particle_types)
                if i not in hydrophobic_particles
            ]
            if hydrophilic_particles:
                hydrophilic_positions = positions[hydrophilic_particles]
                hydrophilic_distances = np.linalg.norm(
                    hydrophilic_positions - com, axis=1
                )
                hydrophilic_shell_thickness = (
                    np.mean(hydrophilic_distances) - hydrophobic_core_radius
                )
            else:
                hydrophilic_shell_thickness = 0.0
        else:
            hydrophobic_core_radius = 0.0
            hydrophilic_shell_thickness = 0.0

        # Order parameter (simplified)
        order_parameter = (
            eigenvals[0] / np.sum(eigenvals) if np.sum(eigenvals) > 0 else 0
        )

        return StructureMetrics(
            structure_id=structure_id,
            structure_type=structure_type,
            morphology=morphology,
            aggregation_number=len(positions),
            center_of_mass=com,
            radius_of_gyration=radius_of_gyration,
            asphericity=asphericity,
            eccentricity=eccentricity,
            surface_area=surface_area,
            volume=volume,
            compactness=compactness,
            hydrophobic_core_radius=hydrophobic_core_radius,
            hydrophilic_shell_thickness=hydrophilic_shell_thickness,
            order_parameter=order_parameter,
        )

    def _analyze_global_phase(
        self, positions: npt.NDArray[np.float64], particle_types: List[str]
    ) -> PhaseAnalysisResult:
        """Analyze global phase behavior."""

        # Simplified phase analysis
        n_particles = len(positions)

        # Calculate radial distribution function
        rdf_r, rdf_g = self._calculate_rdf(positions)

        # Calculate structure factor
        structure_factor = self._calculate_structure_factor(positions)

        # Simple phase classification based on structure
        if np.max(rdf_g) > 3.0:  # Strong ordering
            phase_type = PhaseType.CRYSTAL
        elif np.max(rdf_g) > 2.0:  # Moderate ordering
            phase_type = PhaseType.LIQUID_CRYSTAL
        else:  # Weak ordering
            phase_type = PhaseType.ISOTROPIC

        order_parameters = {
            "translational": float(np.max(rdf_g) - 1.0),
            "orientational": 0.5,  # Placeholder
        }

        correlation_functions = {"rdf": rdf_g, "structure_factor": structure_factor}

        return PhaseAnalysisResult(
            phase_type=phase_type,
            order_parameters=order_parameters,
            correlation_functions=correlation_functions,
        )

    def _calculate_rdf(
        self, positions: npt.NDArray[np.float64], r_max: float = 10.0, n_bins: int = 100
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate radial distribution function."""
        n_particles = len(positions)

        # Calculate all pairwise distances
        distances = cdist(positions, positions)

        # Remove self-distances and take upper triangle
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        distances = distances[mask]

        # Create histogram
        r_bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2

        hist, _ = np.histogram(distances, bins=r_bins)

        # Normalize to get g(r)
        dr = r_bins[1] - r_bins[0]
        ideal_density = n_particles / (4 / 3 * np.pi * r_max**3)

        rdf = np.zeros_like(r_centers)
        for i, r in enumerate(r_centers):
            if r > 0:
                shell_volume = 4 * np.pi * r**2 * dr
                expected_count = ideal_density * shell_volume * n_particles
                rdf[i] = hist[i] / expected_count if expected_count > 0 else 0

        return r_centers, rdf

    def _calculate_structure_factor(
        self, positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate structure factor (simplified)."""
        # This is a simplified calculation
        # In practice, would need proper k-space sampling

        com = np.mean(positions, axis=0)
        centered_positions = positions - com

        # Simple approximation using variance
        structure_factor = np.var(centered_positions, axis=0)
        return structure_factor

    def _calculate_global_metrics(
        self, positions: npt.NDArray[np.float64], cluster_labels: npt.NDArray[np.int32]
    ) -> Dict[str, float]:
        """Calculate global system metrics."""

        n_particles = len(positions)
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

        # Clustering metrics
        clustered_fraction = np.sum(cluster_labels >= 0) / n_particles

        # Average cluster size
        cluster_sizes = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id >= 0:
                cluster_sizes.append(np.sum(cluster_labels == cluster_id))

        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

        # System density
        box_volume = 1000.0  # Placeholder - would get from simulation box
        density = n_particles / box_volume

        return {
            "n_clusters": float(n_clusters),
            "clustered_fraction": clustered_fraction,
            "average_cluster_size": avg_cluster_size,
            "system_density": density,
            "largest_cluster_size": float(max(cluster_sizes)) if cluster_sizes else 0,
        }


# High-level convenience functions
def characterize_trajectory(
    trajectory: List[TrajectoryFrame],
    particle_types: List[str],
    use_gpu: bool = True,
    clustering_method: ClusteringMethod = ClusteringMethod.DBSCAN,
) -> List[CharacterizationResult]:
    """Characterize an entire simulation trajectory."""

    characterizer = SelfAssemblyCharacterizer(use_gpu=use_gpu)
    results = []

    for frame in trajectory:
        result = characterizer.characterize_frame(
            frame, particle_types, clustering_method
        )
        results.append(result)

    return results


def analyze_micelle_formation(
    trajectory: List[TrajectoryFrame], particle_types: List[str]
) -> Dict[str, Any]:
    """Analyze micelle formation process over time."""

    results = characterize_trajectory(trajectory, particle_types)

    # Track micelle formation
    micelle_counts = []
    aggregation_numbers = []

    for result in results:
        micelle_counts.append(len(result.micelles))
        if result.micelles:
            agg_nums = [m.aggregation_number for m in result.micelles]
            aggregation_numbers.append(np.mean(agg_nums))
        else:
            aggregation_numbers.append(0)

    # Calculate critical micelle concentration (CMC) approximation
    times = [r.time for r in results]

    return {
        "times": times,
        "micelle_counts": micelle_counts,
        "aggregation_numbers": aggregation_numbers,
        "cmc_estimate": np.mean(aggregation_numbers) if aggregation_numbers else 0,
    }


def calculate_phase_diagram(
    concentration_range: List[float],
    temperature_range: List[float],
    simulation_function: Callable,
) -> Dict[str, Any]:
    """Calculate phase diagram for self-assembling system."""

    phase_diagram = np.zeros(
        (len(concentration_range), len(temperature_range)), dtype=int
    )

    for i, conc in enumerate(concentration_range):
        for j, temp in enumerate(temperature_range):
            # Run simulation at this point
            trajectory = simulation_function(concentration=conc, temperature=temp)

            # Characterize final frame
            particle_types = ["HEAD", "TAIL", "WATER"] * (
                len(trajectory[-1].positions) // 3
            )
            characterizer = SelfAssemblyCharacterizer()
            result = characterizer.characterize_frame(trajectory[-1], particle_types)

            # Classify phase
            if result.micelles:
                phase_diagram[i, j] = 1  # Micelle phase
            elif result.vesicles:
                phase_diagram[i, j] = 2  # Vesicle phase
            elif result.liquid_crystals:
                phase_diagram[i, j] = 3  # LC phase
            else:
                phase_diagram[i, j] = 0  # Isotropic phase

    return {
        "concentrations": concentration_range,
        "temperatures": temperature_range,
        "phase_diagram": phase_diagram,
    }
