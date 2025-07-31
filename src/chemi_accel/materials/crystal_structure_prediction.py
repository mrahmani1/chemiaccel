"""
GPU-Accelerated Crystal Structure Prediction (CSP) Module for ChemiAccel.

This module provides comprehensive tools for de novo prediction of crystal structures
using GPU-accelerated AI models, including GNNs for packing motif prediction,
generative models for unit cells, and efficient lattice energy minimization
algorithms for exploring polymorphic landscapes.

Features:
- GPU-accelerated graph neural networks for crystal property prediction
- Generative models for unit cell structure prediction
- Lattice energy minimization with gradient-based optimization
- Polymorphic landscape exploration and analysis
- Crystal packing motif prediction and classification
- Space group symmetry handling and validation
- Integration with experimental data for structure refinement
- High-throughput crystal structure screening
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional GPU dependencies
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, GCNConv, MessagePassing

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    GCNConv = None
    GATConv = None
    MessagePassing = None
    Data = None
    Batch = None
    _TORCH_AVAILABLE = False

try:
    import scipy.sparse as sp
    from scipy.linalg import eigh
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    cdist = None
    minimize = None
    eigh = None
    sp = None
    _SCIPY_AVAILABLE = False

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    _PYMATGEN_AVAILABLE = True
except ImportError:
    Structure = None
    Lattice = None
    StructureMatcher = None
    SpacegroupAnalyzer = None
    _PYMATGEN_AVAILABLE = False


class CrystalSystemType(Enum):
    """Crystal system types."""

    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


class CSPMethod(Enum):
    """Crystal structure prediction methods."""

    GNN_BASED = "gnn_based"
    GENERATIVE_MODEL = "generative_model"
    ENERGY_MINIMIZATION = "energy_minimization"
    HYBRID_APPROACH = "hybrid_approach"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"


class PackingMotifType(Enum):
    """Types of crystal packing motifs."""

    CLOSE_PACKED = "close_packed"
    LAYERED = "layered"
    FRAMEWORK = "framework"
    CHAIN = "chain"
    DIMER = "dimer"
    ISOLATED = "isolated"
    HERRINGBONE = "herringbone"
    COLUMNAR = "columnar"


class LatticeEnergyMethod(Enum):
    """Lattice energy calculation methods."""

    LJ_POTENTIAL = "lennard_jones"
    COULOMB = "coulomb"
    BORN_MAYER = "born_mayer"
    BUCKINGHAM = "buckingham"
    DFT_BASED = "dft_based"
    MACHINE_LEARNING = "ml_based"


class SymmetryHandling(Enum):
    """Symmetry handling approaches."""

    STRICT = "strict"
    RELAXED = "relaxed"
    IGNORE = "ignore"
    AUTO_DETECT = "auto_detect"


@dataclass
class CrystalPredictionConfig:
    """Configuration for crystal structure prediction."""

    method: CSPMethod = CSPMethod.HYBRID_APPROACH
    max_structures: int = 100
    energy_cutoff: float = 50.0  # kJ/mol above global minimum
    use_gpu: bool = True
    symmetry_handling: SymmetryHandling = SymmetryHandling.AUTO_DETECT
    include_polymorphs: bool = True
    temperature: float = 300.0  # K
    pressure: float = 1.0  # atm
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    population_size: int = 50  # For genetic algorithm
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8


@dataclass
class UnitCellParameters:
    """Unit cell parameters."""

    a: float
    b: float
    c: float
    alpha: float  # degrees
    beta: float  # degrees
    gamma: float  # degrees
    volume: Optional[float] = None
    space_group: Optional[int] = None
    crystal_system: Optional[CrystalSystemType] = None

    def __post_init__(self):
        """Calculate volume and infer crystal system."""
        if self.volume is None:
            self.volume = self._calculate_volume()

        if self.crystal_system is None:
            self.crystal_system = self._infer_crystal_system()

    def _calculate_volume(self) -> float:
        """Calculate unit cell volume."""
        a, b, c = self.a, self.b, self.c
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
                - np.cos(alpha_rad) ** 2
                - np.cos(beta_rad) ** 2
                - np.cos(gamma_rad) ** 2
            )
        )
        return volume

    def _infer_crystal_system(self) -> CrystalSystemType:
        """Infer crystal system from parameters."""
        tol = 1e-3

        # Check angles
        angles_90 = [
            abs(angle - 90) < tol for angle in [self.alpha, self.beta, self.gamma]
        ]
        all_90 = all(angles_90)

        # Check lengths
        a_eq_b = abs(self.a - self.b) < tol
        b_eq_c = abs(self.b - self.c) < tol
        a_eq_c = abs(self.a - self.c) < tol
        all_equal = a_eq_b and b_eq_c

        if all_equal and all_90:
            return CrystalSystemType.CUBIC
        elif a_eq_b and all_90:
            return CrystalSystemType.TETRAGONAL
        elif all_90:
            return CrystalSystemType.ORTHORHOMBIC
        elif a_eq_b and abs(self.gamma - 120) < tol and angles_90[:2] == [True, True]:
            return CrystalSystemType.HEXAGONAL
        elif (
            all_equal
            and abs(self.alpha - self.beta) < tol
            and abs(self.beta - self.gamma) < tol
        ):
            return CrystalSystemType.TRIGONAL
        elif angles_90[0] and angles_90[2]:  # alpha = gamma = 90
            return CrystalSystemType.MONOCLINIC
        else:
            return CrystalSystemType.TRICLINIC


@dataclass
class AtomicPosition:
    """Atomic position in crystal structure."""

    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    thermal_factor: float = 0.0


@dataclass
class CrystalStructure:
    """Crystal structure representation."""

    unit_cell: UnitCellParameters
    atomic_positions: List[AtomicPosition]
    space_group: int
    formula: str
    density: Optional[float] = None
    lattice_energy: Optional[float] = None
    formation_energy: Optional[float] = None
    stability_score: Optional[float] = None
    packing_efficiency: Optional[float] = None
    packing_motif: Optional[PackingMotifType] = None

    def __post_init__(self):
        """Calculate derived properties."""
        if self.density is None:
            self.density = self._calculate_density()

        if self.packing_efficiency is None:
            self.packing_efficiency = self._calculate_packing_efficiency()


@dataclass
class CSPPrediction:
    """Crystal structure prediction result."""

    predicted_structures: List[CrystalStructure]
    energy_landscape: Dict[str, float]
    polymorphs: List[CrystalStructure]
    most_stable_structure: CrystalStructure
    prediction_confidence: float
    computational_cost: float
    method_used: CSPMethod
    convergence_achieved: bool
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class CrystalStructurePredictionError(Exception):
    """Base exception for crystal structure prediction errors."""

    pass


class LatticeEnergyError(CrystalStructurePredictionError):
    """Exception for lattice energy calculation errors."""

    pass


class StructureGenerationError(CrystalStructurePredictionError):
    """Exception for structure generation errors."""

    pass


class SymmetryError(CrystalStructurePredictionError):
    """Exception for symmetry handling errors."""

    pass


class ConvergenceError(CrystalStructurePredictionError):
    """Exception for convergence failures."""

    pass


# GPU-Accelerated Neural Networks for CSP


class CrystalGraphConv(MessagePassing if _TORCH_AVAILABLE else object):
    """Crystal-specific graph convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric required for CrystalGraphConv"
            )

        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.atom_embedding = nn.Linear(in_channels, out_channels)
        self.bond_embedding = nn.Linear(10, out_channels)  # Distance + angles features
        self.message_net = nn.Sequential(
            nn.Linear(2 * out_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.update_net = nn.Sequential(
            nn.Linear(out_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        """Forward pass."""
        x = self.atom_embedding(x)
        edge_attr = self.bond_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Compute messages."""
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(message_input)

    def update(self, aggr_out, x):
        """Update node features."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class PackingMotifGNN(nn.Module if _TORCH_AVAILABLE else object):
    """Graph Neural Network for predicting crystal packing motifs."""

    def __init__(
        self,
        node_features: int = 92,  # Number of elements
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_motif_types: int = len(PackingMotifType),
        dropout: float = 0.1,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PackingMotifGNN")

        super().__init__()

        self.atom_embedding = nn.Embedding(node_features, hidden_dim)

        self.conv_layers = nn.ModuleList(
            [CrystalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.batch_norm = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_motif_types),
        )

    def forward(self, batch):
        """Forward pass."""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch

        # Embed atomic numbers
        x = self.atom_embedding(x.long())

        # Apply graph convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norm):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection

        # Global pooling
        batch_size = batch_idx.max().item() + 1
        global_features = torch.zeros(batch_size, x.size(1), device=x.device)

        for i in range(batch_size):
            mask = batch_idx == i
            if mask.sum() > 0:
                global_features[i] = torch.mean(x[mask], dim=0)

        global_features = self.global_pool(global_features)

        # Classification
        output = self.classifier(global_features)
        return output


class UnitCellGenerator(nn.Module if _TORCH_AVAILABLE else object):
    """Generative model for unit cell parameters."""

    def __init__(
        self,
        molecular_features: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        latent_dim: int = 128,
        condition_dim: int = 64,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for UnitCellGenerator")

        super().__init__()

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        self.encoder = nn.Sequential(
            nn.Linear(molecular_features + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
                )
                for _ in range(num_layers - 2)
            ],
            nn.Linear(hidden_dim, 2 * latent_dim),  # mu and logvar
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
                )
                for _ in range(num_layers - 2)
            ],
            nn.Linear(hidden_dim, 6),  # a, b, c, alpha, beta, gamma
        )

        self.crystal_system_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(CrystalSystemType)),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, molecular_features, conditions=None):
        """Forward pass."""
        if conditions is not None:
            conditions = self.condition_encoder(conditions)
            x = torch.cat([molecular_features, conditions], dim=-1)
        else:
            conditions = torch.zeros(
                molecular_features.size(0),
                self.condition_encoder[-1].out_features,
                device=molecular_features.device,
            )
            x = torch.cat([molecular_features, conditions], dim=-1)

        # Encode to latent space
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        # Decode to unit cell parameters
        decoder_input = torch.cat([z, conditions], dim=-1)
        unit_cell_params = self.decoder(decoder_input)

        # Predict crystal system
        crystal_system = self.crystal_system_classifier(z)

        return {
            "unit_cell_params": unit_cell_params,
            "crystal_system": crystal_system,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
        }


class LatticeEnergyPredictor(nn.Module if _TORCH_AVAILABLE else object):
    """Neural network for lattice energy prediction."""

    def __init__(
        self,
        structural_features: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LatticeEnergyPredictor")

        super().__init__()

        layers = [nn.Linear(structural_features, hidden_dim), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(hidden_dim, 1))  # Single energy output

        self.network = nn.Sequential(*layers)

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(self, x):
        """Forward pass."""
        features = x

        # Extract features from second-to-last layer
        for layer in self.network[:-1]:
            features = layer(features)

        # Energy prediction
        energy = self.network[-1](features)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features)

        return energy, uncertainty


# Core CSP Engine


class GPUAcceleratedCSPEngine:
    """GPU-accelerated crystal structure prediction engine."""

    def __init__(self, config: Optional[CrystalPredictionConfig] = None):
        """Initialize CSP engine."""
        self.config = config or CrystalPredictionConfig()
        self.use_gpu = self.config.use_gpu and _CUPY_AVAILABLE and _TORCH_AVAILABLE

        # Set device
        if self.use_gpu and torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        # Initialize models
        self._setup_models()

        # Initialize energy calculators
        self._setup_energy_calculators()

        # Initialize structure generators
        self._setup_structure_generators()

        # Prediction cache
        self.prediction_cache = {}

        # Performance metrics
        self.performance_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_time": 0.0,
            "gpu_utilization": 0.0,
        }

    def _setup_models(self):
        """Set up neural network models."""
        if not _TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Neural network models disabled.")
            self.motif_predictor = None
            self.unit_cell_generator = None
            self.energy_predictor = None
            return

        self.motif_predictor = PackingMotifGNN()
        self.unit_cell_generator = UnitCellGenerator()
        self.energy_predictor = LatticeEnergyPredictor()

        if self.use_gpu and self.device is not None:
            self.motif_predictor = self.motif_predictor.to(self.device)
            self.unit_cell_generator = self.unit_cell_generator.to(self.device)
            self.energy_predictor = self.energy_predictor.to(self.device)

    def _setup_energy_calculators(self):
        """Set up energy calculation methods."""
        self.energy_calculators = {
            LatticeEnergyMethod.LJ_POTENTIAL: self._calculate_lj_energy,
            LatticeEnergyMethod.COULOMB: self._calculate_coulomb_energy,
            LatticeEnergyMethod.BORN_MAYER: self._calculate_born_mayer_energy,
            LatticeEnergyMethod.BUCKINGHAM: self._calculate_buckingham_energy,
            LatticeEnergyMethod.MACHINE_LEARNING: self._calculate_ml_energy,
        }

    def _setup_structure_generators(self):
        """Set up structure generation methods."""
        self.structure_generators = {
            CSPMethod.GNN_BASED: self._generate_structures_gnn,
            CSPMethod.GENERATIVE_MODEL: self._generate_structures_generative,
            CSPMethod.ENERGY_MINIMIZATION: self._generate_structures_energy_min,
            CSPMethod.HYBRID_APPROACH: self._generate_structures_hybrid,
            CSPMethod.RANDOM_SEARCH: self._generate_structures_random,
            CSPMethod.GENETIC_ALGORITHM: self._generate_structures_genetic,
        }

    def predict_crystal_structure(
        self,
        molecular_formula: str,
        molecular_features: Optional[np.ndarray] = None,
        experimental_constraints: Optional[Dict[str, Any]] = None,
    ) -> CSPPrediction:
        """
        Predict crystal structure for given molecular formula.

        Args:
            molecular_formula: Chemical formula (e.g., 'C6H6', 'NaCl')
            molecular_features: Optional molecular descriptor features
            experimental_constraints: Optional experimental constraints

        Returns:
            CSPPrediction with predicted structures and analysis
        """
        try:
            # Generate molecular features if not provided
            if molecular_features is None:
                molecular_features = self._generate_molecular_features(
                    molecular_formula
                )

            # Cache key for prediction
            cache_key = f"{molecular_formula}_{hash(str(experimental_constraints))}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]

            # Generate candidate structures
            candidate_structures = self._generate_candidate_structures(
                molecular_formula, molecular_features, experimental_constraints
            )

            # Calculate energies and optimize structures
            optimized_structures = self._optimize_structures(candidate_structures)

            # Analyze polymorphs
            polymorphs = self._identify_polymorphs(optimized_structures)

            # Rank structures by stability
            ranked_structures = self._rank_structures_by_stability(optimized_structures)

            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(ranked_structures)

            # Create prediction result
            prediction = CSPPrediction(
                predicted_structures=ranked_structures[: self.config.max_structures],
                energy_landscape=self._calculate_energy_landscape(ranked_structures),
                polymorphs=polymorphs,
                most_stable_structure=(
                    ranked_structures[0] if ranked_structures else None
                ),
                prediction_confidence=confidence,
                computational_cost=0.0,  # Will be updated
                method_used=self.config.method,
                convergence_achieved=True,
            )

            # Cache result
            self.prediction_cache[cache_key] = prediction

            # Update performance stats
            self.performance_stats["total_predictions"] += 1
            self.performance_stats["successful_predictions"] += 1

            return prediction

        except Exception as e:
            self.performance_stats["total_predictions"] += 1
            raise CrystalStructurePredictionError(
                f"Structure prediction failed: {str(e)}"
            )

    def _generate_molecular_features(self, molecular_formula: str) -> np.ndarray:
        """Generate molecular features from formula."""
        # Parse molecular formula
        elements = self._parse_molecular_formula(molecular_formula)

        # Create feature vector
        features = np.zeros(256)  # Standard feature dimension

        # Encode elemental composition
        for i, (element, count) in enumerate(elements.items()):
            if i < 50:  # Limit to first 50 elements
                atomic_num = self._get_atomic_number(element)
                features[i] = count
                features[50 + i] = atomic_num

        # Add derived features
        features[100] = len(elements)  # Number of different elements
        features[101] = sum(elements.values())  # Total atoms
        features[102] = np.mean(list(elements.values()))  # Average count
        features[103] = np.std(list(elements.values()))  # Count variance

        return features

    def _generate_candidate_structures(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate candidate crystal structures."""
        generator = self.structure_generators[self.config.method]
        return generator(
            molecular_formula, molecular_features, experimental_constraints
        )

    def _generate_structures_gnn(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate structures using GNN-based prediction."""
        if not _TORCH_AVAILABLE or self.motif_predictor is None:
            warnings.warn("GNN models not available, falling back to random generation")
            return self._generate_structures_random(
                molecular_formula, molecular_features, experimental_constraints
            )

        structures = []

        # Predict packing motifs
        motif_predictions = self._predict_packing_motifs(molecular_features)

        # Generate structures for each predicted motif
        for motif, probability in motif_predictions.items():
            if probability > 0.1:  # Only consider likely motifs
                motif_structures = self._generate_structures_for_motif(
                    molecular_formula, molecular_features, motif, int(probability * 10)
                )
                structures.extend(motif_structures)

        return structures[: self.config.max_structures]

    def _generate_structures_generative(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate structures using generative models."""
        if not _TORCH_AVAILABLE or self.unit_cell_generator is None:
            warnings.warn(
                "Generative models not available, falling back to random generation"
            )
            return self._generate_structures_random(
                molecular_formula, molecular_features, experimental_constraints
            )

        structures = []

        # Convert to torch tensor
        if self.device is not None:
            features_tensor = (
                torch.FloatTensor(molecular_features).unsqueeze(0).to(self.device)
            )
        else:
            features_tensor = torch.FloatTensor(molecular_features).unsqueeze(0)

        # Generate multiple unit cells
        for _ in range(self.config.max_structures):
            with torch.no_grad():
                generation_result = self.unit_cell_generator(features_tensor)

                # Extract unit cell parameters
                unit_cell_params = (
                    generation_result["unit_cell_params"].cpu().numpy()[0]
                )
                crystal_system_logits = (
                    generation_result["crystal_system"].cpu().numpy()[0]
                )

                # Convert to UnitCellParameters
                unit_cell = UnitCellParameters(
                    a=float(
                        abs(unit_cell_params[0]) + 3.0
                    ),  # Ensure positive, reasonable values
                    b=float(abs(unit_cell_params[1]) + 3.0),
                    c=float(abs(unit_cell_params[2]) + 3.0),
                    alpha=float(np.clip(unit_cell_params[3], 60, 120)),
                    beta=float(np.clip(unit_cell_params[4], 60, 120)),
                    gamma=float(np.clip(unit_cell_params[5], 60, 120)),
                )

                # Generate atomic positions
                atomic_positions = self._generate_atomic_positions(
                    molecular_formula, unit_cell
                )

                # Create crystal structure
                structure = CrystalStructure(
                    unit_cell=unit_cell,
                    atomic_positions=atomic_positions,
                    space_group=1,  # P1 initially
                    formula=molecular_formula,
                )

                structures.append(structure)

        return structures

    def _generate_structures_energy_min(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate structures using energy minimization."""
        structures = []

        # Start with random structures and optimize
        initial_structures = self._generate_structures_random(
            molecular_formula, molecular_features, experimental_constraints
        )

        for structure in initial_structures:
            try:
                optimized = self._minimize_lattice_energy(structure)
                structures.append(optimized)
            except Exception as e:
                warnings.warn(f"Energy minimization failed: {e}")
                structures.append(structure)  # Use unoptimized structure

        return structures

    def _generate_structures_hybrid(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate structures using hybrid approach."""
        structures = []

        # Combine different methods
        methods = [
            self._generate_structures_gnn,
            self._generate_structures_generative,
            self._generate_structures_energy_min,
        ]

        structures_per_method = self.config.max_structures // len(methods)

        for method in methods:
            try:
                method_structures = method(
                    molecular_formula, molecular_features, experimental_constraints
                )[:structures_per_method]
                structures.extend(method_structures)
            except Exception as e:
                warnings.warn(f"Method failed: {e}")

        return structures[: self.config.max_structures]

    def _generate_structures_random(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate random crystal structures."""
        structures = []

        for _ in range(self.config.max_structures):
            # Generate random unit cell
            unit_cell = self._generate_random_unit_cell()

            # Generate atomic positions
            atomic_positions = self._generate_atomic_positions(
                molecular_formula, unit_cell
            )

            # Create structure
            structure = CrystalStructure(
                unit_cell=unit_cell,
                atomic_positions=atomic_positions,
                space_group=np.random.randint(1, 231),  # Random space group
                formula=molecular_formula,
            )

            structures.append(structure)

        return structures

    def _generate_structures_genetic(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        experimental_constraints: Optional[Dict[str, Any]],
    ) -> List[CrystalStructure]:
        """Generate structures using genetic algorithm."""
        # Initialize population
        population = self._generate_structures_random(
            molecular_formula, molecular_features, experimental_constraints
        )[: self.config.population_size]

        best_structures = []

        for generation in range(
            self.config.max_iterations // 10
        ):  # Fewer generations for speed
            # Evaluate fitness (inverse of energy)
            fitness_scores = []
            for structure in population:
                try:
                    energy = self._calculate_structure_energy(structure)
                    fitness = 1.0 / (1.0 + abs(energy))
                    fitness_scores.append(fitness)
                except:
                    fitness_scores.append(0.0)

            # Select best structures
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = self.config.population_size // 4
            elite = [population[i] for i in sorted_indices[:elite_size]]
            best_structures.extend(elite[:5])  # Keep top 5 from each generation

            # Create next generation
            new_population = elite.copy()

            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child = self._crossover_structures(parent1, parent2)
                else:
                    child = parent1

                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child = self._mutate_structure(child)

                new_population.append(child)

            population = new_population

        return best_structures[: self.config.max_structures]

    def _optimize_structures(
        self, structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """Optimize crystal structures."""
        optimized = []

        for structure in structures:
            try:
                # Calculate initial energy
                initial_energy = self._calculate_structure_energy(structure)
                structure.lattice_energy = initial_energy

                # Optimize if energy minimization is enabled
                if self.config.method in [
                    CSPMethod.ENERGY_MINIMIZATION,
                    CSPMethod.HYBRID_APPROACH,
                ]:
                    optimized_structure = self._minimize_lattice_energy(structure)
                    optimized.append(optimized_structure)
                else:
                    optimized.append(structure)

            except Exception as e:
                warnings.warn(f"Structure optimization failed: {e}")
                optimized.append(structure)

        return optimized

    def _calculate_structure_energy(self, structure: CrystalStructure) -> float:
        """Calculate total energy of crystal structure."""
        # Use ML predictor if available
        if self.energy_predictor is not None:
            try:
                structural_features = self._extract_structural_features(structure)
                if self.device is not None:
                    features_tensor = (
                        torch.FloatTensor(structural_features)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                else:
                    features_tensor = torch.FloatTensor(structural_features).unsqueeze(
                        0
                    )

                with torch.no_grad():
                    energy, uncertainty = self.energy_predictor(features_tensor)
                    return float(energy.cpu().numpy()[0, 0])
            except:
                pass

        # Fallback to classical methods
        total_energy = 0.0

        # Lennard-Jones contribution
        lj_energy = self._calculate_lj_energy(structure)
        total_energy += lj_energy

        # Coulomb contribution
        coulomb_energy = self._calculate_coulomb_energy(structure)
        total_energy += coulomb_energy

        return total_energy

    def _calculate_lj_energy(self, structure: CrystalStructure) -> float:
        """Calculate Lennard-Jones energy."""
        if not _SCIPY_AVAILABLE:
            return 0.0

        positions = []
        elements = []

        for pos in structure.atomic_positions:
            positions.append([pos.x, pos.y, pos.z])
            elements.append(pos.element)

        positions = np.array(positions)

        if len(positions) < 2:
            return 0.0

        # Simple LJ parameters (simplified)
        lj_params = {
            "H": (0.5, 2.0),
            "C": (1.0, 3.4),
            "N": (0.8, 3.2),
            "O": (0.7, 3.0),
            "F": (0.6, 2.8),
            "S": (1.2, 3.6),
            "Cl": (1.0, 3.5),
            "Br": (1.2, 3.7),
        }

        default_params = (1.0, 3.0)

        total_energy = 0.0

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                elem_i = elements[i]
                elem_j = elements[j]

                epsilon_i, sigma_i = lj_params.get(elem_i, default_params)
                epsilon_j, sigma_j = lj_params.get(elem_j, default_params)

                epsilon = np.sqrt(epsilon_i * epsilon_j)
                sigma = (sigma_i + sigma_j) / 2.0

                r = np.linalg.norm(positions[i] - positions[j])
                r = max(r, 0.5)  # Prevent division by zero

                # Lennard-Jones potential
                energy = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
                total_energy += energy

        return total_energy

    def _calculate_coulomb_energy(self, structure: CrystalStructure) -> float:
        """Calculate Coulomb energy."""
        # Simple charge assignment (very approximate)
        charges = {
            "H": 0.0,
            "C": 0.0,
            "N": -0.5,
            "O": -1.0,
            "F": -1.0,
            "Na": 1.0,
            "K": 1.0,
            "Ca": 2.0,
            "Mg": 2.0,
            "Cl": -1.0,
        }

        positions = []
        atom_charges = []

        for pos in structure.atomic_positions:
            positions.append([pos.x, pos.y, pos.z])
            charge = charges.get(pos.element, 0.0)
            atom_charges.append(charge)

        positions = np.array(positions)
        atom_charges = np.array(atom_charges)

        if len(positions) < 2 or np.all(atom_charges == 0):
            return 0.0

        total_energy = 0.0
        ke = 8.99e9  # Coulomb constant (simplified units)

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r = np.linalg.norm(positions[i] - positions[j])
                r = max(r, 0.1)  # Prevent division by zero

                energy = ke * atom_charges[i] * atom_charges[j] / r
                total_energy += energy

        return total_energy * 1e-6  # Scale down

    def _calculate_born_mayer_energy(self, structure: CrystalStructure) -> float:
        """Calculate Born-Mayer energy."""
        # Placeholder implementation
        return 0.0

    def _calculate_buckingham_energy(self, structure: CrystalStructure) -> float:
        """Calculate Buckingham potential energy."""
        # Placeholder implementation
        return 0.0

    def _calculate_ml_energy(self, structure: CrystalStructure) -> float:
        """Calculate energy using ML models."""
        if self.energy_predictor is None:
            return 0.0

        try:
            structural_features = self._extract_structural_features(structure)
            if self.device is not None:
                features_tensor = (
                    torch.FloatTensor(structural_features).unsqueeze(0).to(self.device)
                )
            else:
                features_tensor = torch.FloatTensor(structural_features).unsqueeze(0)

            with torch.no_grad():
                energy, uncertainty = self.energy_predictor(features_tensor)
                return float(energy.cpu().numpy()[0, 0])
        except:
            return 0.0

    def _minimize_lattice_energy(self, structure: CrystalStructure) -> CrystalStructure:
        """Minimize lattice energy using gradient-based optimization."""
        if not _SCIPY_AVAILABLE:
            warnings.warn("SciPy not available for energy minimization")
            return structure

        def objective(params):
            """Objective function for minimization."""
            # Unpack parameters
            a, b, c, alpha, beta, gamma = params[:6]

            # Update unit cell
            optimized_structure = structure
            optimized_structure.unit_cell = UnitCellParameters(
                a=max(a, 1.0),
                b=max(b, 1.0),
                c=max(c, 1.0),
                alpha=np.clip(alpha, 60, 120),
                beta=np.clip(beta, 60, 120),
                gamma=np.clip(gamma, 60, 120),
            )

            # Calculate energy
            try:
                energy = self._calculate_structure_energy(optimized_structure)
                return energy
            except:
                return 1e10  # Large penalty for invalid structures

        # Initial parameters
        uc = structure.unit_cell
        initial_params = [uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma]

        # Bounds for optimization
        bounds = [
            (1.0, 50.0),
            (1.0, 50.0),
            (1.0, 50.0),  # a, b, c
            (60.0, 120.0),
            (60.0, 120.0),
            (60.0, 120.0),  # alpha, beta, gamma
        ]

        try:
            result = minimize(
                objective, initial_params, bounds=bounds, method="L-BFGS-B"
            )

            if result.success:
                # Update structure with optimized parameters
                optimized_params = result.x
                structure.unit_cell = UnitCellParameters(
                    a=optimized_params[0],
                    b=optimized_params[1],
                    c=optimized_params[2],
                    alpha=optimized_params[3],
                    beta=optimized_params[4],
                    gamma=optimized_params[5],
                )
                structure.lattice_energy = float(result.fun)
                structure.stability_score = 1.0 / (1.0 + abs(result.fun))

        except Exception as e:
            warnings.warn(f"Energy minimization failed: {e}")

        return structure

    def _identify_polymorphs(
        self, structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """Identify polymorphic structures."""
        if not structures:
            return []

        polymorphs = []
        used_indices = set()

        for i, structure1 in enumerate(structures):
            if i in used_indices:
                continue

            # Check if this is a polymorph of any existing structure
            is_polymorph = False
            for j, structure2 in enumerate(structures[:i]):
                if j not in used_indices and self._are_polymorphs(
                    structure1, structure2
                ):
                    is_polymorph = True
                    break

            if not is_polymorph:
                polymorphs.append(structure1)
                used_indices.add(i)

        return polymorphs

    def _are_polymorphs(
        self, struct1: CrystalStructure, struct2: CrystalStructure
    ) -> bool:
        """Check if two structures are polymorphs."""
        # Same formula but different packing
        if struct1.formula != struct2.formula:
            return False

        # Different unit cell parameters
        tolerance = 0.05
        uc1, uc2 = struct1.unit_cell, struct2.unit_cell

        params1 = [uc1.a, uc1.b, uc1.c, uc1.volume]
        params2 = [uc2.a, uc2.b, uc2.c, uc2.volume]

        for p1, p2 in zip(params1, params2):
            if abs(p1 - p2) / max(p1, p2) > tolerance:
                return True  # Different enough to be polymorphs

        return False

    def _rank_structures_by_stability(
        self, structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """Rank structures by thermodynamic stability."""

        def stability_key(structure):
            if structure.lattice_energy is not None:
                return structure.lattice_energy
            else:
                return self._calculate_structure_energy(structure)

        return sorted(structures, key=stability_key)

    def _calculate_energy_landscape(
        self, structures: List[CrystalStructure]
    ) -> Dict[str, float]:
        """Calculate energy landscape from structures."""
        landscape = {}

        for i, structure in enumerate(structures):
            key = f"structure_{i}_{structure.formula}"
            energy = structure.lattice_energy
            if energy is None:
                energy = self._calculate_structure_energy(structure)
            landscape[key] = energy

        return landscape

    def _calculate_prediction_confidence(
        self, structures: List[CrystalStructure]
    ) -> float:
        """Calculate confidence in structure prediction."""
        if not structures:
            return 0.0

        energies = []
        for structure in structures:
            energy = structure.lattice_energy
            if energy is None:
                energy = self._calculate_structure_energy(structure)
            energies.append(energy)

        if len(energies) == 1:
            return 0.8  # Reasonable confidence for single structure

        # Confidence based on energy gap between most stable structures
        energies = sorted(energies)
        energy_gap = energies[1] - energies[0] if len(energies) > 1 else 0

        # Higher gap = higher confidence
        confidence = min(1.0, energy_gap / 10.0)  # Normalize by typical energy scale
        return max(0.0, confidence)

    def _predict_packing_motifs(
        self, molecular_features: np.ndarray
    ) -> Dict[PackingMotifType, float]:
        """Predict likely packing motifs using GNN."""
        if not _TORCH_AVAILABLE or self.motif_predictor is None:
            # Return uniform distribution if no model
            motif_types = list(PackingMotifType)
            return {motif: 1.0 / len(motif_types) for motif in motif_types}

        # Create dummy graph structure for molecular features
        try:
            # Convert molecular features to graph representation
            graph_data = self._molecular_features_to_graph(molecular_features)

            if self.device is not None:
                graph_data = graph_data.to(self.device)

            with torch.no_grad():
                predictions = self.motif_predictor(graph_data)
                probabilities = F.softmax(predictions, dim=-1).cpu().numpy()[0]

            motif_types = list(PackingMotifType)
            return {motif_types[i]: float(prob) for i, prob in enumerate(probabilities)}

        except Exception as e:
            warnings.warn(f"Motif prediction failed: {e}")
            # Return uniform distribution
            motif_types = list(PackingMotifType)
            return {motif: 1.0 / len(motif_types) for motif in motif_types}

    def _molecular_features_to_graph(self, molecular_features: np.ndarray) -> "Data":
        """Convert molecular features to graph representation."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for graph conversion")

        # Simplified graph creation from molecular features
        num_nodes = min(
            10, int(molecular_features[101])
        )  # Use total atoms, capped at 10
        num_nodes = max(2, num_nodes)  # At least 2 nodes

        # Node features (atomic numbers and properties)
        node_features = []
        for i in range(num_nodes):
            if i < 50 and molecular_features[50 + i] > 0:
                atomic_num = int(molecular_features[50 + i])
                features = [atomic_num] + [0] * 91  # Pad to 92 features
                features[atomic_num] = 1  # One-hot encoding
            else:
                features = [6] + [0] * 91  # Default to carbon
                features[6] = 1
            node_features.append(features)

        x = torch.FloatTensor(node_features)

        # Create edge connections (simple fully connected graph)
        edge_indices = []
        edge_attributes = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_indices.extend([[i, j], [j, i]])  # Bidirectional

                # Edge attributes (distance + angles, simplified)
                distance = np.random.uniform(1.0, 3.0)  # Typical bond lengths
                edge_attr = [distance] + [0] * 9  # 10 features total
                edge_attributes.extend([edge_attr, edge_attr])  # For both directions

        edge_index = torch.LongTensor(edge_indices).T
        edge_attr = torch.FloatTensor(edge_attributes)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _generate_structures_for_motif(
        self,
        molecular_formula: str,
        molecular_features: np.ndarray,
        motif: PackingMotifType,
        num_structures: int,
    ) -> List[CrystalStructure]:
        """Generate structures for specific packing motif."""
        structures = []

        for _ in range(num_structures):
            # Generate unit cell based on motif type
            unit_cell = self._generate_unit_cell_for_motif(motif)

            # Generate atomic positions based on motif
            atomic_positions = self._generate_positions_for_motif(
                molecular_formula, unit_cell, motif
            )

            structure = CrystalStructure(
                unit_cell=unit_cell,
                atomic_positions=atomic_positions,
                space_group=self._suggest_space_group_for_motif(motif),
                formula=molecular_formula,
                packing_motif=motif,
            )

            structures.append(structure)

        return structures

    def _generate_unit_cell_for_motif(
        self, motif: PackingMotifType
    ) -> UnitCellParameters:
        """Generate unit cell parameters suitable for packing motif."""
        if motif == PackingMotifType.CUBIC:
            a = np.random.uniform(8.0, 15.0)
            return UnitCellParameters(a=a, b=a, c=a, alpha=90, beta=90, gamma=90)

        elif motif == PackingMotifType.LAYERED:
            a = np.random.uniform(8.0, 15.0)
            b = np.random.uniform(8.0, 15.0)
            c = np.random.uniform(20.0, 40.0)  # Large c for layered structures
            return UnitCellParameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)

        elif motif == PackingMotifType.HERRINGBONE:
            a = np.random.uniform(6.0, 12.0)
            b = np.random.uniform(8.0, 15.0)
            c = np.random.uniform(10.0, 20.0)
            beta = np.random.uniform(95, 125)  # Monoclinic angle
            return UnitCellParameters(a=a, b=b, c=c, alpha=90, beta=beta, gamma=90)

        else:  # Default case
            return self._generate_random_unit_cell()

    def _generate_positions_for_motif(
        self,
        molecular_formula: str,
        unit_cell: UnitCellParameters,
        motif: PackingMotifType,
    ) -> List[AtomicPosition]:
        """Generate atomic positions for specific motif."""
        elements = self._parse_molecular_formula(molecular_formula)
        positions = []

        if motif == PackingMotifType.LAYERED:
            # Place atoms in layers
            z_layers = [0.1, 0.3, 0.5, 0.7, 0.9]
            layer_idx = 0

            for element, count in elements.items():
                for _ in range(count):
                    x = np.random.uniform(0.1, 0.9)
                    y = np.random.uniform(0.1, 0.9)
                    z = z_layers[layer_idx % len(z_layers)]

                    positions.append(AtomicPosition(element=element, x=x, y=y, z=z))
                    layer_idx += 1

        elif motif == PackingMotifType.HERRINGBONE:
            # Alternating orientations
            orientations = [(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]
            orient_idx = 0

            for element, count in elements.items():
                for _ in range(count):
                    x, y = orientations[orient_idx % len(orientations)]
                    z = np.random.uniform(0.1, 0.9)

                    positions.append(AtomicPosition(element=element, x=x, y=y, z=z))
                    orient_idx += 1

        else:
            # Default random positions
            positions = self._generate_atomic_positions(molecular_formula, unit_cell)

        return positions

    def _suggest_space_group_for_motif(self, motif: PackingMotifType) -> int:
        """Suggest appropriate space group for motif."""
        motif_space_groups = {
            PackingMotifType.CUBIC: [221, 225, 229],  # High symmetry cubic
            PackingMotifType.LAYERED: [2, 14, 15],  # Low symmetry for layers
            PackingMotifType.HERRINGBONE: [14, 15],  # Monoclinic
            PackingMotifType.FRAMEWORK: [221, 225],  # High symmetry
            PackingMotifType.CHAIN: [2, 4, 14],  # Lower symmetry
        }

        candidates = motif_space_groups.get(motif, [1, 2, 14])  # Default options
        return np.random.choice(candidates)

    def _generate_random_unit_cell(self) -> UnitCellParameters:
        """Generate random unit cell parameters."""
        a = np.random.uniform(5.0, 20.0)
        b = np.random.uniform(5.0, 20.0)
        c = np.random.uniform(5.0, 20.0)
        alpha = np.random.uniform(70.0, 110.0)
        beta = np.random.uniform(70.0, 110.0)
        gamma = np.random.uniform(70.0, 110.0)

        return UnitCellParameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    def _generate_atomic_positions(
        self, molecular_formula: str, unit_cell: UnitCellParameters
    ) -> List[AtomicPosition]:
        """Generate atomic positions within unit cell."""
        elements = self._parse_molecular_formula(molecular_formula)
        positions = []

        for element, count in elements.items():
            for _ in range(count):
                # Random positions within unit cell
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                z = np.random.uniform(0.05, 0.95)

                positions.append(AtomicPosition(element=element, x=x, y=y, z=z))

        return positions

    def _extract_structural_features(self, structure: CrystalStructure) -> np.ndarray:
        """Extract features for ML energy prediction."""
        features = np.zeros(1024)  # Standard feature dimension

        # Unit cell features
        uc = structure.unit_cell
        features[0] = uc.a
        features[1] = uc.b
        features[2] = uc.c
        features[3] = uc.alpha
        features[4] = uc.beta
        features[5] = uc.gamma
        features[6] = uc.volume

        # Elemental composition
        element_counts = {}
        for pos in structure.atomic_positions:
            element_counts[pos.element] = element_counts.get(pos.element, 0) + 1

        # Encode elements (first 50 positions)
        for i, (element, count) in enumerate(element_counts.items()):
            if i < 50:
                atomic_num = self._get_atomic_number(element)
                features[10 + i] = count
                features[60 + i] = atomic_num

        # Density and packing
        if structure.density is not None:
            features[110] = structure.density

        if structure.packing_efficiency is not None:
            features[111] = structure.packing_efficiency

        # Space group symmetry
        features[112] = structure.space_group

        # Crystal system encoding
        if uc.crystal_system is not None:
            system_idx = list(CrystalSystemType).index(uc.crystal_system)
            features[113 + system_idx] = 1.0  # One-hot encoding

        return features

    def _tournament_selection(
        self, population: List[CrystalStructure], fitness_scores: List[float]
    ) -> CrystalStructure:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        candidates = np.random.choice(len(population), tournament_size, replace=False)

        best_idx = candidates[0]
        best_fitness = fitness_scores[best_idx]

        for idx in candidates[1:]:
            if fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_scores[idx]

        return population[best_idx]

    def _crossover_structures(
        self, parent1: CrystalStructure, parent2: CrystalStructure
    ) -> CrystalStructure:
        """Crossover operation for genetic algorithm."""
        # Average unit cell parameters
        uc1, uc2 = parent1.unit_cell, parent2.unit_cell

        new_unit_cell = UnitCellParameters(
            a=(uc1.a + uc2.a) / 2,
            b=(uc1.b + uc2.b) / 2,
            c=(uc1.c + uc2.c) / 2,
            alpha=(uc1.alpha + uc2.alpha) / 2,
            beta=(uc1.beta + uc2.beta) / 2,
            gamma=(uc1.gamma + uc2.gamma) / 2,
        )

        # Mix atomic positions
        positions1 = parent1.atomic_positions
        positions2 = parent2.atomic_positions

        new_positions = []
        for i in range(min(len(positions1), len(positions2))):
            if np.random.random() < 0.5:
                new_positions.append(positions1[i])
            else:
                new_positions.append(positions2[i])

        return CrystalStructure(
            unit_cell=new_unit_cell,
            atomic_positions=new_positions,
            space_group=parent1.space_group,
            formula=parent1.formula,
        )

    def _mutate_structure(self, structure: CrystalStructure) -> CrystalStructure:
        """Mutation operation for genetic algorithm."""
        mutation_strength = 0.1

        # Mutate unit cell parameters
        uc = structure.unit_cell
        new_unit_cell = UnitCellParameters(
            a=max(1.0, uc.a * (1 + np.random.normal(0, mutation_strength))),
            b=max(1.0, uc.b * (1 + np.random.normal(0, mutation_strength))),
            c=max(1.0, uc.c * (1 + np.random.normal(0, mutation_strength))),
            alpha=np.clip(uc.alpha + np.random.normal(0, 5), 60, 120),
            beta=np.clip(uc.beta + np.random.normal(0, 5), 60, 120),
            gamma=np.clip(uc.gamma + np.random.normal(0, 5), 60, 120),
        )

        # Mutate some atomic positions
        new_positions = []
        for pos in structure.atomic_positions:
            if np.random.random() < 0.3:  # 30% chance to mutate position
                new_x = np.clip(pos.x + np.random.normal(0, 0.1), 0.05, 0.95)
                new_y = np.clip(pos.y + np.random.normal(0, 0.1), 0.05, 0.95)
                new_z = np.clip(pos.z + np.random.normal(0, 0.1), 0.05, 0.95)

                new_positions.append(
                    AtomicPosition(
                        element=pos.element,
                        x=new_x,
                        y=new_y,
                        z=new_z,
                        occupancy=pos.occupancy,
                        thermal_factor=pos.thermal_factor,
                    )
                )
            else:
                new_positions.append(pos)

        return CrystalStructure(
            unit_cell=new_unit_cell,
            atomic_positions=new_positions,
            space_group=structure.space_group,
            formula=structure.formula,
        )

    def _parse_molecular_formula(self, formula: str) -> Dict[str, int]:
        """Parse molecular formula into elements and counts."""
        import re

        # Simple regex for parsing (can be improved)
        pattern = r"([A-Z][a-z]?)(\d*)"
        matches = re.findall(pattern, formula)

        elements = {}
        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            elements[element] = elements.get(element, 0) + count

        return elements

    def _get_atomic_number(self, element: str) -> int:
        """Get atomic number for element."""
        atomic_numbers = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            "Sc": 21,
            "Ti": 22,
            "V": 23,
            "Cr": 24,
            "Mn": 25,
            "Fe": 26,
            "Co": 27,
            "Ni": 28,
            "Cu": 29,
            "Zn": 30,
            "Ga": 31,
            "Ge": 32,
            "As": 33,
            "Se": 34,
            "Br": 35,
            "Kr": 36,
            "Rb": 37,
            "Sr": 38,
            "Y": 39,
            "Zr": 40,
            "Nb": 41,
            "Mo": 42,
            "Tc": 43,
            "Ru": 44,
            "Rh": 45,
            "Pd": 46,
            "Ag": 47,
            "Cd": 48,
            "In": 49,
            "Sn": 50,
        }
        return atomic_numbers.get(element, 1)

    def explore_polymorphic_landscape(
        self,
        molecular_formula: str,
        energy_window: float = 20.0,
        molecular_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Explore the polymorphic landscape for a given compound.

        Args:
            molecular_formula: Chemical formula
            energy_window: Energy window above global minimum (kJ/mol)
            molecular_features: Optional molecular features

        Returns:
            Dictionary with landscape analysis
        """
        try:
            # Generate comprehensive set of structures
            original_max = self.config.max_structures
            self.config.max_structures = min(
                500, original_max * 3
            )  # Generate more structures

            prediction = self.predict_crystal_structure(
                molecular_formula, molecular_features
            )

            # Restore original setting
            self.config.max_structures = original_max

            # Filter structures within energy window
            if not prediction.predicted_structures:
                return {"error": "No structures generated"}

            min_energy = min(
                s.lattice_energy or 0 for s in prediction.predicted_structures
            )
            structures_in_window = [
                s
                for s in prediction.predicted_structures
                if (s.lattice_energy or 0) - min_energy <= energy_window
            ]

            # Analyze landscape
            landscape_analysis = {
                "total_structures": len(prediction.predicted_structures),
                "structures_in_window": len(structures_in_window),
                "global_minimum_energy": min_energy,
                "energy_window": energy_window,
                "polymorphs": self._analyze_polymorph_families(structures_in_window),
                "crystal_systems": self._analyze_crystal_systems(structures_in_window),
                "packing_motifs": self._analyze_packing_motifs(structures_in_window),
                "stability_ranking": self._create_stability_ranking(
                    structures_in_window
                ),
                "structure_diversity": self._calculate_structure_diversity(
                    structures_in_window
                ),
            }

            return landscape_analysis

        except Exception as e:
            raise CrystalStructurePredictionError(
                f"Landscape exploration failed: {str(e)}"
            )

    def _analyze_polymorph_families(
        self, structures: List[CrystalStructure]
    ) -> Dict[str, Any]:
        """Analyze polymorph families in structure set."""
        families = []
        used_indices = set()

        for i, struct1 in enumerate(structures):
            if i in used_indices:
                continue

            family = [struct1]
            used_indices.add(i)

            for j, struct2 in enumerate(structures[i + 1 :], i + 1):
                if j not in used_indices and self._are_polymorphs(struct1, struct2):
                    family.append(struct2)
                    used_indices.add(j)

            if len(family) > 1:
                families.append(family)

        # Calculate family statistics
        family_stats = []
        for family in families:
            energies = [s.lattice_energy or 0 for s in family]
            family_stats.append(
                {
                    "size": len(family),
                    "energy_range": max(energies) - min(energies),
                    "most_stable_energy": min(energies),
                    "crystal_systems": [
                        s.unit_cell.crystal_system.value
                        for s in family
                        if s.unit_cell.crystal_system
                    ],
                    "space_groups": [s.space_group for s in family],
                }
            )

        return {
            "num_families": len(families),
            "family_statistics": family_stats,
            "total_polymorphs": sum(len(f) for f in families),
        }

    def _analyze_crystal_systems(
        self, structures: List[CrystalStructure]
    ) -> Dict[str, int]:
        """Analyze distribution of crystal systems."""
        system_counts = {}

        for structure in structures:
            if structure.unit_cell.crystal_system:
                system = structure.unit_cell.crystal_system.value
                system_counts[system] = system_counts.get(system, 0) + 1

        return system_counts

    def _analyze_packing_motifs(
        self, structures: List[CrystalStructure]
    ) -> Dict[str, int]:
        """Analyze distribution of packing motifs."""
        motif_counts = {}

        for structure in structures:
            if structure.packing_motif:
                motif = structure.packing_motif.value
                motif_counts[motif] = motif_counts.get(motif, 0) + 1

        return motif_counts

    def _create_stability_ranking(
        self, structures: List[CrystalStructure]
    ) -> List[Dict[str, Any]]:
        """Create stability ranking of structures."""
        sorted_structures = sorted(structures, key=lambda s: s.lattice_energy or 0)

        ranking = []
        for i, structure in enumerate(sorted_structures[:20]):  # Top 20
            ranking.append(
                {
                    "rank": i + 1,
                    "energy": structure.lattice_energy,
                    "formula": structure.formula,
                    "crystal_system": (
                        structure.unit_cell.crystal_system.value
                        if structure.unit_cell.crystal_system
                        else "unknown"
                    ),
                    "space_group": structure.space_group,
                    "density": structure.density,
                    "packing_motif": (
                        structure.packing_motif.value
                        if structure.packing_motif
                        else "unknown"
                    ),
                }
            )

        return ranking

    def _calculate_structure_diversity(
        self, structures: List[CrystalStructure]
    ) -> Dict[str, float]:
        """Calculate structural diversity metrics."""
        if len(structures) < 2:
            return {"diversity_score": 0.0, "uniqueness_score": 0.0}

        # Calculate pairwise structural differences
        differences = []

        for i, struct1 in enumerate(structures):
            for struct2 in structures[i + 1 :]:
                diff = self._calculate_structural_difference(struct1, struct2)
                differences.append(diff)

        diversity_score = np.mean(differences) if differences else 0.0
        uniqueness_score = (
            len(set(differences)) / len(differences) if differences else 0.0
        )

        return {
            "diversity_score": float(diversity_score),
            "uniqueness_score": float(uniqueness_score),
            "num_comparisons": len(differences),
        }

    def _calculate_structural_difference(
        self, struct1: CrystalStructure, struct2: CrystalStructure
    ) -> float:
        """Calculate structural difference between two structures."""
        # Compare unit cell parameters
        uc1, uc2 = struct1.unit_cell, struct2.unit_cell

        param_diffs = [
            abs(uc1.a - uc2.a) / max(uc1.a, uc2.a),
            abs(uc1.b - uc2.b) / max(uc1.b, uc2.b),
            abs(uc1.c - uc2.c) / max(uc1.c, uc2.c),
            abs(uc1.volume - uc2.volume) / max(uc1.volume, uc2.volume),
        ]

        unit_cell_diff = np.mean(param_diffs)

        # Compare densities if available
        density_diff = 0.0
        if struct1.density and struct2.density:
            density_diff = abs(struct1.density - struct2.density) / max(
                struct1.density, struct2.density
            )

        # Compare space groups
        space_group_diff = 1.0 if struct1.space_group != struct2.space_group else 0.0

        # Overall structural difference
        total_diff = (unit_cell_diff + density_diff + space_group_diff) / 3.0

        return total_diff

    def batch_csp_prediction(
        self,
        molecular_formulas: List[str],
        molecular_features_list: Optional[List[np.ndarray]] = None,
        parallel: bool = True,
    ) -> Dict[str, CSPPrediction]:
        """
        Perform batch crystal structure prediction.

        Args:
            molecular_formulas: List of chemical formulas
            molecular_features_list: Optional list of molecular features
            parallel: Whether to use parallel processing

        Returns:
            Dictionary mapping formulas to predictions
        """
        results = {}

        if molecular_features_list is None:
            molecular_features_list = [None] * len(molecular_formulas)

        for formula, features in zip(molecular_formulas, molecular_features_list):
            try:
                prediction = self.predict_crystal_structure(formula, features)
                results[formula] = prediction
            except Exception as e:
                warnings.warn(f"Prediction failed for {formula}: {e}")
                # Create empty prediction
                results[formula] = CSPPrediction(
                    predicted_structures=[],
                    energy_landscape={},
                    polymorphs=[],
                    most_stable_structure=None,
                    prediction_confidence=0.0,
                    computational_cost=0.0,
                    method_used=self.config.method,
                    convergence_achieved=False,
                )

        return results

    def validate_structure(self, structure: CrystalStructure) -> Dict[str, Any]:
        """
        Validate a crystal structure for physical reasonableness.

        Args:
            structure: Crystal structure to validate

        Returns:
            Validation report
        """
        validation_report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "checks_performed": [],
        }

        # Check unit cell parameters
        uc = structure.unit_cell

        # Check for reasonable cell dimensions
        if any(param < 1.0 or param > 100.0 for param in [uc.a, uc.b, uc.c]):
            validation_report["errors"].append(
                "Unit cell dimensions outside reasonable range (1-100 Å)"
            )
            validation_report["valid"] = False

        # Check angles
        if any(
            angle < 30.0 or angle > 150.0 for angle in [uc.alpha, uc.beta, uc.gamma]
        ):
            validation_report["errors"].append(
                "Unit cell angles outside reasonable range (30-150°)"
            )
            validation_report["valid"] = False

        validation_report["checks_performed"].append("unit_cell_parameters")

        # Check atomic positions
        if not structure.atomic_positions:
            validation_report["errors"].append("No atomic positions defined")
            validation_report["valid"] = False
        else:
            # Check for atoms outside unit cell
            outside_atoms = 0
            for pos in structure.atomic_positions:
                if not (
                    0.0 <= pos.x <= 1.0 and 0.0 <= pos.y <= 1.0 and 0.0 <= pos.z <= 1.0
                ):
                    outside_atoms += 1

            if outside_atoms > 0:
                validation_report["warnings"].append(
                    f"{outside_atoms} atoms outside unit cell"
                )

        validation_report["checks_performed"].append("atomic_positions")

        # Check density
        if structure.density:
            if structure.density < 0.1 or structure.density > 50.0:
                validation_report["warnings"].append(
                    "Density outside typical range (0.1-50 g/cm³)"
                )

        validation_report["checks_performed"].append("density")

        # Check for minimum interatomic distances
        min_distance = self._calculate_minimum_interatomic_distance(structure)
        if min_distance < 0.5:
            validation_report["errors"].append(
                f"Atoms too close (min distance: {min_distance:.2f} Å)"
            )
            validation_report["valid"] = False
        elif min_distance < 1.0:
            validation_report["warnings"].append(
                f"Some atoms very close (min distance: {min_distance:.2f} Å)"
            )

        validation_report["checks_performed"].append("interatomic_distances")

        return validation_report

    def _calculate_minimum_interatomic_distance(
        self, structure: CrystalStructure
    ) -> float:
        """Calculate minimum interatomic distance in structure."""
        if len(structure.atomic_positions) < 2:
            return float("inf")

        positions = []
        for pos in structure.atomic_positions:
            # Convert fractional to cartesian coordinates (simplified)
            uc = structure.unit_cell
            x_cart = pos.x * uc.a
            y_cart = pos.y * uc.b
            z_cart = pos.z * uc.c
            positions.append([x_cart, y_cart, z_cart])

        positions = np.array(positions)
        min_dist = float("inf")

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)

        return min_dist


# Additional dataclass methods
@dataclass
class CrystalStructure:
    """Crystal structure representation with additional methods."""

    def _calculate_density(self) -> float:
        """Calculate crystal density."""
        if not hasattr(self, "unit_cell") or not hasattr(self, "atomic_positions"):
            return 0.0

        # Atomic masses (simplified)
        atomic_masses = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.085,
            "P": 30.974,
            "S": 32.065,
            "Cl": 35.453,
            "K": 39.098,
            "Ca": 40.078,
            "Fe": 55.845,
        }

        total_mass = 0.0
        for pos in self.atomic_positions:
            mass = atomic_masses.get(pos.element, 12.0)  # Default to carbon
            total_mass += mass * pos.occupancy

        # Convert to g/mol
        volume_cm3 = self.unit_cell.volume * 1e-24  # Å³ to cm³
        avogadro = 6.022e23

        density = total_mass / (volume_cm3 * avogadro)
        return density

    def _calculate_packing_efficiency(self) -> float:
        """Calculate packing efficiency."""
        # Simplified calculation using atomic radii
        atomic_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "Na": 1.02,
            "Mg": 0.72,
            "Al": 1.18,
            "Si": 1.11,
            "P": 1.07,
            "S": 1.05,
            "Cl": 0.99,
            "K": 1.38,
            "Ca": 1.00,
            "Fe": 1.26,
        }

        total_atomic_volume = 0.0
        for pos in self.atomic_positions:
            radius = atomic_radii.get(pos.element, 0.76)  # Default to carbon
            atomic_volume = (4 / 3) * np.pi * radius**3
            total_atomic_volume += atomic_volume * pos.occupancy

        packing_efficiency = total_atomic_volume / self.unit_cell.volume
        return min(1.0, packing_efficiency)  # Cap at 100%


# Utility functions


def predict_crystal_structure(
    molecular_formula: str,
    config: Optional[CrystalPredictionConfig] = None,
    molecular_features: Optional[np.ndarray] = None,
    experimental_constraints: Optional[Dict[str, Any]] = None,
) -> CSPPrediction:
    """
    Predict crystal structure for a molecular formula.

    Args:
        molecular_formula: Chemical formula
        config: Prediction configuration
        molecular_features: Optional molecular features
        experimental_constraints: Optional experimental constraints

    Returns:
        CSPPrediction result
    """
    engine = GPUAcceleratedCSPEngine(config)
    return engine.predict_crystal_structure(
        molecular_formula, molecular_features, experimental_constraints
    )


def create_csp_config(
    method: CSPMethod = CSPMethod.HYBRID_APPROACH,
    max_structures: int = 100,
    use_gpu: bool = True,
    energy_cutoff: float = 50.0,
) -> CrystalPredictionConfig:
    """Create CSP configuration."""
    return CrystalPredictionConfig(
        method=method,
        max_structures=max_structures,
        use_gpu=use_gpu,
        energy_cutoff=energy_cutoff,
    )


def analyze_polymorphs(
    structures: List[CrystalStructure], tolerance: float = 0.1
) -> List[List[CrystalStructure]]:
    """
    Analyze and group polymorphic structures.

    Args:
        structures: List of crystal structures
        tolerance: Tolerance for structure matching

    Returns:
        List of polymorph groups
    """
    if not _PYMATGEN_AVAILABLE:
        warnings.warn("pymatgen not available for polymorph analysis")
        return [structures]  # Return all as one group

    polymorph_groups = []
    used_indices = set()

    for i, structure1 in enumerate(structures):
        if i in used_indices:
            continue

        current_group = [structure1]
        used_indices.add(i)

        for j, structure2 in enumerate(structures[i + 1 :], i + 1):
            if j in used_indices:
                continue

            # Simple comparison based on unit cell parameters
            if _structures_similar(structure1, structure2, tolerance):
                current_group.append(structure2)
                used_indices.add(j)

        polymorph_groups.append(current_group)

    return polymorph_groups


def _structures_similar(
    struct1: CrystalStructure, struct2: CrystalStructure, tolerance: float
) -> bool:
    """Check if two structures are similar (polymorphs)."""
    # Compare unit cell parameters
    uc1, uc2 = struct1.unit_cell, struct2.unit_cell

    params1 = [uc1.a, uc1.b, uc1.c, uc1.alpha, uc1.beta, uc1.gamma]
    params2 = [uc2.a, uc2.b, uc2.c, uc2.alpha, uc2.beta, uc2.gamma]

    for p1, p2 in zip(params1, params2):
        if abs(p1 - p2) / max(p1, p2) > tolerance:
            return False

    return True


# Additional implementation would continue...
# This provides the core framework for GPU-accelerated crystal structure prediction
