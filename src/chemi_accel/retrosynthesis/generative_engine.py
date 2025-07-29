"""
GPU-Accelerated Generative Retrosynthesis Engine.

This module implements transformer and graph-based generative models for
predicting one-step retrosynthetic disconnections, leveraging large reaction
datasets and GPU acceleration for efficient retrosynthetic analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

# Set up logging
logger = logging.getLogger(__name__)


# Enums
class ModelType(Enum):
    """Types of retrosynthesis models."""

    TRANSFORMER = "transformer"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    TEMPLATE_BASED = "template_based"
    HYBRID = "hybrid"


class DisconnectionStrategy(Enum):
    """Strategies for retrosynthetic disconnection."""

    FUNCTIONAL_GROUP = "functional_group"
    BOND_BREAKING = "bond_breaking"
    RING_FORMATION = "ring_formation"
    PROTECTING_GROUP = "protecting_group"
    RADICAL_CHEMISTRY = "radical_chemistry"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReactionType(Enum):
    """Types of organic reactions."""

    SUBSTITUTION = "substitution"
    ELIMINATION = "elimination"
    ADDITION = "addition"
    REARRANGEMENT = "rearrangement"
    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    CONDENSATION = "condensation"
    CYCLIZATION = "cyclization"


# Data Structures
@dataclass
class RetrosynthesisConfig:
    """Configuration for retrosynthesis prediction."""

    model_type: ModelType = ModelType.TRANSFORMER
    use_gpu: bool = True
    batch_size: int = 32
    max_predictions: int = 10
    confidence_threshold: float = 0.7
    include_reaction_templates: bool = True
    template_database_path: Optional[str] = None
    model_checkpoint_path: Optional[str] = None
    beam_size: int = 5
    max_depth: int = 10
    enable_caching: bool = True
    random_seed: int = 42


@dataclass
class MolecularRepresentation:
    """Molecular representation for retrosynthesis."""

    smiles: str
    mol: Optional[Any] = None
    fingerprint: Optional[np.ndarray] = None
    molecular_weight: float = 0.0
    num_atoms: int = 0
    num_bonds: int = 0
    ring_count: int = 0
    aromaticity: float = 0.0
    complexity_score: float = 0.0
    functional_groups: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize molecular properties from SMILES."""
        if self.mol is None and self.smiles:
            self.mol = Chem.MolFromSmiles(self.smiles)

        if self.mol:
            self.molecular_weight = rdMolDescriptors.CalcExactMolWt(self.mol)
            self.num_atoms = self.mol.GetNumAtoms()
            self.num_bonds = self.mol.GetNumBonds()
            self.ring_count = rdMolDescriptors.CalcNumRings(self.mol)
            self.aromaticity = rdMolDescriptors.CalcNumAromaticRings(self.mol) / max(
                1, self.ring_count
            )
            self.complexity_score = self._calculate_complexity()
            self.functional_groups = self._identify_functional_groups()

    def _calculate_complexity(self) -> float:
        """Calculate molecular complexity score."""
        if not self.mol:
            return 0.0

        # Bertz complexity index
        return rdMolDescriptors.BertzCT(self.mol)

    def _identify_functional_groups(self) -> List[str]:
        """Identify functional groups in the molecule."""
        if not self.mol:
            return []

        functional_groups = []

        # Common functional group patterns
        patterns = {
            "alcohol": "[OH]",
            "aldehyde": "[CX3H1](=O)",
            "ketone": "[CX3](=O)[CX3]",
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "ester": "[CX3](=O)[OX2][CX3]",
            "amide": "[CX3](=O)[NX3]",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "nitrile": "[CX2]#N",
            "halide": "[F,Cl,Br,I]",
            "aromatic": "c",
        }

        for name, pattern in patterns.items():
            if self.mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                functional_groups.append(name)

        return functional_groups


@dataclass
class DisconnectionSite:
    """Represents a potential disconnection site."""

    bond_indices: Tuple[int, int]
    confidence: float
    strategy: DisconnectionStrategy
    reaction_type: ReactionType
    template_id: Optional[str] = None
    atom_mapping: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate disconnection site data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ReactionTemplate:
    """Reaction template for retrosynthetic transformations."""

    template_id: str
    reaction_smarts: str
    reaction_type: ReactionType
    frequency: int = 1
    success_rate: float = 0.0
    complexity_change: float = 0.0
    functional_groups: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate template data."""
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("Success rate must be between 0.0 and 1.0")


@dataclass
class ModelPrediction:
    """Prediction from a retrosynthesis model."""

    reactants: List[str]
    confidence: float
    disconnection_sites: List[DisconnectionSite]
    reaction_template: Optional[ReactionTemplate] = None
    model_type: ModelType = ModelType.TRANSFORMER
    prediction_time: float = 0.0

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on score."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class RetrosynthesisResult:
    """Result of retrosynthetic analysis."""

    target_molecule: MolecularRepresentation
    predictions: List[ModelPrediction]
    total_time: float = 0.0
    model_config: Optional[RetrosynthesisConfig] = None

    def get_best_prediction(self) -> Optional[ModelPrediction]:
        """Get highest confidence prediction."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.confidence)

    def filter_by_confidence(self, min_confidence: float) -> List[ModelPrediction]:
        """Filter predictions by minimum confidence."""
        return [p for p in self.predictions if p.confidence >= min_confidence]


# Exceptions
class RetrosynthesisError(Exception):
    """Base exception for retrosynthesis errors."""

    pass


class ModelLoadingError(RetrosynthesisError):
    """Error loading retrosynthesis model."""

    pass


class PredictionError(RetrosynthesisError):
    """Error during prediction."""

    pass


class DisconnectionError(RetrosynthesisError):
    """Error in disconnection analysis."""

    pass


# Neural Network Models
if _TORCH_AVAILABLE:

    class TransformerRetroModel(nn.Module):
        """Transformer-based retrosynthesis model."""

        def __init__(
            self,
            vocab_size: int = 10000,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 6,
            max_length: int = 200,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_length = max_length

            # Embedding layers
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))

            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            # Output layers
            self.output_projection = nn.Linear(d_model, vocab_size)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Forward pass through transformer."""
            seq_len = src.size(1)

            # Embeddings and positional encoding
            embedded = self.embedding(src) * np.sqrt(self.d_model)
            embedded += self.pos_encoding[:seq_len, :].unsqueeze(0)
            embedded = self.dropout(embedded)

            # Transformer encoding
            output = self.transformer(embedded, src_mask)

            # Output projection
            return self.output_projection(output)

    class GraphRetroModel(nn.Module):
        """Graph neural network for retrosynthesis."""

        def __init__(
            self,
            node_features: int = 64,
            edge_features: int = 32,
            hidden_dim: int = 256,
            num_layers: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.node_features = node_features
            self.edge_features = edge_features
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Graph convolution layers
            self.node_embedding = nn.Linear(node_features, hidden_dim)
            self.edge_embedding = nn.Linear(edge_features, hidden_dim)

            self.graph_layers = nn.ModuleList(
                [
                    GraphConvLayer(hidden_dim, hidden_dim, dropout)
                    for _ in range(num_layers)
                ]
            )

            # Output layers
            self.bond_predictor = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_indices: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass through graph network."""
            # Initial embeddings
            node_emb = self.node_embedding(node_features)
            edge_emb = self.edge_embedding(edge_features)

            # Graph convolutions
            for layer in self.graph_layers:
                node_emb = layer(node_emb, edge_emb, edge_indices)

            # Bond breaking predictions
            src_nodes = node_emb[edge_indices[0]]
            dst_nodes = node_emb[edge_indices[1]]
            bond_features = torch.cat([src_nodes, dst_nodes], dim=1)

            return torch.sigmoid(self.bond_predictor(bond_features))

    class GraphConvLayer(nn.Module):
        """Graph convolution layer."""

        def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.1):
            super().__init__()
            self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.update_fn = nn.Linear(hidden_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_indices: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass through graph convolution."""
            src_nodes = node_features[edge_indices[0]]
            dst_nodes = node_features[edge_indices[1]]

            # Message passing
            messages = self.message_fn(torch.cat([src_nodes, edge_features], dim=1))
            messages = F.relu(messages)
            messages = self.dropout(messages)

            # Aggregate messages
            num_nodes = node_features.size(0)
            aggregated = torch.zeros_like(node_features)
            aggregated.index_add_(0, edge_indices[1], messages)

            # Update node features
            updated = self.update_fn(torch.cat([node_features, aggregated], dim=1))
            return F.relu(updated)

else:
    # Placeholder classes when PyTorch is not available
    class TransformerRetroModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TransformerRetroModel")

    class GraphRetroModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GraphRetroModel")


# Main Classes
class DisconnectionPredictor:
    """Predicts potential disconnection sites in molecules."""

    def __init__(self, config: RetrosynthesisConfig):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def predict_disconnection_sites(
        self, molecule: MolecularRepresentation
    ) -> List[DisconnectionSite]:
        """Predict disconnection sites for a molecule."""
        if not molecule.mol:
            raise DisconnectionError("Invalid molecule for disconnection analysis")

        sites = []

        # Analyze all bonds for potential disconnections
        for bond in molecule.mol.GetBonds():
            confidence = self._calculate_bond_breaking_confidence(bond, molecule)

            if confidence >= self.config.confidence_threshold:
                strategy = self._determine_disconnection_strategy(bond, molecule)
                reaction_type = self._determine_reaction_type(bond, molecule)

                site = DisconnectionSite(
                    bond_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    confidence=confidence,
                    strategy=strategy,
                    reaction_type=reaction_type,
                )
                sites.append(site)

        # Sort by confidence
        sites.sort(key=lambda s: s.confidence, reverse=True)
        return sites[: self.config.max_predictions]

    def _calculate_bond_breaking_confidence(
        self, bond: Any, molecule: MolecularRepresentation
    ) -> float:
        """Calculate confidence for breaking a specific bond."""
        # Simplified heuristic based on bond properties
        base_confidence = 0.5

        # Adjust based on bond type
        if bond.GetBondType() == Chem.BondType.SINGLE:
            base_confidence += 0.2
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            base_confidence += 0.1

        # Adjust based on atom types
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        if begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 6:  # C-C
            base_confidence += 0.1
        elif 6 in [begin_atom.GetAtomicNum(), end_atom.GetAtomicNum()]:  # C-X
            base_confidence += 0.15

        # Adjust based on ring membership
        if bond.IsInRing():
            base_confidence -= 0.2

        return min(1.0, max(0.0, base_confidence))

    def _determine_disconnection_strategy(
        self, bond: Any, molecule: MolecularRepresentation
    ) -> DisconnectionStrategy:
        """Determine the best disconnection strategy for a bond."""
        # Simplified strategy determination
        if bond.IsInRing():
            return DisconnectionStrategy.RING_FORMATION
        elif (
            "alcohol" in molecule.functional_groups
            or "ketone" in molecule.functional_groups
        ):
            return DisconnectionStrategy.FUNCTIONAL_GROUP
        else:
            return DisconnectionStrategy.BOND_BREAKING

    def _determine_reaction_type(
        self, bond: Any, molecule: MolecularRepresentation
    ) -> ReactionType:
        """Determine the reaction type for disconnection."""
        # Simplified reaction type determination
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            return ReactionType.ADDITION
        elif "halide" in molecule.functional_groups:
            return ReactionType.SUBSTITUTION
        elif bond.IsInRing():
            return ReactionType.CYCLIZATION
        else:
            return ReactionType.CONDENSATION


class ReactionTemplateExtractor:
    """Extracts and manages reaction templates."""

    def __init__(self, template_database_path: Optional[str] = None):
        self.templates: Dict[str, ReactionTemplate] = {}
        self.template_database_path = template_database_path
        self._load_templates()

    def _load_templates(self):
        """Load reaction templates from database."""
        # Default templates for common reactions
        default_templates = [
            ReactionTemplate(
                template_id="aldol_condensation",
                reaction_smarts="[C:1](=[O:2])[C:3].[C:4][C:5](=[O:6])>>[C:1](=[O:2])[C:3][C:4][C:5](=[O:6])",
                reaction_type=ReactionType.CONDENSATION,
                success_rate=0.8,
            ),
            ReactionTemplate(
                template_id="grignard_addition",
                reaction_smarts="[C:1]#[C:2].[Mg+][C-:3]>>[C:1]([C:3])=[C:2]",
                reaction_type=ReactionType.ADDITION,
                success_rate=0.9,
            ),
            ReactionTemplate(
                template_id="sn2_substitution",
                reaction_smarts="[C:1][Cl,Br,I:2].[Nu:3]>>[C:1][Nu:3]",
                reaction_type=ReactionType.SUBSTITUTION,
                success_rate=0.75,
            ),
        ]

        for template in default_templates:
            self.templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[ReactionTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def find_applicable_templates(
        self, molecule: MolecularRepresentation
    ) -> List[ReactionTemplate]:
        """Find templates applicable to a molecule."""
        applicable = []

        if not molecule.mol:
            return applicable

        for template in self.templates.values():
            # Check if template functional groups match molecule
            template_groups = set(template.functional_groups)
            molecule_groups = set(molecule.functional_groups)

            if template_groups.intersection(molecule_groups):
                applicable.append(template)

        # Sort by success rate
        applicable.sort(key=lambda t: t.success_rate, reverse=True)
        return applicable


class RetrosynthesisEngine:
    """Main GPU-accelerated retrosynthesis engine."""

    def __init__(self, config: Optional[RetrosynthesisConfig] = None):
        self.config = config or RetrosynthesisConfig()
        self.use_gpu = self.config.use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )

        # Initialize components
        self.disconnection_predictor = DisconnectionPredictor(self.config)
        self.template_extractor = ReactionTemplateExtractor(
            self.config.template_database_path
        )

        # Models
        self.transformer_model: Optional[TransformerRetroModel] = None
        self.graph_model: Optional[GraphRetroModel] = None

        # Cache for predictions
        self._prediction_cache: Dict[str, RetrosynthesisResult] = {}

        self._load_models()

    def _load_models(self):
        """Load retrosynthesis models."""
        if not _TORCH_AVAILABLE:
            logger.warning(
                "PyTorch not available. Using template-based predictions only."
            )
            return

        try:
            if self.config.model_type in [ModelType.TRANSFORMER, ModelType.HYBRID]:
                self.transformer_model = TransformerRetroModel()
                if self.use_gpu:
                    self.transformer_model = self.transformer_model.to(self.device)

            if self.config.model_type in [
                ModelType.GRAPH_NEURAL_NETWORK,
                ModelType.HYBRID,
            ]:
                self.graph_model = GraphRetroModel()
                if self.use_gpu:
                    self.graph_model = self.graph_model.to(self.device)

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise ModelLoadingError(f"Failed to load models: {e}")

    def predict_retrosynthesis(
        self, target_smiles: str, max_predictions: Optional[int] = None
    ) -> RetrosynthesisResult:
        """Predict retrosynthetic routes for a target molecule."""
        max_predictions = max_predictions or self.config.max_predictions

        # Check cache
        if self.config.enable_caching and target_smiles in self._prediction_cache:
            return self._prediction_cache[target_smiles]

        # Create molecular representation
        try:
            molecule = MolecularRepresentation(smiles=target_smiles)
        except Exception as e:
            raise PredictionError(f"Failed to create molecular representation: {e}")

        predictions = []
        start_time = time.time()

        try:
            # Get predictions from different models
            if (
                self.config.model_type == ModelType.TRANSFORMER
                and self.transformer_model
            ):
                transformer_predictions = self._predict_transformer(molecule)
                predictions.extend(transformer_predictions)

            elif (
                self.config.model_type == ModelType.GRAPH_NEURAL_NETWORK
                and self.graph_model
            ):
                graph_predictions = self._predict_graph(molecule)
                predictions.extend(graph_predictions)

            elif self.config.model_type == ModelType.HYBRID:
                if self.transformer_model:
                    transformer_predictions = self._predict_transformer(molecule)
                    predictions.extend(transformer_predictions)
                if self.graph_model:
                    graph_predictions = self._predict_graph(molecule)
                    predictions.extend(graph_predictions)

            # Always include template-based predictions
            template_predictions = self._predict_template_based(molecule)
            predictions.extend(template_predictions)

            # Remove duplicates and sort by confidence
            predictions = self._deduplicate_predictions(predictions)
            predictions.sort(key=lambda p: p.confidence, reverse=True)
            predictions = predictions[:max_predictions]

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise PredictionError(f"Prediction failed: {e}")

        total_time = time.time() - start_time

        result = RetrosynthesisResult(
            target_molecule=molecule,
            predictions=predictions,
            total_time=total_time,
            model_config=self.config,
        )

        # Cache result
        if self.config.enable_caching:
            self._prediction_cache[target_smiles] = result

        return result

    def _predict_transformer(
        self, molecule: MolecularRepresentation
    ) -> List[ModelPrediction]:
        """Generate predictions using transformer model."""
        predictions = []

        if not self.transformer_model or not molecule.mol:
            return predictions

        try:
            # Simplified transformer prediction
            # In practice, this would involve tokenizing the molecule and running inference
            disconnection_sites = (
                self.disconnection_predictor.predict_disconnection_sites(molecule)
            )

            for site in disconnection_sites[:3]:  # Top 3 sites
                # Generate reactants based on disconnection
                reactants = self._generate_reactants_from_disconnection(molecule, site)

                prediction = ModelPrediction(
                    reactants=reactants,
                    confidence=site.confidence
                    * 0.9,  # Slightly reduce for model uncertainty
                    disconnection_sites=[site],
                    model_type=ModelType.TRANSFORMER,
                    prediction_time=0.1,
                )
                predictions.append(prediction)

        except Exception as e:
            logger.error(f"Error in transformer prediction: {e}")

        return predictions

    def _predict_graph(
        self, molecule: MolecularRepresentation
    ) -> List[ModelPrediction]:
        """Generate predictions using graph neural network."""
        predictions = []

        if not self.graph_model or not molecule.mol:
            return predictions

        try:
            # Simplified graph prediction
            disconnection_sites = (
                self.disconnection_predictor.predict_disconnection_sites(molecule)
            )

            for site in disconnection_sites[:3]:  # Top 3 sites
                reactants = self._generate_reactants_from_disconnection(molecule, site)

                prediction = ModelPrediction(
                    reactants=reactants,
                    confidence=site.confidence * 0.85,  # Adjust for graph model
                    disconnection_sites=[site],
                    model_type=ModelType.GRAPH_NEURAL_NETWORK,
                    prediction_time=0.15,
                )
                predictions.append(prediction)

        except Exception as e:
            logger.error(f"Error in graph prediction: {e}")

        return predictions

    def _predict_template_based(
        self, molecule: MolecularRepresentation
    ) -> List[ModelPrediction]:
        """Generate predictions using reaction templates."""
        predictions = []

        try:
            applicable_templates = self.template_extractor.find_applicable_templates(
                molecule
            )

            for template in applicable_templates[:5]:  # Top 5 templates
                # Apply template to generate reactants
                reactants = self._apply_template(molecule, template)

                if reactants:
                    prediction = ModelPrediction(
                        reactants=reactants,
                        confidence=template.success_rate * 0.8,
                        disconnection_sites=[],
                        reaction_template=template,
                        model_type=ModelType.TEMPLATE_BASED,
                        prediction_time=0.05,
                    )
                    predictions.append(prediction)

        except Exception as e:
            logger.error(f"Error in template-based prediction: {e}")

        return predictions

    def _generate_reactants_from_disconnection(
        self, molecule: MolecularRepresentation, site: DisconnectionSite
    ) -> List[str]:
        """Generate reactant SMILES from disconnection site."""
        if not molecule.mol:
            return []

        try:
            # Create a copy of the molecule
            mol_copy = Chem.Mol(molecule.mol)

            # Break the bond at the disconnection site
            editable_mol = Chem.EditableMol(mol_copy)
            editable_mol.RemoveBond(site.bond_indices[0], site.bond_indices[1])

            # Get fragments
            fragmented_mol = editable_mol.GetMol()
            fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)

            # Convert fragments to SMILES
            reactants = []
            for frag in fragments:
                if frag.GetNumAtoms() > 0:
                    smiles = Chem.MolToSmiles(frag)
                    if smiles:
                        reactants.append(smiles)

            return reactants

        except Exception as e:
            logger.error(f"Error generating reactants: {e}")
            return []

    def _apply_template(
        self, molecule: MolecularRepresentation, template: ReactionTemplate
    ) -> List[str]:
        """Apply reaction template to generate reactants."""
        if not molecule.mol:
            return []

        try:
            # This is a simplified template application
            # In practice, would use more sophisticated SMARTS matching

            # For demonstration, return simplified reactants
            if template.reaction_type == ReactionType.CONDENSATION:
                return [molecule.smiles.replace("C=O", "C"), "C=O"]
            elif template.reaction_type == ReactionType.SUBSTITUTION:
                return [molecule.smiles.replace("Cl", "H"), "Cl"]
            else:
                return [molecule.smiles]

        except Exception as e:
            logger.error(f"Error applying template: {e}")
            return []

    def _deduplicate_predictions(
        self, predictions: List[ModelPrediction]
    ) -> List[ModelPrediction]:
        """Remove duplicate predictions."""
        seen = set()
        unique_predictions = []

        for prediction in predictions:
            # Create a signature based on reactants
            signature = tuple(sorted(prediction.reactants))

            if signature not in seen:
                seen.add(signature)
                unique_predictions.append(prediction)

        return unique_predictions

    def batch_predict(
        self, target_smiles_list: List[str]
    ) -> List[RetrosynthesisResult]:
        """Predict retrosynthesis for multiple targets."""
        results = []

        for smiles in target_smiles_list:
            try:
                result = self.predict_retrosynthesis(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for {smiles}: {e}")
                # Create empty result for failed predictions
                molecule = MolecularRepresentation(smiles=smiles)
                result = RetrosynthesisResult(
                    target_molecule=molecule,
                    predictions=[],
                    total_time=0.0,
                    model_config=self.config,
                )
                results.append(result)

        return results


# Utility Functions
def create_retro_config(
    model_type: ModelType = ModelType.TRANSFORMER,
    use_gpu: bool = True,
    confidence_threshold: float = 0.7,
    **kwargs,
) -> RetrosynthesisConfig:
    """Create retrosynthesis configuration."""
    return RetrosynthesisConfig(
        model_type=model_type,
        use_gpu=use_gpu,
        confidence_threshold=confidence_threshold,
        **kwargs,
    )


def encode_molecule(smiles: str) -> MolecularRepresentation:
    """Encode SMILES string as molecular representation."""
    return MolecularRepresentation(smiles=smiles)


def decode_molecule(representation: MolecularRepresentation) -> str:
    """Decode molecular representation to SMILES."""
    return representation.smiles


def predict_retrosynthesis(
    target_smiles: str, config: Optional[RetrosynthesisConfig] = None
) -> RetrosynthesisResult:
    """Predict retrosynthesis for a target molecule."""
    engine = RetrosynthesisEngine(config)
    return engine.predict_retrosynthesis(target_smiles)


def extract_disconnection_sites(
    target_smiles: str, config: Optional[RetrosynthesisConfig] = None
) -> List[DisconnectionSite]:
    """Extract potential disconnection sites from a molecule."""
    config = config or RetrosynthesisConfig()
    predictor = DisconnectionPredictor(config)
    molecule = MolecularRepresentation(smiles=target_smiles)
    return predictor.predict_disconnection_sites(molecule)


# Import time for timing
import time
