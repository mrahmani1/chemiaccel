"""
Graph Neural Network Features for ChemiAccel.

This module implements GROVER-inspired graph neural network features including
molecular graph embeddings, graph convolution operations, attention mechanisms,
and hierarchical graph pooling operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
    JaxArray = jnp.ndarray
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jit = None
    vmap = None
    # Create a dummy type for type hints when JAX is not available
    from typing import Any

    JaxArray = Any

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from ..descriptors.graph import MolecularGraphBuilder
from ..utils.logging import get_logger

logger = get_logger()


class GraphEmbeddingBase(ABC):
    """Abstract base class for graph embedding implementations."""

    @abstractmethod
    def extract_node_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract node features from molecular graph."""
        pass

    @abstractmethod
    def extract_edge_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract edge features from molecular graph."""
        pass

    @abstractmethod
    def compute_graph_convolution(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute graph convolution operation."""
        pass

    @abstractmethod
    def apply_attention(
        self, features: np.ndarray, attention_weights: np.ndarray
    ) -> np.ndarray:
        """Apply attention mechanism to features."""
        pass

    @abstractmethod
    def hierarchical_pooling(
        self, node_features: np.ndarray, pooling_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply hierarchical graph pooling."""
        pass


class CPUGraphEmbedding(GraphEmbeddingBase):
    """CPU implementation of graph embedding operations."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        """Initialize CPU graph embedding.

        Args:
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of graph convolution layers
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_builder = MolecularGraphBuilder()

        # Initialize learnable parameters (simplified for CPU)
        self.node_embedding_weights = np.random.normal(
            0, 0.1, (200, hidden_dim)
        )  # Assume max 200 atom features
        self.edge_embedding_weights = np.random.normal(
            0, 0.1, (50, hidden_dim)
        )  # Assume max 50 edge features
        self.conv_weights = [
            np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ]
        self.attention_weights = np.random.normal(0, 0.1, (hidden_dim, 1))

        logger.debug(
            f"CPUGraphEmbedding initialized (hidden_dim={hidden_dim}, num_layers={num_layers})"
        )

    def extract_node_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed node features."""
        atom_features = graph_data["atom_features"]
        if atom_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Pad or truncate features to match embedding weights
        num_atoms, num_features = atom_features.shape
        if num_features > self.node_embedding_weights.shape[0]:
            atom_features = atom_features[:, : self.node_embedding_weights.shape[0]]
        elif num_features < self.node_embedding_weights.shape[0]:
            padding = np.zeros(
                (num_atoms, self.node_embedding_weights.shape[0] - num_features)
            )
            atom_features = np.concatenate([atom_features, padding], axis=1)

        # Linear embedding
        node_embeddings = np.dot(atom_features, self.node_embedding_weights)
        return node_embeddings

    def extract_edge_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed edge features."""
        bond_features = graph_data["bond_features"]
        if bond_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Pad or truncate features to match embedding weights
        num_bonds, num_features = bond_features.shape
        if num_features > self.edge_embedding_weights.shape[0]:
            bond_features = bond_features[:, : self.edge_embedding_weights.shape[0]]
        elif num_features < self.edge_embedding_weights.shape[0]:
            padding = np.zeros(
                (num_bonds, self.edge_embedding_weights.shape[0] - num_features)
            )
            bond_features = np.concatenate([bond_features, padding], axis=1)

        # Linear embedding
        edge_embeddings = np.dot(bond_features, self.edge_embedding_weights)
        return edge_embeddings

    def compute_graph_convolution(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute graph convolution with message passing."""
        if node_features.size == 0:
            return node_features

        num_atoms = node_features.shape[0]
        updated_features = node_features.copy()

        # Apply multiple graph convolution layers
        for layer_idx in range(self.num_layers):
            layer_weights = self.conv_weights[layer_idx]
            new_features = np.zeros_like(updated_features)

            # Message passing: aggregate neighbor features
            for i in range(num_atoms):
                # Self-connection
                new_features[i] += np.dot(updated_features[i], layer_weights)

                # Neighbor connections
                neighbors = np.where(adjacency_matrix[i] == 1)[0]
                if len(neighbors) > 0:
                    neighbor_messages = np.mean(updated_features[neighbors], axis=0)
                    new_features[i] += np.dot(neighbor_messages, layer_weights)

            # Apply activation (ReLU)
            updated_features = np.maximum(0, new_features)

        return updated_features

    def apply_attention(
        self, features: np.ndarray, attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply attention mechanism for substructure importance weighting."""
        if features.size == 0:
            return features

        if attention_weights is None:
            attention_weights = self.attention_weights

        # Compute attention scores
        attention_scores = np.dot(features, attention_weights).flatten()
        attention_probs = self._softmax(attention_scores)

        # Apply attention weights
        weighted_features = features * attention_probs.reshape(-1, 1)
        return weighted_features

    def hierarchical_pooling(
        self, node_features: np.ndarray, pooling_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply hierarchical graph pooling operations."""
        if node_features.size == 0:
            return np.zeros((1, self.hidden_dim))

        # Global pooling operations
        pooled_features = []

        # Mean pooling
        mean_pool = np.mean(node_features, axis=0)
        pooled_features.append(mean_pool)

        # Max pooling
        max_pool = np.max(node_features, axis=0)
        pooled_features.append(max_pool)

        # Sum pooling
        sum_pool = np.sum(node_features, axis=0)
        pooled_features.append(sum_pool)

        # Attention-weighted pooling
        attention_features = self.apply_attention(node_features)
        attention_pool = np.sum(attention_features, axis=0)
        pooled_features.append(attention_pool)

        # Concatenate all pooling results
        graph_embedding = np.concatenate(pooled_features)
        return graph_embedding

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)


class CUDAGraphEmbedding(GraphEmbeddingBase):
    """CUDA implementation of graph embedding operations."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        """Initialize CUDA graph embedding.

        Args:
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of graph convolution layers
        """
        if not CUPY_AVAILABLE:
            raise ChemiAccelError(
                "CuPy is required for CUDA graph embedding operations"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_builder = MolecularGraphBuilder()

        # Initialize learnable parameters on GPU
        self.node_embedding_weights = cp.random.normal(
            0, 0.1, (200, hidden_dim), dtype=cp.float32
        )
        self.edge_embedding_weights = cp.random.normal(
            0, 0.1, (50, hidden_dim), dtype=cp.float32
        )
        self.conv_weights = [
            cp.random.normal(0, 0.1, (hidden_dim, hidden_dim), dtype=cp.float32)
            for _ in range(num_layers)
        ]
        self.attention_weights = cp.random.normal(
            0, 0.1, (hidden_dim, 1), dtype=cp.float32
        )

        logger.debug(
            f"CUDAGraphEmbedding initialized (hidden_dim={hidden_dim}, num_layers={num_layers})"
        )

    def extract_node_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed node features using CUDA."""
        atom_features = graph_data["atom_features"]
        if atom_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Convert to CuPy array
        atom_features_gpu = cp.asarray(atom_features, dtype=cp.float32)

        # Pad or truncate features
        num_atoms, num_features = atom_features_gpu.shape
        if num_features > self.node_embedding_weights.shape[0]:
            atom_features_gpu = atom_features_gpu[
                :, : self.node_embedding_weights.shape[0]
            ]
        elif num_features < self.node_embedding_weights.shape[0]:
            padding = cp.zeros(
                (num_atoms, self.node_embedding_weights.shape[0] - num_features),
                dtype=cp.float32,
            )
            atom_features_gpu = cp.concatenate([atom_features_gpu, padding], axis=1)

        # Linear embedding on GPU
        node_embeddings_gpu = cp.dot(atom_features_gpu, self.node_embedding_weights)

        # Convert back to CPU
        return cp.asnumpy(node_embeddings_gpu)

    def extract_edge_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed edge features using CUDA."""
        bond_features = graph_data["bond_features"]
        if bond_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Convert to CuPy array
        bond_features_gpu = cp.asarray(bond_features, dtype=cp.float32)

        # Pad or truncate features
        num_bonds, num_features = bond_features_gpu.shape
        if num_features > self.edge_embedding_weights.shape[0]:
            bond_features_gpu = bond_features_gpu[
                :, : self.edge_embedding_weights.shape[0]
            ]
        elif num_features < self.edge_embedding_weights.shape[0]:
            padding = cp.zeros(
                (num_bonds, self.edge_embedding_weights.shape[0] - num_features),
                dtype=cp.float32,
            )
            bond_features_gpu = cp.concatenate([bond_features_gpu, padding], axis=1)

        # Linear embedding on GPU
        edge_embeddings_gpu = cp.dot(bond_features_gpu, self.edge_embedding_weights)

        # Convert back to CPU
        return cp.asnumpy(edge_embeddings_gpu)

    def compute_graph_convolution(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute graph convolution using CUDA acceleration."""
        if node_features.size == 0:
            return node_features

        # Convert to GPU
        node_features_gpu = cp.asarray(node_features, dtype=cp.float32)
        adjacency_gpu = cp.asarray(adjacency_matrix, dtype=cp.float32)

        updated_features_gpu = node_features_gpu.copy()

        # Apply multiple graph convolution layers
        for layer_idx in range(self.num_layers):
            layer_weights = self.conv_weights[layer_idx]

            # Self-connection transformation
            self_features = cp.dot(updated_features_gpu, layer_weights)

            # Neighbor aggregation using matrix multiplication
            neighbor_features = cp.dot(adjacency_gpu, updated_features_gpu)
            neighbor_features = cp.dot(neighbor_features, layer_weights)

            # Combine self and neighbor features
            new_features = self_features + neighbor_features

            # Apply ReLU activation
            updated_features_gpu = cp.maximum(0, new_features)

        # Convert back to CPU
        return cp.asnumpy(updated_features_gpu)

    def apply_attention(
        self, features: np.ndarray, attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply attention mechanism using CUDA."""
        if features.size == 0:
            return features

        # Convert to GPU
        features_gpu = cp.asarray(features, dtype=cp.float32)

        if attention_weights is None:
            attention_weights_gpu = self.attention_weights
        else:
            attention_weights_gpu = cp.asarray(attention_weights, dtype=cp.float32)

        # Compute attention scores
        attention_scores = cp.dot(features_gpu, attention_weights_gpu).flatten()
        attention_probs = self._softmax_cuda(attention_scores)

        # Apply attention weights
        weighted_features = features_gpu * attention_probs.reshape(-1, 1)

        # Convert back to CPU
        return cp.asnumpy(weighted_features)

    def hierarchical_pooling(
        self, node_features: np.ndarray, pooling_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply hierarchical graph pooling using CUDA."""
        if node_features.size == 0:
            return np.zeros((1, self.hidden_dim))

        # Convert to GPU
        node_features_gpu = cp.asarray(node_features, dtype=cp.float32)

        # Global pooling operations on GPU
        pooled_features = []

        # Mean pooling
        mean_pool = cp.mean(node_features_gpu, axis=0)
        pooled_features.append(mean_pool)

        # Max pooling
        max_pool = cp.max(node_features_gpu, axis=0)
        pooled_features.append(max_pool)

        # Sum pooling
        sum_pool = cp.sum(node_features_gpu, axis=0)
        pooled_features.append(sum_pool)

        # Attention-weighted pooling
        attention_features = cp.asarray(
            self.apply_attention(cp.asnumpy(node_features_gpu)), dtype=cp.float32
        )
        attention_pool = cp.sum(attention_features, axis=0)
        pooled_features.append(attention_pool)

        # Concatenate all pooling results
        graph_embedding = cp.concatenate(pooled_features)

        # Convert back to CPU
        return cp.asnumpy(graph_embedding)

    def _softmax_cuda(self, x):
        """Compute softmax activation on GPU."""
        exp_x = cp.exp(x - cp.max(x))  # Numerical stability
        return exp_x / cp.sum(exp_x)


class JAXGraphEmbedding(GraphEmbeddingBase):
    """JAX implementation of graph embedding operations."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        """Initialize JAX graph embedding.

        Args:
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of graph convolution layers
        """
        if not JAX_AVAILABLE:
            raise ChemiAccelError("JAX is required for JAX graph embedding operations")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_builder = MolecularGraphBuilder()

        # Initialize learnable parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.node_embedding_weights = jax.random.normal(key1, (200, hidden_dim)) * 0.1
        self.edge_embedding_weights = jax.random.normal(key2, (50, hidden_dim)) * 0.1
        self.conv_weights = [
            jax.random.normal(
                jax.random.split(key3, num_layers)[i], (hidden_dim, hidden_dim)
            )
            * 0.1
            for i in range(num_layers)
        ]
        self.attention_weights = jax.random.normal(key4, (hidden_dim, 1)) * 0.1

        # JIT compile functions for performance
        self._jit_extract_node_features = jit(self._extract_node_features_impl)
        self._jit_extract_edge_features = jit(self._extract_edge_features_impl)
        self._jit_graph_convolution = jit(self._graph_convolution_impl)
        self._jit_attention = jit(self._attention_impl)
        self._jit_pooling = jit(self._pooling_impl)

        logger.debug(
            f"JAXGraphEmbedding initialized (hidden_dim={hidden_dim}, num_layers={num_layers})"
        )

    def extract_node_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed node features using JAX."""
        atom_features = graph_data["atom_features"]
        if atom_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Convert to JAX array
        atom_features_jax = jnp.asarray(atom_features, dtype=jnp.float32)

        # Apply JIT-compiled function
        node_embeddings = self._jit_extract_node_features(
            atom_features_jax, self.node_embedding_weights
        )

        # Convert back to numpy
        return np.asarray(node_embeddings)

    def extract_edge_features(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Extract and embed edge features using JAX."""
        bond_features = graph_data["bond_features"]
        if bond_features.size == 0:
            return np.zeros((0, self.hidden_dim))

        # Convert to JAX array
        bond_features_jax = jnp.asarray(bond_features, dtype=jnp.float32)

        # Apply JIT-compiled function
        edge_embeddings = self._jit_extract_edge_features(
            bond_features_jax, self.edge_embedding_weights
        )

        # Convert back to numpy
        return np.asarray(edge_embeddings)

    def compute_graph_convolution(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute graph convolution using JAX."""
        if node_features.size == 0:
            return node_features

        # Convert to JAX arrays
        node_features_jax = jnp.asarray(node_features, dtype=jnp.float32)
        adjacency_jax = jnp.asarray(adjacency_matrix, dtype=jnp.float32)

        # Apply JIT-compiled function
        updated_features = self._jit_graph_convolution(
            node_features_jax, adjacency_jax, self.conv_weights
        )

        # Convert back to numpy
        return np.asarray(updated_features)

    def apply_attention(
        self, features: np.ndarray, attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply attention mechanism using JAX."""
        if features.size == 0:
            return features

        # Convert to JAX array
        features_jax = jnp.asarray(features, dtype=jnp.float32)

        if attention_weights is None:
            attention_weights_jax = self.attention_weights
        else:
            attention_weights_jax = jnp.asarray(attention_weights, dtype=jnp.float32)

        # Apply JIT-compiled function
        weighted_features = self._jit_attention(features_jax, attention_weights_jax)

        # Convert back to numpy
        return np.asarray(weighted_features)

    def hierarchical_pooling(
        self, node_features: np.ndarray, pooling_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply hierarchical graph pooling using JAX."""
        if node_features.size == 0:
            return np.zeros((1, self.hidden_dim))

        # Convert to JAX array
        node_features_jax = jnp.asarray(node_features, dtype=jnp.float32)

        # Apply JIT-compiled function
        graph_embedding = self._jit_pooling(node_features_jax, self.attention_weights)

        # Convert back to numpy
        return np.asarray(graph_embedding)

    def _extract_node_features_impl(
        self, atom_features: JaxArray, embedding_weights: JaxArray
    ) -> JaxArray:
        """JAX implementation of node feature extraction."""
        # Pad or truncate features
        num_atoms, num_features = atom_features.shape
        max_features = embedding_weights.shape[0]

        if num_features > max_features:
            atom_features = atom_features[:, :max_features]
        elif num_features < max_features:
            padding = jnp.zeros((num_atoms, max_features - num_features))
            atom_features = jnp.concatenate([atom_features, padding], axis=1)

        # Linear embedding
        return jnp.dot(atom_features, embedding_weights)

    def _extract_edge_features_impl(
        self, bond_features: JaxArray, embedding_weights: JaxArray
    ) -> JaxArray:
        """JAX implementation of edge feature extraction."""
        # Pad or truncate features
        num_bonds, num_features = bond_features.shape
        max_features = embedding_weights.shape[0]

        if num_features > max_features:
            bond_features = bond_features[:, :max_features]
        elif num_features < max_features:
            padding = jnp.zeros((num_bonds, max_features - num_features))
            bond_features = jnp.concatenate([bond_features, padding], axis=1)

        # Linear embedding
        return jnp.dot(bond_features, embedding_weights)

    def _graph_convolution_impl(
        self,
        node_features: JaxArray,
        adjacency_matrix: JaxArray,
        conv_weights: List[JaxArray],
    ) -> JaxArray:
        """JAX implementation of graph convolution."""
        updated_features = node_features

        for layer_weights in conv_weights:
            # Self-connection
            self_features = jnp.dot(updated_features, layer_weights)

            # Neighbor aggregation
            neighbor_features = jnp.dot(adjacency_matrix, updated_features)
            neighbor_features = jnp.dot(neighbor_features, layer_weights)

            # Combine and activate
            new_features = self_features + neighbor_features
            updated_features = jnp.maximum(0, new_features)  # ReLU

        return updated_features

    def _attention_impl(
        self, features: JaxArray, attention_weights: JaxArray
    ) -> JaxArray:
        """JAX implementation of attention mechanism."""
        # Compute attention scores
        attention_scores = jnp.dot(features, attention_weights).flatten()
        attention_probs = jax.nn.softmax(attention_scores)

        # Apply attention weights
        return features * attention_probs.reshape(-1, 1)

    def _pooling_impl(
        self, node_features: JaxArray, attention_weights: JaxArray
    ) -> JaxArray:
        """JAX implementation of hierarchical pooling."""
        # Global pooling operations
        mean_pool = jnp.mean(node_features, axis=0)
        max_pool = jnp.max(node_features, axis=0)
        sum_pool = jnp.sum(node_features, axis=0)

        # Attention-weighted pooling
        attention_features = self._attention_impl(node_features, attention_weights)
        attention_pool = jnp.sum(attention_features, axis=0)

        # Concatenate all pooling results
        return jnp.concatenate([mean_pool, max_pool, sum_pool, attention_pool])


class MolecularGraphEmbedding:
    """Main class for molecular graph embeddings with backend selection."""

    def __init__(
        self, backend: str = "auto", hidden_dim: int = 128, num_layers: int = 3
    ):
        """Initialize molecular graph embedding.

        Args:
            backend: Backend to use ("cpu", "cuda", "jax", or "auto")
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of graph convolution layers
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for molecular graph embedding")

        self.backend = backend
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Select backend implementation
        if backend == "auto":
            backend = self._select_best_backend()

        if backend == "cuda" and CUPY_AVAILABLE:
            self.embedding_impl = CUDAGraphEmbedding(hidden_dim, num_layers)
        elif backend == "jax" and JAX_AVAILABLE:
            self.embedding_impl = JAXGraphEmbedding(hidden_dim, num_layers)
        else:
            self.embedding_impl = CPUGraphEmbedding(hidden_dim, num_layers)
            backend = "cpu"

        self.actual_backend = backend
        logger.info(f"MolecularGraphEmbedding initialized with {backend} backend")

    def _select_best_backend(self) -> str:
        """Select the best available backend."""
        if CUPY_AVAILABLE:
            return "cuda"
        elif JAX_AVAILABLE:
            return "jax"
        else:
            return "cpu"

    def compute_embeddings(self, mol_data: MolecularData) -> Dict[str, np.ndarray]:
        """Compute graph embeddings for a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary containing various embeddings
        """
        try:
            # Build molecular graph
            graph_data = self.embedding_impl.graph_builder.build_molecular_graph(
                mol_data
            )

            # Extract features
            node_features = self.embedding_impl.extract_node_features(graph_data)
            edge_features = self.embedding_impl.extract_edge_features(graph_data)

            # Compute graph convolution
            conv_features = self.embedding_impl.compute_graph_convolution(
                node_features, edge_features, graph_data["adjacency_matrix"]
            )

            # Apply attention
            attention_features = self.embedding_impl.apply_attention(conv_features)

            # Hierarchical pooling
            graph_embedding = self.embedding_impl.hierarchical_pooling(
                attention_features
            )

            return {
                "node_features": node_features,
                "edge_features": edge_features,
                "conv_features": conv_features,
                "attention_features": attention_features,
                "graph_embedding": graph_embedding,
                "num_atoms": graph_data["num_atoms"],
                "num_bonds": graph_data["num_bonds"],
            }

        except Exception as e:
            logger.error(
                f"Failed to compute embeddings for molecule {mol_data.mol_id}: {e}"
            )
            return {
                "node_features": np.array([]),
                "edge_features": np.array([]),
                "conv_features": np.array([]),
                "attention_features": np.array([]),
                "graph_embedding": np.zeros(
                    self.hidden_dim * 4
                ),  # 4 pooling operations
                "num_atoms": 0,
                "num_bonds": 0,
            }

    def compute_batch_embeddings(
        self, molecules: List[MolecularData]
    ) -> List[Dict[str, np.ndarray]]:
        """Compute embeddings for a batch of molecules.

        Args:
            molecules: List of MolecularData objects

        Returns:
            List of embedding dictionaries
        """
        embeddings = []
        for mol_data in molecules:
            embedding = self.compute_embeddings(mol_data)
            embeddings.append(embedding)

        return embeddings

    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get dimensions of different embedding types.

        Returns:
            Dictionary with embedding dimensions
        """
        return {
            "node_features": self.hidden_dim,
            "edge_features": self.hidden_dim,
            "conv_features": self.hidden_dim,
            "attention_features": self.hidden_dim,
            "graph_embedding": self.hidden_dim * 4,  # 4 pooling operations
        }


class AdvancedGraphFeatures:
    """Advanced graph-based features including learned representations and subgraph analysis."""

    def __init__(self, backend: str = "auto"):
        """Initialize advanced graph features.

        Args:
            backend: Backend to use ("cpu", "cuda", "jax", or "auto")
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for advanced graph features")

        self.backend = backend
        self.graph_builder = MolecularGraphBuilder()

        # Select backend implementation
        if backend == "auto":
            backend = self._select_best_backend()

        self.actual_backend = backend
        logger.info(f"AdvancedGraphFeatures initialized with {backend} backend")

    def _select_best_backend(self) -> str:
        """Select the best available backend."""
        if CUPY_AVAILABLE:
            return "cuda"
        elif JAX_AVAILABLE:
            return "jax"
        else:
            return "cpu"

    def generate_graph_fingerprints(
        self, mol_data: MolecularData, radius: int = 3, n_bits: int = 2048
    ) -> Dict[str, np.ndarray]:
        """Generate molecular graph fingerprints with learned representations.

        Args:
            mol_data: MolecularData with RDKit mol object
            radius: Radius for substructure enumeration
            n_bits: Number of bits in fingerprint

        Returns:
            Dictionary containing different types of graph fingerprints
        """
        try:
            # Build molecular graph
            graph_data = self.graph_builder.build_molecular_graph(mol_data)

            # Traditional Morgan-like fingerprint from graph
            morgan_graph_fp = self._compute_morgan_graph_fingerprint(
                graph_data, radius, n_bits
            )

            # Substructure-based fingerprint
            substructure_fp = self._compute_substructure_fingerprint(
                graph_data, radius, n_bits
            )

            # Path-based fingerprint
            path_fp = self._compute_path_fingerprint(graph_data, radius, n_bits)

            # Atom pair fingerprint
            atom_pair_fp = self._compute_atom_pair_fingerprint(graph_data, n_bits)

            return {
                "morgan_graph": morgan_graph_fp,
                "substructure": substructure_fp,
                "path_based": path_fp,
                "atom_pair": atom_pair_fp,
            }

        except Exception as e:
            logger.error(
                f"Failed to generate graph fingerprints for molecule {mol_data.mol_id}: {e}"
            )
            return {
                "morgan_graph": np.zeros(n_bits, dtype=np.int32),
                "substructure": np.zeros(n_bits, dtype=np.int32),
                "path_based": np.zeros(n_bits, dtype=np.int32),
                "atom_pair": np.zeros(n_bits, dtype=np.int32),
            }

    def _compute_morgan_graph_fingerprint(
        self, graph_data: Dict[str, Any], radius: int, n_bits: int
    ) -> np.ndarray:
        """Compute Morgan-like fingerprint from graph representation."""
        fingerprint = np.zeros(n_bits, dtype=np.int32)

        if graph_data["num_atoms"] == 0:
            return fingerprint

        atom_features = graph_data["atom_features"]
        adjacency_matrix = graph_data["adjacency_matrix"]

        # Initialize atom identifiers from features
        atom_identifiers = []
        for i in range(graph_data["num_atoms"]):
            # Hash atom features to create initial identifier
            feature_hash = hash(tuple(atom_features[i])) % (2**32)
            atom_identifiers.append(feature_hash)

        # Iteratively update identifiers based on neighbors
        for iteration in range(radius):
            new_identifiers = []
            for i in range(graph_data["num_atoms"]):
                # Get neighbor identifiers
                neighbors = np.where(adjacency_matrix[i] == 1)[0]
                neighbor_ids = sorted([atom_identifiers[j] for j in neighbors])

                # Combine current identifier with neighbor identifiers
                combined = [atom_identifiers[i]] + neighbor_ids
                new_hash = hash(tuple(combined)) % (2**32)
                new_identifiers.append(new_hash)

                # Set bit in fingerprint
                bit_index = new_hash % n_bits
                fingerprint[bit_index] = 1

            atom_identifiers = new_identifiers

        return fingerprint

    def _compute_substructure_fingerprint(
        self, graph_data: Dict[str, Any], radius: int, n_bits: int
    ) -> np.ndarray:
        """Compute fingerprint based on substructure enumeration."""
        fingerprint = np.zeros(n_bits, dtype=np.int32)

        if graph_data["num_atoms"] == 0:
            return fingerprint

        adjacency_matrix = graph_data["adjacency_matrix"]
        atom_features = graph_data["atom_features"]

        # Enumerate substructures of different sizes
        for size in range(1, min(radius + 1, graph_data["num_atoms"] + 1)):
            substructures = self._enumerate_substructures(
                adjacency_matrix, atom_features, size
            )

            for substructure in substructures:
                # Hash substructure to get fingerprint bit
                substructure_hash = hash(tuple(sorted(substructure))) % (2**32)
                bit_index = substructure_hash % n_bits
                fingerprint[bit_index] = 1

        return fingerprint

    def _enumerate_substructures(
        self, adjacency_matrix: np.ndarray, atom_features: np.ndarray, size: int
    ) -> List[List[int]]:
        """Enumerate connected substructures of given size."""
        num_atoms = adjacency_matrix.shape[0]
        substructures = []

        # Use DFS to find connected substructures
        def dfs(current_atoms: List[int], remaining_size: int):
            if remaining_size == 0:
                # Create canonical representation
                canonical = self._canonicalize_substructure(
                    current_atoms, adjacency_matrix, atom_features
                )
                if canonical not in substructures:
                    substructures.append(canonical)
                return

            # Find neighbors of current atoms
            neighbors = set()
            for atom in current_atoms:
                for neighbor in range(num_atoms):
                    if (
                        adjacency_matrix[atom, neighbor] == 1
                        and neighbor not in current_atoms
                    ):
                        neighbors.add(neighbor)

            # Extend substructure with each neighbor
            for neighbor in neighbors:
                dfs(current_atoms + [neighbor], remaining_size - 1)

        # Start DFS from each atom
        for start_atom in range(num_atoms):
            dfs([start_atom], size - 1)

        return substructures

    def _canonicalize_substructure(
        self, atoms: List[int], adjacency_matrix: np.ndarray, atom_features: np.ndarray
    ) -> List[int]:
        """Create canonical representation of substructure."""
        # Sort atoms by their features for canonical ordering
        atom_feature_tuples = [(i, tuple(atom_features[i])) for i in atoms]
        atom_feature_tuples.sort(key=lambda x: x[1])
        return [atom for atom, _ in atom_feature_tuples]

    def _compute_path_fingerprint(
        self, graph_data: Dict[str, Any], radius: int, n_bits: int
    ) -> np.ndarray:
        """Compute fingerprint based on molecular paths."""
        fingerprint = np.zeros(n_bits, dtype=np.int32)

        if graph_data["num_atoms"] == 0:
            return fingerprint

        adjacency_matrix = graph_data["adjacency_matrix"]
        atom_features = graph_data["atom_features"]

        # Find all paths up to given radius
        for start_atom in range(graph_data["num_atoms"]):
            paths = self._find_paths_from_atom(start_atom, adjacency_matrix, radius)

            for path in paths:
                # Create path signature from atom features
                path_signature = []
                for atom_idx in path:
                    path_signature.extend(atom_features[atom_idx])

                # Hash path signature
                path_hash = hash(tuple(path_signature)) % (2**32)
                bit_index = path_hash % n_bits
                fingerprint[bit_index] = 1

        return fingerprint

    def _find_paths_from_atom(
        self, start_atom: int, adjacency_matrix: np.ndarray, max_length: int
    ) -> List[List[int]]:
        """Find all paths starting from given atom."""
        paths = []

        def dfs_paths(current_path: List[int], remaining_length: int):
            if remaining_length == 0:
                if len(current_path) > 1:  # Only include paths with at least 2 atoms
                    paths.append(current_path.copy())
                return

            current_atom = current_path[-1]
            for neighbor in range(adjacency_matrix.shape[0]):
                if (
                    adjacency_matrix[current_atom, neighbor] == 1
                    and neighbor not in current_path
                ):
                    current_path.append(neighbor)
                    dfs_paths(current_path, remaining_length - 1)
                    current_path.pop()

        dfs_paths([start_atom], max_length)
        return paths

    def _compute_atom_pair_fingerprint(
        self, graph_data: Dict[str, Any], n_bits: int
    ) -> np.ndarray:
        """Compute atom pair fingerprint."""
        fingerprint = np.zeros(n_bits, dtype=np.int32)

        if graph_data["num_atoms"] < 2:
            return fingerprint

        atom_features = graph_data["atom_features"]

        # Consider all atom pairs
        for i in range(graph_data["num_atoms"]):
            for j in range(i + 1, graph_data["num_atoms"]):
                # Compute distance between atoms (using adjacency matrix)
                distance = self._compute_graph_distance(
                    i, j, graph_data["adjacency_matrix"]
                )

                if distance > 0:  # Only consider connected atoms
                    # Create atom pair signature
                    atom1_features = tuple(atom_features[i])
                    atom2_features = tuple(atom_features[j])

                    # Canonical ordering
                    if atom1_features > atom2_features:
                        atom1_features, atom2_features = atom2_features, atom1_features

                    pair_signature = atom1_features + atom2_features + (distance,)
                    pair_hash = hash(pair_signature) % (2**32)
                    bit_index = pair_hash % n_bits
                    fingerprint[bit_index] = 1

        return fingerprint

    def _compute_graph_distance(
        self, atom1: int, atom2: int, adjacency_matrix: np.ndarray
    ) -> int:
        """Compute shortest path distance between two atoms."""
        if atom1 == atom2:
            return 0

        num_atoms = adjacency_matrix.shape[0]
        visited = [False] * num_atoms
        queue = [(atom1, 0)]
        visited[atom1] = True

        while queue:
            current_atom, distance = queue.pop(0)

            for neighbor in range(num_atoms):
                if adjacency_matrix[current_atom, neighbor] == 1:
                    if neighbor == atom2:
                        return distance + 1

                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, distance + 1))

        return -1  # Not connected

    def compute_graph_similarity_metrics(
        self, mol_data1: MolecularData, mol_data2: MolecularData
    ) -> Dict[str, float]:
        """Compute graph-based similarity metrics between two molecules.

        Args:
            mol_data1: First molecule
            mol_data2: Second molecule

        Returns:
            Dictionary of similarity metrics
        """
        try:
            # Generate fingerprints for both molecules
            fp1 = self.generate_graph_fingerprints(mol_data1)
            fp2 = self.generate_graph_fingerprints(mol_data2)

            similarities = {}

            # Compute similarities for each fingerprint type
            for fp_type in fp1.keys():
                tanimoto = self._compute_tanimoto_similarity(fp1[fp_type], fp2[fp_type])
                dice = self._compute_dice_similarity(fp1[fp_type], fp2[fp_type])
                cosine = self._compute_cosine_similarity(fp1[fp_type], fp2[fp_type])

                similarities[f"{fp_type}_tanimoto"] = tanimoto
                similarities[f"{fp_type}_dice"] = dice
                similarities[f"{fp_type}_cosine"] = cosine

            # Compute graph edit distance (simplified)
            graph_edit_distance = self._compute_graph_edit_distance(
                mol_data1, mol_data2
            )
            similarities["graph_edit_distance"] = graph_edit_distance

            return similarities

        except Exception as e:
            logger.error(f"Failed to compute graph similarities: {e}")
            return {}

    def _compute_tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Tanimoto similarity between two fingerprints."""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return float(intersection / union) if union > 0 else 0.0

    def _compute_dice_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Dice similarity between two fingerprints."""
        intersection = np.sum(fp1 & fp2)
        total = np.sum(fp1) + np.sum(fp2)
        return float(2 * intersection / total) if total > 0 else 0.0

    def _compute_cosine_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute cosine similarity between two fingerprints."""
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    def _compute_graph_edit_distance(
        self, mol_data1: MolecularData, mol_data2: MolecularData
    ) -> float:
        """Compute simplified graph edit distance."""
        try:
            graph1 = self.graph_builder.build_molecular_graph(mol_data1)
            graph2 = self.graph_builder.build_molecular_graph(mol_data2)

            # Simplified edit distance based on size differences
            atom_diff = abs(graph1["num_atoms"] - graph2["num_atoms"])
            bond_diff = abs(graph1["num_bonds"] - graph2["num_bonds"])

            # Normalize by maximum size
            max_atoms = max(graph1["num_atoms"], graph2["num_atoms"])
            max_bonds = max(graph1["num_bonds"], graph2["num_bonds"])

            if max_atoms == 0 and max_bonds == 0:
                return 0.0

            normalized_distance = (atom_diff + bond_diff) / (max_atoms + max_bonds)
            return float(normalized_distance)

        except Exception as e:
            logger.warning(f"Failed to compute graph edit distance: {e}")
            return 1.0  # Maximum distance for error cases

    def analyze_molecular_scaffolds(self, mol_data: MolecularData) -> Dict[str, Any]:
        """Analyze molecular scaffolds and decomposition.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary containing scaffold analysis results
        """
        try:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None:
                raise ChemiAccelError(
                    f"No RDKit mol object found for molecule {mol_data.mol_id}"
                )

            # Build molecular graph for analysis
            graph_data = self.graph_builder.build_molecular_graph(mol_data)

            # Identify ring systems
            ring_systems = self._identify_ring_systems(graph_data)

            # Find largest scaffold
            largest_scaffold = self._find_largest_scaffold(graph_data, ring_systems)

            # Identify side chains
            side_chains = self._identify_side_chains(graph_data, largest_scaffold)

            # Compute scaffold properties
            scaffold_properties = self._compute_scaffold_properties(
                graph_data, largest_scaffold
            )

            return {
                "num_ring_systems": len(ring_systems),
                "ring_systems": ring_systems,
                "largest_scaffold": largest_scaffold,
                "scaffold_size": len(largest_scaffold) if largest_scaffold else 0,
                "num_side_chains": len(side_chains),
                "side_chains": side_chains,
                "scaffold_properties": scaffold_properties,
            }

        except Exception as e:
            logger.error(
                f"Failed to analyze molecular scaffolds for molecule {mol_data.mol_id}: {e}"
            )
            return {
                "num_ring_systems": 0,
                "ring_systems": [],
                "largest_scaffold": [],
                "scaffold_size": 0,
                "num_side_chains": 0,
                "side_chains": [],
                "scaffold_properties": {},
            }

    def _identify_ring_systems(self, graph_data: Dict[str, Any]) -> List[List[int]]:
        """Identify ring systems in the molecular graph."""
        if graph_data["num_atoms"] == 0:
            return []

        adjacency_matrix = graph_data["adjacency_matrix"]
        num_atoms = graph_data["num_atoms"]

        # Find all cycles using DFS
        visited = [False] * num_atoms
        cycles = []

        def dfs_cycles(node: int, parent: int, path: List[int]):
            visited[node] = True
            path.append(node)

            for neighbor in range(num_atoms):
                if adjacency_matrix[node, neighbor] == 1:
                    if neighbor == parent:
                        continue
                    elif neighbor in path:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:]
                        if len(cycle) >= 3:  # Valid cycle
                            cycles.append(sorted(cycle))
                    elif not visited[neighbor]:
                        dfs_cycles(neighbor, node, path.copy())

        # Start DFS from each unvisited node
        for start_node in range(num_atoms):
            if not visited[start_node]:
                dfs_cycles(start_node, -1, [])

        # Remove duplicate cycles
        unique_cycles = []
        for cycle in cycles:
            if cycle not in unique_cycles:
                unique_cycles.append(cycle)

        return unique_cycles

    def _find_largest_scaffold(
        self, graph_data: Dict[str, Any], ring_systems: List[List[int]]
    ) -> List[int]:
        """Find the largest molecular scaffold."""
        if not ring_systems:
            return []

        # Find the largest ring system
        largest_ring = max(ring_systems, key=len)

        # Extend scaffold to include connecting atoms
        scaffold_atoms = set(largest_ring)
        adjacency_matrix = graph_data["adjacency_matrix"]

        # Add atoms that connect ring systems
        for ring in ring_systems:
            if ring != largest_ring:
                # Check if this ring connects to the main scaffold
                for ring_atom in ring:
                    for scaffold_atom in scaffold_atoms:
                        if adjacency_matrix[ring_atom, scaffold_atom] == 1:
                            scaffold_atoms.update(ring)
                            break

        return sorted(list(scaffold_atoms))

    def _identify_side_chains(
        self, graph_data: Dict[str, Any], scaffold: List[int]
    ) -> List[List[int]]:
        """Identify side chains attached to the scaffold."""
        if not scaffold:
            return []

        scaffold_set = set(scaffold)
        adjacency_matrix = graph_data["adjacency_matrix"]
        num_atoms = graph_data["num_atoms"]

        side_chains = []
        visited = [False] * num_atoms

        # Mark scaffold atoms as visited
        for atom in scaffold:
            visited[atom] = True

        # Find side chains starting from scaffold atoms
        for scaffold_atom in scaffold:
            for neighbor in range(num_atoms):
                if (
                    adjacency_matrix[scaffold_atom, neighbor] == 1
                    and neighbor not in scaffold_set
                    and not visited[neighbor]
                ):
                    # Trace the side chain
                    side_chain = self._trace_side_chain(
                        neighbor, scaffold_atom, adjacency_matrix, visited
                    )
                    if side_chain:
                        side_chains.append(side_chain)

        return side_chains

    def _trace_side_chain(
        self,
        start_atom: int,
        scaffold_atom: int,
        adjacency_matrix: np.ndarray,
        visited: List[bool],
    ) -> List[int]:
        """Trace a side chain from the starting atom."""
        side_chain = []
        queue = [start_atom]
        local_visited = visited.copy()

        while queue:
            current_atom = queue.pop(0)
            if local_visited[current_atom]:
                continue

            local_visited[current_atom] = True
            side_chain.append(current_atom)

            # Add unvisited neighbors (except scaffold atoms)
            for neighbor in range(adjacency_matrix.shape[0]):
                if (
                    adjacency_matrix[current_atom, neighbor] == 1
                    and not local_visited[neighbor]
                ):
                    queue.append(neighbor)

        # Update global visited array
        for atom in side_chain:
            visited[atom] = True

        return side_chain

    def _compute_scaffold_properties(
        self, graph_data: Dict[str, Any], scaffold: List[int]
    ) -> Dict[str, float]:
        """Compute properties of the molecular scaffold."""
        if not scaffold:
            return {}

        adjacency_matrix = graph_data["adjacency_matrix"]
        atom_features = graph_data["atom_features"]

        # Extract scaffold subgraph
        scaffold_adj = adjacency_matrix[np.ix_(scaffold, scaffold)]
        scaffold_features = atom_features[scaffold]

        # Compute scaffold properties
        properties = {
            "scaffold_density": (
                float(np.sum(scaffold_adj) / (len(scaffold) * (len(scaffold) - 1)))
                if len(scaffold) > 1
                else 0.0
            ),
            "scaffold_atoms": len(scaffold),
            "scaffold_bonds": int(np.sum(scaffold_adj) // 2),  # Undirected graph
            "average_degree": float(np.mean(np.sum(scaffold_adj, axis=1))),
            "max_degree": (
                float(np.max(np.sum(scaffold_adj, axis=1)))
                if len(scaffold) > 0
                else 0.0
            ),
        }

        return properties


def compute_molecular_graph_embeddings(
    molecules: List[MolecularData],
    backend: str = "auto",
    hidden_dim: int = 128,
    num_layers: int = 3,
) -> pd.DataFrame:
    """Compute graph embeddings for a list of molecules.

    Args:
        molecules: List of MolecularData objects
        backend: Backend to use for computation
        hidden_dim: Hidden dimension for embeddings
        num_layers: Number of graph convolution layers

    Returns:
        DataFrame with graph embeddings
    """
    embedding_calculator = MolecularGraphEmbedding(backend, hidden_dim, num_layers)

    results = []
    for mol_data in molecules:
        embeddings = embedding_calculator.compute_embeddings(mol_data)

        # Flatten graph embedding for DataFrame
        result = {
            "mol_id": mol_data.mol_id,
            "smiles": mol_data.smiles,
            "num_atoms": embeddings["num_atoms"],
            "num_bonds": embeddings["num_bonds"],
        }

        # Add flattened graph embedding features
        graph_emb = embeddings["graph_embedding"]
        for i, value in enumerate(graph_emb):
            result[f"graph_emb_{i}"] = value

        results.append(result)

    df = pd.DataFrame(results)

    # Handle empty results
    if df.empty:
        return df

    # Reorder columns
    id_cols = ["mol_id", "smiles", "num_atoms", "num_bonds"]
    emb_cols = [col for col in df.columns if col.startswith("graph_emb_")]
    df = df[id_cols + emb_cols]

    return df


def compute_advanced_graph_features(
    molecules: List[MolecularData], backend: str = "auto"
) -> pd.DataFrame:
    """Compute advanced graph-based features for a list of molecules.

    Args:
        molecules: List of MolecularData objects
        backend: Backend to use for computation

    Returns:
        DataFrame with advanced graph features
    """
    advanced_features = AdvancedGraphFeatures(backend=backend)

    results = []
    for mol_data in molecules:
        try:
            # Generate graph fingerprints
            fingerprints = advanced_features.generate_graph_fingerprints(mol_data)

            # Analyze molecular scaffolds
            scaffold_analysis = advanced_features.analyze_molecular_scaffolds(mol_data)

            # Combine results
            result = {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
            }

            # Add fingerprint statistics (instead of full fingerprints to keep DataFrame manageable)
            for fp_type, fp_array in fingerprints.items():
                result[f"{fp_type}_bits_set"] = int(np.sum(fp_array))
                result[f"{fp_type}_density"] = float(np.sum(fp_array) / len(fp_array))

            # Add scaffold analysis results
            result.update(
                {
                    "num_ring_systems": scaffold_analysis["num_ring_systems"],
                    "scaffold_size": scaffold_analysis["scaffold_size"],
                    "num_side_chains": scaffold_analysis["num_side_chains"],
                }
            )

            # Add scaffold properties
            scaffold_props = scaffold_analysis["scaffold_properties"]
            for prop_name, prop_value in scaffold_props.items():
                result[f"scaffold_{prop_name}"] = prop_value

            results.append(result)

        except Exception as e:
            logger.error(
                f"Failed to compute advanced graph features for molecule {mol_data.mol_id}: {e}"
            )
            # Add row with default values
            result = {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
                "morgan_graph_bits_set": 0,
                "morgan_graph_density": 0.0,
                "substructure_bits_set": 0,
                "substructure_density": 0.0,
                "path_based_bits_set": 0,
                "path_based_density": 0.0,
                "atom_pair_bits_set": 0,
                "atom_pair_density": 0.0,
                "num_ring_systems": 0,
                "scaffold_size": 0,
                "num_side_chains": 0,
            }
            results.append(result)

    df = pd.DataFrame(results)

    # Handle empty results
    if df.empty:
        return df

    # Reorder columns
    id_cols = ["mol_id", "smiles"]
    feature_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + feature_cols]

    return df


def compute_graph_similarity_matrix(
    molecules: List[MolecularData], backend: str = "auto"
) -> pd.DataFrame:
    """Compute pairwise graph similarity matrix for molecules.

    Args:
        molecules: List of MolecularData objects
        backend: Backend to use for computation

    Returns:
        DataFrame with pairwise similarity matrix
    """
    advanced_features = AdvancedGraphFeatures(backend=backend)

    n_molecules = len(molecules)
    similarity_matrix = np.zeros((n_molecules, n_molecules))

    # Compute pairwise similarities
    for i in range(n_molecules):
        for j in range(i, n_molecules):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                try:
                    similarities = advanced_features.compute_graph_similarity_metrics(
                        molecules[i], molecules[j]
                    )
                    # Use average Tanimoto similarity across fingerprint types
                    tanimoto_scores = [
                        v for k, v in similarities.items() if k.endswith("_tanimoto")
                    ]
                    avg_tanimoto = np.mean(tanimoto_scores) if tanimoto_scores else 0.0

                    similarity_matrix[i, j] = avg_tanimoto
                    similarity_matrix[j, i] = avg_tanimoto  # Symmetric matrix

                except Exception as e:
                    logger.warning(
                        f"Failed to compute similarity between molecules {i} and {j}: {e}"
                    )
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0

    # Create DataFrame with molecule IDs as index and columns
    mol_ids = [mol.mol_id for mol in molecules]
    df = pd.DataFrame(similarity_matrix, index=mol_ids, columns=mol_ids)

    return df


def benchmark_graph_features_performance(
    molecules: List[MolecularData], traditional_method: str = "rdkit"
) -> Dict[str, Any]:
    """Benchmark graph features performance against traditional methods.

    Args:
        molecules: List of MolecularData objects
        traditional_method: Traditional method to compare against ("rdkit", "mordred")

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Benchmark ChemiAccel graph features
    start_time = time.time()
    chemi_accel_features = AdvancedGraphFeatures(backend="cpu")

    chemi_accel_results = []
    for mol_data in molecules:
        fingerprints = chemi_accel_features.generate_graph_fingerprints(mol_data)
        scaffold_analysis = chemi_accel_features.analyze_molecular_scaffolds(mol_data)
        chemi_accel_results.append((fingerprints, scaffold_analysis))

    chemi_accel_time = time.time() - start_time

    # Benchmark traditional method (simplified)
    start_time = time.time()
    traditional_results = []

    if traditional_method == "rdkit":
        try:
            from rdkit.Chem import rdMolDescriptors

            for mol_data in molecules:
                rdkit_mol = mol_data.backend_data.get("rdkit_mol")
                if rdkit_mol:
                    # Compute traditional RDKit fingerprints
                    morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        rdkit_mol, 2
                    )
                    traditional_results.append(morgan_fp)
        except ImportError:
            logger.warning("RDKit not available for benchmarking")

    traditional_time = time.time() - start_time

    # Calculate speedup
    speedup = (
        traditional_time / chemi_accel_time if chemi_accel_time > 0 else float("inf")
    )

    benchmark_results = {
        "num_molecules": len(molecules),
        "chemi_accel_time": chemi_accel_time,
        "traditional_time": traditional_time,
        "speedup": speedup,
        "chemi_accel_features_per_second": (
            len(molecules) / chemi_accel_time if chemi_accel_time > 0 else 0
        ),
        "traditional_features_per_second": (
            len(molecules) / traditional_time if traditional_time > 0 else 0
        ),
        "backend_used": chemi_accel_features.actual_backend,
    }

    logger.info(
        f"Graph features benchmark: {speedup:.2f}x speedup over {traditional_method}"
    )

    return benchmark_results
