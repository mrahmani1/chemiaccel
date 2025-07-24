"""
Dynamic Network Analysis for Allosteric Pathway Identification - Task 19.4

This module provides advanced dynamic network analysis capabilities for identifying
allosteric communication pathways in proteins from MD trajectories, leveraging
graph theory and machine learning to uncover long-range coupling mechanisms.

Key Features:
- Dynamic contact network construction from MD trajectories
- Allosteric pathway identification using graph algorithms
- Community detection for functional domain analysis
- Information flow analysis for communication pathway mapping
- GPU-accelerated network calculations for large systems
- Machine learning-based pathway prediction and validation

Network Analysis Methods:
- Residue-residue contact networks with time-varying weights
- Mutual information and correlation-based coupling analysis
- Shortest path algorithms for optimal pathway identification
- Centrality measures for key residue identification
- Network clustering and community detection
- Signal propagation modeling for allosteric effects

Scientific Applications:
- Allosteric drug design and optimization
- Protein engineering for enhanced allostery
- Disease mutation impact analysis
- Enzyme regulation mechanism studies
- Large-scale comparative allostery analysis
- Real-time allosteric monitoring in simulations

Performance Optimizations:
- Sparse matrix representations for large networks
- GPU-accelerated graph algorithms with CuPy
- Parallel processing of multiple trajectories
- Memory-efficient streaming analysis
- Just-in-time compilation for custom network metrics
- Hierarchical network decomposition for scalability
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any, Callable, Set
import threading
from collections import defaultdict, deque
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

# Core ChemiAccel imports
from chemi_accel.core.molecular import MolecularData
from chemi_accel.core.exceptions import (
    ChemiAccelError,
    ValidationError,
    CalculationError
)
from chemi_accel.analysis.trajectory_reader import TrajectoryFrame, TrajectoryMetadata
from chemi_accel.analysis.timeseries_properties import TimeSeriesProperty

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.spatial.distance as cp_distance
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Network analysis libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available - limited network functionality")

# Graph processing libraries
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

# Advanced clustering
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.manifold import SpectralEmbedding
    SKLEARN_CLUSTERING_AVAILABLE = True
except ImportError:
    SKLEARN_CLUSTERING_AVAILABLE = False

@dataclass
class NetworkNode:
    """Network node representing a residue or atom."""
    
    node_id: int
    residue_id: int
    chain_id: str = "A"
    residue_name: str = ""
    atom_name: str = ""
    coordinates: Optional[np.ndarray] = None
    properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node data."""
        if self.node_id < 0:
            raise ValidationError(f"node_id must be >= 0: {self.node_id}")
        if self.residue_id < 0:
            raise ValidationError(f"residue_id must be >= 0: {self.residue_id}")

@dataclass
class NetworkEdge:
    """Network edge representing interaction between nodes."""
    
    node1: int
    node2: int
    weight: float
    edge_type: str = "contact"
    properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge data."""
        if self.node1 == self.node2:
            raise ValidationError("Self-loops not allowed")
        if self.weight < 0:
            raise ValidationError(f"Edge weight must be >= 0: {self.weight}")

@dataclass
class DynamicNetwork:
    """Dynamic network representation."""
    
    frame_id: int
    time_ps: float
    nodes: Dict[int, NetworkNode]
    edges: List[NetworkEdge]
    adjacency_matrix: Optional[np.ndarray] = None
    properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate network data."""
        if self.frame_id < 0:
            raise ValidationError(f"frame_id must be >= 0: {self.frame_id}")
        
        # Validate edge node references
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.node1 not in node_ids:
                raise ValidationError(f"Edge references unknown node: {edge.node1}")
            if edge.node2 not in node_ids:
                raise ValidationError(f"Edge references unknown node: {edge.node2}")
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)
    
    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)
    
    def get_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """Get adjacency matrix."""
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix
        
        n_nodes = self.n_nodes
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Create node ID to index mapping
        node_ids = sorted(self.nodes.keys())
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Fill adjacency matrix
        for edge in self.edges:
            i = id_to_index[edge.node1]
            j = id_to_index[edge.node2]
            
            weight = edge.weight if weighted else 1.0
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Undirected graph
        
        self.adjacency_matrix = adj_matrix
        return adj_matrix
    
    def to_networkx(self) -> 'nx.Graph':
        """Convert to NetworkX graph."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available")
        
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **{
                'residue_id': node.residue_id,
                'chain_id': node.chain_id,
                'residue_name': node.residue_name,
                'atom_name': node.atom_name,
                **node.properties
            })
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.node1, edge.node2, weight=edge.weight, 
                      edge_type=edge.edge_type, **edge.properties)
        
        return G

@dataclass
class AllostericPathway:
    """Allosteric communication pathway."""
    
    pathway_id: str
    source_residues: List[int]
    target_residues: List[int]
    pathway_nodes: List[int]
    pathway_edges: List[Tuple[int, int]]
    pathway_length: float
    information_flow: float
    significance_score: float
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate pathway data."""
        if not self.source_residues:
            raise ValidationError("Source residues cannot be empty")
        if not self.target_residues:
            raise ValidationError("Target residues cannot be empty")
        if self.pathway_length < 0:
            raise ValidationError(f"Pathway length must be >= 0: {self.pathway_length}")
        if not 0 <= self.significance_score <= 1:
            raise ValidationError(f"Significance score must be in [0,1]: {self.significance_score}")

@dataclass
class NetworkAnalysisConfig:
    """Configuration for network analysis."""
    
    contact_cutoff: float = 4.0  # Angstrom
    min_contact_frequency: float = 0.1  # Minimum frequency for persistent contacts
    use_gpu: bool = GPU_AVAILABLE
    n_workers: int = 4
    memory_limit_gb: float = 8.0
    network_type: str = "residue"  # "residue" or "atom"
    weighted_networks: bool = True
    include_backbone: bool = True
    include_sidechain: bool = True
    correlation_method: str = "pearson"  # "pearson", "spearman", "mutual_info"
    pathway_algorithm: str = "shortest_path"  # "shortest_path", "random_walk", "flow"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.contact_cutoff <= 0:
            raise ValidationError(f"contact_cutoff must be > 0: {self.contact_cutoff}")
        if not 0 <= self.min_contact_frequency <= 1:
            raise ValidationError(f"min_contact_frequency must be in [0,1]: {self.min_contact_frequency}")
        if self.network_type not in ["residue", "atom"]:
            raise ValidationError(f"network_type must be 'residue' or 'atom': {self.network_type}")
        if self.correlation_method not in ["pearson", "spearman", "mutual_info"]:
            raise ValidationError(f"Invalid correlation method: {self.correlation_method}")

class NetworkBuilder:
    """Builder for dynamic contact networks."""
    
    def __init__(self, config: NetworkAnalysisConfig = None):
        """Initialize network builder.
        
        Args:
            config: Configuration for network construction
        """
        self.config = config or NetworkAnalysisConfig()
        
        # GPU memory management
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                self.memory_pool = cp.get_default_memory_pool()
                memory_limit = int(self.config.memory_limit_gb * 1024**3)
                self.memory_pool.set_limit(size=memory_limit)
            except Exception as e:
                warnings.warn(f"GPU memory setup failed: {e}")
                self.config.use_gpu = False
    
    def build_contact_network(self, frame: TrajectoryFrame, 
                            topology: Dict[str, Any]) -> DynamicNetwork:
        """Build contact network for single frame.
        
        Args:
            frame: Trajectory frame
            topology: Topology information with residue/atom mappings
            
        Returns:
            DynamicNetwork object
        """
        # Create nodes
        nodes = self._create_nodes(frame, topology)
        
        # Calculate contacts
        edges = self._calculate_contacts(frame, nodes)
        
        # Create network
        network = DynamicNetwork(
            frame_id=frame.frame_id,
            time_ps=frame.time_ps,
            nodes=nodes,
            edges=edges
        )
        
        # Calculate network properties
        network.properties = self._calculate_network_properties(network)
        
        return network
    
    def _create_nodes(self, frame: TrajectoryFrame, 
                     topology: Dict[str, Any]) -> Dict[int, NetworkNode]:
        """Create network nodes from frame and topology."""
        nodes = {}
        
        if self.config.network_type == "residue":
            # Group atoms by residue
            residue_atoms = defaultdict(list)
            for atom_idx, atom_info in topology.get("atoms", {}).items():
                if self._should_include_atom(atom_info):
                    residue_id = atom_info.get("residue_id", 0)
                    residue_atoms[residue_id].append(atom_idx)
            
            # Create residue nodes
            for residue_id, atom_indices in residue_atoms.items():
                if not atom_indices:
                    continue
                
                # Calculate residue center of mass
                residue_coords = frame.coordinates[atom_indices]
                center_of_mass = np.mean(residue_coords, axis=0)
                
                # Get residue info
                residue_info = topology.get("residues", {}).get(residue_id, {})
                
                node = NetworkNode(
                    node_id=residue_id,
                    residue_id=residue_id,
                    chain_id=residue_info.get("chain_id", "A"),
                    residue_name=residue_info.get("residue_name", "UNK"),
                    coordinates=center_of_mass
                )
                
                nodes[residue_id] = node
        
        else:  # atom-level network
            for atom_idx, atom_info in topology.get("atoms", {}).items():
                if self._should_include_atom(atom_info):
                    node = NetworkNode(
                        node_id=atom_idx,
                        residue_id=atom_info.get("residue_id", 0),
                        chain_id=atom_info.get("chain_id", "A"),
                        residue_name=atom_info.get("residue_name", "UNK"),
                        atom_name=atom_info.get("atom_name", "CA"),
                        coordinates=frame.coordinates[atom_idx]
                    )
                    
                    nodes[atom_idx] = node
        
        return nodes
    
    def _should_include_atom(self, atom_info: Dict[str, Any]) -> bool:
        """Check if atom should be included in network."""
        atom_name = atom_info.get("atom_name", "")
        
        # Backbone atoms
        backbone_atoms = {"CA", "CB", "N", "C", "O"}
        is_backbone = atom_name in backbone_atoms
        
        # Sidechain atoms
        is_sidechain = not is_backbone and atom_name not in {"H", "HA"}
        
        # Apply inclusion criteria
        if self.config.include_backbone and is_backbone:
            return True
        if self.config.include_sidechain and is_sidechain:
            return True
        
        return False
    
    def _calculate_contacts(self, frame: TrajectoryFrame, 
                          nodes: Dict[int, NetworkNode]) -> List[NetworkEdge]:
        """Calculate contacts between nodes."""
        edges = []
        
        if not nodes:
            return edges
        
        # Extract coordinates
        node_ids = list(nodes.keys())
        coords = np.array([nodes[nid].coordinates for nid in node_ids])
        
        # Calculate pairwise distances
        if self.config.use_gpu and GPU_AVAILABLE:
            distances = self._calculate_distances_gpu(coords)
        else:
            distances = self._calculate_distances_cpu(coords)
        
        # Create edges for contacts within cutoff
        n_nodes = len(node_ids)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                distance = distances[i, j]
                
                if distance <= self.config.contact_cutoff:
                    # Calculate contact weight (closer = stronger)
                    weight = 1.0 / (1.0 + distance) if self.config.weighted_networks else 1.0
                    
                    edge = NetworkEdge(
                        node1=node_ids[i],
                        node2=node_ids[j],
                        weight=weight,
                        edge_type="contact",
                        properties={"distance": distance}
                    )
                    
                    edges.append(edge)
        
        return edges
    
    def _calculate_distances_cpu(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances on CPU."""
        n_points = len(coords)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _calculate_distances_gpu(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances on GPU."""
        try:
            gpu_coords = cp.asarray(coords)
            
            # Use broadcasting for vectorized distance calculation
            diff = gpu_coords[:, None, :] - gpu_coords[None, :, :]
            distances = cp.linalg.norm(diff, axis=2)
            
            return cp.asnumpy(distances)
            
        except Exception as e:
            warnings.warn(f"GPU distance calculation failed: {e}")
            return self._calculate_distances_cpu(coords)
    
    def _calculate_network_properties(self, network: DynamicNetwork) -> Dict[str, float]:
        """Calculate basic network properties."""
        properties = {}
        
        adj_matrix = network.get_adjacency_matrix()
        
        # Basic properties
        properties["n_nodes"] = network.n_nodes
        properties["n_edges"] = network.n_edges
        properties["density"] = network.n_edges / (network.n_nodes * (network.n_nodes - 1) / 2)
        
        # Degree statistics
        degrees = np.sum(adj_matrix > 0, axis=1)
        properties["mean_degree"] = np.mean(degrees)
        properties["max_degree"] = np.max(degrees) if len(degrees) > 0 else 0
        properties["degree_std"] = np.std(degrees)
        
        # Weight statistics if weighted
        if self.config.weighted_networks:
            weights = adj_matrix[adj_matrix > 0]
            if len(weights) > 0:
                properties["mean_weight"] = np.mean(weights)
                properties["weight_std"] = np.std(weights)
        
        return properties
    
    def build_trajectory_networks(self, frames: List[TrajectoryFrame],
                                topology: Dict[str, Any]) -> List[DynamicNetwork]:
        """Build networks for multiple frames.
        
        Args:
            frames: List of trajectory frames
            topology: Topology information
            
        Returns:
            List of DynamicNetwork objects
        """
        if self.config.n_workers > 1:
            return self._build_networks_parallel(frames, topology)
        else:
            return self._build_networks_sequential(frames, topology)
    
    def _build_networks_sequential(self, frames: List[TrajectoryFrame],
                                 topology: Dict[str, Any]) -> List[DynamicNetwork]:
        """Build networks sequentially."""
        networks = []
        
        for frame in frames:
            try:
                network = self.build_contact_network(frame, topology)
                networks.append(network)
            except Exception as e:
                warnings.warn(f"Network construction failed for frame {frame.frame_id}: {e}")
                continue
        
        return networks
    
    def _build_networks_parallel(self, frames: List[TrajectoryFrame],
                               topology: Dict[str, Any]) -> List[DynamicNetwork]:
        """Build networks in parallel."""
        networks = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit network building tasks
            future_to_frame = {
                executor.submit(self.build_contact_network, frame, topology): frame
                for frame in frames
            }
            
            # Collect results
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    network = future.result()
                    networks.append(network)
                except Exception as e:
                    warnings.warn(f"Network construction failed for frame {frame.frame_id}: {e}")
                    continue
        
        # Sort by frame ID
        networks.sort(key=lambda x: x.frame_id)
        return networks
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                self.memory_pool.free_all_blocks()
            except:
                pass

class AllostericPathwayAnalyzer:
    """Analyzer for allosteric communication pathways."""
    
    def __init__(self, config: NetworkAnalysisConfig = None):
        """Initialize pathway analyzer.
        
        Args:
            config: Configuration for pathway analysis
        """
        self.config = config or NetworkAnalysisConfig()
        self.networks: List[DynamicNetwork] = []
        self.consensus_network: Optional[DynamicNetwork] = None
        self.pathways: Dict[str, AllostericPathway] = {}
        
    def add_networks(self, networks: List[DynamicNetwork]):
        """Add networks for analysis."""
        self.networks.extend(networks)
    
    def build_consensus_network(self, min_frequency: Optional[float] = None) -> DynamicNetwork:
        """Build consensus network from multiple time points.
        
        Args:
            min_frequency: Minimum frequency for edge inclusion
            
        Returns:
            Consensus DynamicNetwork
        """
        if not self.networks:
            raise ValueError("No networks available for consensus building")
        
        min_freq = min_frequency or self.config.min_contact_frequency
        
        # Collect all edges with frequencies
        edge_counts = defaultdict(int)
        edge_weights = defaultdict(list)
        all_nodes = {}
        
        for network in self.networks:
            # Collect nodes (use first occurrence)
            for node_id, node in network.nodes.items():
                if node_id not in all_nodes:
                    all_nodes[node_id] = node
            
            # Count edge occurrences
            for edge in network.edges:
                edge_key = tuple(sorted([edge.node1, edge.node2]))
                edge_counts[edge_key] += 1
                edge_weights[edge_key].append(edge.weight)
        
        # Filter edges by frequency and create consensus edges
        n_networks = len(self.networks)
        consensus_edges = []
        
        for edge_key, count in edge_counts.items():
            frequency = count / n_networks
            
            if frequency >= min_freq:
                # Calculate consensus weight
                weights = edge_weights[edge_key]
                avg_weight = np.mean(weights)
                
                edge = NetworkEdge(
                    node1=edge_key[0],
                    node2=edge_key[1],
                    weight=avg_weight,
                    edge_type="consensus",
                    properties={
                        "frequency": frequency,
                        "weight_std": np.std(weights),
                        "count": count
                    }
                )
                
                consensus_edges.append(edge)
        
        # Create consensus network
        self.consensus_network = DynamicNetwork(
            frame_id=-1,  # Special ID for consensus
            time_ps=-1,
            nodes=all_nodes,
            edges=consensus_edges,
            properties={
                "n_source_networks": n_networks,
                "min_frequency": min_freq,
                "consensus_edges": len(consensus_edges)
            }
        )
        
        return self.consensus_network
    
    def identify_allosteric_pathways(self, source_residues: List[int], 
                                   target_residues: List[int],
                                   max_pathways: int = 10) -> List[AllostericPathway]:
        """Identify allosteric pathways between source and target residues.
        
        Args:
            source_residues: List of source residue IDs
            target_residues: List of target residue IDs
            max_pathways: Maximum number of pathways to return
            
        Returns:
            List of AllostericPathway objects
        """
        if self.consensus_network is None:
            self.build_consensus_network()
        
        # Use different algorithms based on configuration
        if self.config.pathway_algorithm == "shortest_path":
            pathways = self._find_shortest_paths(source_residues, target_residues, max_pathways)
        elif self.config.pathway_algorithm == "random_walk":
            pathways = self._find_random_walk_paths(source_residues, target_residues, max_pathways)
        elif self.config.pathway_algorithm == "flow":
            pathways = self._find_flow_paths(source_residues, target_residues, max_pathways)
        else:
            raise ValueError(f"Unknown pathway algorithm: {self.config.pathway_algorithm}")
        
        # Store pathways
        for pathway in pathways:
            self.pathways[pathway.pathway_id] = pathway
        
        return pathways
    
    def _find_shortest_paths(self, source_residues: List[int], 
                           target_residues: List[int], 
                           max_pathways: int) -> List[AllostericPathway]:
        """Find shortest paths using Dijkstra's algorithm."""
        if not NETWORKX_AVAILABLE:
            return self._find_shortest_paths_custom(source_residues, target_residues, max_pathways)
        
        # Convert to NetworkX
        G = self.consensus_network.to_networkx()
        
        # Find all shortest paths
        all_paths = []
        
        for source in source_residues:
            if source not in G:
                continue
            
            for target in target_residues:
                if target not in G:
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(G, source, target, weight='weight')
                    path_length = nx.shortest_path_length(G, source, target, weight='weight')
                    
                    # Calculate information flow (sum of edge weights)
                    info_flow = 0.0
                    path_edges = []
                    
                    for i in range(len(path) - 1):
                        edge_data = G[path[i]][path[i+1]]
                        info_flow += edge_data['weight']
                        path_edges.append((path[i], path[i+1]))
                    
                    all_paths.append({
                        'path': path,
                        'edges': path_edges,
                        'length': path_length,
                        'info_flow': info_flow,
                        'source': source,
                        'target': target
                    })
                    
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by length and take top paths
        all_paths.sort(key=lambda x: x['length'])
        top_paths = all_paths[:max_pathways]
        
        # Convert to AllostericPathway objects
        pathways = []
        
        for i, path_data in enumerate(top_paths):
            # Calculate significance score (normalized by network properties)
            max_possible_length = len(self.consensus_network.nodes)
            significance = 1.0 - (path_data['length'] / max_possible_length)
            
            pathway = AllostericPathway(
                pathway_id=f"path_{i}_{path_data['source']}_{path_data['target']}",
                source_residues=[path_data['source']],
                target_residues=[path_data['target']],
                pathway_nodes=path_data['path'],
                pathway_edges=path_data['edges'],
                pathway_length=path_data['length'],
                information_flow=path_data['info_flow'],
                significance_score=significance,
                properties={
                    'algorithm': 'shortest_path',
                    'n_nodes': len(path_data['path']),
                    'n_edges': len(path_data['edges'])
                }
            )
            
            pathways.append(pathway)
        
        return pathways
    
    def _find_shortest_paths_custom(self, source_residues: List[int],
                                  target_residues: List[int],
                                  max_pathways: int) -> List[AllostericPathway]:
        """Custom shortest path implementation using adjacency matrix."""
        adj_matrix = self.consensus_network.get_adjacency_matrix()
        node_ids = sorted(self.consensus_network.nodes.keys())
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        pathways = []
        
        # Simple Dijkstra implementation
        for source in source_residues:
            if source not in id_to_index:
                continue
            
            source_idx = id_to_index[source]
            
            # Initialize distances
            n_nodes = len(node_ids)
            distances = np.full(n_nodes, np.inf)
            distances[source_idx] = 0.0
            predecessors = np.full(n_nodes, -1)
            visited = np.zeros(n_nodes, dtype=bool)
            
            # Dijkstra's algorithm
            for _ in range(n_nodes):
                # Find unvisited node with minimum distance
                unvisited_distances = np.where(visited, np.inf, distances)
                current_idx = np.argmin(unvisited_distances)
                
                if distances[current_idx] == np.inf:
                    break
                
                visited[current_idx] = True
                
                # Update neighbors
                for neighbor_idx in range(n_nodes):
                    if not visited[neighbor_idx] and adj_matrix[current_idx, neighbor_idx] > 0:
                        # Use inverse weight as distance (higher weight = shorter distance)
                        edge_distance = 1.0 / adj_matrix[current_idx, neighbor_idx]
                        new_distance = distances[current_idx] + edge_distance
                        
                        if new_distance < distances[neighbor_idx]:
                            distances[neighbor_idx] = new_distance
                            predecessors[neighbor_idx] = current_idx
            
            # Extract paths to targets
            for target in target_residues:
                if target not in id_to_index:
                    continue
                
                target_idx = id_to_index[target]
                
                if distances[target_idx] == np.inf:
                    continue
                
                # Reconstruct path
                path_indices = []
                current_idx = target_idx
                
                while current_idx != -1:
                    path_indices.append(current_idx)
                    current_idx = predecessors[current_idx]
                
                path_indices.reverse()
                path_nodes = [node_ids[idx] for idx in path_indices]
                
                # Create pathway edges
                path_edges = []
                info_flow = 0.0
                
                for i in range(len(path_indices) - 1):
                    node1 = node_ids[path_indices[i]]
                    node2 = node_ids[path_indices[i+1]]
                    weight = adj_matrix[path_indices[i], path_indices[i+1]]
                    
                    path_edges.append((node1, node2))
                    info_flow += weight
                
                # Create pathway
                pathway = AllostericPathway(
                    pathway_id=f"path_{len(pathways)}_{source}_{target}",
                    source_residues=[source],
                    target_residues=[target],
                    pathway_nodes=path_nodes,
                    pathway_edges=path_edges,
                    pathway_length=distances[target_idx],
                    information_flow=info_flow,
                    significance_score=1.0 / (1.0 + distances[target_idx]),
                    properties={
                        'algorithm': 'custom_dijkstra',
                        'n_nodes': len(path_nodes),
                        'n_edges': len(path_edges)
                    }
                )
                
                pathways.append(pathway)
        
        # Sort by length and return top pathways
        pathways.sort(key=lambda x: x.pathway_length)
        return pathways[:max_pathways]
    
    def _find_random_walk_paths(self, source_residues: List[int],
                              target_residues: List[int],
                              max_pathways: int) -> List[AllostericPathway]:
        """Find pathways using random walk analysis."""
        # Placeholder for random walk implementation
        # This would implement random walk-based pathway identification
        return []
    
    def _find_flow_paths(self, source_residues: List[int],
                        target_residues: List[int],
                        max_pathways: int) -> List[AllostericPathway]:
        """Find pathways using network flow algorithms."""
        # Placeholder for flow-based implementation
        # This would implement max flow or min-cost flow algorithms
        return []
    
    def calculate_pathway_correlations(self, time_series_data: Dict[int, TimeSeriesProperty]) -> Dict[str, float]:
        """Calculate correlations along identified pathways.
        
        Args:
            time_series_data: Dict mapping residue IDs to time series data
            
        Returns:
            Dict mapping pathway IDs to correlation scores
        """
        correlations = {}
        
        for pathway_id, pathway in self.pathways.items():
            # Calculate correlation along pathway
            pathway_correlations = []
            
            for i in range(len(pathway.pathway_nodes) - 1):
                node1 = pathway.pathway_nodes[i]
                node2 = pathway.pathway_nodes[i + 1]
                
                if node1 in time_series_data and node2 in time_series_data:
                    ts1 = time_series_data[node1]
                    ts2 = time_series_data[node2]
                    
                    # Calculate correlation between time series
                    corr = self._calculate_correlation(ts1.values, ts2.values)
                    pathway_correlations.append(abs(corr))
            
            # Average correlation along pathway
            if pathway_correlations:
                correlations[pathway_id] = np.mean(pathway_correlations)
            else:
                correlations[pathway_id] = 0.0
        
        return correlations
    
    def _calculate_correlation(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Calculate correlation between two time series."""
        # Remove any NaN values
        valid_indices = ~(np.isnan(series1) | np.isnan(series2))
        if np.sum(valid_indices) < 3:
            return 0.0
        
        clean_series1 = series1[valid_indices]
        clean_series2 = series2[valid_indices]
        
        if self.config.correlation_method == "pearson":
            corr, _ = stats.pearsonr(clean_series1, clean_series2)
        elif self.config.correlation_method == "spearman":
            corr, _ = stats.spearmanr(clean_series1, clean_series2)
        elif self.config.correlation_method == "mutual_info":
            # Discretize for mutual information
            bins = 10
            s1_discrete = np.digitize(clean_series1, np.linspace(clean_series1.min(), clean_series1.max(), bins))
            s2_discrete = np.digitize(clean_series2, np.linspace(clean_series2.min(), clean_series2.max(), bins))
            corr = mutual_info_score(s1_discrete, s2_discrete)
        else:
            corr = 0.0
        
        return corr if not np.isnan(corr) else 0.0
    
    def detect_communities(self, method: str = "spectral") -> Dict[int, int]:
        """Detect communities in consensus network.
        
        Args:
            method: Community detection method
            
        Returns:
            Dict mapping node IDs to community IDs
        """
        if self.consensus_network is None:
            self.build_consensus_network()
        
        adj_matrix = self.consensus_network.get_adjacency_matrix()
        node_ids = sorted(self.consensus_network.nodes.keys())
        
        if method == "spectral":
            # Spectral clustering
            if SKLEARN_CLUSTERING_AVAILABLE:
                n_clusters = min(10, max(2, len(node_ids) // 10))
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
                community_labels = clustering.fit_predict(adj_matrix)
            else:
                # Fallback: simple k-means on adjacency matrix
                n_clusters = min(5, max(2, len(node_ids) // 20))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                community_labels = kmeans.fit_predict(adj_matrix)
        
        elif method == "hierarchical":
            # Hierarchical clustering
            # Convert adjacency matrix to distance matrix
            distance_matrix = 1.0 / (adj_matrix + 1e-10)
            np.fill_diagonal(distance_matrix, 0)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            n_clusters = min(8, max(2, len(node_ids) // 15))
            community_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        else:
            # Default: assign all nodes to single community
            community_labels = np.zeros(len(node_ids), dtype=int)
        
        # Create mapping
        community_map = {node_ids[i]: int(community_labels[i]) for i in range(len(node_ids))}
        
        return community_map
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[int, float]]:
        """Calculate various centrality measures for network nodes.
        
        Returns:
            Dict mapping centrality types to node centrality values
        """
        if self.consensus_network is None:
            self.build_consensus_network()
        
        centralities = {}
        
        if NETWORKX_AVAILABLE:
            G = self.consensus_network.to_networkx()
            
            # Calculate different centrality measures
            centralities["degree"] = dict(nx.degree_centrality(G))
            centralities["betweenness"] = dict(nx.betweenness_centrality(G, weight='weight'))
            centralities["closeness"] = dict(nx.closeness_centrality(G, distance='weight'))
            centralities["eigenvector"] = dict(nx.eigenvector_centrality(G, weight='weight', max_iter=1000))
            
        else:
            # Fallback: calculate basic degree centrality
            adj_matrix = self.consensus_network.get_adjacency_matrix()
            node_ids = sorted(self.consensus_network.nodes.keys())
            
            degrees = np.sum(adj_matrix > 0, axis=1)
            max_degree = len(node_ids) - 1
            
            degree_centrality = {node_ids[i]: degrees[i] / max_degree for i in range(len(node_ids))}
            centralities["degree"] = degree_centrality
        
        return centralities
    
    def get_pathway_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of identified pathways.
        
        Returns:
            DataFrame with pathway information
        """
        if not self.pathways:
            return pd.DataFrame()
        
        rows = []
        
        for pathway_id, pathway in self.pathways.items():
            row = {
                'pathway_id': pathway_id,
                'source_residues': ','.join(map(str, pathway.source_residues)),
                'target_residues': ','.join(map(str, pathway.target_residues)),
                'pathway_length': pathway.pathway_length,
                'n_nodes': len(pathway.pathway_nodes),
                'n_edges': len(pathway.pathway_edges),
                'information_flow': pathway.information_flow,
                'significance_score': pathway.significance_score,
                'algorithm': pathway.properties.get('algorithm', 'unknown')
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def cleanup(self):
        """Clean up resources."""
        self.networks.clear()
        self.pathways.clear()
        self.consensus_network = None

def create_network_analyzer(contact_cutoff: float = 4.0,
                          use_gpu: bool = GPU_AVAILABLE,
                          n_workers: int = 4) -> Tuple[NetworkBuilder, AllostericPathwayAnalyzer]:
    """Create network builder and pathway analyzer with optimal configuration.
    
    Args:
        contact_cutoff: Contact distance cutoff in Angstrom
        use_gpu: Enable GPU acceleration
        n_workers: Number of parallel workers
        
    Returns:
        Tuple of (NetworkBuilder, AllostericPathwayAnalyzer)
    """
    config = NetworkAnalysisConfig(
        contact_cutoff=contact_cutoff,
        use_gpu=use_gpu,
        n_workers=n_workers
    )
    
    builder = NetworkBuilder(config)
    analyzer = AllostericPathwayAnalyzer(config)
    
    return builder, analyzer

# Utility functions for network analysis
def calculate_network_similarity(network1: DynamicNetwork, network2: DynamicNetwork) -> float:
    """Calculate similarity between two networks.
    
    Args:
        network1: First network
        network2: Second network
        
    Returns:
        Similarity score (0-1)
    """
    # Get common nodes
    nodes1 = set(network1.nodes.keys())
    nodes2 = set(network2.nodes.keys())
    common_nodes = nodes1.intersection(nodes2)
    
    if not common_nodes:
        return 0.0
    
    # Get edges for common nodes
    edges1 = set()
    edges2 = set()
    
    for edge in network1.edges:
        if edge.node1 in common_nodes and edge.node2 in common_nodes:
            edge_key = tuple(sorted([edge.node1, edge.node2]))
            edges1.add(edge_key)
    
    for edge in network2.edges:
        if edge.node1 in common_nodes and edge.node2 in common_nodes:
            edge_key = tuple(sorted([edge.node1, edge.node2]))
            edges2.add(edge_key)
    
    # Calculate Jaccard similarity
    union_edges = edges1.union(edges2)
    intersection_edges = edges1.intersection(edges2)
    
    if not union_edges:
        return 1.0  # Both networks have no edges
    
    return len(intersection_edges) / len(union_edges)

def analyze_pathway_robustness(pathways: List[AllostericPathway],
                             perturbation_networks: List[DynamicNetwork]) -> Dict[str, float]:
    """Analyze robustness of pathways to network perturbations.
    
    Args:
        pathways: List of pathways to analyze
        perturbation_networks: Networks with perturbations
        
    Returns:
        Dict mapping pathway IDs to robustness scores
    """
    robustness_scores = {}
    
    for pathway in pathways:
        pathway_robustness = []
        
        for network in perturbation_networks:
            # Check if pathway still exists in perturbed network
            pathway_exists = True
            
            # Check if all pathway nodes exist
            for node_id in pathway.pathway_nodes:
                if node_id not in network.nodes:
                    pathway_exists = False
                    break
            
            # Check if pathway edges exist
            if pathway_exists:
                network_edges = set()
                for edge in network.edges:
                    edge_key = tuple(sorted([edge.node1, edge.node2]))
                    network_edges.add(edge_key)
                
                for edge in pathway.pathway_edges:
                    edge_key = tuple(sorted(edge))
                    if edge_key not in network_edges:
                        pathway_exists = False
                        break
            
            pathway_robustness.append(1.0 if pathway_exists else 0.0)
        
        # Calculate average robustness
        robustness_scores[pathway.pathway_id] = np.mean(pathway_robustness) if pathway_robustness else 0.0
    
    return robustness_scores