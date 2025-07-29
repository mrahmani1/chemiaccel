"""
Multi-Step Route Enumeration and Graph Construction.

This module implements algorithms for iteratively applying retrosynthetic steps,
constructing multi-step synthesis trees, and identifying purchasable starting
materials through chemical building block database queries.
"""

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False
    nx = None

from .generative_engine import (
    ModelPrediction,
    MolecularRepresentation,
    RetrosynthesisConfig,
    RetrosynthesisEngine,
    RetrosynthesisResult,
)

# Set up logging
logger = logging.getLogger(__name__)


# Enums
class RouteStrategy(Enum):
    """Strategies for route enumeration."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    MONTE_CARLO = "monte_carlo"
    BEAM_SEARCH = "beam_search"


class PruningCriteria(Enum):
    """Criteria for pruning synthesis routes."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    COMPLEXITY_INCREASE = "complexity_increase"
    MAXIMUM_STEPS = "maximum_steps"
    AVAILABILITY_SCORE = "availability_score"
    COST_THRESHOLD = "cost_threshold"


class BuildingBlockSource(Enum):
    """Sources for chemical building blocks."""

    COMMERCIAL = "commercial"
    ACADEMIC = "academic"
    CUSTOM = "custom"
    ENAMINE = "enamine"
    CHEMBRIDGE = "chembridge"
    MCULE = "mcule"
    ZINC = "zinc"


class SynthesisComplexity(Enum):
    """Complexity levels for synthesis routes."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


# Data Structures
@dataclass
class SynthesisNode:
    """Node in a synthesis tree."""

    smiles: str
    depth: int = 0
    confidence: float = 1.0
    is_purchasable: bool = False
    cost_estimate: float = 0.0
    complexity_score: float = 0.0
    parent_prediction: Optional[ModelPrediction] = None
    children: List["SynthesisNode"] = field(default_factory=list)
    node_id: str = field(default="")

    def __post_init__(self):
        """Initialize node properties."""
        if not self.node_id:
            self.node_id = f"node_{abs(hash(self.smiles + str(self.depth)))}"

    def add_child(self, child: "SynthesisNode"):
        """Add a child node."""
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if node is a leaf (terminal)."""
        return len(self.children) == 0

    def get_molecular_representation(self) -> MolecularRepresentation:
        """Get molecular representation for this node."""
        return MolecularRepresentation(smiles=self.smiles)


@dataclass
class SynthesisTree:
    """Tree structure representing synthesis routes."""

    root: SynthesisNode
    max_depth: int = 10
    total_nodes: int = 0

    def __post_init__(self):
        """Initialize tree properties."""
        self.total_nodes = self._count_nodes(self.root)

    def _count_nodes(self, node: SynthesisNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def get_leaf_nodes(self) -> List[SynthesisNode]:
        """Get all leaf nodes in the tree."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: SynthesisNode, leaves: List[SynthesisNode]):
        """Recursively collect leaf nodes."""
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def get_complete_routes(self) -> List[List[SynthesisNode]]:
        """Get all complete routes (root to purchasable leaves)."""
        routes = []
        self._find_complete_routes(self.root, [], routes)
        return routes

    def _find_complete_routes(
        self,
        node: SynthesisNode,
        current_path: List[SynthesisNode],
        routes: List[List[SynthesisNode]],
    ):
        """Recursively find complete routes."""
        current_path = current_path + [node]

        if node.is_leaf() and node.is_purchasable:
            routes.append(current_path)
        else:
            for child in node.children:
                self._find_complete_routes(child, current_path, routes)

    def to_networkx(self) -> Any:
        """Convert tree to NetworkX graph."""
        if not _NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for graph conversion")

        G = nx.DiGraph()
        self._add_node_to_graph(self.root, G)
        return G

    def _add_node_to_graph(self, node: SynthesisNode, G: Any):
        """Recursively add nodes to NetworkX graph."""
        G.add_node(
            node.node_id,
            smiles=node.smiles,
            depth=node.depth,
            confidence=node.confidence,
            is_purchasable=node.is_purchasable,
            cost_estimate=node.cost_estimate,
        )

        for child in node.children:
            G.add_edge(node.node_id, child.node_id)
            self._add_node_to_graph(child, G)


@dataclass
class RouteEnumerationConfig:
    """Configuration for route enumeration."""

    strategy: RouteStrategy = RouteStrategy.BEST_FIRST
    max_depth: int = 10
    max_routes: int = 100
    confidence_threshold: float = 0.5
    complexity_threshold: float = 10.0
    cost_threshold: float = 1000.0
    beam_width: int = 5
    monte_carlo_iterations: int = 1000
    pruning_criteria: List[PruningCriteria] = field(
        default_factory=lambda: [
            PruningCriteria.CONFIDENCE_THRESHOLD,
            PruningCriteria.MAXIMUM_STEPS,
        ]
    )
    building_block_sources: List[BuildingBlockSource] = field(
        default_factory=lambda: [
            BuildingBlockSource.COMMERCIAL,
            BuildingBlockSource.ENAMINE,
        ]
    )
    enable_caching: bool = True
    parallel_workers: int = 4


@dataclass
class SynthesisRoute:
    """Complete synthesis route from target to starting materials."""

    target_smiles: str
    route_nodes: List[SynthesisNode]
    total_steps: int = 0
    overall_confidence: float = 0.0
    total_cost: float = 0.0
    complexity: SynthesisComplexity = SynthesisComplexity.MODERATE
    route_id: str = field(default="")

    def __post_init__(self):
        """Initialize route properties."""
        self.total_steps = len(self.route_nodes) - 1
        self.overall_confidence = self._calculate_overall_confidence()
        self.total_cost = self._calculate_total_cost()
        self.complexity = self._determine_complexity()

        if not self.route_id:
            self.route_id = (
                f"route_{abs(hash(self.target_smiles + str(self.total_steps)))}"
            )

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall route confidence."""
        if not self.route_nodes:
            return 0.0

        # Multiply individual confidences
        confidence = 1.0
        for node in self.route_nodes:
            confidence *= node.confidence

        return confidence

    def _calculate_total_cost(self) -> float:
        """Calculate total route cost."""
        return sum(node.cost_estimate for node in self.route_nodes)

    def _determine_complexity(self) -> SynthesisComplexity:
        """Determine route complexity."""
        if self.total_steps <= 2:
            return SynthesisComplexity.SIMPLE
        elif self.total_steps <= 4:
            return SynthesisComplexity.MODERATE
        elif self.total_steps <= 6:
            return SynthesisComplexity.COMPLEX
        else:
            return SynthesisComplexity.VERY_COMPLEX


@dataclass
class RouteMetrics:
    """Metrics for evaluating synthesis routes."""

    convergence: float = 0.0
    atom_economy: float = 0.0
    step_economy: float = 0.0
    green_chemistry_score: float = 0.0
    synthetic_accessibility: float = 0.0
    novelty_score: float = 0.0

    def overall_score(self) -> float:
        """Calculate overall route quality score."""
        weights = {
            "convergence": 0.2,
            "atom_economy": 0.2,
            "step_economy": 0.2,
            "green_chemistry_score": 0.15,
            "synthetic_accessibility": 0.15,
            "novelty_score": 0.1,
        }

        return (
            self.convergence * weights["convergence"]
            + self.atom_economy * weights["atom_economy"]
            + self.step_economy * weights["step_economy"]
            + self.green_chemistry_score * weights["green_chemistry_score"]
            + self.synthetic_accessibility * weights["synthetic_accessibility"]
            + self.novelty_score * weights["novelty_score"]
        )


@dataclass
class BuildingBlockQuery:
    """Query for building block database."""

    smiles: str
    max_price: float = 1000.0
    min_purity: float = 95.0
    sources: List[BuildingBlockSource] = field(
        default_factory=lambda: [BuildingBlockSource.COMMERCIAL]
    )
    max_delivery_time: int = 30  # days


@dataclass
class BuildingBlockResult:
    """Result from building block database query."""

    smiles: str
    is_available: bool = False
    price: float = 0.0
    purity: float = 0.0
    supplier: str = ""
    catalog_id: str = ""
    delivery_time: int = 0
    source: BuildingBlockSource = BuildingBlockSource.COMMERCIAL


# Exceptions
class RouteEnumerationError(Exception):
    """Base exception for route enumeration errors."""

    pass


class TreeConstructionError(RouteEnumerationError):
    """Error in tree construction."""

    pass


class DatabaseQueryError(RouteEnumerationError):
    """Error querying building block database."""

    pass


class OptimizationError(RouteEnumerationError):
    """Error in route optimization."""

    pass


# Main Classes
class BuildingBlockDatabase:
    """Database for querying chemical building blocks."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or ":memory:"
        self.connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize building block database."""
        try:
            self.connection = sqlite3.connect(self.database_path)
            self._create_tables()
            self._populate_sample_data()
        except Exception as e:
            raise DatabaseQueryError(f"Failed to initialize database: {e}")

    def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS building_blocks (
                id INTEGER PRIMARY KEY,
                smiles TEXT UNIQUE,
                price REAL,
                purity REAL,
                supplier TEXT,
                catalog_id TEXT,
                delivery_time INTEGER,
                source TEXT,
                molecular_weight REAL,
                complexity_score REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_smiles ON building_blocks(smiles)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_price ON building_blocks(price)
        """
        )

        self.connection.commit()

    def _populate_sample_data(self):
        """Populate database with sample building blocks."""
        cursor = self.connection.cursor()

        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM building_blocks")
        if cursor.fetchone()[0] > 0:
            return

        # Sample building blocks
        sample_blocks = [
            ("CCO", 10.0, 99.0, "Sigma-Aldrich", "E7023", 1, "commercial", 46.07, 1.0),
            ("C1=CC=CC=C1", 15.0, 98.0, "Fisher", "B313", 2, "commercial", 78.11, 2.5),
            ("CC(=O)O", 8.0, 99.5, "VWR", "A100", 1, "commercial", 60.05, 1.5),
            ("CC(C)O", 12.0, 99.0, "Merck", "I9516", 3, "commercial", 60.10, 1.8),
            ("C1CCCCC1", 20.0, 97.0, "TCI", "C0298", 5, "commercial", 84.16, 2.0),
            ("CC(=O)C", 18.0, 99.0, "Alfa", "A11696", 2, "commercial", 58.08, 2.2),
            ("CCN", 25.0, 98.5, "Acros", "E0245", 3, "commercial", 45.08, 1.9),
            ("C=C", 50.0, 99.9, "Praxair", "E1012", 7, "commercial", 28.05, 1.0),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO building_blocks
            (smiles, price, purity, supplier, catalog_id, delivery_time, source, molecular_weight, complexity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            sample_blocks,
        )

        self.connection.commit()

    def query_building_block(self, query: BuildingBlockQuery) -> BuildingBlockResult:
        """Query for a specific building block."""
        if not self.connection:
            raise DatabaseQueryError("Database connection not available")

        cursor = self.connection.cursor()

        # Build query
        sql = """
            SELECT smiles, price, purity, supplier, catalog_id, delivery_time, source
            FROM building_blocks
            WHERE smiles = ? AND price <= ? AND purity >= ? AND delivery_time <= ?
        """

        params = [
            query.smiles,
            query.max_price,
            query.min_purity,
            query.max_delivery_time,
        ]

        if query.sources:
            source_placeholders = ",".join("?" * len(query.sources))
            sql += f" AND source IN ({source_placeholders})"
            params.extend([source.value for source in query.sources])

        sql += " ORDER BY price ASC LIMIT 1"

        try:
            cursor.execute(sql, params)
            result = cursor.fetchone()

            if result:
                return BuildingBlockResult(
                    smiles=result[0],
                    is_available=True,
                    price=result[1],
                    purity=result[2],
                    supplier=result[3],
                    catalog_id=result[4],
                    delivery_time=result[5],
                    source=BuildingBlockSource(result[6]),
                )
            else:
                return BuildingBlockResult(smiles=query.smiles, is_available=False)

        except Exception as e:
            logger.error(f"Error querying building block: {e}")
            return BuildingBlockResult(smiles=query.smiles, is_available=False)

    def batch_query(
        self, queries: List[BuildingBlockQuery]
    ) -> List[BuildingBlockResult]:
        """Query multiple building blocks."""
        results = []
        for query in queries:
            result = self.query_building_block(query)
            results.append(result)
        return results

    def add_building_block(
        self,
        smiles: str,
        price: float,
        purity: float,
        supplier: str,
        catalog_id: str,
        delivery_time: int,
        source: BuildingBlockSource,
    ):
        """Add a new building block to the database."""
        if not self.connection:
            raise DatabaseQueryError("Database connection not available")

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                molecular_weight = Descriptors.MolWt(mol)
                complexity_score = rdMolDescriptors.BertzCT(mol)
            else:
                molecular_weight = 0.0
                complexity_score = 0.0

            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO building_blocks
                (smiles, price, purity, supplier, catalog_id, delivery_time, source, molecular_weight, complexity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    smiles,
                    price,
                    purity,
                    supplier,
                    catalog_id,
                    delivery_time,
                    source.value,
                    molecular_weight,
                    complexity_score,
                ),
            )

            self.connection.commit()

        except Exception as e:
            raise DatabaseQueryError(f"Failed to add building block: {e}")


class ComplexityCalculator:
    """Calculates synthesis complexity metrics."""

    def __init__(self):
        self.complexity_cache: Dict[str, float] = {}

    def calculate_molecular_complexity(self, smiles: str) -> float:
        """Calculate molecular complexity score."""
        if smiles in self.complexity_cache:
            return self.complexity_cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.0

            # Bertz complexity index
            complexity = rdMolDescriptors.BertzCT(mol)

            # Normalize to 0-10 scale
            normalized_complexity = min(10.0, complexity / 100.0)

            self.complexity_cache[smiles] = normalized_complexity
            return normalized_complexity

        except Exception as e:
            logger.error(f"Error calculating complexity for {smiles}: {e}")
            return 0.0

    def calculate_route_complexity(self, route: SynthesisRoute) -> float:
        """Calculate overall route complexity."""
        if not route.route_nodes:
            return 0.0

        # Weighted complexity based on steps and molecular complexity
        step_penalty = route.total_steps * 0.5

        # Average molecular complexity
        total_complexity = sum(
            self.calculate_molecular_complexity(node.smiles)
            for node in route.route_nodes
        )
        avg_molecular_complexity = total_complexity / len(route.route_nodes)

        return step_penalty + avg_molecular_complexity


class RouteOptimizer:
    """Optimizes synthesis routes based on various criteria."""

    def __init__(self, config: RouteEnumerationConfig):
        self.config = config
        self.complexity_calculator = ComplexityCalculator()

    def optimize_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        """Optimize and rank synthesis routes."""
        if not routes:
            return routes

        # Calculate metrics for each route
        for route in routes:
            route.metrics = self._calculate_route_metrics(route)

        # Sort by overall score
        optimized_routes = sorted(
            routes,
            key=lambda r: getattr(r, "metrics", RouteMetrics()).overall_score(),
            reverse=True,
        )

        return optimized_routes[: self.config.max_routes]

    def _calculate_route_metrics(self, route: SynthesisRoute) -> RouteMetrics:
        """Calculate comprehensive metrics for a route."""
        metrics = RouteMetrics()

        # Step economy
        metrics.step_economy = 1.0 / (1.0 + route.total_steps * 0.2)

        # Synthetic accessibility (based on complexity)
        complexity = self.complexity_calculator.calculate_route_complexity(route)
        metrics.synthetic_accessibility = 1.0 / (1.0 + complexity * 0.1)

        # Convergence (simplified)
        metrics.convergence = route.overall_confidence

        # Atom economy (simplified)
        if route.route_nodes:
            target_mol = Chem.MolFromSmiles(route.target_smiles)
            if target_mol:
                target_mw = Descriptors.MolWt(target_mol)

                # Sum of starting material molecular weights
                starting_mw = 0.0
                for node in route.route_nodes:
                    if node.is_purchasable:
                        mol = Chem.MolFromSmiles(node.smiles)
                        if mol:
                            starting_mw += Descriptors.MolWt(mol)

                if starting_mw > 0:
                    metrics.atom_economy = target_mw / starting_mw

        # Green chemistry score (simplified)
        metrics.green_chemistry_score = 0.5  # Default value

        # Novelty score (simplified)
        metrics.novelty_score = 0.5  # Default value

        return metrics


class SynthesisTreeBuilder:
    """Builds synthesis trees through iterative retrosynthetic analysis."""

    def __init__(
        self,
        retro_engine: RetrosynthesisEngine,
        building_block_db: BuildingBlockDatabase,
        config: RouteEnumerationConfig,
    ):
        self.retro_engine = retro_engine
        self.building_block_db = building_block_db
        self.config = config
        self.complexity_calculator = ComplexityCalculator()

        # Cache for synthesis trees
        self._tree_cache: Dict[str, SynthesisTree] = {}

    def build_tree(self, target_smiles: str) -> SynthesisTree:
        """Build synthesis tree for target molecule."""
        # Check cache
        if self.config.enable_caching and target_smiles in self._tree_cache:
            return self._tree_cache[target_smiles]

        # Create root node
        root = SynthesisNode(smiles=target_smiles, depth=0, confidence=1.0)

        # Check if target is already purchasable
        query = BuildingBlockQuery(smiles=target_smiles)
        result = self.building_block_db.query_building_block(query)

        if result.is_available:
            root.is_purchasable = True
            root.cost_estimate = result.price
            tree = SynthesisTree(root=root, max_depth=0)
        else:
            # Build tree recursively
            self._expand_node(root, 0)
            tree = SynthesisTree(root=root, max_depth=self.config.max_depth)

        # Cache result
        if self.config.enable_caching:
            self._tree_cache[target_smiles] = tree

        return tree

    def _expand_node(self, node: SynthesisNode, current_depth: int):
        """Recursively expand synthesis tree node."""
        if current_depth >= self.config.max_depth:
            return

        # Check if molecule is purchasable
        query = BuildingBlockQuery(
            smiles=node.smiles, sources=self.config.building_block_sources
        )
        result = self.building_block_db.query_building_block(query)

        if result.is_available:
            node.is_purchasable = True
            node.cost_estimate = result.price
            return

        # Get retrosynthetic predictions
        try:
            retro_result = self.retro_engine.predict_retrosynthesis(node.smiles)

            for prediction in retro_result.predictions[: self.config.beam_width]:
                # Apply pruning criteria
                if not self._should_expand_prediction(prediction, current_depth):
                    continue

                # Create child nodes for reactants
                for reactant_smiles in prediction.reactants:
                    if self._is_valid_reactant(reactant_smiles):
                        child = SynthesisNode(
                            smiles=reactant_smiles,
                            depth=current_depth + 1,
                            confidence=node.confidence * prediction.confidence,
                            parent_prediction=prediction,
                        )

                        # Calculate complexity
                        child.complexity_score = (
                            self.complexity_calculator.calculate_molecular_complexity(
                                reactant_smiles
                            )
                        )

                        node.add_child(child)

                        # Recursively expand child
                        self._expand_node(child, current_depth + 1)

        except Exception as e:
            logger.error(f"Error expanding node {node.smiles}: {e}")

    def _should_expand_prediction(
        self, prediction: ModelPrediction, depth: int
    ) -> bool:
        """Check if prediction should be expanded based on pruning criteria."""
        for criteria in self.config.pruning_criteria:
            if criteria == PruningCriteria.CONFIDENCE_THRESHOLD:
                if prediction.confidence < self.config.confidence_threshold:
                    return False

            elif criteria == PruningCriteria.MAXIMUM_STEPS:
                if depth >= self.config.max_depth:
                    return False

            elif criteria == PruningCriteria.COMPLEXITY_INCREASE:
                # Check if reactants are significantly more complex
                for reactant in prediction.reactants:
                    reactant_complexity = (
                        self.complexity_calculator.calculate_molecular_complexity(
                            reactant
                        )
                    )
                    if reactant_complexity > self.config.complexity_threshold:
                        return False

        return True

    def _is_valid_reactant(self, smiles: str) -> bool:
        """Check if reactant is chemically valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() > 0
        except:
            return False


class RouteEnumerator:
    """Main class for enumerating synthesis routes."""

    def __init__(
        self,
        config: Optional[RouteEnumerationConfig] = None,
        retro_config: Optional[RetrosynthesisConfig] = None,
    ):
        self.config = config or RouteEnumerationConfig()

        # Initialize components
        self.retro_engine = RetrosynthesisEngine(retro_config)
        self.building_block_db = BuildingBlockDatabase()
        self.tree_builder = SynthesisTreeBuilder(
            self.retro_engine, self.building_block_db, self.config
        )
        self.route_optimizer = RouteOptimizer(self.config)

        # Cache for enumerated routes
        self._route_cache: Dict[str, List[SynthesisRoute]] = {}

    def enumerate_routes(self, target_smiles: str) -> List[SynthesisRoute]:
        """Enumerate synthesis routes for target molecule."""
        # Check cache
        if self.config.enable_caching and target_smiles in self._route_cache:
            return self._route_cache[target_smiles]

        start_time = time.time()

        try:
            # Build synthesis tree
            tree = self.tree_builder.build_tree(target_smiles)

            # Extract complete routes
            complete_routes = tree.get_complete_routes()

            # Convert to SynthesisRoute objects
            synthesis_routes = []
            for route_nodes in complete_routes:
                route = SynthesisRoute(
                    target_smiles=target_smiles, route_nodes=route_nodes
                )
                synthesis_routes.append(route)

            # Optimize and rank routes
            optimized_routes = self.route_optimizer.optimize_routes(synthesis_routes)

            # Apply final filtering
            filtered_routes = self._filter_routes(optimized_routes)

            # Cache result
            if self.config.enable_caching:
                self._route_cache[target_smiles] = filtered_routes

            elapsed_time = time.time() - start_time
            logger.info(
                f"Enumerated {len(filtered_routes)} routes for {target_smiles} in {elapsed_time:.2f}s"
            )

            return filtered_routes

        except Exception as e:
            logger.error(f"Error enumerating routes for {target_smiles}: {e}")
            raise RouteEnumerationError(f"Route enumeration failed: {e}")

    def _filter_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        """Apply final filtering to routes."""
        filtered = []

        for route in routes:
            # Apply cost threshold
            if (
                PruningCriteria.COST_THRESHOLD in self.config.pruning_criteria
                and route.total_cost > self.config.cost_threshold
            ):
                continue

            # Apply confidence threshold
            if route.overall_confidence < self.config.confidence_threshold:
                continue

            filtered.append(route)

        return filtered[: self.config.max_routes]

    def get_route_statistics(self, routes: List[SynthesisRoute]) -> Dict[str, Any]:
        """Get statistics for enumerated routes."""
        if not routes:
            return {}

        return {
            "total_routes": len(routes),
            "avg_steps": np.mean([r.total_steps for r in routes]),
            "avg_confidence": np.mean([r.overall_confidence for r in routes]),
            "avg_cost": np.mean([r.total_cost for r in routes]),
            "complexity_distribution": {
                complexity.value: sum(1 for r in routes if r.complexity == complexity)
                for complexity in SynthesisComplexity
            },
            "max_steps": max([r.total_steps for r in routes]),
            "min_steps": min([r.total_steps for r in routes]),
            "best_confidence": max([r.overall_confidence for r in routes]),
            "lowest_cost": min([r.total_cost for r in routes]),
        }


# Utility Functions
def create_route_config(
    strategy: RouteStrategy = RouteStrategy.BEST_FIRST,
    max_depth: int = 10,
    max_routes: int = 100,
    **kwargs,
) -> RouteEnumerationConfig:
    """Create route enumeration configuration."""
    return RouteEnumerationConfig(
        strategy=strategy, max_depth=max_depth, max_routes=max_routes, **kwargs
    )


def build_synthesis_tree(
    target_smiles: str, config: Optional[RouteEnumerationConfig] = None
) -> SynthesisTree:
    """Build synthesis tree for target molecule."""
    enumerator = RouteEnumerator(config)
    return enumerator.tree_builder.build_tree(target_smiles)


def enumerate_routes(
    target_smiles: str, config: Optional[RouteEnumerationConfig] = None
) -> List[SynthesisRoute]:
    """Enumerate synthesis routes for target molecule."""
    enumerator = RouteEnumerator(config)
    return enumerator.enumerate_routes(target_smiles)


def optimize_route(
    routes: List[SynthesisRoute], config: Optional[RouteEnumerationConfig] = None
) -> List[SynthesisRoute]:
    """Optimize and rank synthesis routes."""
    config = config or RouteEnumerationConfig()
    optimizer = RouteOptimizer(config)
    return optimizer.optimize_routes(routes)


def calculate_complexity(smiles: str) -> float:
    """Calculate molecular complexity score."""
    calculator = ComplexityCalculator()
    return calculator.calculate_molecular_complexity(smiles)


def query_building_blocks(
    smiles_list: List[str], sources: Optional[List[BuildingBlockSource]] = None
) -> List[BuildingBlockResult]:
    """Query building block availability for multiple compounds."""
    db = BuildingBlockDatabase()
    sources = sources or [BuildingBlockSource.COMMERCIAL]

    queries = [
        BuildingBlockQuery(smiles=smiles, sources=sources) for smiles in smiles_list
    ]

    return db.batch_query(queries)
