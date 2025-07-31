"""
High-Performance Pharmacophore Screening Module

This module provides a unified interface for pharmacophore screening with automatic
backend selection (CPU/GPU) and integration with the ChemiAccel ecosystem.

Features:
- Automatic GPU/CPU backend selection
- Integration with existing molecular descriptors
- High-level screening workflows for drug discovery
- Pharmacophore query builder and optimization
- Real-time screening capabilities
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.molecular import MolecularData
from ..utils.logging import get_logger
from .advanced import PharmacophoreAnalyzer

# Try to import GPU backend
try:
    from ..backends.cuda.pharmacophore_screening import (
        GPUPharmacophoreScreener,
        PharmacophoreQuery,
        ScreeningResult,
        create_pharmacophore_query,
        screen_database_gpu,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUPharmacophoreScreener = None
    PharmacophoreQuery = None
    ScreeningResult = None

logger = get_logger()


class PharmacophoreScreeningEngine:
    """Unified pharmacophore screening engine with automatic backend selection."""

    def __init__(
        self,
        backend: str = "auto",
        batch_size: int = 10000,
        use_gpu_acceleration: bool = True,
        memory_limit_gb: float = 4.0,
    ):
        """Initialize pharmacophore screening engine.

        Args:
            backend: Backend to use ("auto", "gpu", "cpu")
            batch_size: Number of molecules per batch
            use_gpu_acceleration: Whether to use GPU if available
            memory_limit_gb: GPU memory limit
        """
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb

        # Select backend
        if backend == "auto":
            self.backend = "gpu" if (GPU_AVAILABLE and use_gpu_acceleration) else "cpu"
        else:
            self.backend = backend

        if self.backend == "gpu" and not GPU_AVAILABLE:
            logger.warning(
                "GPU backend requested but not available, falling back to CPU"
            )
            self.backend = "cpu"

        # Initialize backend-specific components
        if self.backend == "gpu":
            self.gpu_screener = GPUPharmacophoreScreener(
                batch_size=batch_size, memory_limit_gb=memory_limit_gb
            )
            self.cpu_analyzer = None
        else:
            self.gpu_screener = None
            self.cpu_analyzer = PharmacophoreAnalyzer()

        # Screening statistics
        self.screening_stats = {
            "total_molecules_screened": 0,
            "total_hits_found": 0,
            "total_screening_time": 0.0,
            "queries_run": 0,
        }

        logger.info(
            f"Pharmacophore screening engine initialized with {self.backend.upper()} backend"
        )

    def create_query(
        self,
        feature_requirements: Dict[str, Union[int, Tuple[int, int]]],
        distance_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        tolerance: float = 0.5,
        query_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pharmacophore query with flexible feature specification.

        Args:
            feature_requirements: Feature counts or ranges (e.g., {"donor": 2, "acceptor": (1,3)})
            distance_constraints: 3D distance constraints between features
            tolerance: Matching tolerance for continuous features
            query_name: Optional name for the query

        Returns:
            Query object compatible with both backends
        """
        # Convert range specifications to exact counts (use minimum)
        feature_counts = {}
        for feature, requirement in feature_requirements.items():
            if isinstance(requirement, tuple):
                feature_counts[feature] = requirement[0]  # Use minimum count
            else:
                feature_counts[feature] = requirement

        if self.backend == "gpu":
            return create_pharmacophore_query(
                feature_counts=feature_counts,
                distance_constraints=distance_constraints,
                query_id=query_name,
            )
        else:
            # CPU backend query format
            return {
                "feature_counts": feature_counts,
                "distance_constraints": distance_constraints,
                "tolerance": tolerance,
                "query_name": query_name or f"cpu_query_{int(time.time())}",
            }

    def screen_molecules(
        self,
        molecules: List[MolecularData],
        query: Union[Dict[str, Any], Any],
        similarity_threshold: float = 0.6,
        max_results: Optional[int] = None,
        return_details: bool = True,
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """Screen molecules against pharmacophore query.

        Args:
            molecules: Molecules to screen
            query: Pharmacophore query (backend-specific format)
            similarity_threshold: Minimum similarity for hits
            max_results: Maximum number of results
            return_details: Whether to return detailed results

        Returns:
            Screening results as list of dictionaries or DataFrame
        """
        start_time = time.time()

        logger.info(
            f"Starting pharmacophore screening of {len(molecules)} molecules "
            f"using {self.backend.upper()} backend"
        )

        if self.backend == "gpu":
            results = self._screen_gpu(
                molecules, query, similarity_threshold, max_results
            )
        else:
            results = self._screen_cpu(
                molecules, query, similarity_threshold, max_results
            )

        screening_time = time.time() - start_time
        hits = len([r for r in results if self._is_hit(r, similarity_threshold)])

        # Update statistics
        self.screening_stats["total_molecules_screened"] += len(molecules)
        self.screening_stats["total_hits_found"] += hits
        self.screening_stats["total_screening_time"] += screening_time
        self.screening_stats["queries_run"] += 1

        logger.info(
            f"Screening completed: {hits}/{len(molecules)} hits in {screening_time:.2f}s "
            f"({len(molecules)/screening_time:.0f} molecules/sec)"
        )

        if return_details:
            return results
        else:
            # Return simplified DataFrame
            return self._results_to_dataframe(results)

    def _screen_gpu(self, molecules, query, threshold, max_results):
        """Screen using GPU backend."""
        try:
            gpu_results = self.gpu_screener.screen_molecules(
                molecules, query, threshold, max_results
            )

            # Convert to common format
            return [
                {
                    "mol_id": r.mol_id,
                    "smiles": r.smiles,
                    "similarity_score": r.similarity_score,
                    "feature_matches": r.feature_match_scores,
                    "distance_matches": r.distance_match_scores or {},
                    "matches_query": r.matches_query,
                    "screening_time_ms": r.screening_time_ms,
                    "backend": "gpu",
                }
                for r in gpu_results
            ]

        except Exception as e:
            logger.error(f"GPU screening failed: {e}")
            logger.info("Falling back to CPU backend")
            return self._screen_cpu(molecules, query, threshold, max_results)

    def _screen_cpu(self, molecules, query, threshold, max_results):
        """Screen using CPU backend."""
        results = []

        for mol in molecules:
            try:
                start_time = time.time()

                # Extract pharmacophore fingerprint
                fingerprint = self.cpu_analyzer.calculate_pharmacophore_fingerprint(
                    mol, include_3d=True
                )

                # Calculate similarity to query
                similarity = self._calculate_cpu_similarity(fingerprint, query)

                screening_time = (time.time() - start_time) * 1000  # ms

                result = {
                    "mol_id": mol.mol_id,
                    "smiles": mol.smiles,
                    "similarity_score": similarity,
                    "feature_matches": fingerprint.get("feature_counts", {}),
                    "distance_matches": fingerprint.get("3d_distances", {}),
                    "matches_query": similarity >= threshold,
                    "screening_time_ms": screening_time,
                    "backend": "cpu",
                }

                results.append(result)

            except Exception as e:
                logger.warning(f"CPU screening failed for molecule {mol.mol_id}: {e}")
                result = {
                    "mol_id": mol.mol_id,
                    "smiles": mol.smiles,
                    "similarity_score": 0.0,
                    "feature_matches": {},
                    "distance_matches": {},
                    "matches_query": False,
                    "screening_time_ms": 0.0,
                    "backend": "cpu",
                    "error": str(e),
                }
                results.append(result)

        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Apply max_results limit
        if max_results:
            results = results[:max_results]

        return results

    def _calculate_cpu_similarity(self, fingerprint, query):
        """Calculate similarity for CPU backend."""
        query_counts = query.get("feature_counts", {})
        mol_counts = fingerprint.get("feature_counts", {})

        if not query_counts or not mol_counts:
            return 0.0

        # Tanimoto similarity for feature counts
        intersection = 0
        union = 0

        all_features = set(query_counts.keys()) | set(mol_counts.keys())

        for feature in all_features:
            q_count = query_counts.get(feature, 0)
            m_count = mol_counts.get(feature, 0)

            intersection += min(q_count, m_count)
            union += max(q_count, m_count)

        if union == 0:
            return 0.0

        return intersection / union

    def _is_hit(self, result, threshold):
        """Check if result is a hit."""
        if isinstance(result, dict):
            return (
                result.get("matches_query", False)
                or result.get("similarity_score", 0) >= threshold
            )
        else:
            return result.matches_query

    def _results_to_dataframe(self, results):
        """Convert results to pandas DataFrame."""
        if not results:
            return pd.DataFrame()

        # Flatten nested dictionaries for DataFrame compatibility
        flattened = []
        for result in results:
            flat_result = {
                "mol_id": result["mol_id"],
                "smiles": result["smiles"],
                "similarity_score": result["similarity_score"],
                "matches_query": result["matches_query"],
                "screening_time_ms": result["screening_time_ms"],
                "backend": result["backend"],
            }

            # Add feature match scores
            for feature, score in result.get("feature_matches", {}).items():
                flat_result[f"feature_{feature}"] = score

            flattened.append(flat_result)

        return pd.DataFrame(flattened)

    def batch_screen_database(
        self,
        molecules: List[MolecularData],
        queries: List[Dict[str, Any]],
        similarity_threshold: float = 0.6,
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """Screen large molecular database against multiple queries.

        Args:
            molecules: Large list of molecules to screen
            queries: List of pharmacophore queries
            similarity_threshold: Minimum similarity threshold
            output_file: Optional file to save results

        Returns:
            Combined results DataFrame
        """
        logger.info(
            f"Starting batch screening: {len(molecules)} molecules x {len(queries)} queries"
        )

        all_results = []

        for i, query in enumerate(queries):
            query_name = query.get("query_name", f"query_{i}")
            logger.info(f"Processing query {i+1}/{len(queries)}: {query_name}")

            query_results = self.screen_molecules(
                molecules, query, similarity_threshold, return_details=False
            )

            # Add query information
            query_results["query_name"] = query_name
            query_results["query_index"] = i

            all_results.append(query_results)

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() == ".csv":
                combined_results.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in [".xlsx", ".xls"]:
                combined_results.to_excel(output_path, index=False)
            else:
                # Default to CSV
                combined_results.to_csv(output_path.with_suffix(".csv"), index=False)

            logger.info(f"Results saved to {output_path}")

        return combined_results

    def optimize_query(
        self,
        training_molecules: List[MolecularData],
        active_labels: List[bool],
        initial_query: Dict[str, Any],
        optimization_method: str = "feature_weighting",
    ) -> Dict[str, Any]:
        """Optimize pharmacophore query using training data.

        Args:
            training_molecules: Training molecules
            active_labels: Activity labels (True = active)
            initial_query: Initial query to optimize
            optimization_method: Optimization approach

        Returns:
            Optimized query
        """
        logger.info(
            f"Optimizing pharmacophore query with {len(training_molecules)} training molecules"
        )

        if self.backend == "gpu" and hasattr(self.gpu_screener, "optimize_query"):
            # Convert CPU query format to GPU format if needed
            if isinstance(initial_query, dict):
                gpu_query = create_pharmacophore_query(
                    feature_counts=initial_query.get("feature_counts", {}),
                    distance_constraints=initial_query.get("distance_constraints"),
                    query_id=initial_query.get("query_name", "optimization_query"),
                )
            else:
                gpu_query = initial_query

            optimized_gpu_query = self.gpu_screener.optimize_query(
                training_molecules, active_labels, gpu_query
            )

            # Convert back to common format
            return {
                "feature_counts": optimized_gpu_query.feature_counts,
                "distance_constraints": optimized_gpu_query.distance_constraints,
                "weight_vector": (
                    optimized_gpu_query.weight_vector.tolist()
                    if optimized_gpu_query.weight_vector is not None
                    else None
                ),
                "tolerance": optimized_gpu_query.tolerance,
                "query_name": optimized_gpu_query.query_id,
            }

        else:
            # CPU-based optimization (simplified)
            return self._optimize_query_cpu(
                training_molecules, active_labels, initial_query
            )

    def _optimize_query_cpu(self, training_molecules, active_labels, initial_query):
        """CPU-based query optimization."""
        # Extract features for all training molecules
        active_features = []
        inactive_features = []

        for i, mol in enumerate(training_molecules):
            fingerprint = self.cpu_analyzer.calculate_pharmacophore_fingerprint(mol)
            feature_counts = fingerprint.get("feature_counts", {})

            if active_labels[i]:
                active_features.append(feature_counts)
            else:
                inactive_features.append(feature_counts)

        # Calculate discriminative features
        optimized_counts = initial_query["feature_counts"].copy()

        if active_features and inactive_features:
            for feature in optimized_counts.keys():
                active_values = [f.get(feature, 0) for f in active_features]
                inactive_values = [f.get(feature, 0) for f in inactive_features]

                if active_values and inactive_values:
                    active_mean = np.mean(active_values)
                    inactive_mean = np.mean(inactive_values)

                    # Adjust required count based on discrimination
                    if active_mean > inactive_mean:
                        optimized_counts[feature] = max(1, int(active_mean * 0.8))
                    else:
                        optimized_counts[feature] = max(0, int(inactive_mean * 0.5))

        optimized_query = initial_query.copy()
        optimized_query["feature_counts"] = optimized_counts
        optimized_query[
            "query_name"
        ] = f"{initial_query.get('query_name', 'query')}_optimized"

        return optimized_query

    def get_screening_statistics(self) -> Dict[str, Any]:
        """Get cumulative screening statistics."""
        stats = self.screening_stats.copy()

        if stats["total_molecules_screened"] > 0:
            stats["average_hit_rate"] = (
                stats["total_hits_found"] / stats["total_molecules_screened"]
            )
            stats["average_molecules_per_second"] = stats[
                "total_molecules_screened"
            ] / max(stats["total_screening_time"], 0.001)
        else:
            stats["average_hit_rate"] = 0.0
            stats["average_molecules_per_second"] = 0.0

        stats["backend"] = self.backend
        stats["gpu_available"] = GPU_AVAILABLE

        return stats

    def benchmark_performance(self, num_molecules: int = 10000) -> Dict[str, float]:
        """Benchmark screening performance."""
        if self.backend == "gpu" and self.gpu_screener:
            return self.gpu_screener.benchmark_performance(num_molecules)
        else:
            # Simple CPU benchmark
            from rdkit import Chem

            # Create test molecules
            test_molecules = []
            for i in range(min(num_molecules, 1000)):  # Limit CPU test size
                smiles = f"C{'C' * (i % 10)}O"
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol_data = MolecularData(
                        mol_id=f"test_{i}",
                        smiles=smiles,
                        backend_data={"rdkit_mol": mol},
                    )
                    test_molecules.append(mol_data)

            # Create simple query
            query = self.create_query({"donor": 1, "acceptor": 1})

            # Run benchmark
            start_time = time.time()
            results = self.screen_molecules(test_molecules, query, return_details=False)
            end_time = time.time()

            total_time = end_time - start_time

            return {
                "backend": "cpu",
                "total_molecules": len(test_molecules),
                "total_time_seconds": total_time,
                "molecules_per_second": len(test_molecules) / total_time,
                "hits_found": (
                    len(results[results["matches_query"]]) if not results.empty else 0
                ),
            }

    def cleanup(self):
        """Clean up resources."""
        if self.gpu_screener:
            self.gpu_screener.cleanup()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


# High-level convenience functions
def create_screening_engine(
    backend: str = "auto", **kwargs
) -> PharmacophoreScreeningEngine:
    """Create a pharmacophore screening engine with automatic configuration."""
    return PharmacophoreScreeningEngine(backend=backend, **kwargs)


def quick_screen(
    molecules: List[MolecularData],
    feature_requirements: Dict[str, Union[int, Tuple[int, int]]],
    similarity_threshold: float = 0.6,
) -> pd.DataFrame:
    """Quick pharmacophore screening with automatic backend selection."""
    engine = create_screening_engine()
    query = engine.create_query(feature_requirements)
    return engine.screen_molecules(
        molecules, query, similarity_threshold, return_details=False
    )
