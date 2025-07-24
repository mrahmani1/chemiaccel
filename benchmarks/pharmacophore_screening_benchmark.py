"""
GPU-Accelerated Pharmacophore Screening Benchmarks

This module provides comprehensive benchmarking and validation for the
GPU pharmacophore screening implementation, comparing performance against
traditional CPU methods and validating accuracy.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict

# Import core modules
from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.pharmacophore_screening import (
    PharmacophoreScreeningEngine,
    create_screening_engine
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Import GPU modules if available
try:
    from src.chemi_accel.backends.cuda.pharmacophore_screening import (
        GPUPharmacophoreScreener,
        create_pharmacophore_query
    )
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@dataclass
class BenchmarkConfig:
    """Configuration for pharmacophore screening benchmarks."""
    
    dataset_sizes: List[int]
    query_complexities: List[str]  # ["simple", "medium", "complex"]
    similarity_thresholds: List[float]
    iterations: int
    backends: List[str]  # ["cpu", "gpu", "auto"]
    batch_sizes: List[int]
    include_3d_constraints: bool
    save_detailed_results: bool
    output_directory: str

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    backend: str
    dataset_size: int
    query_complexity: str
    batch_size: int
    similarity_threshold: float
    iteration: int
    
    # Performance metrics
    total_time_seconds: float
    molecules_per_second: float
    hits_found: int
    hit_rate: float
    
    # Resource usage
    memory_used_mb: float
    gpu_memory_used_mb: float
    
    # Quality metrics
    average_similarity_score: float
    similarity_score_std: float
    
    # Additional metadata
    timestamp: str
    error_occurred: bool = False
    error_message: str = ""

class PharmacophoreScreeningBenchmark:
    """Comprehensive benchmarking suite for pharmacophore screening."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite with configuration."""
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized pharmacophore screening benchmark suite")
        self.logger.info(f"Dataset sizes: {config.dataset_sizes}")
        self.logger.info(f"Backends: {config.backends}")
        self.logger.info(f"GPU available: {GPU_AVAILABLE}")

    def generate_test_datasets(self) -> Dict[int, List[MolecularData]]:
        """Generate test molecular datasets of different sizes."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for test dataset generation")
        
        self.logger.info("Generating test molecular datasets...")
        
        # Base SMILES patterns for diversity
        base_patterns = [
            # Simple molecules
            "CCO", "CC(C)O", "CCCCO", "CC(C)(C)O",
            
            # Aromatic compounds
            "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C(=O)O)cc1",
            
            # Heterocycles
            "c1cccnc1", "c1ccoc1", "c1ccsc1", "c1c[nH]cn1",
            
            # Drug-like molecules
            "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",  # Ibuprofen-like
            "CC(=O)Nc1ccc(cc1)O",  # Paracetamol-like
            "CC1=CC=CC=C1C(=O)O",  # Aspirin-like
            
            # Complex structures
            "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C(=O)NC3=CC=C(C=C3)Cl",
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
            
            # Peptide-like
            "CC(C)CC(=O)NC(CC1=CNC2=CC=CC=C12)C(=O)O",
            
            # Natural product-like
            "CC1CC2C(C(C(C(O2)OC3C(C(C(C(O3)CO)O)O)O)O)O)OC1",
        ]
        
        datasets = {}
        
        for size in self.config.dataset_sizes:
            self.logger.info(f"Generating dataset of size {size}...")
            
            molecules = []
            for i in range(size):
                # Select base pattern and add variations
                base_smiles = base_patterns[i % len(base_patterns)]
                
                # Add structural variations
                if i >= len(base_patterns):
                    variation_factor = i // len(base_patterns)
                    smiles = self._add_structural_variation(base_smiles, variation_factor)
                else:
                    smiles = base_smiles
                
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetNumAtoms() <= 100:  # Reasonable size limit
                    try:
                        # Generate 3D coordinates
                        AllChem.EmbedMolecule(mol, randomSeed=42 + i)
                        AllChem.MMFFOptimizeMolecule(mol)
                        
                        mol_data = MolecularData(
                            mol_id=f"mol_{size}_{i:06d}",
                            smiles=Chem.MolToSmiles(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        molecules.append(mol_data)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to generate molecule {i}: {e}")
                        continue
                
                if len(molecules) >= size:
                    break
            
            datasets[size] = molecules
            self.logger.info(f"Generated {len(molecules)} valid molecules for size {size}")
        
        return datasets

    def _add_structural_variation(self, base_smiles: str, variation_factor: int) -> str:
        """Add structural variations to base SMILES."""
        variations = [
            lambda s: s.replace("C", "CC", 1) if "C" in s else s,  # Add carbon
            lambda s: s.replace("O", "N", 1) if "O" in s else s,   # Bioisostere
            lambda s: f"F{s}" if not s.startswith("F") else s,     # Add fluorine
            lambda s: s.replace("c1ccccc1", "c1ccc(Cl)cc1", 1),   # Add chlorine to benzene
            lambda s: s.replace("C(=O)O", "C(=O)N", 1),           # Acid to amide
        ]
        
        variation_index = variation_factor % len(variations)
        try:
            return variations[variation_index](base_smiles)
        except:
            return base_smiles

    def create_benchmark_queries(self) -> Dict[str, Dict[str, Any]]:
        """Create pharmacophore queries of different complexities."""
        queries = {}
        
        # Simple query - basic hydrogen bonding
        queries["simple"] = {
            "feature_counts": {"donor": 1, "acceptor": 1},
            "description": "Simple hydrogen bonding pharmacophore"
        }
        
        # Medium complexity - multi-feature
        queries["medium"] = {
            "feature_counts": {"donor": 1, "acceptor": 2, "aromatic": 1, "hydrophobic": 1},
            "description": "Multi-feature pharmacophore"
        }
        
        # Complex query - with distance constraints
        queries["complex"] = {
            "feature_counts": {"donor": 2, "acceptor": 2, "aromatic": 1, "hydrophobic": 2},
            "distance_constraints": {
                "donor_acceptor": (2.0, 8.0),
                "aromatic_hydrophobic": (3.0, 10.0)
            } if self.config.include_3d_constraints else None,
            "description": "Complex multi-feature pharmacophore with distance constraints"
        }
        
        return queries

    def run_single_benchmark(self,
                           molecules: List[MolecularData],
                           backend: str,
                           query_spec: Dict[str, Any],
                           query_complexity: str,
                           batch_size: int,
                           similarity_threshold: float,
                           iteration: int) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Create screening engine
            engine = PharmacophoreScreeningEngine(
                backend=backend,
                batch_size=batch_size,
                memory_limit_gb=4.0
            )
            
            # Create query
            query = engine.create_query(
                feature_requirements=query_spec["feature_counts"],
                distance_constraints=query_spec.get("distance_constraints"),
                query_name=f"{query_complexity}_{backend}_{iteration}"
            )
            
            # Measure memory usage before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            gpu_memory_before = 0
            if backend == "gpu" and GPU_AVAILABLE:
                try:
                    gpu_memory_before = cp.get_default_memory_pool().used_bytes() / 1024 / 1024
                except:
                    pass
            
            # Run screening
            start_time = time.time()
            
            results = engine.screen_molecules(
                molecules,
                query,
                similarity_threshold=similarity_threshold,
                return_details=True
            )
            
            end_time = time.time()
            
            # Measure memory usage after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            gpu_memory_after = 0
            if backend == "gpu" and GPU_AVAILABLE:
                try:
                    gpu_memory_after = cp.get_default_memory_pool().used_bytes() / 1024 / 1024
                except:
                    pass
            
            # Calculate metrics
            total_time = end_time - start_time
            molecules_per_second = len(molecules) / total_time if total_time > 0 else 0
            
            hits = [r for r in results if r.get("matches_query", False)]
            hit_rate = len(hits) / len(molecules) if molecules else 0
            
            similarity_scores = [r.get("similarity_score", 0.0) for r in results]
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
            
            # Cleanup
            engine.cleanup()
            
            return BenchmarkResult(
                backend=backend,
                dataset_size=len(molecules),
                query_complexity=query_complexity,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
                iteration=iteration,
                total_time_seconds=total_time,
                molecules_per_second=molecules_per_second,
                hits_found=len(hits),
                hit_rate=hit_rate,
                memory_used_mb=memory_after - memory_before,
                gpu_memory_used_mb=gpu_memory_after - gpu_memory_before,
                average_similarity_score=avg_similarity,
                similarity_score_std=similarity_std,
                timestamp=timestamp,
                error_occurred=False
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            
            return BenchmarkResult(
                backend=backend,
                dataset_size=len(molecules) if molecules else 0,
                query_complexity=query_complexity,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
                iteration=iteration,
                total_time_seconds=0.0,
                molecules_per_second=0.0,
                hits_found=0,
                hit_rate=0.0,
                memory_used_mb=0.0,
                gpu_memory_used_mb=0.0,
                average_similarity_score=0.0,
                similarity_score_std=0.0,
                timestamp=timestamp,
                error_occurred=True,
                error_message=str(e)
            )

    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations."""
        self.logger.info("Starting comprehensive pharmacophore screening benchmark...")
        
        # Generate test datasets
        datasets = self.generate_test_datasets()
        
        # Create benchmark queries
        queries = self.create_benchmark_queries()
        
        # Run all benchmark combinations
        total_runs = (
            len(self.config.dataset_sizes) *
            len(self.config.query_complexities) *
            len(self.config.backends) *
            len(self.config.batch_sizes) *
            len(self.config.similarity_thresholds) *
            self.config.iterations
        )
        
        self.logger.info(f"Running {total_runs} benchmark configurations...")
        
        run_count = 0
        
        for dataset_size in self.config.dataset_sizes:
            molecules = datasets[dataset_size]
            
            for query_complexity in self.config.query_complexities:
                if query_complexity not in queries:
                    self.logger.warning(f"Skipping undefined query complexity: {query_complexity}")
                    continue
                
                query_spec = queries[query_complexity]
                
                for backend in self.config.backends:
                    # Skip GPU backend if not available
                    if backend == "gpu" and not GPU_AVAILABLE:
                        self.logger.info(f"Skipping GPU backend - not available")
                        continue
                    
                    for batch_size in self.config.batch_sizes:
                        for similarity_threshold in self.config.similarity_thresholds:
                            for iteration in range(self.config.iterations):
                                run_count += 1
                                
                                self.logger.info(
                                    f"Run {run_count}/{total_runs}: "
                                    f"{backend} backend, {dataset_size} molecules, "
                                    f"{query_complexity} query, batch_size={batch_size}, "
                                    f"threshold={similarity_threshold}, iter={iteration}"
                                )
                                
                                result = self.run_single_benchmark(
                                    molecules=molecules,
                                    backend=backend,
                                    query_spec=query_spec,
                                    query_complexity=query_complexity,
                                    batch_size=batch_size,
                                    similarity_threshold=similarity_threshold,
                                    iteration=iteration
                                )
                                
                                self.results.append(result)
                                
                                # Log progress
                                if not result.error_occurred:
                                    self.logger.info(
                                        f"  → {result.molecules_per_second:.1f} mol/sec, "
                                        f"{result.hits_found} hits ({result.hit_rate:.3f})"
                                    )
                                else:
                                    self.logger.warning(f"  → ERROR: {result.error_message}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in self.results])
        
        self.logger.info("Benchmark completed!")
        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights."""
        self.logger.info("Analyzing benchmark results...")
        
        analysis = {}
        
        # Performance analysis
        analysis["performance_summary"] = self._analyze_performance(results_df)
        
        # Scalability analysis
        analysis["scalability_analysis"] = self._analyze_scalability(results_df)
        
        # Backend comparison
        analysis["backend_comparison"] = self._compare_backends(results_df)
        
        # Query complexity impact
        analysis["query_complexity_impact"] = self._analyze_query_complexity(results_df)
        
        # Batch size optimization
        analysis["batch_size_optimization"] = self._analyze_batch_sizes(results_df)
        
        # Error analysis
        analysis["error_analysis"] = self._analyze_errors(results_df)
        
        return analysis

    def _analyze_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall performance metrics."""
        successful_runs = df[~df["error_occurred"]]
        
        if len(successful_runs) == 0:
            return {"error": "No successful runs to analyze"}
        
        return {
            "average_molecules_per_second": successful_runs["molecules_per_second"].mean(),
            "max_molecules_per_second": successful_runs["molecules_per_second"].max(),
            "min_molecules_per_second": successful_runs["molecules_per_second"].min(),
            "average_hit_rate": successful_runs["hit_rate"].mean(),
            "total_molecules_screened": successful_runs["dataset_size"].sum(),
            "total_screening_time": successful_runs["total_time_seconds"].sum(),
            "successful_runs": len(successful_runs),
            "failed_runs": len(df[df["error_occurred"]])
        }

    def _analyze_scalability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how performance scales with dataset size."""
        successful_runs = df[~df["error_occurred"]]
        
        scalability = {}
        
        for backend in successful_runs["backend"].unique():
            backend_data = successful_runs[successful_runs["backend"] == backend]
            
            # Group by dataset size and calculate average performance
            size_performance = backend_data.groupby("dataset_size")["molecules_per_second"].agg([
                "mean", "std", "count"
            ]).to_dict()
            
            scalability[backend] = size_performance
        
        return scalability

    def _compare_backends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance between different backends."""
        successful_runs = df[~df["error_occurred"]]
        
        comparison = {}
        
        for backend in successful_runs["backend"].unique():
            backend_data = successful_runs[successful_runs["backend"] == backend]
            
            comparison[backend] = {
                "average_molecules_per_second": backend_data["molecules_per_second"].mean(),
                "average_hit_rate": backend_data["hit_rate"].mean(),
                "average_memory_usage_mb": backend_data["memory_used_mb"].mean(),
                "runs_completed": len(backend_data),
                "error_rate": len(df[(df["backend"] == backend) & df["error_occurred"]]) / 
                             len(df[df["backend"] == backend])
            }
        
        # Calculate speedup ratios
        if "cpu" in comparison and "gpu" in comparison:
            gpu_speed = comparison["gpu"]["average_molecules_per_second"]
            cpu_speed = comparison["cpu"]["average_molecules_per_second"]
            
            if cpu_speed > 0:
                comparison["gpu_speedup"] = gpu_speed / cpu_speed
        
        return comparison

    def _analyze_query_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of query complexity on performance."""
        successful_runs = df[~df["error_occurred"]]
        
        complexity_impact = {}
        
        for complexity in successful_runs["query_complexity"].unique():
            complexity_data = successful_runs[successful_runs["query_complexity"] == complexity]
            
            complexity_impact[complexity] = {
                "average_molecules_per_second": complexity_data["molecules_per_second"].mean(),
                "average_hit_rate": complexity_data["hit_rate"].mean(),
                "average_similarity_score": complexity_data["average_similarity_score"].mean(),
                "runs_completed": len(complexity_data)
            }
        
        return complexity_impact

    def _analyze_batch_sizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimal batch sizes for different backends."""
        successful_runs = df[~df["error_occurred"]]
        
        batch_analysis = {}
        
        for backend in successful_runs["backend"].unique():
            backend_data = successful_runs[successful_runs["backend"] == backend]
            
            batch_performance = backend_data.groupby("batch_size").agg({
                "molecules_per_second": ["mean", "std"],
                "memory_used_mb": ["mean", "std"],
                "hit_rate": "mean"
            }).to_dict()
            
            # Find optimal batch size (highest molecules/sec)
            optimal_batch = backend_data.loc[
                backend_data["molecules_per_second"].idxmax(),
                "batch_size"
            ]
            
            batch_analysis[backend] = {
                "performance_by_batch_size": batch_performance,
                "optimal_batch_size": optimal_batch
            }
        
        return batch_analysis

    def _analyze_errors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze error patterns and failure modes."""
        error_runs = df[df["error_occurred"]]
        
        if len(error_runs) == 0:
            return {"message": "No errors occurred during benchmarking"}
        
        error_analysis = {
            "total_errors": len(error_runs),
            "error_rate": len(error_runs) / len(df),
            "errors_by_backend": error_runs["backend"].value_counts().to_dict(),
            "errors_by_dataset_size": error_runs["dataset_size"].value_counts().to_dict(),
            "common_error_messages": error_runs["error_message"].value_counts().head(5).to_dict()
        }
        
        return error_analysis

    def save_results(self, 
                    results_df: pd.DataFrame,
                    analysis: Dict[str, Any]) -> Tuple[str, str]:
        """Save benchmark results and analysis to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = Path(self.config.output_directory) / f"pharmacophore_benchmark_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save analysis
        analysis_file = Path(self.config.output_directory) / f"pharmacophore_benchmark_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Analysis saved to: {analysis_file}")
        
        return str(results_file), str(analysis_file)

    def generate_report(self, 
                       results_df: pd.DataFrame,
                       analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        
        report.append("# GPU-Accelerated Pharmacophore Screening Benchmark Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Configuration summary
        report.append("## Configuration")
        report.append(f"- Dataset sizes: {self.config.dataset_sizes}")
        report.append(f"- Query complexities: {self.config.query_complexities}")
        report.append(f"- Backends tested: {self.config.backends}")
        report.append(f"- Batch sizes: {self.config.batch_sizes}")
        report.append(f"- Iterations per config: {self.config.iterations}")
        report.append(f"- GPU available: {GPU_AVAILABLE}")
        report.append("")
        
        # Performance summary
        perf = analysis.get("performance_summary", {})
        report.append("## Performance Summary")
        report.append(f"- Average screening rate: {perf.get('average_molecules_per_second', 0):.1f} molecules/second")
        report.append(f"- Maximum screening rate: {perf.get('max_molecules_per_second', 0):.1f} molecules/second")
        report.append(f"- Average hit rate: {perf.get('average_hit_rate', 0):.3f}")
        report.append(f"- Total molecules screened: {perf.get('total_molecules_screened', 0):,}")
        report.append(f"- Successful runs: {perf.get('successful_runs', 0)}")
        report.append(f"- Failed runs: {perf.get('failed_runs', 0)}")
        report.append("")
        
        # Backend comparison
        backend_comp = analysis.get("backend_comparison", {})
        report.append("## Backend Performance Comparison")
        
        for backend, metrics in backend_comp.items():
            if backend == "gpu_speedup":
                continue
            report.append(f"### {backend.upper()}")
            report.append(f"- Average rate: {metrics.get('average_molecules_per_second', 0):.1f} mol/sec")
            report.append(f"- Average hit rate: {metrics.get('average_hit_rate', 0):.3f}")
            report.append(f"- Memory usage: {metrics.get('average_memory_usage_mb', 0):.1f} MB")
            report.append(f"- Error rate: {metrics.get('error_rate', 0):.3f}")
            report.append("")
        
        if "gpu_speedup" in backend_comp:
            report.append(f"**GPU Speedup: {backend_comp['gpu_speedup']:.2f}x over CPU**")
            report.append("")
        
        # Query complexity impact
        complexity_impact = analysis.get("query_complexity_impact", {})
        report.append("## Query Complexity Impact")
        
        for complexity, metrics in complexity_impact.items():
            report.append(f"### {complexity.capitalize()}")
            report.append(f"- Average rate: {metrics.get('average_molecules_per_second', 0):.1f} mol/sec")
            report.append(f"- Average hit rate: {metrics.get('average_hit_rate', 0):.3f}")
            report.append(f"- Average similarity: {metrics.get('average_similarity_score', 0):.3f}")
            report.append("")
        
        # Error analysis
        error_analysis = analysis.get("error_analysis", {})
        if "total_errors" in error_analysis and error_analysis["total_errors"] > 0:
            report.append("## Error Analysis")
            report.append(f"- Total errors: {error_analysis['total_errors']}")
            report.append(f"- Error rate: {error_analysis['error_rate']:.3f}")
            
            if "errors_by_backend" in error_analysis:
                report.append("- Errors by backend:")
                for backend, count in error_analysis["errors_by_backend"].items():
                    report.append(f"  - {backend}: {count}")
            
            report.append("")
        
        return "\n".join(report)


def create_default_benchmark_config(output_dir: str = "benchmark_results") -> BenchmarkConfig:
    """Create a default benchmark configuration."""
    return BenchmarkConfig(
        dataset_sizes=[100, 500, 1000, 2000],
        query_complexities=["simple", "medium", "complex"],
        similarity_thresholds=[0.5, 0.7],
        iterations=3,
        backends=["cpu", "gpu", "auto"] if GPU_AVAILABLE else ["cpu", "auto"],
        batch_sizes=[100, 500, 1000],
        include_3d_constraints=True,
        save_detailed_results=True,
        output_directory=output_dir
    )


def run_performance_benchmark(config: Optional[BenchmarkConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run performance benchmark with given or default configuration."""
    if config is None:
        config = create_default_benchmark_config()
    
    benchmark = PharmacophoreScreeningBenchmark(config)
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    analysis = benchmark.analyze_results(results_df)
    
    # Save results
    results_file, analysis_file = benchmark.save_results(results_df, analysis)
    
    # Generate and save report
    report = benchmark.generate_report(results_df, analysis)
    report_file = Path(config.output_directory) / f"pharmacophore_benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("PHARMACOPHORE SCREENING BENCHMARK COMPLETED")
    print("="*80)
    print(report)
    print("="*80)
    print(f"Detailed results: {results_file}")
    print(f"Analysis: {analysis_file}")
    print(f"Report: {report_file}")
    
    return results_df, analysis


if __name__ == "__main__":
    # Run default benchmark
    results, analysis = run_performance_benchmark()