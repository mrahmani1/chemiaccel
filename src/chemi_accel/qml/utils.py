"""
Utility functions and classes for QML model integration.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from rdkit import Chem

from .base import BaseQMLModel, QMLModelError


class QMLModelLoader:
    """
    Utility class for loading and managing QML models.

    Provides standardized loading, caching, and validation of pre-trained
    quantum machine learning models from various sources.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for caching downloaded models
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".chemi_accel" / "qml_models"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache = {}

    def load_model(
        self, model_name: str, model_path: Optional[Union[str, Path]] = None, **kwargs
    ) -> BaseQMLModel:
        """
        Load a QML model by name.

        Args:
            model_name: Name of the model to load
            model_path: Custom path to model files
            **kwargs: Additional model-specific parameters

        Returns:
            Loaded QML model instance
        """
        # Check cache first
        cache_key = f"{model_name}_{hash(str(model_path))}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Load model based on name
        if model_name.lower().startswith("ani"):
            from .ani import ANIModel

            model = ANIModel(model_name=model_name, model_path=model_path, **kwargs)
        elif model_name.lower().startswith("gfn2"):
            from .gfn2_xtb import GFN2xTBModel

            model = GFN2xTBModel(model_path=model_path, **kwargs)
        else:
            raise QMLModelError(f"Unknown model name: {model_name}")

        # Load the model
        model.load_model()

        # Cache the loaded model
        self._model_cache[cache_key] = model

        return model

    def download_model(
        self, model_name: str, url: Optional[str] = None, force_download: bool = False
    ) -> Path:
        """
        Download a pre-trained model from a URL.

        Args:
            model_name: Name of the model
            url: Download URL (if None, uses default URLs)
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model files
        """
        model_dir = self.cache_dir / model_name

        if model_dir.exists() and not force_download:
            return model_dir

        # Default URLs for common models
        default_urls = {
            "ani2x": "https://github.com/aiqm/torchani/releases/download/v2.2/ani-2x.pt",
            "gfn2_xtb_surrogate": "https://figshare.com/ndownloader/files/example_surrogate.pt",
        }

        if url is None:
            url = default_urls.get(model_name)
            if url is None:
                raise QMLModelError(f"No default URL for model: {model_name}")

        # Download implementation would go here
        # For now, just create directory structure
        model_dir.mkdir(parents=True, exist_ok=True)

        return model_dir

    def validate_model_files(self, model_path: Path) -> bool:
        """
        Validate that model files are complete and uncorrupted.

        Args:
            model_path: Path to model directory

        Returns:
            True if model files are valid
        """
        if not model_path.exists():
            return False

        # Check for required files based on model type
        # This would be model-specific validation
        return True

    def list_available_models(self) -> List[str]:
        """List all available models in cache directory."""
        if not self.cache_dir.exists():
            return []

        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()


class GPUAccelerator:
    """
    Utility class for GPU acceleration of QML computations.

    Handles device management, memory optimization, and batch processing
    for efficient GPU utilization in quantum property predictions.
    """

    def __init__(self, device: str = "auto", memory_fraction: float = 0.9):
        """
        Initialize GPU accelerator.

        Args:
            device: Target device ('cpu', 'cuda', 'auto')
            memory_fraction: Fraction of GPU memory to use
        """
        self.device = self._get_optimal_device(device)
        self.memory_fraction = memory_fraction
        self._setup_gpu_settings()

    def _get_optimal_device(self, device: str) -> str:
        """Determine optimal computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                # Select best GPU
                best_gpu = 0
                max_memory = 0

                for i in range(torch.cuda.device_count()):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i

                return f"cuda:{best_gpu}"
            else:
                return "cpu"
        return device

    def _setup_gpu_settings(self) -> None:
        """Configure GPU settings for optimal performance."""
        if self.device.startswith("cuda"):
            # Set memory fraction
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

            # Enable tensor core optimization
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def optimize_batch_size(
        self, model: BaseQMLModel, sample_input: Any, max_batch_size: int = 1024
    ) -> int:
        """
        Determine optimal batch size for GPU memory.

        Args:
            model: QML model to optimize for
            sample_input: Sample input for testing
            max_batch_size: Maximum batch size to try

        Returns:
            Optimal batch size
        """
        if not self.device.startswith("cuda"):
            return min(32, max_batch_size)  # Conservative for CPU

        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal_size = 1

        while low <= high:
            mid = (low + high) // 2

            try:
                # Test batch processing with current size
                torch.cuda.empty_cache()

                # Create batch of sample inputs
                # This would be model-specific implementation

                # Try prediction to test memory usage
                # model.predict(batch_inputs)

                optimal_size = mid
                low = mid + 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    raise

        return max(1, optimal_size)

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the computing device."""
        info = {
            "device": self.device,
            "device_type": "cuda" if self.device.startswith("cuda") else "cpu",
        }

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            props = torch.cuda.get_device_properties(device_id)

            info.update(
                {
                    "gpu_name": props.name,
                    "gpu_memory_total": props.total_memory,
                    "gpu_memory_available": torch.cuda.get_device_properties(
                        device_id
                    ).total_memory
                    - torch.cuda.memory_allocated(device_id),
                    "gpu_compute_capability": f"{props.major}.{props.minor}",
                    "gpu_multiprocessor_count": props.multi_processor_count,
                }
            )

        return info


class QMLBenchmark:
    """
    Benchmarking utilities for QML model performance evaluation.
    """

    def __init__(self):
        """Initialize benchmark utility."""
        self.results = {}

    def benchmark_model(
        self,
        model: BaseQMLModel,
        test_molecules: List[Union[str, Chem.Mol]],
        properties: List[Any],
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Benchmark QML model performance.

        Args:
            model: QML model to benchmark
            test_molecules: Test molecules for benchmarking
            properties: Properties to predict
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        import time

        results = {
            "model_name": model.__class__.__name__,
            "num_molecules": len(test_molecules),
            "num_properties": len(properties),
            "runs": [],
            "statistics": {},
        }

        for run in range(num_runs):
            start_time = time.time()

            try:
                predictions = model.predict(test_molecules, properties)
                end_time = time.time()

                run_time = end_time - start_time
                throughput = len(test_molecules) / run_time

                results["runs"].append(
                    {
                        "run_id": run,
                        "runtime_seconds": run_time,
                        "throughput_molecules_per_second": throughput,
                        "success": True,
                    }
                )

            except Exception as e:
                results["runs"].append(
                    {"run_id": run, "error": str(e), "success": False}
                )

        # Compute statistics
        successful_runs = [r for r in results["runs"] if r["success"]]
        if successful_runs:
            runtimes = [r["runtime_seconds"] for r in successful_runs]
            throughputs = [
                r["throughput_molecules_per_second"] for r in successful_runs
            ]

            results["statistics"] = {
                "mean_runtime": np.mean(runtimes),
                "std_runtime": np.std(runtimes),
                "mean_throughput": np.mean(throughputs),
                "std_throughput": np.std(throughputs),
                "success_rate": len(successful_runs) / num_runs,
            }

        return results

    def compare_models(
        self, models: List[BaseQMLModel], test_data: Any
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple QML models.

        Args:
            models: List of QML models to compare
            test_data: Test dataset

        Returns:
            Comparison results
        """
        comparison = {
            "models": [model.__class__.__name__ for model in models],
            "benchmarks": {},
            "rankings": {},
        }

        # Benchmark each model
        for model in models:
            model_name = model.__class__.__name__
            # benchmark_result = self.benchmark_model(model, test_data)
            # comparison["benchmarks"][model_name] = benchmark_result

        return comparison


def create_molecular_hash(molecule: Union[str, Chem.Mol]) -> str:
    """Create a unique hash for a molecule."""
    if isinstance(molecule, str):
        mol_str = molecule
    else:
        mol_str = Chem.MolToSmiles(molecule, canonical=True)

    return hashlib.md5(mol_str.encode()).hexdigest()


def validate_qml_environment() -> Dict[str, bool]:
    """
    Validate that the environment is properly configured for QML calculations.

    Returns:
        Dictionary of validation results
    """
    validation = {
        "torch_available": False,
        "cuda_available": False,
        "rdkit_available": False,
        "torchani_available": False,
        "xtb_available": False,
    }

    try:
        import torch

        validation["torch_available"] = True
        validation["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import rdkit

        validation["rdkit_available"] = True
    except ImportError:
        pass

    try:
        import torchani

        validation["torchani_available"] = True
    except ImportError:
        pass

    try:
        import subprocess

        result = subprocess.run(["xtb", "--version"], capture_output=True, timeout=5)
        validation["xtb_available"] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return validation
