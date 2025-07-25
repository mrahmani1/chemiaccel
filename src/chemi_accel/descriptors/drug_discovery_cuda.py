"""
CUDA-accelerated drug discovery descriptors.

This module provides GPU-accelerated implementations of drug discovery
descriptors, particularly focusing on synthetic accessibility scoring
and other computationally intensive drug discovery metrics.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..utils.logging import get_logger
from .drug_discovery import DrugDiscoveryDescriptorCalculator

logger = get_logger()


class CUDADrugDiscoveryCalculator(DrugDiscoveryDescriptorCalculator):
    """CUDA-accelerated drug discovery descriptor calculator."""

    def __init__(self, device_id: int = 0):
        """
        Initialize CUDA-accelerated drug discovery calculator.

        Args:
            device_id: CUDA device ID to use
        """
        super().__init__()

        if not CUPY_AVAILABLE:
            raise ChemiAccelError("CuPy is required for CUDA acceleration")

        self.device_id = device_id

        # Set CUDA device
        cp.cuda.Device(device_id).use()

        # Override CPU implementations with GPU versions where beneficial
        self.descriptor_registry.update(
            {
                "synthetic_accessibility": {
                    "function": self._calculate_synthetic_accessibility_cuda,
                    "description": "GPU-accelerated Synthetic Accessibility Score",
                    "category": "synthetic",
                    "units": "score",
                },
                "batch_qed_score": {
                    "function": self._calculate_batch_qed_cuda,
                    "description": "GPU-accelerated batch QED calculation",
                    "category": "drug_likeness",
                    "units": "score",
                },
                "batch_rule_violations": {
                    "function": self._calculate_batch_violations_cuda,
                    "description": "GPU-accelerated rule violations",
                    "category": "drug_likeness",
                    "units": "violations",
                },
            }
        )

    def calculate_descriptors_batch_cuda(
        self,
        molecules: Union[List[str], pd.Series],
        descriptor_names: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Calculate descriptors using GPU acceleration for large batches.

        Args:
            molecules: List of SMILES strings or pandas Series
            descriptor_names: List of descriptor names to calculate
            batch_size: Batch size for GPU processing

        Returns:
            DataFrame with calculated descriptors
        """
        if isinstance(molecules, pd.Series):
            molecules = molecules.tolist()

        if descriptor_names is None:
            descriptor_names = list(self.descriptor_registry.keys())

        # Process in batches to manage GPU memory
        results = []

        for i in range(0, len(molecules), batch_size):
            batch_molecules = molecules[i : i + batch_size]
            batch_results = self._process_batch_cuda(batch_molecules, descriptor_names)
            results.append(batch_results)

        # Combine all batch results
        final_results = pd.concat(results, ignore_index=True)

        return final_results

    def _process_batch_cuda(
        self, molecules: List[str], descriptor_names: List[str]
    ) -> pd.DataFrame:
        """Process a batch of molecules using CUDA acceleration."""

        # Parse molecules and extract features
        molecular_features = self._extract_molecular_features_batch(molecules)

        if molecular_features is None:
            # Fallback to CPU processing
            return super().calculate_descriptors(molecules, descriptor_names)

        # Transfer to GPU
        gpu_features = self._transfer_to_gpu(molecular_features)

        # Calculate descriptors on GPU
        results = {"smiles": molecules}

        for desc_name in descriptor_names:
            if desc_name in [
                "synthetic_accessibility",
                "batch_qed_score",
                "batch_rule_violations",
            ]:
                # Use GPU-accelerated versions
                desc_func = self.descriptor_registry[desc_name]["function"]
                results[desc_name] = desc_func(gpu_features)
            else:
                # Use CPU versions for descriptors not yet GPU-accelerated
                desc_func = self.descriptor_registry[desc_name]["function"]
                desc_values = []
                for smiles in molecules:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            desc_values.append(desc_func(mol))
                        else:
                            desc_values.append(np.nan)
                    except Exception:
                        desc_values.append(np.nan)
                results[desc_name] = desc_values

        return pd.DataFrame(results)

    def _extract_molecular_features_batch(
        self, molecules: List[str]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract molecular features for batch processing.

        Returns:
            Dictionary of feature arrays or None if extraction fails
        """
        try:
            features = {
                "molecular_weights": [],
                "num_atoms": [],
                "num_bonds": [],
                "num_rings": [],
                "num_heteroatoms": [],
                "num_rotatable": [],
                "num_stereo": [],
                "logp_values": [],
                "tpsa_values": [],
                "hbd_counts": [],
                "hba_counts": [],
            }

            for smiles in molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Add NaN values for invalid molecules
                    for key in features:
                        features[key].append(np.nan)
                    continue

                # Extract features
                features["molecular_weights"].append(Descriptors.MolWt(mol))
                features["num_atoms"].append(mol.GetNumAtoms())
                features["num_bonds"].append(mol.GetNumBonds())
                features["num_rings"].append(rdMolDescriptors.CalcNumRings(mol))
                features["num_heteroatoms"].append(
                    rdMolDescriptors.CalcNumHeteroatoms(mol)
                )
                features["num_rotatable"].append(
                    rdMolDescriptors.CalcNumRotatableBonds(mol)
                )
                features["num_stereo"].append(
                    rdMolDescriptors.CalcNumAtomStereoCenters(mol)
                )
                features["logp_values"].append(Descriptors.MolLogP(mol))
                features["tpsa_values"].append(rdMolDescriptors.CalcTPSA(mol))
                features["hbd_counts"].append(rdMolDescriptors.CalcNumHBD(mol))
                features["hba_counts"].append(rdMolDescriptors.CalcNumHBA(mol))

            # Convert to numpy arrays
            for key in features:
                features[key] = np.array(features[key], dtype=np.float32)

            return features

        except Exception as e:
            logger.warning(f"Error extracting molecular features: {e}")
            return None

    def _transfer_to_gpu(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, cp.ndarray]:
        """Transfer molecular features to GPU memory."""
        gpu_features = {}

        for key, values in features.items():
            gpu_features[key] = cp.asarray(values)

        return gpu_features

    def _calculate_synthetic_accessibility_cuda(
        self, gpu_features: Dict[str, cp.ndarray]
    ) -> List[float]:
        """
        Calculate synthetic accessibility scores using GPU acceleration.

        This implements a simplified SA score calculation that can be
        efficiently computed on GPU using molecular features.
        """
        try:
            # Extract features
            mw = gpu_features["molecular_weights"]
            num_rings = gpu_features["num_rings"]
            num_heteroatoms = gpu_features["num_heteroatoms"]
            num_rotatable = gpu_features["num_rotatable"]
            num_stereo = gpu_features["num_stereo"]

            # Calculate complexity components on GPU
            mw_complexity = cp.minimum(mw / 100.0, 3.0)
            ring_complexity = num_rings * 0.5
            hetero_complexity = num_heteroatoms * 0.3
            rotatable_complexity = num_rotatable * 0.1
            stereo_complexity = num_stereo * 0.4

            # Combine complexity factors
            total_complexity = (
                mw_complexity
                + ring_complexity
                + hetero_complexity
                + rotatable_complexity
                + stereo_complexity
            )

            # Convert to SA score (1-10 scale)
            sa_scores = cp.minimum(1.0 + total_complexity, 10.0)

            # Handle NaN values
            sa_scores = cp.where(cp.isnan(mw), cp.nan, sa_scores)

            # Transfer back to CPU
            return sa_scores.get().tolist()

        except Exception as e:
            logger.warning(f"Error in CUDA SA calculation: {e}")
            # Fallback to CPU calculation
            return [np.nan] * len(gpu_features["molecular_weights"])

    def _calculate_batch_qed_cuda(
        self, gpu_features: Dict[str, cp.ndarray]
    ) -> List[float]:
        """
        Calculate QED scores for batch using GPU acceleration.

        This implements a simplified QED-like calculation using
        available molecular features.
        """
        try:
            # Extract features
            mw = gpu_features["molecular_weights"]
            logp = gpu_features["logp_values"]
            hbd = gpu_features["hbd_counts"]
            hba = gpu_features["hba_counts"]
            tpsa = gpu_features["tpsa_values"]

            # QED-like scoring functions (simplified)
            # Molecular weight component (optimal around 300-500)
            mw_score = cp.where(mw <= 500, 1.0, cp.exp(-(mw - 500) / 100))
            mw_score = cp.where(mw <= 200, mw / 200, mw_score)

            # LogP component (optimal around 1-3)
            logp_score = cp.where(
                (logp >= 1) & (logp <= 3), 1.0, cp.exp(-cp.abs(logp - 2) / 2)
            )

            # HBD component (optimal <= 5)
            hbd_score = cp.where(hbd <= 5, 1.0, cp.exp(-(hbd - 5) / 2))

            # HBA component (optimal <= 10)
            hba_score = cp.where(hba <= 10, 1.0, cp.exp(-(hba - 10) / 3))

            # TPSA component (optimal <= 140)
            tpsa_score = cp.where(tpsa <= 140, 1.0, cp.exp(-(tpsa - 140) / 50))

            # Combine scores (geometric mean)
            qed_scores = (
                mw_score * logp_score * hbd_score * hba_score * tpsa_score
            ) ** 0.2

            # Handle NaN values
            qed_scores = cp.where(cp.isnan(mw), cp.nan, qed_scores)

            # Transfer back to CPU
            return qed_scores.get().tolist()

        except Exception as e:
            logger.warning(f"Error in CUDA QED calculation: {e}")
            return [np.nan] * len(gpu_features["molecular_weights"])

    def _calculate_batch_violations_cuda(
        self, gpu_features: Dict[str, cp.ndarray]
    ) -> List[int]:
        """
        Calculate Lipinski rule violations using GPU acceleration.
        """
        try:
            # Extract features
            mw = gpu_features["molecular_weights"]
            logp = gpu_features["logp_values"]
            hbd = gpu_features["hbd_counts"]
            hba = gpu_features["hba_counts"]

            # Count violations
            violations = cp.zeros_like(mw, dtype=cp.int32)

            violations += cp.where(mw > 500, 1, 0)
            violations += cp.where(logp > 5, 1, 0)
            violations += cp.where(hbd > 5, 1, 0)
            violations += cp.where(hba > 10, 1, 0)

            # Handle NaN values
            violations = cp.where(cp.isnan(mw), -1, violations)

            # Transfer back to CPU
            return violations.get().tolist()

        except Exception as e:
            logger.warning(f"Error in CUDA violations calculation: {e}")
            return [-1] * len(gpu_features["molecular_weights"])

    def benchmark_gpu_vs_cpu(
        self,
        molecules: List[str],
        descriptor_names: List[str] = None,
        num_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance for descriptor calculations.

        Args:
            molecules: List of SMILES strings
            descriptor_names: Descriptors to benchmark
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with timing results
        """
        import time

        if descriptor_names is None:
            descriptor_names = [
                "synthetic_accessibility",
                "batch_qed_score",
                "batch_rule_violations",
            ]

        # CPU benchmark
        cpu_times = []
        for _ in range(num_runs):
            start_time = time.time()
            cpu_results = super().calculate_descriptors(molecules, descriptor_names)
            cpu_times.append(time.time() - start_time)

        # GPU benchmark
        gpu_times = []
        for _ in range(num_runs):
            start_time = time.time()
            gpu_results = self.calculate_descriptors_batch_cuda(
                molecules, descriptor_names
            )
            gpu_times.append(time.time() - start_time)

        avg_cpu_time = np.mean(cpu_times)
        avg_gpu_time = np.mean(gpu_times)
        speedup = avg_cpu_time / avg_gpu_time

        return {
            "cpu_time": avg_cpu_time,
            "gpu_time": avg_gpu_time,
            "speedup": speedup,
            "molecules_processed": len(molecules),
            "throughput_cpu": len(molecules) / avg_cpu_time,
            "throughput_gpu": len(molecules) / avg_gpu_time,
        }

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not CUPY_AVAILABLE:
            return {}

        mempool = cp.get_default_memory_pool()

        return {
            "used_bytes": mempool.used_bytes(),
            "total_bytes": mempool.total_bytes(),
            "used_mb": mempool.used_bytes() / 1024 / 1024,
            "total_mb": mempool.total_bytes() / 1024 / 1024,
        }

    def clear_gpu_memory(self):
        """Clear GPU memory pool."""
        if CUPY_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


def create_cuda_calculator(device_id: int = 0) -> CUDADrugDiscoveryCalculator:
    """
    Factory function to create CUDA-accelerated drug discovery calculator.

    Args:
        device_id: CUDA device ID

    Returns:
        CUDADrugDiscoveryCalculator instance
    """
    try:
        return CUDADrugDiscoveryCalculator(device_id)
    except Exception as e:
        logger.warning(f"Could not create CUDA calculator: {e}")
        logger.info("Falling back to CPU calculator")
        return DrugDiscoveryDescriptorCalculator()


# Convenience functions for GPU-accelerated calculations
def calculate_synthetic_accessibility_gpu(
    molecules: List[str], device_id: int = 0
) -> List[float]:
    """Calculate synthetic accessibility scores using GPU acceleration."""
    calc = create_cuda_calculator(device_id)
    results = calc.calculate_descriptors_batch_cuda(
        molecules, ["synthetic_accessibility"]
    )
    return results["synthetic_accessibility"].tolist()


def calculate_drug_likeness_gpu(
    molecules: List[str], device_id: int = 0
) -> pd.DataFrame:
    """Calculate drug-likeness descriptors using GPU acceleration."""
    calc = create_cuda_calculator(device_id)
    descriptors = [
        "batch_qed_score",
        "batch_rule_violations",
        "synthetic_accessibility",
    ]
    return calc.calculate_descriptors_batch_cuda(molecules, descriptors)


def benchmark_drug_discovery_gpu(
    molecules: List[str], device_id: int = 0
) -> Dict[str, float]:
    """Benchmark GPU vs CPU performance for drug discovery descriptors."""
    calc = create_cuda_calculator(device_id)
    return calc.benchmark_gpu_vs_cpu(molecules)
