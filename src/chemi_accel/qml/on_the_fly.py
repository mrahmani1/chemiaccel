"""
On-the-fly QML feature and property prediction for batch molecular processing.

This module provides rapid, GPU-accelerated prediction of quantum properties
for batches of molecules with efficient handling of 3D geometries and conformers.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

from .base import BaseQMLModel, QMLModelError, QMLPrediction, QuantumProperty
from .utils import GPUAccelerator


class PredictionMode(Enum):
    """Prediction execution modes."""

    BATCH = "batch"
    STREAMING = "streaming"
    PIPELINE = "pipeline"


class ConformerStrategy(Enum):
    """Strategies for handling multiple conformers."""

    SINGLE = "single"  # Use only lowest energy conformer
    AVERAGE = "average"  # Average properties across conformers
    ENSEMBLE = "ensemble"  # Keep all conformer predictions
    WEIGHTED = "weighted"  # Weight by Boltzmann factors


@dataclass
class GeometryInput:
    """Container for molecular geometry data."""

    coordinates: np.ndarray  # Shape: (n_atoms, 3)
    atomic_numbers: np.ndarray  # Shape: (n_atoms,)
    charge: int = 0
    multiplicity: int = 1
    conformer_id: Optional[int] = None
    energy: Optional[float] = None

    def to_tensor(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise QMLModelError("PyTorch not available for tensor operations")

        return {
            "coordinates": torch.tensor(
                self.coordinates, dtype=torch.float32, device=device
            ),
            "atomic_numbers": torch.tensor(
                self.atomic_numbers, dtype=torch.long, device=device
            ),
            "charge": torch.tensor([self.charge], dtype=torch.long, device=device),
            "multiplicity": torch.tensor(
                [self.multiplicity], dtype=torch.long, device=device
            ),
        }


@dataclass
class BatchPredictionRequest:
    """Request for batch prediction."""

    molecules: List[Union[str, Chem.Mol, GeometryInput]]
    properties: List[QuantumProperty]
    geometries: Optional[List[List[GeometryInput]]] = None  # For conformer ensembles
    conformer_strategy: ConformerStrategy = ConformerStrategy.SINGLE
    batch_size: int = 32
    max_conformers: int = 10
    energy_threshold: float = 10.0  # kcal/mol
    use_cache: bool = True
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchPredictionResult:
    """Result from batch prediction."""

    predictions: List[QMLPrediction]
    request: BatchPredictionRequest
    execution_time: float
    memory_usage: Dict[str, float]
    statistics: Dict[str, Any]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


class ConformerGenerator:
    """Generate and manage molecular conformers for QML prediction."""

    def __init__(
        self,
        max_conformers: int = 10,
        energy_threshold: float = 10.0,
        rms_threshold: float = 0.5,
        random_seed: int = 42,
    ):
        """
        Initialize conformer generator.

        Args:
            max_conformers: Maximum number of conformers to generate
            energy_threshold: Energy threshold in kcal/mol for conformer filtering
            rms_threshold: RMS threshold for conformer pruning
            random_seed: Random seed for reproducibility
        """
        if not RDKIT_AVAILABLE:
            raise QMLModelError("RDKit required for conformer generation")

        self.max_conformers = max_conformers
        self.energy_threshold = energy_threshold
        self.rms_threshold = rms_threshold
        self.random_seed = random_seed

    def generate_conformers(
        self, molecule: Union[str, Chem.Mol], embed_params: Optional[Dict] = None
    ) -> List[GeometryInput]:
        """
        Generate conformers for a molecule.

        Args:
            molecule: Input molecule (SMILES or RDKit Mol)
            embed_params: Parameters for conformer embedding

        Returns:
            List of GeometryInput objects representing conformers
        """
        # Convert to RDKit molecule
        if isinstance(molecule, str):
            mol = Chem.MolFromSmiles(molecule)
            if mol is None:
                raise QMLModelError(f"Invalid SMILES: {molecule}")
        else:
            mol = molecule

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Set up embedding parameters
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.maxAttempts = 1000
        params.numThreads = 0  # Use all available threads

        if embed_params:
            for key, value in embed_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)

        # Generate conformers
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=self.max_conformers, params=params
        )

        if not conformer_ids:
            # Fallback to single conformer
            conformer_id = AllChem.EmbedMolecule(mol, params=params)
            if conformer_id == -1:
                raise QMLModelError("Failed to generate conformers")
            conformer_ids = [conformer_id]

        # Optimize geometries
        results = []
        for conf_id in conformer_ids:
            try:
                # Minimize with UFF
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

                # Get geometry
                conf = mol.GetConformer(conf_id)
                coordinates = np.array(
                    [
                        [
                            conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z,
                        ]
                        for i in range(mol.GetNumAtoms())
                    ]
                )

                atomic_numbers = np.array(
                    [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                )

                # Calculate energy (UFF)
                energy = AllChem.UFFGetMoleculeForceField(
                    mol, confId=conf_id
                ).CalcEnergy()

                geometry = GeometryInput(
                    coordinates=coordinates,
                    atomic_numbers=atomic_numbers,
                    charge=Chem.rdmolops.GetFormalCharge(mol),
                    conformer_id=conf_id,
                    energy=energy,
                )

                results.append(geometry)

            except Exception as e:
                logging.warning(f"Failed to optimize conformer {conf_id}: {e}")
                continue

        if not results:
            raise QMLModelError("No valid conformers generated")

        # Filter by energy threshold
        if len(results) > 1:
            min_energy = min(geom.energy for geom in results)
            results = [
                geom
                for geom in results
                if (geom.energy - min_energy) * 627.509
                <= self.energy_threshold  # Convert to kcal/mol
            ]

        # Sort by energy
        results.sort(key=lambda x: x.energy)

        return results[: self.max_conformers]

    def filter_conformers_by_rms(
        self, geometries: List[GeometryInput]
    ) -> List[GeometryInput]:
        """Filter conformers by RMS similarity."""
        if len(geometries) <= 1:
            return geometries

        filtered = [geometries[0]]  # Keep lowest energy conformer

        for geom in geometries[1:]:
            # Check RMS against all kept conformers
            keep = True
            for kept_geom in filtered:
                rms = self._calculate_rms(geom.coordinates, kept_geom.coordinates)
                if rms < self.rms_threshold:
                    keep = False
                    break

            if keep:
                filtered.append(geom)

        return filtered

    def _calculate_rms(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMS distance between two coordinate sets."""
        if coords1.shape != coords2.shape:
            return float("inf")

        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


class GPUBatchProcessor:
    """GPU-accelerated batch processor for molecular predictions."""

    def __init__(
        self,
        device: str = "auto",
        max_batch_size: int = 128,
        memory_fraction: float = 0.8,
    ):
        """
        Initialize GPU batch processor.

        Args:
            device: Device to use ("auto", "cpu", "cuda:0", etc.)
            max_batch_size: Maximum batch size for GPU processing
            memory_fraction: Fraction of GPU memory to use
        """
        self.accelerator = GPUAccelerator(device=device)
        self.device = self.accelerator.device
        self.max_batch_size = max_batch_size
        self.memory_fraction = memory_fraction
        self._memory_pool = {}
        self._peak_memory = 0

    def process_batch(
        self,
        geometries: List[GeometryInput],
        model: BaseQMLModel,
        properties: List[QuantumProperty],
    ) -> List[QMLPrediction]:
        """
        Process a batch of geometries through QML model.

        Args:
            geometries: List of molecular geometries
            model: QML model for prediction
            properties: Properties to predict

        Returns:
            List of predictions
        """
        if not TORCH_AVAILABLE:
            raise QMLModelError("PyTorch required for GPU batch processing")

        batch_size = min(len(geometries), self.max_batch_size)
        predictions = []

        # Process in chunks
        for i in range(0, len(geometries), batch_size):
            chunk = geometries[i : i + batch_size]
            chunk_predictions = self._process_chunk(chunk, model, properties)
            predictions.extend(chunk_predictions)

            # Clear GPU cache periodically
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return predictions

    def _process_chunk(
        self,
        geometries: List[GeometryInput],
        model: BaseQMLModel,
        properties: List[QuantumProperty],
    ) -> List[QMLPrediction]:
        """Process a single chunk of geometries."""
        try:
            # Convert to tensors and batch
            batch_data = self._prepare_batch_tensors(geometries)

            # Predict using model
            with torch.no_grad():
                if hasattr(model, "predict_batch_tensors"):
                    # Model supports direct tensor prediction
                    predictions = model.predict_batch_tensors(batch_data, properties)
                else:
                    # Fallback to individual predictions
                    predictions = []
                    for geom in geometries:
                        pred = model.predict(geom, properties)
                        predictions.append(pred)

            return predictions

        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            if len(geometries) > 1:
                mid = len(geometries) // 2
                left_preds = self._process_chunk(geometries[:mid], model, properties)
                right_preds = self._process_chunk(geometries[mid:], model, properties)
                return left_preds + right_preds
            else:
                raise QMLModelError("GPU out of memory even with single molecule")

    def _prepare_batch_tensors(
        self, geometries: List[GeometryInput]
    ) -> Dict[str, torch.Tensor]:
        """Prepare batched tensors from geometries."""
        max_atoms = max(len(geom.atomic_numbers) for geom in geometries)
        batch_size = len(geometries)

        # Initialize padded tensors
        batch_coords = torch.zeros((batch_size, max_atoms, 3), device=self.device)
        batch_numbers = torch.zeros(
            (batch_size, max_atoms), dtype=torch.long, device=self.device
        )
        batch_mask = torch.zeros(
            (batch_size, max_atoms), dtype=torch.bool, device=self.device
        )
        batch_charges = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i, geom in enumerate(geometries):
            n_atoms = len(geom.atomic_numbers)
            batch_coords[i, :n_atoms] = torch.tensor(
                geom.coordinates, device=self.device
            )
            batch_numbers[i, :n_atoms] = torch.tensor(
                geom.atomic_numbers, device=self.device
            )
            batch_mask[i, :n_atoms] = True
            batch_charges[i] = geom.charge

        return {
            "coordinates": batch_coords,
            "atomic_numbers": batch_numbers,
            "atom_mask": batch_mask,
            "charges": batch_charges,
            "batch_size": batch_size,
            "max_atoms": max_atoms,
        }

    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage information."""
        if not self.device.startswith("cuda"):
            return {"device": "cpu", "memory_used": 0, "memory_total": 0}

        if TORCH_AVAILABLE:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3

            return {
                "device": self.device,
                "memory_allocated": allocated,
                "memory_reserved": reserved,
                "memory_max_allocated": max_allocated,
                "memory_peak": self._peak_memory,
            }

        return {"device": self.device, "memory_info": "unavailable"}


class OnTheFlyPredictor:
    """
    Main class for on-the-fly QML property prediction.

    Provides rapid, GPU-accelerated prediction of quantum properties
    for batches of molecules with efficient conformer handling.
    """

    def __init__(
        self,
        model: BaseQMLModel,
        device: str = "auto",
        max_batch_size: int = 64,
        conformer_strategy: ConformerStrategy = ConformerStrategy.SINGLE,
        cache_size: int = 1000,
        num_workers: int = 4,
    ):
        """
        Initialize on-the-fly predictor.

        Args:
            model: QML model for predictions
            device: Computing device
            max_batch_size: Maximum batch size
            conformer_strategy: Strategy for handling conformers
            cache_size: Size of prediction cache
            num_workers: Number of worker threads
        """
        self.model = model
        self.conformer_strategy = conformer_strategy
        self.cache_size = cache_size
        self.num_workers = num_workers

        # Initialize components
        self.gpu_processor = GPUBatchProcessor(
            device=device, max_batch_size=max_batch_size
        )
        self.conformer_generator = ConformerGenerator()

        # Cache and threading
        self._prediction_cache = {}
        self._cache_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Statistics
        self._stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "average_batch_size": 0.0,
        }

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResult:
        """
        Perform batch prediction on molecules.

        Args:
            request: Batch prediction request

        Returns:
            Batch prediction result
        """
        start_time = time.time()
        predictions = []
        errors = []
        cache_hits = 0
        cache_misses = 0

        try:
            # Prepare molecular inputs
            molecular_inputs = self._prepare_molecular_inputs(request)

            # Process in batches
            batch_size = request.batch_size
            for i in range(0, len(molecular_inputs), batch_size):
                batch = molecular_inputs[i : i + batch_size]

                try:
                    (
                        batch_preds,
                        batch_cache_hits,
                        batch_cache_misses,
                    ) = self._process_molecular_batch(
                        batch,
                        request.properties,
                        request.conformer_strategy,
                        request.use_cache,
                    )
                    predictions.extend(batch_preds)
                    cache_hits += batch_cache_hits
                    cache_misses += batch_cache_misses

                except Exception as e:
                    error_info = {
                        "batch_index": i // batch_size,
                        "error": str(e),
                        "molecules": len(batch),
                    }
                    errors.append(error_info)

                    # Add empty predictions for failed molecules
                    for _ in batch:
                        predictions.append(
                            QMLPrediction(
                                properties={},
                                model_name=self.model.__class__.__name__,
                                confidence=0.0,
                            )
                        )

            execution_time = time.time() - start_time

            # Update statistics
            self._update_statistics(
                len(molecular_inputs), execution_time, cache_hits, cache_misses
            )

            # Gather memory usage info
            memory_usage = self.gpu_processor.get_memory_info()

            # Compile statistics
            statistics = {
                "total_molecules": len(molecular_inputs),
                "successful_predictions": len(predictions) - len(errors),
                "failed_predictions": len(errors),
                "average_prediction_time": (
                    execution_time / len(predictions) if predictions else 0
                ),
                "throughput": (
                    len(predictions) / execution_time if execution_time > 0 else 0
                ),
                "cache_hit_rate": (
                    cache_hits / (cache_hits + cache_misses)
                    if (cache_hits + cache_misses) > 0
                    else 0
                ),
            }

            return BatchPredictionResult(
                predictions=predictions,
                request=request,
                execution_time=execution_time,
                memory_usage=memory_usage,
                statistics=statistics,
                errors=errors,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            raise QMLModelError(f"Batch prediction failed: {e}")

    def predict_streaming(
        self,
        molecules: Iterator[Union[str, Chem.Mol, GeometryInput]],
        properties: List[QuantumProperty],
        buffer_size: int = 32,
    ) -> Iterator[QMLPrediction]:
        """
        Streaming prediction interface.

        Args:
            molecules: Iterator of molecules
            properties: Properties to predict
            buffer_size: Internal buffer size

        Yields:
            QML predictions
        """
        buffer = []

        for molecule in molecules:
            buffer.append(molecule)

            if len(buffer) >= buffer_size:
                # Process buffer
                request = BatchPredictionRequest(
                    molecules=buffer, properties=properties, batch_size=buffer_size
                )

                result = self.predict_batch(request)
                for prediction in result.predictions:
                    yield prediction

                buffer = []

        # Process remaining molecules
        if buffer:
            request = BatchPredictionRequest(
                molecules=buffer, properties=properties, batch_size=len(buffer)
            )

            result = self.predict_batch(request)
            for prediction in result.predictions:
                yield prediction

    def _prepare_molecular_inputs(
        self, request: BatchPredictionRequest
    ) -> List[List[GeometryInput]]:
        """Prepare molecular inputs with conformer generation."""
        molecular_inputs = []

        for i, molecule in enumerate(request.molecules):
            try:
                if request.geometries and i < len(request.geometries):
                    # Use provided geometries
                    geometries = request.geometries[i]
                else:
                    # Generate conformers
                    if isinstance(molecule, GeometryInput):
                        geometries = [molecule]
                    else:
                        geometries = self.conformer_generator.generate_conformers(
                            molecule
                        )

                        # Apply conformer strategy
                        if request.conformer_strategy == ConformerStrategy.SINGLE:
                            geometries = geometries[:1]
                        elif len(geometries) > request.max_conformers:
                            geometries = geometries[: request.max_conformers]

                molecular_inputs.append(geometries)

            except Exception as e:
                logging.warning(f"Failed to prepare molecule {i}: {e}")
                # Add empty geometry list for failed molecules
                molecular_inputs.append([])

        return molecular_inputs

    def _process_molecular_batch(
        self,
        molecular_batch: List[List[GeometryInput]],
        properties: List[QuantumProperty],
        conformer_strategy: ConformerStrategy,
        use_cache: bool,
    ) -> Tuple[List[QMLPrediction], int, int]:
        """Process a batch of molecules with their conformers."""
        predictions = []
        cache_hits = 0
        cache_misses = 0

        for geometries in molecular_batch:
            if not geometries:
                # Empty geometry list - create empty prediction
                predictions.append(
                    QMLPrediction(
                        properties={},
                        model_name=self.model.__class__.__name__,
                        confidence=0.0,
                    )
                )
                continue

            # Check cache
            cache_key = None
            if use_cache:
                cache_key = self._generate_cache_key(geometries, properties)
                cached_pred = self._get_cached_prediction(cache_key)
                if cached_pred:
                    predictions.append(cached_pred)
                    cache_hits += 1
                    continue

            cache_misses += 1

            # Process conformers
            conformer_predictions = self.gpu_processor.process_batch(
                geometries, self.model, properties
            )

            # Apply conformer strategy
            final_prediction = self._apply_conformer_strategy(
                conformer_predictions, conformer_strategy, geometries
            )

            predictions.append(final_prediction)

            # Cache result
            if use_cache and cache_key:
                self._cache_prediction(cache_key, final_prediction)

        return predictions, cache_hits, cache_misses

    def _apply_conformer_strategy(
        self,
        conformer_predictions: List[QMLPrediction],
        strategy: ConformerStrategy,
        geometries: List[GeometryInput],
    ) -> QMLPrediction:
        """Apply conformer aggregation strategy."""
        if not conformer_predictions:
            return QMLPrediction(
                properties={}, model_name=self.model.__class__.__name__, confidence=0.0
            )

        if strategy == ConformerStrategy.SINGLE or len(conformer_predictions) == 1:
            return conformer_predictions[0]

        elif strategy == ConformerStrategy.AVERAGE:
            return self._average_predictions(conformer_predictions)

        elif strategy == ConformerStrategy.WEIGHTED:
            return self._weighted_average_predictions(conformer_predictions, geometries)

        elif strategy == ConformerStrategy.ENSEMBLE:
            # Return prediction with ensemble information
            ensemble_pred = conformer_predictions[0]  # Start with first prediction
            ensemble_pred.metadata = {
                "ensemble_size": len(conformer_predictions),
                "conformer_predictions": conformer_predictions,
            }
            return ensemble_pred

        else:
            raise QMLModelError(f"Unknown conformer strategy: {strategy}")

    def _average_predictions(self, predictions: List[QMLPrediction]) -> QMLPrediction:
        """Average predictions across conformers."""
        if not predictions:
            raise QMLModelError("No predictions to average")

        # Find common properties
        common_props = set(predictions[0].properties.keys())
        for pred in predictions[1:]:
            common_props &= set(pred.properties.keys())

        # Average common properties
        averaged_props = {}
        for prop in common_props:
            values = [pred.properties[prop] for pred in predictions]
            if isinstance(values[0], np.ndarray):
                averaged_props[prop] = np.mean(values, axis=0)
            else:
                averaged_props[prop] = np.mean(values)

        # Average confidence
        avg_confidence = np.mean(
            [pred.confidence for pred in predictions if pred.confidence is not None]
        )

        return QMLPrediction(
            properties=averaged_props,
            confidence=avg_confidence,
            model_name=predictions[0].model_name,
            metadata={"conformer_average": True, "n_conformers": len(predictions)},
        )

    def _weighted_average_predictions(
        self, predictions: List[QMLPrediction], geometries: List[GeometryInput]
    ) -> QMLPrediction:
        """Weighted average using Boltzmann factors."""
        if len(predictions) != len(geometries):
            return self._average_predictions(predictions)

        # Calculate Boltzmann weights
        energies = [geom.energy for geom in geometries if geom.energy is not None]
        if not energies:
            return self._average_predictions(predictions)

        kT = 0.59616  # kcal/mol at 298K
        min_energy = min(energies)
        weights = np.exp(-(np.array(energies) - min_energy) / kT)
        weights /= weights.sum()

        # Find common properties
        common_props = set(predictions[0].properties.keys())
        for pred in predictions[1:]:
            common_props &= set(pred.properties.keys())

        # Weighted average
        weighted_props = {}
        for prop in common_props:
            values = [pred.properties[prop] for pred in predictions]
            if isinstance(values[0], np.ndarray):
                weighted_props[prop] = np.average(values, weights=weights, axis=0)
            else:
                weighted_props[prop] = np.average(values, weights=weights)

        # Weighted average confidence
        confidences = [
            pred.confidence for pred in predictions if pred.confidence is not None
        ]
        if confidences:
            avg_confidence = np.average(
                confidences, weights=weights[: len(confidences)]
            )
        else:
            avg_confidence = None

        return QMLPrediction(
            properties=weighted_props,
            confidence=avg_confidence,
            model_name=predictions[0].model_name,
            metadata={
                "boltzmann_weighted": True,
                "n_conformers": len(predictions),
                "weights": weights.tolist(),
            },
        )

    def _generate_cache_key(
        self, geometries: List[GeometryInput], properties: List[QuantumProperty]
    ) -> str:
        """Generate cache key for geometries and properties."""
        import hashlib

        # Create key from geometry hashes and properties
        geom_hashes = []
        for geom in geometries:
            geom_str = f"{geom.coordinates.tostring()}{geom.atomic_numbers.tostring()}{geom.charge}"
            geom_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
            geom_hashes.append(geom_hash)

        prop_str = "_".join(
            prop.value for prop in sorted(properties, key=lambda x: x.value)
        )

        combined = f"{'_'.join(geom_hashes)}_{prop_str}_{self.conformer_strategy.value}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached_prediction(self, cache_key: str) -> Optional[QMLPrediction]:
        """Get prediction from cache."""
        with self._cache_lock:
            return self._prediction_cache.get(cache_key)

    def _cache_prediction(self, cache_key: str, prediction: QMLPrediction):
        """Cache a prediction."""
        with self._cache_lock:
            if len(self._prediction_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._prediction_cache))
                del self._prediction_cache[oldest_key]

            self._prediction_cache[cache_key] = prediction

    def _update_statistics(
        self,
        num_molecules: int,
        execution_time: float,
        cache_hits: int,
        cache_misses: int,
    ):
        """Update internal statistics."""
        self._stats["total_predictions"] += num_molecules
        self._stats["cache_hits"] += cache_hits
        self._stats["cache_misses"] += cache_misses
        self._stats["total_execution_time"] += execution_time

        total_preds = self._stats["total_predictions"]
        if total_preds > 0:
            self._stats["average_batch_size"] = total_preds / (
                self._stats["cache_misses"] / max(1, num_molecules)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._stats.copy()

        if stats["total_predictions"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (
                stats["cache_hits"] + stats["cache_misses"]
            )
            stats["average_prediction_time"] = (
                stats["total_execution_time"] / stats["total_predictions"]
            )
            stats["throughput"] = (
                stats["total_predictions"] / stats["total_execution_time"]
            )

        return stats

    def clear_cache(self):
        """Clear prediction cache."""
        with self._cache_lock:
            self._prediction_cache.clear()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
