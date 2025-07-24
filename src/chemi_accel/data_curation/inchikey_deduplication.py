"""
GPU-Based InChIKey Generation and Molecular Deduplication.

This module provides high-performance InChIKey generation and molecular deduplication
using GPU acceleration, parallel processing, and intelligent caching strategies.
Optimized for large-scale molecular datasets with comprehensive error handling.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import pickle
import hashlib
from collections import defaultdict, Counter

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    inchi = None

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, MolecularDataError, GPUMemoryError
from ..utils.logging import get_logger
from ..backends.cuda.memory import get_memory_manager

logger = get_logger()


class InChIKeyGenerationStatus(Enum):
    """Status of InChIKey generation process."""
    SUCCESS = "success"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"
    INCHI_ERROR = "inchi_error"
    TIMEOUT = "timeout"


class DeduplicationStrategy(Enum):
    """Strategy for molecular deduplication."""
    INCHIKEY_ONLY = "inchikey_only"  # Use only InChIKey
    INCHIKEY_AND_SMILES = "inchikey_and_smiles"  # Use InChIKey + canonical SMILES
    MOLECULAR_FORMULA = "molecular_formula"  # Group by molecular formula first
    CONNECTIVITY_ONLY = "connectivity_only"  # Use connectivity-only InChI


@dataclass
class InChIKeyResult:
    """Result of InChIKey generation."""
    original_smiles: str
    canonical_smiles: Optional[str]
    inchi: Optional[str]
    inchikey: Optional[str]
    molecular_formula: Optional[str]
    status: InChIKeyGenerationStatus
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if InChIKey generation was successful."""
        return self.status == InChIKeyGenerationStatus.SUCCESS
    
    @property
    def deduplication_key(self) -> Optional[str]:
        """Get the key used for deduplication."""
        return self.inchikey


@dataclass
class DeduplicationResult:
    """Result of molecular deduplication."""
    original_count: int
    unique_count: int
    duplicate_count: int
    duplicate_groups: Dict[str, List[int]]  # InChIKey -> list of indices
    unique_indices: List[int]
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def deduplication_rate(self) -> float:
        """Calculate deduplication rate as percentage."""
        return (self.duplicate_count / self.original_count * 100) if self.original_count > 0 else 0.0


@dataclass
class InChIKeyConfig:
    """Configuration for InChIKey generation and deduplication."""
    # InChI generation settings
    include_stereochemistry: bool = True
    include_isotopes: bool = True
    remove_hydrogens: bool = False
    
    # Deduplication settings
    deduplication_strategy: DeduplicationStrategy = DeduplicationStrategy.INCHIKEY_ONLY
    keep_first_duplicate: bool = True
    preserve_order: bool = True
    
    # Performance settings
    parallel_backend: str = "threading"  # "threading", "multiprocessing", "joblib"
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    timeout_per_molecule: float = 2.0
    use_gpu_acceleration: bool = True
    
    # Memory management
    use_gpu_memory: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size_optimization: bool = True
    
    # Caching settings
    cache_enabled: bool = True
    cache_size: int = 50000
    persistent_cache: bool = False
    cache_file: Optional[str] = None
    
    # Error handling
    skip_on_error: bool = True
    max_retries: int = 2
    validate_inchikey: bool = True


class InChIKeyGenerator:
    """High-performance InChIKey generator with GPU acceleration."""
    
    def __init__(self, config: Optional[InChIKeyConfig] = None):
        """Initialize InChIKey generator.
        
        Args:
            config: InChIKey generation configuration
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for InChIKey generation")
        
        self.config = config or InChIKeyConfig()
        
        # Initialize GPU memory manager if available
        self.memory_manager = None
        if self.config.use_gpu_acceleration and CUPY_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager()
                logger.info("GPU acceleration enabled for InChIKey generation")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory manager: {e}")
                self.config.use_gpu_acceleration = False
        
        # Initialize cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Load persistent cache if configured
        if self.config.persistent_cache and self.config.cache_file:
            self._load_persistent_cache()
        
        # Thread-local storage for RDKit objects
        self._local = threading.local()
        
        logger.info(f"Initialized InChIKey generator with config: {self.config}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from file."""
        try:
            if self.config.cache_file and Path(self.config.cache_file).exists():
                with open(self.config.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} entries from persistent cache")
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self) -> None:
        """Save cache to persistent file."""
        try:
            if self.config.persistent_cache and self.config.cache_file:
                with open(self.config.cache_file, 'wb') as f:
                    pickle.dump(self._cache, f)
                logger.debug(f"Saved {len(self._cache)} entries to persistent cache")
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    @lru_cache(maxsize=10000)
    def _get_cache_key(self, smiles: str) -> str:
        """Generate cache key for SMILES."""
        # Include config parameters that affect InChI generation
        config_hash = hashlib.md5(
            f"{self.config.include_stereochemistry}_{self.config.include_isotopes}_{self.config.remove_hydrogens}".encode()
        ).hexdigest()[:8]
        return hashlib.md5(f"{smiles}_{config_hash}".encode()).hexdigest()
    
    def _get_from_cache(self, smiles: str) -> Optional[InChIKeyResult]:
        """Get result from cache."""
        if not self.config.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(smiles)
        with self._cache_lock:
            return self._cache.get(cache_key)
    
    def _store_in_cache(self, smiles: str, result: InChIKeyResult) -> None:
        """Store result in cache."""
        if not self.config.cache_enabled:
            return
        
        cache_key = self._get_cache_key(smiles)
        with self._cache_lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.config.cache_size:
                # Remove oldest entries (simple FIFO for now)
                oldest_keys = list(self._cache.keys())[:len(self._cache) // 2]
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[cache_key] = result
    
    def _validate_inchikey(self, inchikey: str) -> bool:
        """Validate InChIKey format."""
        if not inchikey:
            return False
        
        # Standard InChIKey format: 14-10-1 characters separated by hyphens
        parts = inchikey.split('-')
        if len(parts) != 3:
            return False
        
        if len(parts[0]) != 14 or len(parts[1]) != 10 or len(parts[2]) != 1:
            return False
        
        # Check that all parts contain only valid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for part in parts:
            if not all(c in valid_chars for c in part):
                return False
        
        return True
    
    def generate_single(self, smiles: str) -> InChIKeyResult:
        """Generate InChIKey for a single SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            InChIKey generation result
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._get_from_cache(smiles)
        if cached_result is not None:
            logger.debug(f"Retrieved from cache: {smiles}")
            return cached_result
        
        result = InChIKeyResult(
            original_smiles=smiles,
            canonical_smiles=None,
            inchi=None,
            inchikey=None,
            molecular_formula=None,
            status=InChIKeyGenerationStatus.FAILED
        )
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.status = InChIKeyGenerationStatus.INVALID_INPUT
                result.error_message = "Failed to parse SMILES"
                return result
            
            # Generate canonical SMILES
            try:
                result.canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                logger.warning(f"Failed to generate canonical SMILES: {e}")
            
            # Get molecular formula
            try:
                result.molecular_formula = CalcMolFormula(mol)
            except Exception as e:
                logger.warning(f"Failed to calculate molecular formula: {e}")
            
            # Generate InChI
            inchi_options = []
            if not self.config.include_stereochemistry:
                inchi_options.append("SNon")
            if not self.config.include_isotopes:
                inchi_options.append("FixedH")
            if self.config.remove_hydrogens:
                inchi_options.append("RemH")
            
            options_string = " ".join(inchi_options) if inchi_options else ""
            
            try:
                if options_string:
                    result.inchi = inchi.MolToInchi(mol, options=options_string)
                else:
                    result.inchi = inchi.MolToInchi(mol)
                
                if not result.inchi:
                    result.status = InChIKeyGenerationStatus.INCHI_ERROR
                    result.error_message = "InChI generation returned empty result"
                    return result
                
            except Exception as e:
                result.status = InChIKeyGenerationStatus.INCHI_ERROR
                result.error_message = f"InChI generation failed: {e}"
                return result
            
            # Generate InChIKey
            try:
                result.inchikey = inchi.InchiToInchiKey(result.inchi)
                
                if not result.inchikey:
                    result.status = InChIKeyGenerationStatus.INCHI_ERROR
                    result.error_message = "InChIKey generation returned empty result"
                    return result
                
                # Validate InChIKey format if requested
                if self.config.validate_inchikey and not self._validate_inchikey(result.inchikey):
                    result.status = InChIKeyGenerationStatus.INCHI_ERROR
                    result.error_message = f"Invalid InChIKey format: {result.inchikey}"
                    return result
                
            except Exception as e:
                result.status = InChIKeyGenerationStatus.INCHI_ERROR
                result.error_message = f"InChIKey generation failed: {e}"
                return result
            
            # Add metadata
            result.metadata = {
                'inchi_options': options_string,
                'atom_count': mol.GetNumAtoms(),
                'bond_count': mol.GetNumBonds(),
                'heavy_atom_count': mol.GetNumHeavyAtoms()
            }
            
            result.status = InChIKeyGenerationStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"InChIKey generation failed for {smiles}: {e}")
            result.status = InChIKeyGenerationStatus.FAILED
            result.error_message = str(e)
        
        finally:
            result.processing_time = time.time() - start_time
        
        # Store in cache
        self._store_in_cache(smiles, result)
        
        return result
    
    def _process_chunk_threading(self, smiles_chunk: List[str]) -> List[InChIKeyResult]:
        """Process chunk using threading."""
        results = []
        for smiles in smiles_chunk:
            try:
                result = self.generate_single(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {smiles}: {e}")
                if not self.config.skip_on_error:
                    raise
                results.append(InChIKeyResult(
                    original_smiles=smiles,
                    canonical_smiles=None,
                    inchi=None,
                    inchikey=None,
                    molecular_formula=None,
                    status=InChIKeyGenerationStatus.FAILED,
                    error_message=str(e)
                ))
        return results
    
    def _process_chunk_multiprocessing(self, smiles_chunk: List[str]) -> List[InChIKeyResult]:
        """Process chunk using multiprocessing."""
        # Create a new generator instance to avoid pickling issues
        local_generator = InChIKeyGenerator(self.config)
        return [local_generator.generate_single(smiles) for smiles in smiles_chunk]
    
    def generate_batch(self, smiles_list: List[str]) -> List[InChIKeyResult]:
        """Generate InChIKeys for a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of InChIKey generation results
        """
        if not smiles_list:
            return []
        
        logger.info(f"Generating InChIKeys for batch of {len(smiles_list)} molecules using {self.config.parallel_backend}")
        
        # Optimize batch size if GPU acceleration is enabled
        if self.config.use_gpu_acceleration and self.memory_manager and self.config.batch_size_optimization:
            optimal_batch_size = self._estimate_optimal_batch_size(len(smiles_list))
            if optimal_batch_size < len(smiles_list):
                logger.info(f"Using optimized batch size: {optimal_batch_size}")
                return self._process_in_optimized_batches(smiles_list, optimal_batch_size)
        
        # Check cache for all molecules first
        uncached_smiles = []
        cached_results = {}
        
        if self.config.cache_enabled:
            for smiles in smiles_list:
                cached_result = self._get_from_cache(smiles)
                if cached_result is not None:
                    cached_results[smiles] = cached_result
                else:
                    uncached_smiles.append(smiles)
        else:
            uncached_smiles = smiles_list
        
        logger.debug(f"Found {len(cached_results)} cached results, processing {len(uncached_smiles)} new molecules")
        
        # Process uncached molecules
        new_results = []
        if uncached_smiles:
            if self.config.parallel_backend == "multiprocessing" and len(uncached_smiles) > 500:
                new_results = self._generate_multiprocessing(uncached_smiles)
            elif self.config.parallel_backend == "joblib" and JOBLIB_AVAILABLE:
                new_results = self._generate_joblib(uncached_smiles)
            else:
                new_results = self._generate_threading(uncached_smiles)
        
        # Combine cached and new results in original order
        results = []
        new_results_dict = {result.original_smiles: result for result in new_results}
        
        for smiles in smiles_list:
            if smiles in cached_results:
                results.append(cached_results[smiles])
            elif smiles in new_results_dict:
                results.append(new_results_dict[smiles])
            else:
                # Should not happen, but create a failed result as fallback
                results.append(InChIKeyResult(
                    original_smiles=smiles,
                    canonical_smiles=None,
                    inchi=None,
                    inchikey=None,
                    molecular_formula=None,
                    status=InChIKeyGenerationStatus.FAILED,
                    error_message="Result not found"
                ))
        
        # Save persistent cache if configured
        if self.config.persistent_cache:
            self._save_persistent_cache()
        
        # Log statistics
        successful = len([r for r in results if r.is_successful])
        logger.info(f"InChIKey generation complete: {successful}/{len(results)} successful")
        
        return results
    
    def _estimate_optimal_batch_size(self, total_size: int) -> int:
        """Estimate optimal batch size based on available GPU memory."""
        if not self.memory_manager:
            return total_size
        
        try:
            memory_info = self.memory_manager.get_memory_info()
            available_gb = memory_info.free / (1024**3)
            
            # Estimate memory per molecule (rough heuristic)
            memory_per_molecule_mb = 0.1  # Very rough estimate
            max_molecules = int((available_gb * 1024 * self.config.max_memory_usage_gb) / memory_per_molecule_mb)
            
            return min(total_size, max_molecules, 100000)  # Reasonable upper limit
            
        except Exception as e:
            logger.warning(f"Failed to estimate optimal batch size: {e}")
            return total_size
    
    def _process_in_optimized_batches(self, smiles_list: List[str], batch_size: int) -> List[InChIKeyResult]:
        """Process large dataset in optimized batches."""
        all_results = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(smiles_list) + batch_size - 1)//batch_size}")
            
            batch_results = self.generate_batch(batch)
            all_results.extend(batch_results)
            
            # Clear GPU memory between batches if available
            if self.memory_manager:
                try:
                    if hasattr(cp, 'get_default_memory_pool'):
                        cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        
        return all_results
    
    def _generate_threading(self, smiles_list: List[str]) -> List[InChIKeyResult]:
        """Generate InChIKeys using threading."""
        max_workers = self.config.max_workers or min(32, len(smiles_list))
        chunk_size = max(1, len(smiles_list) // max_workers)
        
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk_threading, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result(timeout=self.config.timeout_per_molecule * len(future_to_chunk[future]))
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    if not self.config.skip_on_error:
                        raise
        
        return results
    
    def _generate_multiprocessing(self, smiles_list: List[str]) -> List[InChIKeyResult]:
        """Generate InChIKeys using multiprocessing."""
        max_workers = self.config.max_workers or min(8, len(smiles_list) // 500)
        chunk_size = max(500, len(smiles_list) // max_workers)
        
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk_multiprocessing, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result(timeout=self.config.timeout_per_molecule * len(future_to_chunk[future]) * 3)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Multiprocessing chunk failed: {e}")
                    if not self.config.skip_on_error:
                        raise
        
        return results
    
    def _generate_joblib(self, smiles_list: List[str]) -> List[InChIKeyResult]:
        """Generate InChIKeys using joblib."""
        n_jobs = self.config.max_workers or -1
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
            delayed(self.generate_single)(smiles) for smiles in smiles_list
        )
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the InChIKey cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cleared InChIKey cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.config.cache_size,
                'cache_enabled': self.config.cache_enabled,
                'persistent_cache': self.config.persistent_cache
            }


class MolecularDeduplicator:
    """High-performance molecular deduplication using InChIKeys."""
    
    def __init__(self, config: Optional[InChIKeyConfig] = None):
        """Initialize molecular deduplicator.
        
        Args:
            config: Configuration for InChIKey generation
        """
        self.config = config or InChIKeyConfig()
        self.inchikey_generator = InChIKeyGenerator(config)
        
        logger.info(f"Initialized molecular deduplicator with strategy: {self.config.deduplication_strategy}")
    
    def deduplicate_smiles(self, smiles_list: List[str]) -> DeduplicationResult:
        """Deduplicate a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Deduplication result with unique indices and duplicate groups
        """
        start_time = time.time()
        
        if not smiles_list:
            return DeduplicationResult(
                original_count=0,
                unique_count=0,
                duplicate_count=0,
                duplicate_groups={},
                unique_indices=[],
                processing_time=0.0
            )
        
        logger.info(f"Deduplicating {len(smiles_list)} molecules using strategy: {self.config.deduplication_strategy}")
        
        # Generate InChIKeys for all molecules
        inchikey_results = self.inchikey_generator.generate_batch(smiles_list)
        
        # Extract deduplication keys based on strategy
        dedup_keys = []
        for result in inchikey_results:
            if self.config.deduplication_strategy == DeduplicationStrategy.INCHIKEY_ONLY:
                key = result.inchikey if result.is_successful else None
            elif self.config.deduplication_strategy == DeduplicationStrategy.INCHIKEY_AND_SMILES:
                key = f"{result.inchikey}:{result.canonical_smiles}" if result.is_successful else None
            elif self.config.deduplication_strategy == DeduplicationStrategy.MOLECULAR_FORMULA:
                key = result.molecular_formula if result.is_successful else None
            elif self.config.deduplication_strategy == DeduplicationStrategy.CONNECTIVITY_ONLY:
                # Use first part of InChIKey (connectivity layer)
                key = result.inchikey.split('-')[0] if result.is_successful and result.inchikey else None
            else:
                key = result.inchikey if result.is_successful else None
            
            dedup_keys.append(key)
        
        # Find duplicates
        key_to_indices = defaultdict(list)
        for i, key in enumerate(dedup_keys):
            if key is not None:  # Only consider successful results
                key_to_indices[key].append(i)
        
        # Separate unique and duplicate groups
        unique_indices = []
        duplicate_groups = {}
        
        for key, indices in key_to_indices.items():
            if len(indices) == 1:
                unique_indices.extend(indices)
            else:
                # Keep first occurrence if configured
                if self.config.keep_first_duplicate:
                    unique_indices.append(indices[0])
                
                duplicate_groups[key] = indices
        
        # Sort unique indices to preserve order if requested
        if self.config.preserve_order:
            unique_indices.sort()
        
        # Calculate statistics
        original_count = len(smiles_list)
        unique_count = len(unique_indices)
        total_duplicates = sum(len(indices) for indices in duplicate_groups.values())
        duplicate_count = total_duplicates - len(duplicate_groups) if duplicate_groups else 0
        
        processing_time = time.time() - start_time
        
        result = DeduplicationResult(
            original_count=original_count,
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            duplicate_groups=duplicate_groups,
            unique_indices=unique_indices,
            processing_time=processing_time,
            metadata={
                'strategy': self.config.deduplication_strategy.value,
                'failed_inchikey_count': len([r for r in inchikey_results if not r.is_successful]),
                'successful_inchikey_count': len([r for r in inchikey_results if r.is_successful]),
                'duplicate_group_count': len(duplicate_groups),
                'largest_duplicate_group': max(len(indices) for indices in duplicate_groups.values()) if duplicate_groups else 0
            }
        )
        
        logger.info(f"Deduplication complete: {unique_count} unique out of {original_count} molecules "
                   f"({result.deduplication_rate:.1f}% duplicates) in {processing_time:.2f}s")
        
        return result
    
    def deduplicate_dataframe(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> Tuple[pd.DataFrame, DeduplicationResult]:
        """Deduplicate a DataFrame containing SMILES.
        
        Args:
            df: DataFrame with molecular data
            smiles_column: Name of the column containing SMILES
            
        Returns:
            Tuple of (deduplicated_dataframe, deduplication_result)
        """
        if not PANDAS_AVAILABLE:
            raise RuntimeError("Pandas is required for DataFrame deduplication")
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in DataFrame")
        
        smiles_list = df[smiles_column].tolist()
        dedup_result = self.deduplicate_smiles(smiles_list)
        
        # Create deduplicated DataFrame
        if dedup_result.unique_indices:
            unique_df = df.iloc[dedup_result.unique_indices].copy()
            
            # Add deduplication metadata
            inchikey_results = self.inchikey_generator.generate_batch(smiles_list)
            unique_df['inchikey'] = [inchikey_results[i].inchikey for i in dedup_result.unique_indices]
            unique_df['canonical_smiles'] = [inchikey_results[i].canonical_smiles for i in dedup_result.unique_indices]
            
            # Reset index if preserving order
            if self.config.preserve_order:
                unique_df = unique_df.sort_index().reset_index(drop=True)
        else:
            # No unique molecules found
            unique_df = df.iloc[:0].copy()  # Empty DataFrame with same structure
        
        return unique_df, dedup_result
    
    def get_duplicate_report(self, smiles_list: List[str], dedup_result: DeduplicationResult) -> Dict[str, Any]:
        """Generate a detailed duplicate report.
        
        Args:
            smiles_list: Original list of SMILES
            dedup_result: Deduplication result
            
        Returns:
            Detailed duplicate report
        """
        report = {
            'summary': {
                'total_molecules': dedup_result.original_count,
                'unique_molecules': dedup_result.unique_count,
                'duplicate_molecules': dedup_result.duplicate_count,
                'deduplication_rate': dedup_result.deduplication_rate,
                'processing_time': dedup_result.processing_time
            },
            'duplicate_groups': []
        }
        
        # Add detailed information about duplicate groups
        for key, indices in dedup_result.duplicate_groups.items():
            group_info = {
                'deduplication_key': key,
                'group_size': len(indices),
                'indices': indices,
                'smiles': [smiles_list[i] for i in indices]
            }
            report['duplicate_groups'].append(group_info)
        
        # Sort by group size (largest first)
        report['duplicate_groups'].sort(key=lambda x: x['group_size'], reverse=True)
        
        return report


# Standalone functions for easy import
def generate_inchikeys_batch(smiles_list: List[str], 
                           config: Optional[InChIKeyConfig] = None) -> List[InChIKeyResult]:
    """Generate InChIKeys for a batch of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        config: InChIKey generation configuration
        
    Returns:
        List of InChIKey generation results
    """
    generator = InChIKeyGenerator(config)
    return generator.generate_batch(smiles_list)


def deduplicate_molecules(smiles_list: List[str],
                         config: Optional[InChIKeyConfig] = None) -> DeduplicationResult:
    """Deduplicate a list of molecules using InChIKeys.
    
    Args:
        smiles_list: List of SMILES strings
        config: Configuration for deduplication
        
    Returns:
        Deduplication result
    """
    deduplicator = MolecularDeduplicator(config)
    return deduplicator.deduplicate_smiles(smiles_list)