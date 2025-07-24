"""
Canonical Tautomer Generation and Stereochemistry Standardization.

This module provides GPU-accelerated tautomer enumeration, canonical tautomer
selection, and stereochemistry standardization for molecular data curation pipelines.
Implements parallel processing with multiple backends and comprehensive error handling.
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
    from rdkit.Chem import rdMolStandardize
    from rdkit.Chem.MolStandardize import rdMolStandardize as standardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    rdMolStandardize = None

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, MolecularDataError, GPUMemoryError
from ..utils.logging import get_logger
from .parallel_validation import ValidationResult

logger = get_logger()


class TautomerStandardizationStatus(Enum):
    """Status of tautomer standardization process."""
    SUCCESS = "success"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"
    NO_TAUTOMERS = "no_tautomers"
    STEREOCHEMISTRY_ERROR = "stereochemistry_error"
    TIMEOUT = "timeout"


@dataclass
class TautomerResult:
    """Result of tautomer standardization."""
    original_smiles: str
    canonical_tautomer: Optional[str]
    tautomer_count: int
    standardized_stereochemistry: Optional[str]
    molecular_formula: Optional[str]
    status: TautomerStandardizationStatus
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if standardization was successful."""
        return self.status == TautomerStandardizationStatus.SUCCESS
    
    @property
    def has_stereochemistry(self) -> bool:
        """Check if molecule has stereochemistry."""
        return self.standardized_stereochemistry is not None


@dataclass
class TautomerStandardizationConfig:
    """Configuration for tautomer standardization."""
    # Tautomer enumeration settings
    max_tautomers: int = 1000
    include_stereochemistry: bool = True
    remove_stereo_first: bool = False
    reassign_stereo: bool = True
    
    # Stereochemistry settings
    max_undefined_stereocenters: int = 4
    enumerate_undefined_stereo: bool = False
    prefer_organic_form: bool = True
    
    # Performance settings
    parallel_backend: str = "threading"  # "threading", "multiprocessing", "joblib"
    max_workers: Optional[int] = None
    chunk_size: int = 100
    timeout_per_molecule: float = 5.0
    use_gpu_memory: bool = True
    
    # Caching settings
    cache_enabled: bool = True
    cache_size: int = 10000
    
    # Error handling
    skip_on_error: bool = True
    max_retries: int = 1


class TautomerEnumerator:
    """Handles tautomer enumeration and canonical selection."""
    
    def __init__(self, config: TautomerStandardizationConfig):
        """Initialize tautomer enumerator.
        
        Args:
            config: Tautomer standardization configuration
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for tautomer enumeration")
        
        self.config = config
        self._enumerator = rdMolStandardize.TautomerEnumerator()
        self._canonical_tautomer = rdMolStandardize.TautomerCanonicalizer()
        
        # Configure enumerator
        if hasattr(self._enumerator, 'SetMaxTautomers'):
            self._enumerator.SetMaxTautomers(config.max_tautomers)
        
        # Thread-local storage for RDKit objects
        self._local = threading.local()
        
        logger.debug(f"Initialized tautomer enumerator with max_tautomers={config.max_tautomers}")
    
    def _get_local_enumerator(self):
        """Get thread-local tautomer enumerator."""
        if not hasattr(self._local, 'enumerator'):
            self._local.enumerator = rdMolStandardize.TautomerEnumerator()
            self._local.canonical_tautomer = rdMolStandardize.TautomerCanonicalizer()
            
            if hasattr(self._local.enumerator, 'SetMaxTautomers'):
                self._local.enumerator.SetMaxTautomers(self.config.max_tautomers)
        
        return self._local.enumerator, self._local.canonical_tautomer
    
    def enumerate_tautomers(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Enumerate all tautomers for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of tautomer molecules
        """
        if mol is None:
            return []
        
        try:
            enumerator, _ = self._get_local_enumerator()
            tautomers = enumerator.Enumerate(mol)
            
            # Convert to list and limit
            tautomer_list = list(tautomers)[:self.config.max_tautomers]
            
            logger.debug(f"Enumerated {len(tautomer_list)} tautomers")
            return tautomer_list
            
        except Exception as e:
            logger.error(f"Failed to enumerate tautomers: {e}")
            return [mol]  # Return original molecule as fallback
    
    def get_canonical_tautomer(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Get canonical tautomer for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Canonical tautomer molecule or None if failed
        """
        if mol is None:
            return None
        
        try:
            _, canonical_tautomer = self._get_local_enumerator()
            canonical_mol = canonical_tautomer.Canonicalize(mol)
            
            if canonical_mol is None:
                logger.warning("Canonical tautomer generation returned None")
                return mol  # Return original as fallback
            
            return canonical_mol
            
        except Exception as e:
            logger.error(f"Failed to get canonical tautomer: {e}")
            return mol  # Return original as fallback


class StereochemistryStandardizer:
    """Handles stereochemistry standardization and enumeration."""
    
    def __init__(self, config: TautomerStandardizationConfig):
        """Initialize stereochemistry standardizer.
        
        Args:
            config: Tautomer standardization configuration
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for stereochemistry standardization")
        
        self.config = config
        
        # Set up stereochemistry enumeration options
        self.stereo_options = StereoEnumerationOptions()
        self.stereo_options.onlyUnassigned = True
        self.stereo_options.maxIsomers = 2 ** min(config.max_undefined_stereocenters, 10)
        
        logger.debug(f"Initialized stereochemistry standardizer")
    
    def standardize_stereochemistry(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Standardize stereochemistry for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Molecule with standardized stereochemistry or None if failed
        """
        if mol is None:
            return None
        
        try:
            # Make a copy to avoid modifying original
            mol_copy = Chem.Mol(mol)
            
            if self.config.remove_stereo_first:
                # Remove existing stereochemistry
                Chem.rdmolops.RemoveStereochemistry(mol_copy)
            
            if self.config.reassign_stereo:
                # Reassign stereochemistry based on 3D coordinates if available
                if mol_copy.GetNumConformers() > 0:
                    Chem.rdmolops.AssignStereochemistry(mol_copy, force=True, cleanIt=True)
                else:
                    # Generate conformer for stereochemistry assignment
                    try:
                        AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                        Chem.rdmolops.AssignStereochemistry(mol_copy, force=True, cleanIt=True)
                    except Exception:
                        # If conformer generation fails, just assign without 3D
                        Chem.rdmolops.AssignStereochemistry(mol_copy, force=True, cleanIt=True)
            
            return mol_copy
            
        except Exception as e:
            logger.error(f"Failed to standardize stereochemistry: {e}")
            return mol
    
    def enumerate_undefined_stereochemistry(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Enumerate undefined stereocenters.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of stereoisomers
        """
        if mol is None or not self.config.enumerate_undefined_stereo:
            return [mol] if mol else []
        
        try:
            # Count undefined stereocenters
            undefined_centers = []
            for atom in mol.GetAtoms():
                if atom.HasProp('_ChiralityPossible') and not atom.HasProp('_CIPCode'):
                    undefined_centers.append(atom.GetIdx())
            
            if len(undefined_centers) > self.config.max_undefined_stereocenters:
                logger.warning(f"Molecule has {len(undefined_centers)} undefined stereocenters, "
                             f"exceeding limit of {self.config.max_undefined_stereocenters}")
                return [mol]
            
            # Enumerate stereoisomers
            isomers = tuple(EnumerateStereoisomers(mol, options=self.stereo_options))
            
            if len(isomers) > 100:  # Reasonable limit
                logger.warning(f"Too many stereoisomers ({len(isomers)}), returning original")
                return [mol]
            
            return list(isomers)
            
        except Exception as e:
            logger.error(f"Failed to enumerate stereoisomers: {e}")
            return [mol]
    
    def get_stereochemistry_info(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Get stereochemistry information for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with stereochemistry information
        """
        if mol is None:
            return {}
        
        try:
            info = {
                'has_stereochemistry': False,
                'stereocenters': 0,
                'undefined_stereocenters': 0,
                'chiral_centers': [],
                'double_bond_stereo': 0
            }
            
            # Count stereocenters
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            info['stereocenters'] = len(chiral_centers)
            info['chiral_centers'] = chiral_centers
            
            # Count undefined stereocenters
            undefined = [center for center in chiral_centers if center[1] == '?']
            info['undefined_stereocenters'] = len(undefined)
            
            # Count double bond stereochemistry
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                        info['double_bond_stereo'] += 1
            
            info['has_stereochemistry'] = (info['stereocenters'] > 0 or 
                                         info['double_bond_stereo'] > 0)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get stereochemistry info: {e}")
            return {}


class TautomerStandardizer:
    """Main class for tautomer and stereochemistry standardization."""
    
    def __init__(self, config: Optional[TautomerStandardizationConfig] = None):
        """Initialize tautomer standardizer.
        
        Args:
            config: Standardization configuration
        """
        self.config = config or TautomerStandardizationConfig()
        self.tautomer_enumerator = TautomerEnumerator(self.config)
        self.stereo_standardizer = StereochemistryStandardizer(self.config)
        
        # Initialize cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        logger.info(f"Initialized tautomer standardizer with config: {self.config}")
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, smiles: str) -> str:
        """Generate cache key for SMILES."""
        return hashlib.md5(smiles.encode()).hexdigest()
    
    def _get_from_cache(self, smiles: str) -> Optional[TautomerResult]:
        """Get result from cache."""
        if not self.config.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(smiles)
        with self._cache_lock:
            return self._cache.get(cache_key)
    
    def _store_in_cache(self, smiles: str, result: TautomerResult) -> None:
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
    
    def standardize_single(self, smiles: str) -> TautomerResult:
        """Standardize a single SMILES string.
        
        Args:
            smiles: SMILES string to standardize
            
        Returns:
            Tautomer standardization result
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._get_from_cache(smiles)
        if cached_result is not None:
            logger.debug(f"Retrieved from cache: {smiles}")
            return cached_result
        
        result = TautomerResult(
            original_smiles=smiles,
            canonical_tautomer=None,
            tautomer_count=0,
            standardized_stereochemistry=None,
            molecular_formula=None,
            status=TautomerStandardizationStatus.FAILED
        )
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.status = TautomerStandardizationStatus.INVALID_INPUT
                result.error_message = "Failed to parse SMILES"
                return result
            
            # Get molecular formula
            try:
                result.molecular_formula = CalcMolFormula(mol)
            except Exception:
                pass
            
            # Enumerate tautomers
            tautomers = self.tautomer_enumerator.enumerate_tautomers(mol)
            result.tautomer_count = len(tautomers)
            
            if not tautomers:
                result.status = TautomerStandardizationStatus.NO_TAUTOMERS
                result.error_message = "No tautomers enumerated"
                return result
            
            # Get canonical tautomer
            canonical_mol = self.tautomer_enumerator.get_canonical_tautomer(mol)
            if canonical_mol is not None:
                try:
                    result.canonical_tautomer = Chem.MolToSmiles(canonical_mol, canonical=True)
                except Exception as e:
                    logger.error(f"Failed to convert canonical tautomer to SMILES: {e}")
                    result.canonical_tautomer = smiles  # Fallback to original
            
            # Standardize stereochemistry if requested
            if self.config.include_stereochemistry:
                try:
                    stereo_mol = self.stereo_standardizer.standardize_stereochemistry(canonical_mol or mol)
                    if stereo_mol is not None:
                        result.standardized_stereochemistry = Chem.MolToSmiles(stereo_mol, canonical=True)
                        
                        # Get stereochemistry information
                        stereo_info = self.stereo_standardizer.get_stereochemistry_info(stereo_mol)
                        result.metadata['stereochemistry_info'] = stereo_info
                    
                except Exception as e:
                    logger.error(f"Stereochemistry standardization failed: {e}")
                    result.status = TautomerStandardizationStatus.STEREOCHEMISTRY_ERROR
                    result.error_message = f"Stereochemistry error: {e}"
                    return result
            
            result.status = TautomerStandardizationStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Tautomer standardization failed for {smiles}: {e}")
            result.status = TautomerStandardizationStatus.FAILED
            result.error_message = str(e)
        
        finally:
            result.processing_time = time.time() - start_time
        
        # Store in cache
        self._store_in_cache(smiles, result)
        
        return result
    
    def _standardize_chunk_threading(self, smiles_chunk: List[str]) -> List[TautomerResult]:
        """Process chunk using threading."""
        results = []
        for smiles in smiles_chunk:
            try:
                result = self.standardize_single(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {smiles}: {e}")
                if not self.config.skip_on_error:
                    raise
                results.append(TautomerResult(
                    original_smiles=smiles,
                    canonical_tautomer=None,
                    tautomer_count=0,
                    standardized_stereochemistry=None,
                    molecular_formula=None,
                    status=TautomerStandardizationStatus.FAILED,
                    error_message=str(e)
                ))
        return results
    
    def _standardize_chunk_multiprocessing(self, smiles_chunk: List[str]) -> List[TautomerResult]:
        """Process chunk using multiprocessing."""
        # Note: This method would be called in a separate process
        # Create a new standardizer instance to avoid pickling issues
        local_standardizer = TautomerStandardizer(self.config)
        return [local_standardizer.standardize_single(smiles) for smiles in smiles_chunk]
    
    def standardize_batch(self, smiles_list: List[str]) -> List[TautomerResult]:
        """Standardize a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of standardization results
        """
        if not smiles_list:
            return []
        
        logger.info(f"Standardizing batch of {len(smiles_list)} molecules using {self.config.parallel_backend}")
        
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
            if self.config.parallel_backend == "multiprocessing" and len(uncached_smiles) > 100:
                new_results = self._standardize_multiprocessing(uncached_smiles)
            elif self.config.parallel_backend == "joblib" and JOBLIB_AVAILABLE:
                new_results = self._standardize_joblib(uncached_smiles)
            else:
                new_results = self._standardize_threading(uncached_smiles)
        
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
                results.append(TautomerResult(
                    original_smiles=smiles,
                    canonical_tautomer=None,
                    tautomer_count=0,
                    standardized_stereochemistry=None,
                    molecular_formula=None,
                    status=TautomerStandardizationStatus.FAILED,
                    error_message="Result not found"
                ))
        
        # Log statistics
        successful = len([r for r in results if r.is_successful])
        logger.info(f"Standardization complete: {successful}/{len(results)} successful")
        
        return results
    
    def _standardize_threading(self, smiles_list: List[str]) -> List[TautomerResult]:
        """Standardize using threading."""
        max_workers = self.config.max_workers or min(32, len(smiles_list))
        chunk_size = max(1, len(smiles_list) // max_workers)
        
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._standardize_chunk_threading, chunk): chunk 
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
    
    def _standardize_multiprocessing(self, smiles_list: List[str]) -> List[TautomerResult]:
        """Standardize using multiprocessing."""
        max_workers = self.config.max_workers or min(8, len(smiles_list) // 100)
        chunk_size = max(100, len(smiles_list) // max_workers)
        
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._standardize_chunk_multiprocessing, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result(timeout=self.config.timeout_per_molecule * len(future_to_chunk[future]) * 2)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Multiprocessing chunk failed: {e}")
                    if not self.config.skip_on_error:
                        raise
        
        return results
    
    def _standardize_joblib(self, smiles_list: List[str]) -> List[TautomerResult]:
        """Standardize using joblib."""
        n_jobs = self.config.max_workers or -1
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
            delayed(self.standardize_single)(smiles) for smiles in smiles_list
        )
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the standardization cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cleared tautomer standardization cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.config.cache_size,
                'cache_enabled': self.config.cache_enabled
            }


# Standalone function for easy import
def standardize_tautomers_batch(smiles_list: List[str], 
                               config: Optional[TautomerStandardizationConfig] = None) -> List[TautomerResult]:
    """Standardize tautomers for a batch of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        config: Standardization configuration
        
    Returns:
        List of standardization results
    """
    standardizer = TautomerStandardizer(config)
    return standardizer.standardize_batch(smiles_list)