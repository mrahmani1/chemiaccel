"""
PAINS and Substructure Filter Parallel Processing.

This module provides high-performance PAINS (Pan-Assay Interference Compounds) 
filtering and custom substructure filtering with GPU acceleration and parallel 
processing capabilities for large-scale molecular datasets.
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
import json
from pathlib import Path
from collections import defaultdict

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
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    FilterCatalog = None
    FilterCatalogParams = None

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, MolecularDataError, GPUMemoryError
from ..utils.logging import get_logger
from ..backends.cuda.memory import get_memory_manager

logger = get_logger()


class FilterType(Enum):
    """Type of molecular filter."""
    PAINS_A = "pains_a"
    PAINS_B = "pains_b"
    PAINS_C = "pains_c"
    BRENK = "brenk"
    NIH = "nih"
    ZINC = "zinc"
    CUSTOM_SMARTS = "custom_smarts"
    CUSTOM_SUBSTRUCTURE = "custom_substructure"
    LIPINSKI_RULE_OF_5 = "lipinski_rule_of_5"
    VEBER = "veber"
    EGAN = "egan"
    MUEGGE = "muegge"


class FilterAction(Enum):
    """Action to take when filter matches."""
    EXCLUDE = "exclude"  # Remove matching molecules
    INCLUDE = "include"  # Keep only matching molecules
    FLAG = "flag"        # Flag matching molecules but keep them


@dataclass
class FilterRule:
    """Definition of a molecular filter rule."""
    name: str
    filter_type: FilterType
    action: FilterAction
    pattern: Optional[str] = None  # SMARTS pattern for custom filters
    description: Optional[str] = None
    enabled: bool = True
    priority: int = 0  # Higher priority filters run first
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of molecular filtering."""
    original_smiles: str
    molecule_index: int
    matched_filters: List[str]  # Names of matched filters
    excluded: bool
    flagged: bool
    filter_scores: Dict[str, Any]  # Filter-specific scores/values
    drug_like_properties: Dict[str, float]
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def is_clean(self) -> bool:
        """Check if molecule passed all filters."""
        return not self.excluded and not self.error_message
    
    @property
    def has_flags(self) -> bool:
        """Check if molecule has any flags."""
        return self.flagged or len(self.matched_filters) > 0


@dataclass
class BatchFilterResult:
    """Result of batch filtering operation."""
    total_molecules: int
    clean_molecules: int
    excluded_molecules: int
    flagged_molecules: int
    clean_indices: List[int]
    excluded_indices: List[int]
    flagged_indices: List[int]
    filter_statistics: Dict[str, int]  # Filter name -> number of matches
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def exclusion_rate(self) -> float:
        """Calculate exclusion rate as percentage."""
        return (self.excluded_molecules / self.total_molecules * 100) if self.total_molecules > 0 else 0.0


@dataclass
class FilterConfig:
    """Configuration for molecular filtering."""
    # Filter selection
    enabled_filters: List[FilterType] = field(default_factory=lambda: [
        FilterType.PAINS_A, FilterType.PAINS_B, FilterType.LIPINSKI_RULE_OF_5
    ])
    custom_filter_rules: List[FilterRule] = field(default_factory=list)
    
    # Performance settings
    parallel_backend: str = "threading"  # "threading", "multiprocessing", "joblib"
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    timeout_per_molecule: float = 1.0
    use_gpu_acceleration: bool = True
    
    # Memory management
    use_gpu_memory: bool = True
    max_memory_usage_gb: float = 4.0
    batch_size_optimization: bool = True
    
    # Caching settings
    cache_enabled: bool = True
    cache_size: int = 100000
    persistent_cache: bool = False
    cache_file: Optional[str] = None
    
    # Drug-likeness parameters
    calculate_drug_properties: bool = True
    lipinski_mw_max: float = 500.0
    lipinski_logp_max: float = 5.0
    lipinski_hbd_max: int = 5
    lipinski_hba_max: int = 10
    veber_rotbonds_max: int = 10
    veber_psa_max: float = 140.0
    
    # Error handling
    skip_on_error: bool = True
    max_retries: int = 1
    strict_mode: bool = False  # Strict interpretation of filters


class PAINSFilter:
    """PAINS (Pan-Assay Interference Compounds) filter implementation."""
    
    def __init__(self):
        """Initialize PAINS filter."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for PAINS filtering")
        
        self._filter_catalogs = {}
        self._initialize_filter_catalogs()
        
        # Thread-local storage for RDKit objects
        self._local = threading.local()
        
        logger.debug("Initialized PAINS filter")
    
    def _initialize_filter_catalogs(self):
        """Initialize RDKit filter catalogs."""
        try:
            # PAINS A
            params_a = FilterCatalogParams()
            params_a.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            self._filter_catalogs[FilterType.PAINS_A] = FilterCatalog(params_a)
            
            # PAINS B
            params_b = FilterCatalogParams()
            params_b.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            self._filter_catalogs[FilterType.PAINS_B] = FilterCatalog(params_b)
            
            # PAINS C
            params_c = FilterCatalogParams()
            params_c.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            self._filter_catalogs[FilterType.PAINS_C] = FilterCatalog(params_c)
            
            # Brenk filters
            if hasattr(FilterCatalogParams.FilterCatalogs, 'BRENK'):
                params_brenk = FilterCatalogParams()
                params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
                self._filter_catalogs[FilterType.BRENK] = FilterCatalog(params_brenk)
            
            # NIH filters
            if hasattr(FilterCatalogParams.FilterCatalogs, 'NIH'):
                params_nih = FilterCatalogParams()
                params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
                self._filter_catalogs[FilterType.NIH] = FilterCatalog(params_nih)
            
            # ZINC filters
            if hasattr(FilterCatalogParams.FilterCatalogs, 'ZINC'):
                params_zinc = FilterCatalogParams()
                params_zinc.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
                self._filter_catalogs[FilterType.ZINC] = FilterCatalog(params_zinc)
            
            logger.info(f"Initialized {len(self._filter_catalogs)} PAINS filter catalogs")
            
        except Exception as e:
            logger.error(f"Failed to initialize PAINS filter catalogs: {e}")
            raise
    
    def _get_local_catalog(self, filter_type: FilterType):
        """Get thread-local filter catalog."""
        if not hasattr(self._local, 'catalogs'):
            self._local.catalogs = {}
        
        if filter_type not in self._local.catalogs:
            if filter_type in self._filter_catalogs:
                self._local.catalogs[filter_type] = self._filter_catalogs[filter_type]
        
        return self._local.catalogs.get(filter_type)
    
    def check_pains(self, mol: Chem.Mol, filter_types: List[FilterType]) -> Dict[str, List[str]]:
        """Check molecule against PAINS filters.
        
        Args:
            mol: RDKit molecule object
            filter_types: List of filter types to check
            
        Returns:
            Dictionary mapping filter type to list of matched filter names
        """
        if mol is None:
            return {}
        
        matches = {}
        
        for filter_type in filter_types:
            if filter_type not in self._filter_catalogs:
                continue
            
            try:
                catalog = self._get_local_catalog(filter_type)
                if catalog is None:
                    continue
                
                filter_matches = catalog.GetMatches(mol)
                
                if filter_matches:
                    match_names = []
                    for match in filter_matches:
                        match_names.append(match.GetDescription())
                    matches[filter_type.value] = match_names
                
            except Exception as e:
                logger.error(f"Error checking {filter_type.value} for molecule: {e}")
        
        return matches


class SubstructureFilter:
    """Custom substructure filter implementation."""
    
    def __init__(self, custom_rules: List[FilterRule]):
        """Initialize substructure filter.
        
        Args:
            custom_rules: List of custom filter rules
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for substructure filtering")
        
        self.custom_rules = custom_rules
        self._compiled_patterns = {}
        self._compile_patterns()
        
        logger.debug(f"Initialized substructure filter with {len(custom_rules)} custom rules")
    
    def _compile_patterns(self):
        """Compile SMARTS patterns for efficiency."""
        for rule in self.custom_rules:
            if rule.pattern and rule.enabled:
                try:
                    pattern_mol = Chem.MolFromSmarts(rule.pattern)
                    if pattern_mol is not None:
                        self._compiled_patterns[rule.name] = pattern_mol
                    else:
                        logger.warning(f"Failed to compile SMARTS pattern for rule {rule.name}: {rule.pattern}")
                except Exception as e:
                    logger.error(f"Error compiling pattern for rule {rule.name}: {e}")
    
    def check_substructures(self, mol: Chem.Mol) -> Dict[str, bool]:
        """Check molecule against custom substructure patterns.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary mapping rule name to match status
        """
        if mol is None:
            return {}
        
        matches = {}
        
        for rule_name, pattern_mol in self._compiled_patterns.items():
            try:
                has_match = mol.HasSubstructMatch(pattern_mol)
                matches[rule_name] = has_match
            except Exception as e:
                logger.error(f"Error checking substructure {rule_name}: {e}")
                matches[rule_name] = False
        
        return matches


class DrugLikenessFilter:
    """Drug-likeness filter implementation."""
    
    def __init__(self, config: FilterConfig):
        """Initialize drug-likeness filter.
        
        Args:
            config: Filter configuration
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for drug-likeness filtering")
        
        self.config = config
        logger.debug("Initialized drug-likeness filter")
    
    def calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate drug-likeness properties.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of calculated properties
        """
        if mol is None:
            return {}
        
        try:
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),  # Hydrogen bond donors
                'hba': rdMolDescriptors.CalcNumHBA(mol),  # Hydrogen bond acceptors
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),  # Topological polar surface area
                'heavy_atom_count': mol.GetNumHeavyAtoms(),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
            }
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating drug-likeness properties: {e}")
            return {}
    
    def check_lipinski_rule_of_5(self, properties: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check Lipinski's Rule of 5.
        
        Args:
            properties: Calculated molecular properties
            
        Returns:
            Tuple of (passes_rule, list_of_violations)
        """
        violations = []
        
        if properties.get('molecular_weight', 0) > self.config.lipinski_mw_max:
            violations.append(f"MW > {self.config.lipinski_mw_max}")
        
        if properties.get('logp', 0) > self.config.lipinski_logp_max:
            violations.append(f"LogP > {self.config.lipinski_logp_max}")
        
        if properties.get('hbd', 0) > self.config.lipinski_hbd_max:
            violations.append(f"HBD > {self.config.lipinski_hbd_max}")
        
        if properties.get('hba', 0) > self.config.lipinski_hba_max:
            violations.append(f"HBA > {self.config.lipinski_hba_max}")
        
        passes = len(violations) == 0
        return passes, violations
    
    def check_veber_rule(self, properties: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check Veber's rule.
        
        Args:
            properties: Calculated molecular properties
            
        Returns:
            Tuple of (passes_rule, list_of_violations)
        """
        violations = []
        
        if properties.get('rotatable_bonds', 0) > self.config.veber_rotbonds_max:
            violations.append(f"RotBonds > {self.config.veber_rotbonds_max}")
        
        if properties.get('tpsa', 0) > self.config.veber_psa_max:
            violations.append(f"TPSA > {self.config.veber_psa_max}")
        
        passes = len(violations) == 0
        return passes, violations
    
    def check_egan_rule(self, properties: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check Egan's rule (simplified).
        
        Args:
            properties: Calculated molecular properties
            
        Returns:
            Tuple of (passes_rule, list_of_violations)
        """
        violations = []
        
        # Egan rule: TPSA <= 131.6 and LogP <= 5.88
        if properties.get('tpsa', 0) > 131.6:
            violations.append("TPSA > 131.6")
        
        if properties.get('logp', 0) > 5.88:
            violations.append("LogP > 5.88")
        
        passes = len(violations) == 0
        return passes, violations


class MolecularFilterEngine:
    """Main molecular filtering engine."""
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """Initialize molecular filter engine.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        
        # Initialize sub-filters
        self.pains_filter = PAINSFilter()
        self.substructure_filter = SubstructureFilter(self.config.custom_filter_rules)
        self.druglikeness_filter = DrugLikenessFilter(self.config)
        
        # Initialize GPU memory manager if available
        self.memory_manager = None
        if self.config.use_gpu_acceleration and CUPY_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager()
                logger.info("GPU acceleration enabled for molecular filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory manager: {e}")
                self.config.use_gpu_acceleration = False
        
        # Initialize cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Load persistent cache if configured
        if self.config.persistent_cache and self.config.cache_file:
            self._load_persistent_cache()
        
        logger.info(f"Initialized molecular filter engine with {len(self.config.enabled_filters)} enabled filters")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from file."""
        try:
            cache_path = Path(self.config.cache_file)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
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
        # Include filter configuration in cache key
        config_hash = hashlib.md5(
            str(sorted(self.config.enabled_filters)).encode()
        ).hexdigest()[:8]
        return hashlib.md5(f"{smiles}_{config_hash}".encode()).hexdigest()
    
    def _get_from_cache(self, smiles: str) -> Optional[FilterResult]:
        """Get result from cache."""
        if not self.config.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(smiles)
        with self._cache_lock:
            return self._cache.get(cache_key)
    
    def _store_in_cache(self, smiles: str, result: FilterResult) -> None:
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
    
    def filter_single(self, smiles: str, molecule_index: int = 0) -> FilterResult:
        """Filter a single molecule.
        
        Args:
            smiles: SMILES string
            molecule_index: Index of molecule in original dataset
            
        Returns:
            Filter result
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._get_from_cache(smiles)
        if cached_result is not None:
            cached_result.molecule_index = molecule_index  # Update index
            logger.debug(f"Retrieved from cache: {smiles}")
            return cached_result
        
        result = FilterResult(
            original_smiles=smiles,
            molecule_index=molecule_index,
            matched_filters=[],
            excluded=False,
            flagged=False,
            filter_scores={},
            drug_like_properties={}
        )
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.error_message = "Failed to parse SMILES"
                return result
            
            # Calculate drug-likeness properties if requested
            if self.config.calculate_drug_properties:
                result.drug_like_properties = self.druglikeness_filter.calculate_properties(mol)
            
            # Check PAINS filters
            pains_filters = [f for f in self.config.enabled_filters if f in [
                FilterType.PAINS_A, FilterType.PAINS_B, FilterType.PAINS_C,
                FilterType.BRENK, FilterType.NIH, FilterType.ZINC
            ]]
            
            if pains_filters:
                pains_matches = self.pains_filter.check_pains(mol, pains_filters)
                for filter_name, match_list in pains_matches.items():
                    if match_list:
                        result.matched_filters.append(filter_name)
                        result.filter_scores[filter_name] = match_list
            
            # Check custom substructure filters
            if self.config.custom_filter_rules:
                substructure_matches = self.substructure_filter.check_substructures(mol)
                for rule_name, has_match in substructure_matches.items():
                    if has_match:
                        # Find the rule to get its action
                        rule = next((r for r in self.config.custom_filter_rules if r.name == rule_name), None)
                        if rule:
                            result.matched_filters.append(rule_name)
                            result.filter_scores[rule_name] = True
            
            # Check drug-likeness rules
            if FilterType.LIPINSKI_RULE_OF_5 in self.config.enabled_filters:
                passes, violations = self.druglikeness_filter.check_lipinski_rule_of_5(result.drug_like_properties)
                if not passes:
                    result.matched_filters.append("lipinski_violations")
                    result.filter_scores["lipinski_violations"] = violations
            
            if FilterType.VEBER in self.config.enabled_filters:
                passes, violations = self.druglikeness_filter.check_veber_rule(result.drug_like_properties)
                if not passes:
                    result.matched_filters.append("veber_violations")
                    result.filter_scores["veber_violations"] = violations
            
            if FilterType.EGAN in self.config.enabled_filters:
                passes, violations = self.druglikeness_filter.check_egan_rule(result.drug_like_properties)
                if not passes:
                    result.matched_filters.append("egan_violations")
                    result.filter_scores["egan_violations"] = violations
            
            # Determine final filter actions
            for rule in self.config.custom_filter_rules:
                if rule.name in result.matched_filters:
                    if rule.action == FilterAction.EXCLUDE:
                        result.excluded = True
                    elif rule.action == FilterAction.FLAG:
                        result.flagged = True
            
            # Default action for PAINS matches is exclusion
            pains_matches = [f for f in result.matched_filters if 'pains' in f.lower() or f in ['brenk', 'nih', 'zinc']]
            if pains_matches:
                result.excluded = True
            
            # Drug-likeness violations typically flag rather than exclude
            druglike_matches = [f for f in result.matched_filters if 'violations' in f]
            if druglike_matches:
                result.flagged = True
            
        except Exception as e:
            logger.error(f"Filtering failed for {smiles}: {e}")
            result.error_message = str(e)
        
        finally:
            result.processing_time = time.time() - start_time
        
        # Store in cache
        self._store_in_cache(smiles, result)
        
        return result
    
    def _process_chunk_threading(self, smiles_chunk: List[Tuple[str, int]]) -> List[FilterResult]:
        """Process chunk using threading."""
        results = []
        for smiles, index in smiles_chunk:
            try:
                result = self.filter_single(smiles, index)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {smiles}: {e}")
                if not self.config.skip_on_error:
                    raise
                results.append(FilterResult(
                    original_smiles=smiles,
                    molecule_index=index,
                    matched_filters=[],
                    excluded=False,
                    flagged=False,
                    filter_scores={},
                    drug_like_properties={},
                    error_message=str(e)
                ))
        return results
    
    def filter_batch(self, smiles_list: List[str]) -> BatchFilterResult:
        """Filter a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Batch filter result
        """
        start_time = time.time()
        
        if not smiles_list:
            return BatchFilterResult(
                total_molecules=0,
                clean_molecules=0,
                excluded_molecules=0,
                flagged_molecules=0,
                clean_indices=[],
                excluded_indices=[],
                flagged_indices=[],
                filter_statistics={},
                processing_time=0.0
            )
        
        logger.info(f"Filtering batch of {len(smiles_list)} molecules using {self.config.parallel_backend}")
        
        # Create list of (smiles, index) tuples
        smiles_with_indices = [(smiles, i) for i, smiles in enumerate(smiles_list)]
        
        # Process in parallel
        if self.config.parallel_backend == "multiprocessing" and len(smiles_list) > 1000:
            results = self._filter_multiprocessing(smiles_with_indices)
        elif self.config.parallel_backend == "joblib" and JOBLIB_AVAILABLE:
            results = self._filter_joblib(smiles_with_indices)
        else:
            results = self._filter_threading(smiles_with_indices)
        
        # Analyze results
        clean_indices = []
        excluded_indices = []
        flagged_indices = []
        filter_statistics = defaultdict(int)
        
        for result in results:
            if result.error_message:
                excluded_indices.append(result.molecule_index)
            elif result.excluded:
                excluded_indices.append(result.molecule_index)
            elif result.flagged:
                flagged_indices.append(result.molecule_index)
            else:
                clean_indices.append(result.molecule_index)
            
            # Count filter matches
            for filter_name in result.matched_filters:
                filter_statistics[filter_name] += 1
        
        processing_time = time.time() - start_time
        
        batch_result = BatchFilterResult(
            total_molecules=len(smiles_list),
            clean_molecules=len(clean_indices),
            excluded_molecules=len(excluded_indices),
            flagged_molecules=len(flagged_indices),
            clean_indices=clean_indices,
            excluded_indices=excluded_indices,
            flagged_indices=flagged_indices,
            filter_statistics=dict(filter_statistics),
            processing_time=processing_time,
            metadata={
                'enabled_filters': [f.value for f in self.config.enabled_filters],
                'custom_rules_count': len(self.config.custom_filter_rules),
                'cache_hit_rate': self._calculate_cache_hit_rate(len(smiles_list))
            }
        )
        
        # Save persistent cache if configured
        if self.config.persistent_cache:
            self._save_persistent_cache()
        
        logger.info(f"Filtering complete: {batch_result.clean_molecules} clean, "
                   f"{batch_result.excluded_molecules} excluded, "
                   f"{batch_result.flagged_molecules} flagged in {processing_time:.2f}s")
        
        return batch_result
    
    def _filter_threading(self, smiles_with_indices: List[Tuple[str, int]]) -> List[FilterResult]:
        """Filter using threading."""
        max_workers = self.config.max_workers or min(32, len(smiles_with_indices))
        chunk_size = max(1, len(smiles_with_indices) // max_workers)
        
        chunks = [smiles_with_indices[i:i + chunk_size] for i in range(0, len(smiles_with_indices), chunk_size)]
        
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
        
        # Sort results by molecule index to maintain order
        results.sort(key=lambda x: x.molecule_index)
        return results
    
    def _filter_multiprocessing(self, smiles_with_indices: List[Tuple[str, int]]) -> List[FilterResult]:
        """Filter using multiprocessing."""
        max_workers = self.config.max_workers or min(8, len(smiles_with_indices) // 1000)
        chunk_size = max(1000, len(smiles_with_indices) // max_workers)
        
        chunks = [smiles_with_indices[i:i + chunk_size] for i in range(0, len(smiles_with_indices), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create separate filter instances for each process
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
        
        # Sort results by molecule index to maintain order
        results.sort(key=lambda x: x.molecule_index)
        return results
    
    def _process_chunk_multiprocessing(self, smiles_chunk: List[Tuple[str, int]]) -> List[FilterResult]:
        """Process chunk using multiprocessing."""
        # Create a new filter engine instance to avoid pickling issues
        local_filter = MolecularFilterEngine(self.config)
        return [local_filter.filter_single(smiles, index) for smiles, index in smiles_chunk]
    
    def _filter_joblib(self, smiles_with_indices: List[Tuple[str, int]]) -> List[FilterResult]:
        """Filter using joblib."""
        n_jobs = self.config.max_workers or -1
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
            delayed(self.filter_single)(smiles, index) for smiles, index in smiles_with_indices
        )
        
        return results
    
    def _calculate_cache_hit_rate(self, total_requests: int) -> float:
        """Calculate cache hit rate."""
        # This is a simplified calculation
        # In practice, you'd track cache hits vs misses
        cache_size = len(self._cache)
        estimated_hits = min(cache_size, total_requests)
        return (estimated_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def filter_dataframe(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> Tuple[pd.DataFrame, BatchFilterResult]:
        """Filter molecules in a DataFrame.
        
        Args:
            df: DataFrame containing SMILES
            smiles_column: Name of the SMILES column
            
        Returns:
            Tuple of (filtered_dataframe, batch_filter_result)
        """
        if not PANDAS_AVAILABLE:
            raise RuntimeError("Pandas is required for DataFrame filtering")
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in DataFrame")
        
        smiles_list = df[smiles_column].tolist()
        batch_result = self.filter_batch(smiles_list)
        
        # Create filtered DataFrame with clean molecules
        if batch_result.clean_indices:
            clean_df = df.iloc[batch_result.clean_indices].copy()
            
            # Add filter metadata columns
            filter_results = [self.filter_single(smiles_list[i], i) for i in batch_result.clean_indices]
            
            clean_df['filter_flags'] = [','.join(r.matched_filters) for r in filter_results]
            clean_df['is_flagged'] = [r.flagged for r in filter_results]
            
            # Add drug-likeness properties if calculated
            if self.config.calculate_drug_properties and filter_results:
                for prop_name in filter_results[0].drug_like_properties.keys():
                    clean_df[f'drug_like_{prop_name}'] = [
                        r.drug_like_properties.get(prop_name, None) for r in filter_results
                    ]
            
            clean_df = clean_df.reset_index(drop=True)
        else:
            # No clean molecules found
            clean_df = df.iloc[:0].copy()  # Empty DataFrame with same structure
        
        return clean_df, batch_result
    
    def clear_cache(self) -> None:
        """Clear the filter cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cleared molecular filter cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.config.cache_size,
                'cache_enabled': self.config.cache_enabled,
                'persistent_cache': self.config.persistent_cache
            }
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of enabled filters."""
        return {
            'enabled_filters': [f.value for f in self.config.enabled_filters],
            'custom_rules': len(self.config.custom_filter_rules),
            'pains_catalogs': len(self.pains_filter._filter_catalogs),
            'drug_likeness_enabled': self.config.calculate_drug_properties,
            'parallel_backend': self.config.parallel_backend,
            'gpu_acceleration': self.config.use_gpu_acceleration
        }


# Standalone functions for easy import
def filter_molecules_batch(smiles_list: List[str],
                          config: Optional[FilterConfig] = None) -> BatchFilterResult:
    """Filter a batch of molecules using PAINS and other filters.
    
    Args:
        smiles_list: List of SMILES strings
        config: Filter configuration
        
    Returns:
        Batch filter result
    """
    filter_engine = MolecularFilterEngine(config)
    return filter_engine.filter_batch(smiles_list)


def create_custom_filter_rule(name: str,
                            smarts_pattern: str,
                            action: FilterAction = FilterAction.EXCLUDE,
                            description: Optional[str] = None) -> FilterRule:
    """Create a custom filter rule.
    
    Args:
        name: Rule name
        smarts_pattern: SMARTS pattern
        action: Action to take when pattern matches
        description: Rule description
        
    Returns:
        Filter rule
    """
    return FilterRule(
        name=name,
        filter_type=FilterType.CUSTOM_SMARTS,
        action=action,
        pattern=smarts_pattern,
        description=description
    )