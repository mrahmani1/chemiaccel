"""
Molecular fingerprint generation.

This module provides functionality to generate various types of molecular
fingerprints including Morgan (circular), MACCS keys, topological, and
other fingerprint types using RDKit.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, AllChem, MACCSkeys
    from rdkit.Chem.AtomPairs import Pairs, Torsions
    from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.molecular import MolecularData
from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..utils.logging import get_logger

# GPU acceleration imports
try:
    from ..backends.cuda.fingerprint_kernels import (
        GPUFingerprintGenerator,
        GPUMorganFingerprintGenerator,  # Backward compatibility
        get_fingerprint_kernel_manager,
        CUDA_AVAILABLE as CUDA_FP_AVAILABLE
    )
    import cupy as cp
except ImportError:
    CUDA_FP_AVAILABLE = False
    GPUFingerprintGenerator = None
    GPUMorganFingerprintGenerator = None
    cp = None

logger = get_logger()


class FingerprintGenerator:
    """Generate various types of molecular fingerprints."""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize fingerprint generator.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for fingerprint generation")
        
        # Initialize pharmacophore factory for 2D pharmacophore fingerprints
        try:
            self.pharm_factory = Gobbi_Pharm2D.factory
        except:
            self.pharm_factory = None
            logger.warning("Pharmacophore fingerprints not available")
        
        # Initialize GPU acceleration if available and requested
        self.use_gpu = use_gpu and CUDA_FP_AVAILABLE
        if self.use_gpu:
            try:
                self.gpu_generator = GPUFingerprintGenerator()
                logger.info("GPU-accelerated fingerprint generation enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU fingerprint generator: {e}")
                self.use_gpu = False
                self.gpu_generator = None
        else:
            self.gpu_generator = None
        
        logger.debug(f"FingerprintGenerator initialized (GPU: {self.use_gpu})")
    
    def generate_morgan_fingerprint(self, 
                                  mol_data: MolecularData,
                                  radius: int = 2,
                                  n_bits: int = 2048,
                                  use_features: bool = False,
                                  use_chirality: bool = False,
                                  force_cpu: bool = False) -> np.ndarray:
        """Generate Morgan (circular) fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            radius: Radius for circular fingerprint (default: 2)
            n_bits: Number of bits in fingerprint (default: 2048)
            use_features: Use feature-based Morgan fingerprint
            use_chirality: Include chirality information
            force_cpu: Force CPU computation even if GPU is available
            
        Returns:
            Binary fingerprint as numpy array
        """
        # Try GPU acceleration first if available and not forced to CPU
        if self.use_gpu and not force_cpu and not use_features and not use_chirality:
            try:
                return self._generate_morgan_fingerprint_gpu(mol_data, radius, n_bits)
            except Exception as e:
                logger.warning(f"GPU Morgan fingerprint failed, falling back to CPU: {e}")
        
        # Fall back to CPU implementation
        return self._generate_morgan_fingerprint_cpu(mol_data, radius, n_bits, use_features, use_chirality)
    
    def _generate_morgan_fingerprint_cpu(self, 
                                       mol_data: MolecularData,
                                       radius: int,
                                       n_bits: int,
                                       use_features: bool,
                                       use_chirality: bool) -> np.ndarray:
        """Generate Morgan fingerprint using CPU/RDKit implementation."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="morgan_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            if use_features:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    rdkit_mol, radius, nBits=n_bits, 
                    useFeatures=True, useChirality=use_chirality
                )
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    rdkit_mol, radius, nBits=n_bits, 
                    useChirality=use_chirality
                )
            
            # Convert to numpy array
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="morgan_fingerprint",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate Morgan fingerprint: {e}"
            )
    
    def _generate_morgan_fingerprint_gpu(self, 
                                       mol_data: MolecularData,
                                       radius: int,
                                       n_bits: int) -> np.ndarray:
        """Generate Morgan fingerprint using GPU acceleration."""
        if not self.gpu_generator:
            raise RuntimeError("GPU generator not initialized")
        
        # Check if molecular data has GPU-compatible format
        gpu_data = mol_data.backend_data.get("gpu_molecular_data")
        if gpu_data is None:
            raise ValueError("No GPU molecular data found - molecule needs to be processed for GPU")
        
        # Generate fingerprint using GPU
        results = self.gpu_generator.generate_batch_fingerprints(
            gpu_data, radius=radius, n_bits=n_bits, return_counts=False
        )
        
        # Convert packed fingerprints to numpy bit array
        packed_fp = results['morgan_fingerprints']
        if len(packed_fp.shape) == 2 and packed_fp.shape[0] == 1:
            # Single molecule case
            bit_fp = self.gpu_generator.convert_to_numpy_bits(packed_fp, n_bits)
            return bit_fp[0]  # Return single fingerprint
        else:
            raise ValueError("Expected single molecule fingerprint")
    
    def generate_morgan_fingerprints_batch_gpu(self,
                                             molecular_data_batch: Dict[str, Any],
                                             radius: int = 2,
                                             n_bits: int = 2048,
                                             return_counts: bool = False) -> Dict[str, np.ndarray]:
        """Generate Morgan fingerprints for a batch of molecules using GPU.
        
        Args:
            molecular_data_batch: Dictionary containing batch molecular data
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint
            return_counts: Whether to return count-based fingerprints
            
        Returns:
            Dictionary containing fingerprint results
        """
        if not self.use_gpu or not self.gpu_generator:
            raise RuntimeError("GPU fingerprint generation not available")
        
        # Generate fingerprints using GPU
        gpu_results = self.gpu_generator.generate_batch_fingerprints(
            molecular_data_batch, radius=radius, n_bits=n_bits, return_counts=return_counts
        )
        
        # Convert results to numpy arrays
        results = {}
        
        if return_counts:
            # Convert count-based results
            results['hash_values'] = cp.asnumpy(gpu_results['morgan_hash_values'])
            results['hash_counts'] = cp.asnumpy(gpu_results['morgan_hash_counts'])
            results['result_sizes'] = cp.asnumpy(gpu_results['morgan_result_sizes'])
        else:
            # Convert binary fingerprints
            packed_fp = gpu_results['morgan_fingerprints']
            results['fingerprints'] = self.gpu_generator.convert_to_numpy_bits(packed_fp, n_bits)
        
        # Include substructure data if available
        if 'substructure_hashes' in gpu_results:
            results['substructure_hashes'] = cp.asnumpy(gpu_results['substructure_hashes'])
            results['substructure_counts'] = cp.asnumpy(gpu_results['substructure_counts'])
        
        return results
    
    def generate_morgan_counts(self,
                             mol_data: MolecularData,
                             radius: int = 2,
                             use_features: bool = False,
                             use_chirality: bool = False) -> Dict[int, int]:
        """Generate Morgan fingerprint with counts (not binary).
        
        Args:
            mol_data: MolecularData with RDKit mol object
            radius: Radius for circular fingerprint
            use_features: Use feature-based Morgan fingerprint
            use_chirality: Include chirality information
            
        Returns:
            Dictionary mapping hash values to counts
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="morgan_counts",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            if use_features:
                fp = AllChem.GetMorganFingerprint(
                    rdkit_mol, radius, useFeatures=True, useChirality=use_chirality
                )
            else:
                fp = AllChem.GetMorganFingerprint(
                    rdkit_mol, radius, useChirality=use_chirality
                )
            
            return fp.GetNonzeroElements()
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="morgan_counts",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate Morgan counts: {e}"
            )
    
    def generate_maccs_keys(self, mol_data: MolecularData, force_cpu: bool = False) -> np.ndarray:
        """Generate MACCS keys fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            force_cpu: Force CPU computation even if GPU is available
            
        Returns:
            MACCS keys as numpy array (167 bits)
        """
        # Try GPU acceleration first if available and not forced to CPU
        if self.use_gpu and not force_cpu:
            try:
                return self._generate_maccs_keys_gpu(mol_data)
            except Exception as e:
                logger.warning(f"GPU MACCS keys failed, falling back to CPU: {e}")
        
        # Fall back to CPU implementation
        return self._generate_maccs_keys_cpu(mol_data)
    
    def _generate_maccs_keys_cpu(self, mol_data: MolecularData) -> np.ndarray:
        """Generate MACCS keys using CPU/RDKit implementation."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="maccs_keys",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            maccs = MACCSkeys.GenMACCSKeys(rdkit_mol)
            
            # Convert to numpy array
            maccs_array = np.zeros((167,), dtype=np.int8)
            for i in range(167):
                maccs_array[i] = maccs[i]
            
            return maccs_array
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="maccs_keys",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate MACCS keys: {e}"
            )
    
    def _generate_maccs_keys_gpu(self, mol_data: MolecularData) -> np.ndarray:
        """Generate MACCS keys using GPU acceleration."""
        if not self.gpu_generator:
            raise RuntimeError("GPU generator not initialized")
        
        # Check if molecular data has GPU-compatible format
        gpu_data = mol_data.backend_data.get("gpu_molecular_data")
        if gpu_data is None:
            raise ValueError("No GPU molecular data found - molecule needs to be processed for GPU")
        
        # Generate MACCS keys using GPU
        results = self.gpu_generator.generate_batch_maccs_keys(gpu_data)
        
        # Convert packed MACCS keys to numpy bit array
        packed_maccs = results['maccs_keys']
        if len(packed_maccs.shape) == 2 and packed_maccs.shape[0] == 1:
            # Single molecule case
            bit_maccs = self.gpu_generator.convert_maccs_to_numpy_bits(packed_maccs)
            return bit_maccs[0]  # Return single MACCS keys
        else:
            raise ValueError("Expected single molecule MACCS keys")
    
    def generate_topological_fingerprint(self, 
                                       mol_data: MolecularData,
                                       n_bits: int = 2048,
                                       force_cpu: bool = False) -> np.ndarray:
        """Generate topological (Daylight-like) fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            n_bits: Number of bits in fingerprint
            force_cpu: Force CPU computation even if GPU is available
            
        Returns:
            Topological fingerprint as numpy array
        """
        # Try GPU acceleration first if available and not forced to CPU
        if self.use_gpu and not force_cpu:
            try:
                return self._generate_topological_fingerprint_gpu(mol_data, n_bits)
            except Exception as e:
                logger.warning(f"GPU topological fingerprint failed, falling back to CPU: {e}")
        
        # Fall back to CPU implementation
        return self._generate_topological_fingerprint_cpu(mol_data, n_bits)
    
    def _generate_topological_fingerprint_cpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate topological fingerprint using CPU/RDKit implementation."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="topological_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            fp = Chem.RDKFingerprint(rdkit_mol, fpSize=n_bits)
            
            # Convert to numpy array
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="topological_fingerprint",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate topological fingerprint: {e}"
            )
    
    def _generate_topological_fingerprint_gpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate topological fingerprint using GPU acceleration."""
        if not self.gpu_generator:
            raise RuntimeError("GPU generator not initialized")
        
        # Check if molecular data has GPU-compatible format
        gpu_data = mol_data.backend_data.get("gpu_molecular_data")
        if gpu_data is None:
            raise ValueError("No GPU molecular data found - molecule needs to be processed for GPU")
        
        # Generate topological fingerprint using GPU
        results = self.gpu_generator.generate_batch_topological_fingerprints(gpu_data, n_bits=n_bits)
        
        # Convert packed fingerprints to numpy bit array
        packed_fp = results['topological_fingerprints']
        if len(packed_fp.shape) == 2 and packed_fp.shape[0] == 1:
            # Single molecule case
            bit_fp = self.gpu_generator.convert_to_numpy_bits(packed_fp, n_bits)
            return bit_fp[0]  # Return single fingerprint
        else:
            raise ValueError("Expected single molecule fingerprint")
    
    def generate_atom_pair_fingerprint(self, 
                                     mol_data: MolecularData,
                                     n_bits: int = 2048,
                                     force_cpu: bool = False) -> np.ndarray:
        """Generate atom pair fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            n_bits: Number of bits in fingerprint
            force_cpu: Force CPU computation even if GPU is available
            
        Returns:
            Atom pair fingerprint as numpy array
        """
        # Try GPU acceleration first if available and not forced to CPU
        if self.use_gpu and not force_cpu:
            try:
                return self._generate_atom_pair_fingerprint_gpu(mol_data, n_bits)
            except Exception as e:
                logger.warning(f"GPU atom pair fingerprint failed, falling back to CPU: {e}")
        
        # Fall back to CPU implementation
        return self._generate_atom_pair_fingerprint_cpu(mol_data, n_bits)
    
    def _generate_atom_pair_fingerprint_cpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate atom pair fingerprint using CPU/RDKit implementation."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="atom_pair_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            # Use the correct RDKit function
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(rdkit_mol, nBits=n_bits)
            
            # Convert to numpy array
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="atom_pair_fingerprint",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate atom pair fingerprint: {e}"
            )
    
    def _generate_atom_pair_fingerprint_gpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate atom pair fingerprint using GPU acceleration."""
        if not self.gpu_generator:
            raise RuntimeError("GPU generator not initialized")
        
        # Check if molecular data has GPU-compatible format
        gpu_data = mol_data.backend_data.get("gpu_molecular_data")
        if gpu_data is None:
            raise ValueError("No GPU molecular data found - molecule needs to be processed for GPU")
        
        # Generate atom pair fingerprint using GPU
        results = self.gpu_generator.generate_batch_atom_pair_fingerprints(gpu_data, n_bits=n_bits)
        
        # Convert packed fingerprints to numpy bit array
        packed_fp = results['atom_pair_fingerprints']
        if len(packed_fp.shape) == 2 and packed_fp.shape[0] == 1:
            # Single molecule case
            bit_fp = self.gpu_generator.convert_to_numpy_bits(packed_fp, n_bits)
            return bit_fp[0]  # Return single fingerprint
        else:
            raise ValueError("Expected single molecule fingerprint")
    
    def generate_topological_torsion_fingerprint(self, 
                                               mol_data: MolecularData,
                                               n_bits: int = 2048,
                                               force_cpu: bool = False) -> np.ndarray:
        """Generate topological torsion fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            n_bits: Number of bits in fingerprint
            force_cpu: Force CPU computation even if GPU is available
            
        Returns:
            Topological torsion fingerprint as numpy array
        """
        # Try GPU acceleration first if available and not forced to CPU
        if self.use_gpu and not force_cpu:
            try:
                return self._generate_topological_torsion_fingerprint_gpu(mol_data, n_bits)
            except Exception as e:
                logger.warning(f"GPU topological torsion fingerprint failed, falling back to CPU: {e}")
        
        # Fall back to CPU implementation
        return self._generate_topological_torsion_fingerprint_cpu(mol_data, n_bits)
    
    def _generate_topological_torsion_fingerprint_cpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate topological torsion fingerprint using CPU/RDKit implementation."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="topological_torsion_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            # Use the correct RDKit function name
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(rdkit_mol, nBits=n_bits)
            
            # Convert to numpy array
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="topological_torsion_fingerprint",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to generate topological torsion fingerprint: {e}"
            )
    
    def _generate_topological_torsion_fingerprint_gpu(self, mol_data: MolecularData, n_bits: int) -> np.ndarray:
        """Generate topological torsion fingerprint using GPU acceleration."""
        if not self.gpu_generator:
            raise RuntimeError("GPU generator not initialized")
        
        # Check if molecular data has GPU-compatible format
        gpu_data = mol_data.backend_data.get("gpu_molecular_data")
        if gpu_data is None:
            raise ValueError("No GPU molecular data found - molecule needs to be processed for GPU")
        
        # Generate topological torsion fingerprint using GPU
        results = self.gpu_generator.generate_batch_topological_torsion_fingerprints(gpu_data, n_bits=n_bits)
        
        # Convert packed fingerprints to numpy bit array
        packed_fp = results['topological_torsion_fingerprints']
        if len(packed_fp.shape) == 2 and packed_fp.shape[0] == 1:
            # Single molecule case
            bit_fp = self.gpu_generator.convert_to_numpy_bits(packed_fp, n_bits)
            return bit_fp[0]  # Return single fingerprint
        else:
            raise ValueError("Expected single molecule fingerprint")
    
    def generate_avalon_fingerprint(self, 
                                  mol_data: MolecularData,
                                  n_bits: int = 512) -> Optional[np.ndarray]:
        """Generate Avalon fingerprint (if available).
        
        Args:
            mol_data: MolecularData with RDKit mol object
            n_bits: Number of bits in fingerprint
            
        Returns:
            Avalon fingerprint as numpy array or None if not available
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="avalon_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            # Try to import Avalon toolkit
            from rdkit.Avalon import pyAvalonTools
            
            fp = pyAvalonTools.GetAvalonFP(rdkit_mol, nBits=n_bits)
            
            # Convert to numpy array
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except ImportError:
            logger.warning("Avalon toolkit not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to generate Avalon fingerprint: {e}")
            return None
    
    def generate_pharmacophore_fingerprint(self, mol_data: MolecularData) -> Optional[np.ndarray]:
        """Generate 2D pharmacophore fingerprint.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            
        Returns:
            Pharmacophore fingerprint as numpy array or None if not available
        """
        if self.pharm_factory is None:
            return None
        
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor="pharmacophore_fingerprint",
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found"
            )
        
        try:
            fp = Generate.Gen2DFingerprint(rdkit_mol, self.pharm_factory)
            
            # Convert to numpy array
            fp_array = np.zeros((len(fp),), dtype=np.int8)
            for i in range(len(fp)):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            logger.warning(f"Failed to generate pharmacophore fingerprint: {e}")
            return None
    
    def generate_all_fingerprints(self, 
                                mol_data: MolecularData,
                                morgan_radius: int = 2,
                                morgan_bits: int = 2048,
                                topo_bits: int = 2048) -> Dict[str, np.ndarray]:
        """Generate all available fingerprints for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            morgan_radius: Radius for Morgan fingerprint
            morgan_bits: Number of bits for Morgan fingerprint
            topo_bits: Number of bits for topological fingerprints
            
        Returns:
            Dictionary of fingerprint arrays
        """
        fingerprints = {}
        
        # Morgan fingerprint
        try:
            fingerprints["morgan"] = self.generate_morgan_fingerprint(
                mol_data, radius=morgan_radius, n_bits=morgan_bits
            )
        except Exception as e:
            logger.warning(f"Failed to generate Morgan fingerprint: {e}")
        
        # MACCS keys
        try:
            fingerprints["maccs"] = self.generate_maccs_keys(mol_data)
        except Exception as e:
            logger.warning(f"Failed to generate MACCS keys: {e}")
        
        # Topological fingerprint
        try:
            fingerprints["topological"] = self.generate_topological_fingerprint(
                mol_data, n_bits=topo_bits
            )
        except Exception as e:
            logger.warning(f"Failed to generate topological fingerprint: {e}")
        
        # Atom pair fingerprint
        try:
            fingerprints["atom_pair"] = self.generate_atom_pair_fingerprint(
                mol_data, n_bits=topo_bits
            )
        except Exception as e:
            logger.warning(f"Failed to generate atom pair fingerprint: {e}")
        
        # Topological torsion fingerprint
        try:
            fingerprints["torsion"] = self.generate_topological_torsion_fingerprint(
                mol_data, n_bits=topo_bits
            )
        except Exception as e:
            logger.warning(f"Failed to generate torsion fingerprint: {e}")
        
        # Avalon fingerprint (optional)
        avalon_fp = self.generate_avalon_fingerprint(mol_data)
        if avalon_fp is not None:
            fingerprints["avalon"] = avalon_fp
        
        # Pharmacophore fingerprint (optional)
        pharm_fp = self.generate_pharmacophore_fingerprint(mol_data)
        if pharm_fp is not None:
            fingerprints["pharmacophore"] = pharm_fp
        
        return fingerprints


class FingerprintSimilarity:
    """Calculate similarity between molecular fingerprints."""
    
    @staticmethod
    def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Tanimoto similarity coefficient (0-1)
        """
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must have the same length")
        
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(intersection) / float(union)
    
    @staticmethod
    def dice_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Dice similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Dice similarity coefficient (0-1)
        """
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must have the same length")
        
        intersection = np.sum(fp1 & fp2)
        total_bits = np.sum(fp1) + np.sum(fp2)
        
        if total_bits == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2.0 * float(intersection)) / float(total_bits)
    
    @staticmethod
    def cosine_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate cosine similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Cosine similarity coefficient (0-1)
        """
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must have the same length")
        
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if dot_product == 0 else 0.0
        
        return float(dot_product) / (norm1 * norm2)
    
    @staticmethod
    def hamming_distance(fp1: np.ndarray, fp2: np.ndarray) -> int:
        """Calculate Hamming distance between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Hamming distance (number of differing bits)
        """
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must have the same length")
        
        return int(np.sum(fp1 != fp2))
    
    @classmethod
    def calculate_similarity_matrix(cls, 
                                  fingerprints: List[np.ndarray],
                                  method: str = "tanimoto") -> np.ndarray:
        """Calculate pairwise similarity matrix for a list of fingerprints.
        
        Args:
            fingerprints: List of fingerprint arrays
            method: Similarity method ("tanimoto", "dice", "cosine")
            
        Returns:
            Symmetric similarity matrix
        """
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        similarity_func = {
            "tanimoto": cls.tanimoto_similarity,
            "dice": cls.dice_similarity,
            "cosine": cls.cosine_similarity
        }.get(method, cls.tanimoto_similarity)
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = similarity_func(fingerprints[i], fingerprints[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        return similarity_matrix


# Convenience functions
def generate_morgan_fingerprints(molecules: List[MolecularData], 
                               radius: int = 2, 
                               n_bits: int = 2048,
                               use_gpu: bool = True,
                               batch_size: Optional[int] = None) -> List[np.ndarray]:
    """Generate Morgan fingerprints for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU processing (None for auto)
        
    Returns:
        List of Morgan fingerprint arrays
    """
    generator = FingerprintGenerator(use_gpu=use_gpu)
    
    # Try GPU batch processing if available and beneficial
    if (generator.use_gpu and len(molecules) > 10 and 
        all(mol.backend_data.get("gpu_molecular_data") for mol in molecules)):
        
        try:
            return _generate_morgan_fingerprints_gpu_batch(
                generator, molecules, radius, n_bits, batch_size
            )
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to individual: {e}")
    
    # Fall back to individual processing
    fingerprints = []
    for mol_data in molecules:
        try:
            fp = generator.generate_morgan_fingerprint(mol_data, radius, n_bits)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to generate fingerprint for {mol_data.mol_id}: {e}")
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return fingerprints


def _generate_morgan_fingerprints_gpu_batch(generator: FingerprintGenerator,
                                          molecules: List[MolecularData],
                                          radius: int,
                                          n_bits: int,
                                          batch_size: Optional[int]) -> List[np.ndarray]:
    """Generate Morgan fingerprints using GPU batch processing."""
    if batch_size is None:
        batch_size = min(1000, len(molecules))  # Auto batch size
    
    all_fingerprints = []
    
    for i in range(0, len(molecules), batch_size):
        batch_molecules = molecules[i:i + batch_size]
        
        # Combine molecular data for batch processing
        batch_data = _combine_molecular_data_for_batch(batch_molecules)
        
        # Generate batch fingerprints
        batch_results = generator.generate_morgan_fingerprints_batch_gpu(
            batch_data, radius=radius, n_bits=n_bits, return_counts=False
        )
        
        # Extract individual fingerprints
        batch_fps = batch_results['fingerprints']
        for j in range(len(batch_molecules)):
            all_fingerprints.append(batch_fps[j])
    
    return all_fingerprints


def _combine_molecular_data_for_batch(molecules: List[MolecularData]) -> Dict[str, Any]:
    """Combine individual molecular data into batch format for GPU processing."""
    # This is a placeholder - in practice, this would combine the GPU molecular data
    # from individual molecules into the batch format expected by GPU kernels
    
    # For now, we'll extract the first molecule's GPU data as a template
    # In a real implementation, this would properly concatenate all molecular data
    if not molecules:
        raise ValueError("No molecules provided")
    
    first_mol_gpu_data = molecules[0].backend_data.get("gpu_molecular_data")
    if first_mol_gpu_data is None:
        raise ValueError("No GPU molecular data found")
    
    # This is a simplified version - real implementation would properly batch the data
    return first_mol_gpu_data


def generate_maccs_fingerprints(molecules: List[MolecularData], 
                              use_gpu: bool = True,
                              batch_size: Optional[int] = None) -> List[np.ndarray]:
    """Generate MACCS keys for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU processing (None for auto)
        
    Returns:
        List of MACCS key arrays
    """
    generator = FingerprintGenerator(use_gpu=use_gpu)
    
    # Try GPU batch processing if available and beneficial
    if (generator.use_gpu and len(molecules) > 10 and 
        all(mol.backend_data.get("gpu_molecular_data") for mol in molecules)):
        
        try:
            return _generate_maccs_fingerprints_gpu_batch(
                generator, molecules, batch_size
            )
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to individual: {e}")
    
    # Fall back to individual processing
    fingerprints = []
    for mol_data in molecules:
        try:
            fp = generator.generate_maccs_keys(mol_data)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to generate MACCS keys for {mol_data.mol_id}: {e}")
            fingerprints.append(np.zeros(167, dtype=np.int8))
    
    return fingerprints


def generate_topological_fingerprints(molecules: List[MolecularData],
                                    n_bits: int = 2048,
                                    use_gpu: bool = True,
                                    batch_size: Optional[int] = None) -> List[np.ndarray]:
    """Generate topological fingerprints for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        n_bits: Number of bits in fingerprint
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU processing (None for auto)
        
    Returns:
        List of topological fingerprint arrays
    """
    generator = FingerprintGenerator(use_gpu=use_gpu)
    
    # Try GPU batch processing if available and beneficial
    if (generator.use_gpu and len(molecules) > 10 and 
        all(mol.backend_data.get("gpu_molecular_data") for mol in molecules)):
        
        try:
            return _generate_topological_fingerprints_gpu_batch(
                generator, molecules, n_bits, batch_size
            )
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to individual: {e}")
    
    # Fall back to individual processing
    fingerprints = []
    for mol_data in molecules:
        try:
            fp = generator.generate_topological_fingerprint(mol_data, n_bits=n_bits)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to generate topological fingerprint for {mol_data.mol_id}: {e}")
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return fingerprints


def generate_atom_pair_fingerprints(molecules: List[MolecularData],
                                  n_bits: int = 2048,
                                  use_gpu: bool = True,
                                  batch_size: Optional[int] = None) -> List[np.ndarray]:
    """Generate atom pair fingerprints for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        n_bits: Number of bits in fingerprint
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU processing (None for auto)
        
    Returns:
        List of atom pair fingerprint arrays
    """
    generator = FingerprintGenerator(use_gpu=use_gpu)
    
    # Try GPU batch processing if available and beneficial
    if (generator.use_gpu and len(molecules) > 10 and 
        all(mol.backend_data.get("gpu_molecular_data") for mol in molecules)):
        
        try:
            return _generate_atom_pair_fingerprints_gpu_batch(
                generator, molecules, n_bits, batch_size
            )
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to individual: {e}")
    
    # Fall back to individual processing
    fingerprints = []
    for mol_data in molecules:
        try:
            fp = generator.generate_atom_pair_fingerprint(mol_data, n_bits=n_bits)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to generate atom pair fingerprint for {mol_data.mol_id}: {e}")
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return fingerprints


def generate_topological_torsion_fingerprints(molecules: List[MolecularData],
                                            n_bits: int = 2048,
                                            use_gpu: bool = True,
                                            batch_size: Optional[int] = None) -> List[np.ndarray]:
    """Generate topological torsion fingerprints for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        n_bits: Number of bits in fingerprint
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU processing (None for auto)
        
    Returns:
        List of topological torsion fingerprint arrays
    """
    generator = FingerprintGenerator(use_gpu=use_gpu)
    
    # Try GPU batch processing if available and beneficial
    if (generator.use_gpu and len(molecules) > 10 and 
        all(mol.backend_data.get("gpu_molecular_data") for mol in molecules)):
        
        try:
            return _generate_topological_torsion_fingerprints_gpu_batch(
                generator, molecules, n_bits, batch_size
            )
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to individual: {e}")
    
    # Fall back to individual processing
    fingerprints = []
    for mol_data in molecules:
        try:
            fp = generator.generate_topological_torsion_fingerprint(mol_data, n_bits=n_bits)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to generate topological torsion fingerprint for {mol_data.mol_id}: {e}")
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return fingerprints


def _generate_maccs_fingerprints_gpu_batch(generator: FingerprintGenerator,
                                         molecules: List[MolecularData],
                                         batch_size: Optional[int]) -> List[np.ndarray]:
    """Generate MACCS keys using GPU batch processing."""
    if batch_size is None:
        batch_size = min(1000, len(molecules))  # Auto batch size
    
    all_fingerprints = []
    
    for i in range(0, len(molecules), batch_size):
        batch_molecules = molecules[i:i + batch_size]
        
        # Combine molecular data for batch processing
        batch_data = _combine_molecular_data_for_batch(batch_molecules)
        
        # Generate batch fingerprints
        batch_results = generator.gpu_generator.generate_batch_maccs_keys(batch_data)
        
        # Extract individual fingerprints
        batch_fps = generator.gpu_generator.convert_maccs_to_numpy_bits(batch_results['maccs_keys'])
        for j in range(len(batch_molecules)):
            all_fingerprints.append(batch_fps[j])
    
    return all_fingerprints


def _generate_topological_fingerprints_gpu_batch(generator: FingerprintGenerator,
                                               molecules: List[MolecularData],
                                               n_bits: int,
                                               batch_size: Optional[int]) -> List[np.ndarray]:
    """Generate topological fingerprints using GPU batch processing."""
    if batch_size is None:
        batch_size = min(1000, len(molecules))  # Auto batch size
    
    all_fingerprints = []
    
    for i in range(0, len(molecules), batch_size):
        batch_molecules = molecules[i:i + batch_size]
        
        # Combine molecular data for batch processing
        batch_data = _combine_molecular_data_for_batch(batch_molecules)
        
        # Generate batch fingerprints
        batch_results = generator.gpu_generator.generate_batch_topological_fingerprints(
            batch_data, n_bits=n_bits
        )
        
        # Extract individual fingerprints
        batch_fps = generator.gpu_generator.convert_to_numpy_bits(
            batch_results['topological_fingerprints'], n_bits
        )
        for j in range(len(batch_molecules)):
            all_fingerprints.append(batch_fps[j])
    
    return all_fingerprints


def _generate_atom_pair_fingerprints_gpu_batch(generator: FingerprintGenerator,
                                             molecules: List[MolecularData],
                                             n_bits: int,
                                             batch_size: Optional[int]) -> List[np.ndarray]:
    """Generate atom pair fingerprints using GPU batch processing."""
    if batch_size is None:
        batch_size = min(500, len(molecules))  # Smaller batch size due to memory usage
    
    all_fingerprints = []
    
    for i in range(0, len(molecules), batch_size):
        batch_molecules = molecules[i:i + batch_size]
        
        # Combine molecular data for batch processing
        batch_data = _combine_molecular_data_for_batch(batch_molecules)
        
        # Generate batch fingerprints
        batch_results = generator.gpu_generator.generate_batch_atom_pair_fingerprints(
            batch_data, n_bits=n_bits
        )
        
        # Extract individual fingerprints
        batch_fps = generator.gpu_generator.convert_to_numpy_bits(
            batch_results['atom_pair_fingerprints'], n_bits
        )
        for j in range(len(batch_molecules)):
            all_fingerprints.append(batch_fps[j])
    
    return all_fingerprints


def _generate_topological_torsion_fingerprints_gpu_batch(generator: FingerprintGenerator,
                                                       molecules: List[MolecularData],
                                                       n_bits: int,
                                                       batch_size: Optional[int]) -> List[np.ndarray]:
    """Generate topological torsion fingerprints using GPU batch processing."""
    if batch_size is None:
        batch_size = min(1000, len(molecules))  # Auto batch size
    
    all_fingerprints = []
    
    for i in range(0, len(molecules), batch_size):
        batch_molecules = molecules[i:i + batch_size]
        
        # Combine molecular data for batch processing
        batch_data = _combine_molecular_data_for_batch(batch_molecules)
        
        # Generate batch fingerprints
        batch_results = generator.gpu_generator.generate_batch_topological_torsion_fingerprints(
            batch_data, n_bits=n_bits
        )
        
        # Extract individual fingerprints
        batch_fps = generator.gpu_generator.convert_to_numpy_bits(
            batch_results['topological_torsion_fingerprints'], n_bits
        )
        for j in range(len(batch_molecules)):
            all_fingerprints.append(batch_fps[j])
    
    return all_fingerprints


def calculate_fingerprint_similarities(fingerprints: List[np.ndarray], 
                                     method: str = "tanimoto") -> np.ndarray:
    """Calculate pairwise similarity matrix for fingerprints.
    
    Args:
        fingerprints: List of fingerprint arrays
        method: Similarity method ("tanimoto", "dice", "cosine")
        
    Returns:
        Symmetric similarity matrix
    """
    return FingerprintSimilarity.calculate_similarity_matrix(fingerprints, method)