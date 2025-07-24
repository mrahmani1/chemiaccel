"""
Geometrical and 3D molecular descriptors for ChemiAccel.

This module implements comprehensive 3D shape descriptors, electrostatic descriptors,
surface area calculations, and geometric diameter/eccentricity calculations.
Provides GPU-accelerated alternatives to traditional 3D descriptor calculations.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
    from rdkit.Chem import Crippen, rdPartialCharges, rdFreeSASA
    from rdkit.Chem.rdMolAlign import AlignMol
    from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist, ShapeTanimotoDist
    from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
    RDKIT_AVAILABLE = True
    
    # Check for optional 3D descriptor modules
    try:
        from rdkit.Chem.Descriptors3D import (
            RadiusOfGyration, Asphericity, Eccentricity, InertialShapeFactor,
            NPR1, NPR2, PMI1, PMI2, PMI3, SpherocityIndex
        )
        DESCRIPTORS_3D_AVAILABLE = True
    except ImportError:
        DESCRIPTORS_3D_AVAILABLE = False
    
    try:
        from rdkit.Chem.rdFreeSASA import CalcSASA
        FREESASA_AVAILABLE = True
    except ImportError:
        FREESASA_AVAILABLE = False
        
except ImportError:
    RDKIT_AVAILABLE = False
    DESCRIPTORS_3D_AVAILABLE = False
    FREESASA_AVAILABLE = False

from ..core.molecular import MolecularData
from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..utils.logging import get_logger

logger = get_logger()


def check_rdkit_availability():
    """Check if RDKit is available and raise error if not."""
    if not RDKIT_AVAILABLE:
        raise ChemiAccelError(
            "RDKit is not available. Please install RDKit: conda install -c conda-forge rdkit"
        )


class ConformerGenerator:
    """Generate 3D conformers for molecular structures."""
    
    def __init__(self, 
                 num_conformers: int = 10,
                 max_attempts: int = 1000,
                 prune_rms_thresh: float = 0.5,
                 use_random_coords: bool = True,
                 optimize_conformers: bool = True,
                 force_field: str = "UFF"):
        """Initialize conformer generator.
        
        Args:
            num_conformers: Number of conformers to generate
            max_attempts: Maximum attempts for conformer generation
            prune_rms_thresh: RMS threshold for conformer pruning
            use_random_coords: Whether to use random coordinates
            optimize_conformers: Whether to optimize conformers with force field
            force_field: Force field to use ("UFF" or "MMFF")
        """
        check_rdkit_availability()
        
        self.num_conformers = num_conformers
        self.max_attempts = max_attempts
        self.prune_rms_thresh = prune_rms_thresh
        self.use_random_coords = use_random_coords
        self.optimize_conformers = optimize_conformers
        self.force_field = force_field.upper()
        
        if self.force_field not in ["UFF", "MMFF"]:
            raise ValueError("force_field must be 'UFF' or 'MMFF'")
        
        logger.debug(f"ConformerGenerator initialized: {num_conformers} conformers, "
                    f"{force_field} force field, optimize={optimize_conformers}")
    
    def generate_conformers(self, mol_data: MolecularData) -> MolecularData:
        """Generate 3D conformers for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            
        Returns:
            MolecularData with 3D conformers added
            
        Raises:
            ChemiAccelError: If conformer generation fails
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for molecule {mol_data.mol_id}")
        
        # Create a copy to avoid modifying original
        mol_copy = Chem.Mol(rdkit_mol)
        
        try:
            # Add hydrogens for better conformer generation
            mol_with_hs = Chem.AddHs(mol_copy)
            
            # Generate conformers
            conf_ids = EmbedMultipleConfs(
                mol_with_hs,
                numConfs=self.num_conformers,
                maxAttempts=self.max_attempts,
                pruneRmsThresh=self.prune_rms_thresh,
                useRandomCoords=self.use_random_coords,
                numThreads=1  # Single thread for reproducibility
            )
            
            if not conf_ids:
                # Fallback to single conformer
                conf_id = EmbedMolecule(mol_with_hs, maxAttempts=self.max_attempts)
                if conf_id == -1:
                    raise ChemiAccelError(f"Failed to generate conformers for molecule {mol_data.mol_id}")
                conf_ids = [conf_id]
            
            # Optimize conformers if requested
            if self.optimize_conformers:
                optimized_ids = []
                for conf_id in conf_ids:
                    try:
                        if self.force_field == "UFF":
                            result = UFFOptimizeMolecule(mol_with_hs, confId=conf_id)
                        else:  # MMFF
                            result = MMFFOptimizeMolecule(mol_with_hs, confId=conf_id)
                        
                        if result == 0:  # Optimization successful
                            optimized_ids.append(conf_id)
                    except Exception as e:
                        logger.warning(f"Failed to optimize conformer {conf_id}: {e}")
                        continue
                
                if optimized_ids:
                    conf_ids = optimized_ids
            
            # Remove hydrogens to match original molecule
            mol_final = Chem.RemoveHs(mol_with_hs)
            
            # Update molecular data
            updated_mol_data = MolecularData(
                smiles=mol_data.smiles,
                mol_id=mol_data.mol_id,
                num_atoms=mol_data.num_atoms,
                num_bonds=mol_data.num_bonds,
                molecular_weight=mol_data.molecular_weight,
                backend_data=mol_data.backend_data.copy()
            )
            updated_mol_data.backend_data["rdkit_mol"] = mol_final
            updated_mol_data.backend_data["num_conformers"] = len(conf_ids)
            updated_mol_data.backend_data["conformer_ids"] = conf_ids
            
            logger.debug(f"Generated {len(conf_ids)} conformers for molecule {mol_data.mol_id}")
            return updated_mol_data
            
        except Exception as e:
            raise ChemiAccelError(f"Conformer generation failed for molecule {mol_data.mol_id}: {e}")
    
    def generate_conformers_batch(self, molecules: List[MolecularData]) -> List[MolecularData]:
        """Generate conformers for batch of molecules.
        
        Args:
            molecules: List of MolecularData objects
            
        Returns:
            List of MolecularData with conformers
        """
        results = []
        
        for mol_data in molecules:
            try:
                mol_with_conformers = self.generate_conformers(mol_data)
                results.append(mol_with_conformers)
            except Exception as e:
                logger.warning(f"Failed to generate conformers for {mol_data.mol_id}: {e}")
                # Return original molecule without conformers
                results.append(mol_data)
        
        logger.info(f"Generated conformers for {len(results)} molecules")
        return results


class Shape3DDescriptors:
    """Calculate 3D shape-based molecular descriptors."""
    
    def __init__(self):
        """Initialize 3D shape descriptor calculator."""
        check_rdkit_availability()
        
        if not DESCRIPTORS_3D_AVAILABLE:
            logger.warning("RDKit 3D descriptors module not fully available")
        
        # Define available 3D descriptors
        self.descriptor_functions = {}
        
        if DESCRIPTORS_3D_AVAILABLE:
            self.descriptor_functions.update({
                "radius_of_gyration": RadiusOfGyration,
                "asphericity": Asphericity,
                "eccentricity": Eccentricity,
                "inertial_shape_factor": InertialShapeFactor,
                "npr1": NPR1,
                "npr2": NPR2,
                "pmi1": PMI1,
                "pmi2": PMI2,
                "pmi3": PMI3,
                "spherocity_index": SpherocityIndex
            })
        
        # Add custom descriptors
        self.descriptor_functions.update({
            "molecular_volume": self._calculate_molecular_volume,
            "molecular_surface_area": self._calculate_molecular_surface_area,
            "geometric_diameter": self._calculate_geometric_diameter,
            "shape_anisotropy": self._calculate_shape_anisotropy,
            "compactness": self._calculate_compactness
        })
        
        logger.debug(f"Shape3DDescriptors initialized with {len(self.descriptor_functions)} descriptors")
    
    def _calculate_molecular_volume(self, mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate molecular volume using van der Waals radii."""
        try:
            # Use RDKit's built-in volume calculation if available
            if hasattr(AllChem, 'ComputeMolVolume'):
                return AllChem.ComputeMolVolume(mol, confId=conf_id)
            
            # Fallback: estimate from atomic volumes
            conf = mol.GetConformer(conf_id)
            total_volume = 0.0
            
            # Van der Waals radii (Angstrom)
            vdw_radii = {
                1: 1.20,   # H
                6: 1.70,   # C
                7: 1.55,   # N
                8: 1.52,   # O
                9: 1.47,   # F
                15: 1.80,  # P
                16: 1.80,  # S
                17: 1.75,  # Cl
                35: 1.85,  # Br
                53: 1.98   # I
            }
            
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                radius = vdw_radii.get(atomic_num, 2.0)  # Default radius
                volume = (4.0 / 3.0) * np.pi * (radius ** 3)
                total_volume += volume
            
            return total_volume
            
        except Exception as e:
            logger.warning(f"Failed to calculate molecular volume: {e}")
            return 0.0
    
    def _calculate_molecular_surface_area(self, mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate molecular surface area."""
        try:
            # Always use fallback method for now due to RDKit API issues
            # Fallback: estimate from atomic surface areas
            conf = mol.GetConformer(conf_id)
            total_area = 0.0
            
            # Van der Waals radii for surface area calculation
            vdw_radii = {
                1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
                15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
            }
            
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                radius = vdw_radii.get(atomic_num, 2.0)
                area = 4.0 * np.pi * (radius ** 2)
                total_area += area
            
            # Apply rough correction for overlapping spheres
            return total_area * 0.7  # Empirical correction factor
            
        except Exception as e:
            logger.warning(f"Failed to calculate molecular surface area: {e}")
            return 0.0
    
    def _calculate_geometric_diameter(self, mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate geometric diameter (maximum distance between atoms)."""
        try:
            conf = mol.GetConformer(conf_id)
            max_distance = 0.0
            
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            
            positions = np.array(positions)
            
            # Calculate all pairwise distances
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    max_distance = max(max_distance, distance)
            
            return max_distance
            
        except Exception as e:
            logger.warning(f"Failed to calculate geometric diameter: {e}")
            return 0.0
    
    def _calculate_shape_anisotropy(self, mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate shape anisotropy based on principal moments of inertia."""
        try:
            if DESCRIPTORS_3D_AVAILABLE:
                pmi1_val = PMI1(mol, confId=conf_id)
                pmi2_val = PMI2(mol, confId=conf_id)
                pmi3_val = PMI3(mol, confId=conf_id)
                
                if pmi1_val > 0 and pmi2_val > 0 and pmi3_val > 0:
                    # Calculate anisotropy as relative difference in PMI values
                    pmi_values = sorted([pmi1_val, pmi2_val, pmi3_val])
                    anisotropy = (pmi_values[2] - pmi_values[0]) / pmi_values[2]
                    return anisotropy
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate shape anisotropy: {e}")
            return 0.0
    
    def _calculate_compactness(self, mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate molecular compactness (volume/surface_area ratio)."""
        try:
            volume = self._calculate_molecular_volume(mol, conf_id)
            surface_area = self._calculate_molecular_surface_area(mol, conf_id)
            
            if surface_area > 0:
                return volume / surface_area
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate compactness: {e}")
            return 0.0
    
    def calculate_shape_descriptors(self, 
                                  mol_data: MolecularData,
                                  descriptor_names: Optional[List[str]] = None,
                                  conf_id: int = -1) -> Dict[str, float]:
        """Calculate 3D shape descriptors for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object and conformers
            descriptor_names: List of descriptor names to calculate (None for all)
            conf_id: Conformer ID to use (-1 for first conformer)
            
        Returns:
            Dictionary of descriptor values
            
        Raises:
            ChemiAccelError: If calculation fails
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for molecule {mol_data.mol_id}")
        
        if rdkit_mol.GetNumConformers() == 0:
            raise ChemiAccelError(f"No conformers found for molecule {mol_data.mol_id}")
        
        if descriptor_names is None:
            descriptor_names = list(self.descriptor_functions.keys())
        
        descriptors = {}
        
        for desc_name in descriptor_names:
            if desc_name not in self.descriptor_functions:
                logger.warning(f"Unknown 3D descriptor: {desc_name}")
                continue
            
            try:
                desc_func = self.descriptor_functions[desc_name]
                
                # Check if it's a custom descriptor function (starts with _calculate_)
                if desc_name in ["molecular_volume", "molecular_surface_area", "geometric_diameter", 
                                "shape_anisotropy", "compactness"]:
                    # Custom descriptor function
                    value = desc_func(rdkit_mol, conf_id)
                else:
                    # RDKit descriptor function
                    value = desc_func(rdkit_mol, confId=conf_id)
                
                descriptors[desc_name] = float(value) if value is not None else 0.0
                
            except Exception as e:
                logger.warning(f"Failed to calculate {desc_name} for molecule {mol_data.mol_id}: {e}")
                descriptors[desc_name] = 0.0
        
        return descriptors
    
    def calculate_shape_descriptors_batch(self, 
                                        molecules: List[MolecularData],
                                        descriptor_names: Optional[List[str]] = None,
                                        use_all_conformers: bool = False) -> pd.DataFrame:
        """Calculate 3D shape descriptors for batch of molecules.
        
        Args:
            molecules: List of MolecularData objects with conformers
            descriptor_names: List of descriptor names to calculate
            use_all_conformers: Whether to calculate for all conformers or just first
            
        Returns:
            DataFrame with calculated descriptors
        """
        results = []
        
        for mol_data in molecules:
            try:
                rdkit_mol = mol_data.backend_data.get("rdkit_mol")
                if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
                    logger.warning(f"No conformers for molecule {mol_data.mol_id}")
                    continue
                
                if use_all_conformers:
                    # Calculate for all conformers
                    for conf_id in range(rdkit_mol.GetNumConformers()):
                        descriptors = self.calculate_shape_descriptors(mol_data, descriptor_names, conf_id)
                        descriptors["mol_id"] = mol_data.mol_id
                        descriptors["smiles"] = mol_data.smiles
                        descriptors["conformer_id"] = conf_id
                        results.append(descriptors)
                else:
                    # Calculate for first conformer only
                    descriptors = self.calculate_shape_descriptors(mol_data, descriptor_names, 0)
                    descriptors["mol_id"] = mol_data.mol_id
                    descriptors["smiles"] = mol_data.smiles
                    results.append(descriptors)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate 3D descriptors for molecule {mol_data.mol_id}: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            # Reorder columns
            id_cols = ["mol_id", "smiles"]
            if "conformer_id" in df.columns:
                id_cols.append("conformer_id")
            desc_cols = [col for col in df.columns if col not in id_cols]
            df = df[id_cols + desc_cols]
        else:
            df = pd.DataFrame()
        
        logger.info(f"Calculated 3D shape descriptors for {len(results)} conformers")
        return df


class ElectrostaticDescriptors:
    """Calculate electrostatic molecular descriptors."""
    
    def __init__(self, charge_model: str = "gasteiger"):
        """Initialize electrostatic descriptor calculator.
        
        Args:
            charge_model: Charge model to use ("gasteiger", "mmff")
        """
        check_rdkit_availability()
        
        self.charge_model = charge_model.lower()
        if self.charge_model not in ["gasteiger", "mmff"]:
            raise ValueError("charge_model must be 'gasteiger' or 'mmff'")
        
        logger.debug(f"ElectrostaticDescriptors initialized with {charge_model} charges")
    
    def _calculate_partial_charges(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate partial charges for atoms in molecule."""
        try:
            if self.charge_model == "gasteiger":
                rdPartialCharges.ComputeGasteigerCharges(mol)
                charges = []
                for atom in mol.GetAtoms():
                    charge = atom.GetDoubleProp('_GasteigerCharge')
                    if np.isnan(charge) or np.isinf(charge):
                        charge = 0.0
                    charges.append(charge)
                return np.array(charges)
            
            elif self.charge_model == "mmff":
                # MMFF charges require force field setup
                try:
                    from rdkit.Chem import ChemicalForceFields
                    mmff_props = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
                    if mmff_props is not None:
                        charges = []
                        for i in range(mol.GetNumAtoms()):
                            charge = mmff_props.GetMMFFPartialCharge(i)
                            charges.append(charge)
                        return np.array(charges)
                except:
                    pass
                
                # Fallback to Gasteiger
                logger.warning("MMFF charges failed, falling back to Gasteiger")
                return self._calculate_partial_charges_gasteiger(mol)
            
        except Exception as e:
            logger.warning(f"Failed to calculate partial charges: {e}")
            return np.zeros(mol.GetNumAtoms())
    
    def _calculate_partial_charges_gasteiger(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate Gasteiger partial charges."""
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp('_GasteigerCharge')
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
        return np.array(charges)
    
    def calculate_electrostatic_descriptors(self, 
                                          mol_data: MolecularData,
                                          conf_id: int = -1) -> Dict[str, float]:
        """Calculate electrostatic descriptors for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object and conformers
            conf_id: Conformer ID to use (-1 for first conformer)
            
        Returns:
            Dictionary of electrostatic descriptors
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for molecule {mol_data.mol_id}")
        
        if rdkit_mol.GetNumConformers() == 0:
            raise ChemiAccelError(f"No conformers found for molecule {mol_data.mol_id}")
        
        try:
            # Calculate partial charges
            charges = self._calculate_partial_charges(rdkit_mol)
            
            # Get atomic positions
            conf = rdkit_mol.GetConformer(conf_id)
            positions = []
            for i in range(rdkit_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            positions = np.array(positions)
            
            # Calculate electrostatic descriptors
            descriptors = {}
            
            # Total charge
            descriptors["total_charge"] = float(np.sum(charges))
            
            # Absolute total charge
            descriptors["abs_total_charge"] = float(np.sum(np.abs(charges)))
            
            # Charge distribution statistics
            descriptors["charge_mean"] = float(np.mean(charges))
            descriptors["charge_std"] = float(np.std(charges))
            descriptors["charge_min"] = float(np.min(charges))
            descriptors["charge_max"] = float(np.max(charges))
            descriptors["charge_range"] = descriptors["charge_max"] - descriptors["charge_min"]
            
            # Positive and negative charge sums
            positive_charges = charges[charges > 0]
            negative_charges = charges[charges < 0]
            
            descriptors["positive_charge_sum"] = float(np.sum(positive_charges)) if len(positive_charges) > 0 else 0.0
            descriptors["negative_charge_sum"] = float(np.sum(negative_charges)) if len(negative_charges) > 0 else 0.0
            descriptors["charge_polarization"] = descriptors["positive_charge_sum"] - descriptors["negative_charge_sum"]
            
            # Dipole moment calculation
            dipole_vector = np.sum(charges[:, np.newaxis] * positions, axis=0)
            descriptors["dipole_moment"] = float(np.linalg.norm(dipole_vector))
            
            # Electrostatic potential statistics (simplified)
            if len(charges) > 1:
                # Calculate pairwise electrostatic interactions
                electrostatic_energies = []
                for i in range(len(charges)):
                    for j in range(i + 1, len(charges)):
                        distance = np.linalg.norm(positions[i] - positions[j])
                        if distance > 0:
                            # Coulomb interaction (simplified, no units conversion)
                            energy = charges[i] * charges[j] / distance
                            electrostatic_energies.append(energy)
                
                if electrostatic_energies:
                    descriptors["electrostatic_energy_sum"] = float(np.sum(electrostatic_energies))
                    descriptors["electrostatic_energy_mean"] = float(np.mean(electrostatic_energies))
                    descriptors["electrostatic_energy_std"] = float(np.std(electrostatic_energies))
                else:
                    descriptors["electrostatic_energy_sum"] = 0.0
                    descriptors["electrostatic_energy_mean"] = 0.0
                    descriptors["electrostatic_energy_std"] = 0.0
            else:
                descriptors["electrostatic_energy_sum"] = 0.0
                descriptors["electrostatic_energy_mean"] = 0.0
                descriptors["electrostatic_energy_std"] = 0.0
            
            return descriptors
            
        except Exception as e:
            logger.warning(f"Failed to calculate electrostatic descriptors for {mol_data.mol_id}: {e}")
            # Return default values
            return {
                "total_charge": 0.0,
                "abs_total_charge": 0.0,
                "charge_mean": 0.0,
                "charge_std": 0.0,
                "charge_min": 0.0,
                "charge_max": 0.0,
                "charge_range": 0.0,
                "positive_charge_sum": 0.0,
                "negative_charge_sum": 0.0,
                "charge_polarization": 0.0,
                "dipole_moment": 0.0,
                "electrostatic_energy_sum": 0.0,
                "electrostatic_energy_mean": 0.0,
                "electrostatic_energy_std": 0.0
            }


class Geometrical3DCalculator:
    """Main calculator for geometrical and 3D molecular descriptors."""
    
    def __init__(self, 
                 generate_conformers: bool = True,
                 num_conformers: int = 10,
                 charge_model: str = "gasteiger"):
        """Initialize 3D descriptor calculator.
        
        Args:
            generate_conformers: Whether to generate conformers if not present
            num_conformers: Number of conformers to generate
            charge_model: Charge model for electrostatic descriptors
        """
        check_rdkit_availability()
        
        self.generate_conformers = generate_conformers
        self.conformer_generator = ConformerGenerator(num_conformers=num_conformers) if generate_conformers else None
        self.shape_calculator = Shape3DDescriptors()
        self.electrostatic_calculator = ElectrostaticDescriptors(charge_model=charge_model)
        
        logger.debug(f"Geometrical3DCalculator initialized: conformers={generate_conformers}, "
                    f"charge_model={charge_model}")
    
    def calculate_3d_descriptors(self, 
                               mol_data: MolecularData,
                               include_shape: bool = True,
                               include_electrostatic: bool = True,
                               shape_descriptors: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate comprehensive 3D descriptors for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            include_shape: Whether to include shape descriptors
            include_electrostatic: Whether to include electrostatic descriptors
            shape_descriptors: Specific shape descriptors to calculate
            
        Returns:
            Dictionary of 3D descriptors
        """
        # Ensure molecule has conformers
        if self.generate_conformers:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
                mol_data = self.conformer_generator.generate_conformers(mol_data)
        
        descriptors = {}
        
        # Calculate shape descriptors
        if include_shape:
            try:
                shape_desc = self.shape_calculator.calculate_shape_descriptors(
                    mol_data, shape_descriptors
                )
                descriptors.update(shape_desc)
            except Exception as e:
                logger.warning(f"Failed to calculate shape descriptors: {e}")
        
        # Calculate electrostatic descriptors
        if include_electrostatic:
            try:
                electrostatic_desc = self.electrostatic_calculator.calculate_electrostatic_descriptors(mol_data)
                descriptors.update(electrostatic_desc)
            except Exception as e:
                logger.warning(f"Failed to calculate electrostatic descriptors: {e}")
        
        return descriptors
    
    def calculate_3d_descriptors_batch(self, 
                                     molecules: List[MolecularData],
                                     include_shape: bool = True,
                                     include_electrostatic: bool = True,
                                     shape_descriptors: Optional[List[str]] = None,
                                     n_jobs: int = 1) -> pd.DataFrame:
        """Calculate 3D descriptors for batch of molecules.
        
        Args:
            molecules: List of MolecularData objects
            include_shape: Whether to include shape descriptors
            include_electrostatic: Whether to include electrostatic descriptors
            shape_descriptors: Specific shape descriptors to calculate
            n_jobs: Number of parallel jobs
            
        Returns:
            DataFrame with calculated descriptors
        """
        if not molecules:
            return pd.DataFrame()
        
        # Generate conformers for all molecules if needed
        if self.generate_conformers:
            molecules = self.conformer_generator.generate_conformers_batch(molecules)
        
        def calculate_single_molecule(mol_data):
            try:
                descriptors = self.calculate_3d_descriptors(
                    mol_data, include_shape, include_electrostatic, shape_descriptors
                )
                descriptors["mol_id"] = mol_data.mol_id
                descriptors["smiles"] = mol_data.smiles
                return descriptors
            except Exception as e:
                logger.warning(f"Failed to calculate 3D descriptors for {mol_data.mol_id}: {e}")
                return {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    "error": str(e)
                }
        
        # Calculate descriptors
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(calculate_single_molecule, molecules))
        else:
            results = [calculate_single_molecule(mol) for mol in molecules]
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            # Reorder columns
            id_cols = ["mol_id", "smiles"]
            desc_cols = [col for col in df.columns if col not in id_cols and col != "error"]
            error_cols = ["error"] if "error" in df.columns else []
            df = df[id_cols + desc_cols + error_cols]
        else:
            df = pd.DataFrame()
        
        logger.info(f"Calculated 3D descriptors for {len(results)} molecules")
        return df
    
    def get_available_descriptors(self) -> Dict[str, List[str]]:
        """Get list of available 3D descriptors.
        
        Returns:
            Dictionary with descriptor categories and names
        """
        return {
            "shape_descriptors": list(self.shape_calculator.descriptor_functions.keys()),
            "electrostatic_descriptors": [
                "total_charge", "abs_total_charge", "charge_mean", "charge_std",
                "charge_min", "charge_max", "charge_range", "positive_charge_sum",
                "negative_charge_sum", "charge_polarization", "dipole_moment",
                "electrostatic_energy_sum", "electrostatic_energy_mean", "electrostatic_energy_std"
            ]
        }


# Convenience functions
def generate_conformers(molecules: List[MolecularData], 
                       num_conformers: int = 10,
                       **kwargs) -> List[MolecularData]:
    """Generate conformers for molecules (convenience function).
    
    Args:
        molecules: List of MolecularData objects
        num_conformers: Number of conformers to generate
        **kwargs: Additional arguments for ConformerGenerator
        
    Returns:
        List of MolecularData with conformers
    """
    generator = ConformerGenerator(num_conformers=num_conformers, **kwargs)
    return generator.generate_conformers_batch(molecules)


def calculate_3d_descriptors(molecules: List[MolecularData],
                           include_shape: bool = True,
                           include_electrostatic: bool = True,
                           **kwargs) -> pd.DataFrame:
    """Calculate 3D descriptors for molecules (convenience function).
    
    Args:
        molecules: List of MolecularData objects
        include_shape: Whether to include shape descriptors
        include_electrostatic: Whether to include electrostatic descriptors
        **kwargs: Additional arguments for Geometrical3DCalculator
        
    Returns:
        DataFrame with calculated descriptors
    """
    calculator = Geometrical3DCalculator(**kwargs)
    return calculator.calculate_3d_descriptors_batch(
        molecules, include_shape, include_electrostatic
    )


def calculate_shape_descriptors(molecules: List[MolecularData],
                              descriptor_names: Optional[List[str]] = None,
                              **kwargs) -> pd.DataFrame:
    """Calculate shape descriptors for molecules (convenience function).
    
    Args:
        molecules: List of MolecularData objects with conformers
        descriptor_names: Specific descriptors to calculate
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with shape descriptors
    """
    calculator = Shape3DDescriptors()
    return calculator.calculate_shape_descriptors_batch(molecules, descriptor_names, **kwargs)


def calculate_electrostatic_descriptors(molecules: List[MolecularData],
                                      charge_model: str = "gasteiger") -> pd.DataFrame:
    """Calculate electrostatic descriptors for molecules (convenience function).
    
    Args:
        molecules: List of MolecularData objects with conformers
        charge_model: Charge model to use
        
    Returns:
        DataFrame with electrostatic descriptors
    """
    calculator = ElectrostaticDescriptors(charge_model=charge_model)
    results = []
    
    for mol_data in molecules:
        try:
            descriptors = calculator.calculate_electrostatic_descriptors(mol_data)
            descriptors["mol_id"] = mol_data.mol_id
            descriptors["smiles"] = mol_data.smiles
            results.append(descriptors)
        except Exception as e:
            logger.warning(f"Failed to calculate electrostatic descriptors for {mol_data.mol_id}: {e}")
    
    return pd.DataFrame(results) if results else pd.DataFrame()