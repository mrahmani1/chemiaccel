"""
Integrated 3D descriptor calculation system for ChemiAccel.

This module provides a complete 3D descriptor calculation pipeline that combines
conformer generation with GPU-accelerated descriptor calculations including
radius of gyration, principal moments of inertia, asphericity, and shape-based
descriptors.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from ..utils.geometry import (
    calculate_asphericity,
    calculate_eccentricity,
    calculate_principal_moments,
    calculate_radius_of_gyration,
)
from ..utils.logging import get_logger
from .conformer_generation import GPUConformerGenerator, validate_conformer_quality
from .geometrical_3d import (
    ConformerGenerator,
    ElectrostaticDescriptors,
    Shape3DDescriptors,
)

if CUDA_AVAILABLE:
    from ..backends.cuda.descriptors_3d_kernels import GPU3DDescriptorCalculator

logger = get_logger()


class Integrated3DDescriptorCalculator:
    """Integrated 3D descriptor calculator with conformer generation."""

    def __init__(
        self,
        generate_conformers: bool = True,
        num_conformers: int = 10,
        use_gpu: bool = True,
        energy_minimize: bool = True,
        force_field: str = "UFF",
        prune_rms_thresh: float = 0.5,
    ):
        """Initialize integrated 3D descriptor calculator.

        Args:
            generate_conformers: Whether to generate conformers if not present
            num_conformers: Number of conformers to generate
            use_gpu: Whether to use GPU acceleration
            energy_minimize: Whether to perform energy minimization
            force_field: Force field for energy minimization
            prune_rms_thresh: RMS threshold for conformer pruning
        """
        self.generate_conformers = generate_conformers
        self.num_conformers = num_conformers
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.energy_minimize = energy_minimize
        self.force_field = force_field
        self.prune_rms_thresh = prune_rms_thresh

        # Initialize conformer generator
        if self.generate_conformers:
            if self.use_gpu:
                self.conformer_generator = GPUConformerGenerator(
                    num_conformers=num_conformers,
                    use_gpu=True,
                    energy_minimize=energy_minimize,
                    force_field=force_field,
                    prune_rms_thresh=prune_rms_thresh,
                )
            else:
                self.conformer_generator = ConformerGenerator(
                    num_conformers=num_conformers,
                    optimize_conformers=energy_minimize,
                    force_field=force_field,
                )

        # Initialize descriptor calculators
        self.shape_calculator = Shape3DDescriptors()
        self.electrostatic_calculator = ElectrostaticDescriptors()

        # Initialize GPU calculator if available
        if self.use_gpu and CUDA_AVAILABLE:
            try:
                self.gpu_calculator = GPU3DDescriptorCalculator()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU calculator: {e}")
                self.gpu_calculator = None
        else:
            self.gpu_calculator = None

        logger.info(
            f"Integrated3DDescriptorCalculator initialized: GPU={self.use_gpu}, "
            f"conformers={generate_conformers}"
        )

    def calculate_3d_descriptors(
        self,
        mol_data: MolecularData,
        descriptor_names: Optional[List[str]] = None,
        use_all_conformers: bool = False,
    ) -> Dict[str, Any]:
        """Calculate comprehensive 3D descriptors for a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            descriptor_names: Specific descriptors to calculate (None for all)
            use_all_conformers: Whether to calculate for all conformers

        Returns:
            Dictionary of 3D descriptors
        """
        start_time = time.time()

        try:
            # Step 1: Ensure molecule has conformers
            if self.generate_conformers:
                mol_data = self._ensure_conformers(mol_data)

            # Step 2: Validate conformers
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
                logger.warning(f"No conformers available for {mol_data.mol_id}")
                return self._get_default_descriptors()

            # Step 3: Calculate descriptors
            if self.use_gpu and self.gpu_calculator is not None:
                descriptors = self._calculate_gpu_descriptors(
                    mol_data, descriptor_names, use_all_conformers
                )
            else:
                descriptors = self._calculate_cpu_descriptors(
                    mol_data, descriptor_names, use_all_conformers
                )

            # Step 4: Add metadata
            descriptors["calculation_time"] = time.time() - start_time
            descriptors["num_conformers_used"] = rdkit_mol.GetNumConformers()
            descriptors["calculation_method"] = "gpu" if self.use_gpu else "cpu"

            logger.debug(
                f"Calculated 3D descriptors for {mol_data.mol_id} in {descriptors['calculation_time']:.3f}s"
            )
            return descriptors

        except Exception as e:
            logger.error(f"3D descriptor calculation failed for {mol_data.mol_id}: {e}")
            return self._get_default_descriptors()

    def _ensure_conformers(self, mol_data: MolecularData) -> MolecularData:
        """Ensure molecule has conformers, generating if necessary."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for {mol_data.mol_id}")

        if rdkit_mol.GetNumConformers() == 0:
            logger.debug(f"Generating conformers for {mol_data.mol_id}")
            mol_data = self.conformer_generator.generate_conformers(mol_data)

        return mol_data

    def _calculate_gpu_descriptors(
        self,
        mol_data: MolecularData,
        descriptor_names: Optional[List[str]],
        use_all_conformers: bool,
    ) -> Dict[str, Any]:
        """Calculate descriptors using GPU acceleration."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        descriptors = {}

        try:
            # Extract molecular data for GPU calculation
            molecular_data = self._extract_molecular_data_for_gpu(
                rdkit_mol, use_all_conformers
            )

            if molecular_data:
                # Calculate using GPU kernels
                gpu_results = self.gpu_calculator.calculate_3d_descriptors(
                    molecular_data
                )

                # Convert GPU results to CPU and aggregate
                for desc_name, gpu_values in gpu_results.items():
                    cpu_values = cp.asnumpy(gpu_values)
                    if use_all_conformers:
                        descriptors[f"{desc_name}_mean"] = float(np.mean(cpu_values))
                        descriptors[f"{desc_name}_std"] = float(np.std(cpu_values))
                        descriptors[f"{desc_name}_min"] = float(np.min(cpu_values))
                        descriptors[f"{desc_name}_max"] = float(np.max(cpu_values))
                    else:
                        descriptors[desc_name] = (
                            float(cpu_values[0]) if len(cpu_values) > 0 else 0.0
                        )

            # Add additional descriptors not available in GPU kernels
            descriptors.update(
                self._calculate_additional_descriptors(mol_data, use_all_conformers)
            )

        except Exception as e:
            logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
            descriptors = self._calculate_cpu_descriptors(
                mol_data, descriptor_names, use_all_conformers
            )

        return descriptors

    def _calculate_cpu_descriptors(
        self,
        mol_data: MolecularData,
        descriptor_names: Optional[List[str]],
        use_all_conformers: bool,
    ) -> Dict[str, Any]:
        """Calculate descriptors using CPU methods."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        descriptors = {}

        # Calculate shape descriptors
        shape_descriptors = self.shape_calculator.calculate_shape_descriptors_batch(
            [mol_data], descriptor_names, use_all_conformers
        )

        if not shape_descriptors.empty:
            # Aggregate results
            for col in shape_descriptors.columns:
                if col not in ["mol_id", "smiles", "conformer_id"]:
                    values = shape_descriptors[col].values
                    if use_all_conformers and len(values) > 1:
                        descriptors[f"{col}_mean"] = float(np.mean(values))
                        descriptors[f"{col}_std"] = float(np.std(values))
                        descriptors[f"{col}_min"] = float(np.min(values))
                        descriptors[f"{col}_max"] = float(np.max(values))
                    else:
                        descriptors[col] = float(values[0]) if len(values) > 0 else 0.0

        # Calculate electrostatic descriptors (first conformer only)
        try:
            electrostatic_descriptors = (
                self.electrostatic_calculator.calculate_electrostatic_descriptors(
                    mol_data, conf_id=0
                )
            )
            descriptors.update(electrostatic_descriptors)
        except Exception as e:
            logger.warning(f"Electrostatic descriptor calculation failed: {e}")

        # Calculate additional geometry descriptors
        descriptors.update(
            self._calculate_geometry_descriptors(mol_data, use_all_conformers)
        )

        return descriptors

    def _calculate_geometry_descriptors(
        self, mol_data: MolecularData, use_all_conformers: bool
    ) -> Dict[str, Any]:
        """Calculate geometry descriptors using utility functions."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        descriptors = {}

        if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
            return descriptors

        # Get atomic masses
        atomic_masses = []
        for atom in rdkit_mol.GetAtoms():
            atomic_masses.append(atom.GetMass())
        masses = np.array(atomic_masses)

        # Calculate for each conformer
        conformer_results = []

        num_conformers = rdkit_mol.GetNumConformers() if use_all_conformers else 1

        for conf_id in range(num_conformers):
            try:
                conf = rdkit_mol.GetConformer(conf_id)
                coords = np.array(
                    [conf.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
                )

                # Calculate individual descriptors
                conf_descriptors = {}
                conf_descriptors["radius_of_gyration"] = calculate_radius_of_gyration(
                    coords, masses
                )
                conf_descriptors["asphericity"] = calculate_asphericity(coords, masses)
                conf_descriptors["eccentricity"] = calculate_eccentricity(
                    coords, masses
                )

                # Principal moments
                eigenvalues, _ = calculate_principal_moments(coords, masses)
                conf_descriptors["pmi1"] = float(eigenvalues[0])
                conf_descriptors["pmi2"] = float(eigenvalues[1])
                conf_descriptors["pmi3"] = float(eigenvalues[2])

                # PMI ratios
                if eigenvalues[2] > 0:
                    conf_descriptors["pmi_ratio_12"] = float(
                        eigenvalues[0] / eigenvalues[2]
                    )
                    conf_descriptors["pmi_ratio_13"] = float(
                        eigenvalues[1] / eigenvalues[2]
                    )
                else:
                    conf_descriptors["pmi_ratio_12"] = 0.0
                    conf_descriptors["pmi_ratio_13"] = 0.0

                conformer_results.append(conf_descriptors)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate geometry descriptors for conformer {conf_id}: {e}"
                )
                continue

        # Aggregate results
        if conformer_results:
            if use_all_conformers and len(conformer_results) > 1:
                # Calculate statistics across conformers
                for desc_name in conformer_results[0].keys():
                    values = [result[desc_name] for result in conformer_results]
                    descriptors[f"{desc_name}_mean"] = float(np.mean(values))
                    descriptors[f"{desc_name}_std"] = float(np.std(values))
                    descriptors[f"{desc_name}_min"] = float(np.min(values))
                    descriptors[f"{desc_name}_max"] = float(np.max(values))
            else:
                # Use first conformer results
                descriptors.update(conformer_results[0])

        return descriptors

    def _calculate_additional_descriptors(
        self, mol_data: MolecularData, use_all_conformers: bool
    ) -> Dict[str, Any]:
        """Calculate additional descriptors not available in GPU kernels."""
        descriptors = {}

        # Add conformer quality metrics
        quality_metrics = validate_conformer_quality(mol_data)
        if "error" not in quality_metrics:
            descriptors.update(
                {f"conformer_{k}": v for k, v in quality_metrics.items()}
            )

        return descriptors

    def _extract_molecular_data_for_gpu(
        self, rdkit_mol: Chem.Mol, use_all_conformers: bool
    ) -> Optional[Dict[str, Any]]:
        """Extract molecular data for GPU calculation."""
        if not CUDA_AVAILABLE:
            return None

        try:
            num_conformers = rdkit_mol.GetNumConformers() if use_all_conformers else 1

            # Extract coordinates and atomic data
            all_coords = []
            atomic_numbers = []

            for conf_id in range(num_conformers):
                conf = rdkit_mol.GetConformer(conf_id)
                coords = []

                for atom_idx in range(rdkit_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    coords.extend([pos.x, pos.y, pos.z])

                    if conf_id == 0:  # Only collect atomic numbers once
                        atomic_numbers.append(
                            rdkit_mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
                        )

                all_coords.extend(coords)

            # Create GPU arrays
            coordinates = cp.array(all_coords, dtype=cp.float32).reshape(-1, 3)
            atomic_numbers_gpu = cp.array(
                atomic_numbers * num_conformers, dtype=cp.int32
            )

            # Create molecule boundaries
            atoms_per_molecule = rdkit_mol.GetNumAtoms()
            molecule_boundaries = cp.array(
                [(i + 1) * atoms_per_molecule for i in range(num_conformers)],
                dtype=cp.int32,
            )

            return {
                "coordinates": coordinates,
                "atomic_numbers": atomic_numbers_gpu,
                "molecule_boundaries": molecule_boundaries,
            }

        except Exception as e:
            logger.warning(f"Failed to extract molecular data for GPU: {e}")
            return None

    def _get_default_descriptors(self) -> Dict[str, Any]:
        """Get default descriptor values for failed calculations."""
        return {
            "radius_of_gyration": 0.0,
            "asphericity": 0.0,
            "eccentricity": 0.0,
            "pmi1": 0.0,
            "pmi2": 0.0,
            "pmi3": 0.0,
            "pmi_ratio_12": 0.0,
            "pmi_ratio_13": 0.0,
            "molecular_volume": 0.0,
            "geometric_diameter": 0.0,
            "calculation_time": 0.0,
            "num_conformers_used": 0,
            "calculation_method": "failed",
        }

    def calculate_3d_descriptors_batch(
        self,
        molecules: List[MolecularData],
        descriptor_names: Optional[List[str]] = None,
        use_all_conformers: bool = False,
    ) -> pd.DataFrame:
        """Calculate 3D descriptors for a batch of molecules.

        Args:
            molecules: List of MolecularData objects
            descriptor_names: Specific descriptors to calculate
            use_all_conformers: Whether to calculate for all conformers

        Returns:
            DataFrame with calculated 3D descriptors
        """
        results = []

        for mol_data in molecules:
            try:
                descriptors = self.calculate_3d_descriptors(
                    mol_data, descriptor_names, use_all_conformers
                )
                descriptors["mol_id"] = mol_data.mol_id
                descriptors["smiles"] = mol_data.smiles
                results.append(descriptors)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate 3D descriptors for {mol_data.mol_id}: {e}"
                )
                default_descriptors = self._get_default_descriptors()
                default_descriptors["mol_id"] = mol_data.mol_id
                default_descriptors["smiles"] = mol_data.smiles
                results.append(default_descriptors)

        if results:
            df = pd.DataFrame(results)
            # Reorder columns
            id_cols = ["mol_id", "smiles"]
            desc_cols = [col for col in df.columns if col not in id_cols]
            df = df[id_cols + desc_cols]
        else:
            df = pd.DataFrame()

        logger.info(f"Calculated 3D descriptors for {len(results)} molecules")
        return df

    def get_available_descriptors(self) -> List[str]:
        """Get list of available 3D descriptors."""
        base_descriptors = [
            "radius_of_gyration",
            "asphericity",
            "eccentricity",
            "pmi1",
            "pmi2",
            "pmi3",
            "pmi_ratio_12",
            "pmi_ratio_13",
            "molecular_volume",
            "geometric_diameter",
            "inertial_shape_factor",
            "npr1",
            "npr2",
            "spherocity_index",
            "compactness",
        ]

        # Add electrostatic descriptors
        electrostatic_descriptors = [
            "total_charge",
            "abs_total_charge",
            "charge_mean",
            "charge_std",
            "charge_range",
            "positive_charge_sum",
            "negative_charge_sum",
            "charge_polarization",
            "dipole_moment",
            "electrostatic_energy_sum",
            "electrostatic_energy_mean",
        ]

        # Add conformer quality metrics
        conformer_descriptors = [
            "conformer_num_conformers",
            "conformer_rmsd_mean",
            "conformer_rmsd_std",
            "conformer_diversity_score",
        ]

        return base_descriptors + electrostatic_descriptors + conformer_descriptors
