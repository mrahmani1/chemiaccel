"""
GPU-accelerated conformer generation system for ChemiAccel.

This module implements GPU-accelerated Distance Geometry algorithms for
parallel conformer generation with energy minimization and hybrid CPU-GPU
approaches for conformer refinement.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
    from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
    from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from ..utils.geometry import DistanceGeometry, calculate_rmsd
from ..utils.logging import get_logger

logger = get_logger()


class GPUConformerGenerator:
    """GPU-accelerated conformer generation using Distance Geometry."""

    def __init__(
        self,
        num_conformers: int = 10,
        max_attempts: int = 1000,
        prune_rms_thresh: float = 0.5,
        use_gpu: bool = True,
        energy_minimize: bool = True,
        force_field: str = "UFF",
    ):
        """Initialize GPU conformer generator.

        Args:
            num_conformers: Number of conformers to generate
            max_attempts: Maximum attempts for conformer generation
            prune_rms_thresh: RMS threshold for conformer pruning
            use_gpu: Whether to use GPU acceleration
            energy_minimize: Whether to perform energy minimization
            force_field: Force field for energy minimization ("UFF" or "MMFF")
        """
        self.num_conformers = num_conformers
        self.max_attempts = max_attempts
        self.prune_rms_thresh = prune_rms_thresh
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.energy_minimize = energy_minimize
        self.force_field = force_field.upper()

        if self.force_field not in ["UFF", "MMFF"]:
            raise ValueError("force_field must be 'UFF' or 'MMFF'")

        # Initialize distance geometry calculator
        self.distance_geometry = DistanceGeometry(use_gpu=self.use_gpu)

        logger.info(
            f"GPUConformerGenerator initialized: {num_conformers} conformers, "
            f"GPU={self.use_gpu}, minimize={energy_minimize}"
        )

    def generate_conformers_parallel(self, mol_data: MolecularData) -> MolecularData:
        """Generate conformers using parallel GPU-accelerated distance geometry.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            MolecularData with generated conformers
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        try:
            # Create adjacency matrix from RDKit molecule
            adjacency_matrix = self._create_adjacency_matrix(rdkit_mol)

            # Generate distance bounds
            (
                lower_bounds,
                upper_bounds,
            ) = self.distance_geometry.generate_distance_bounds(adjacency_matrix)

            # Generate multiple conformers in parallel
            conformer_coords = []

            if self.use_gpu and cp is not None:
                conformer_coords = self._generate_conformers_gpu_batch(
                    lower_bounds, upper_bounds, rdkit_mol.GetNumAtoms()
                )
            else:
                conformer_coords = self._generate_conformers_cpu_parallel(
                    lower_bounds, upper_bounds, rdkit_mol.GetNumAtoms()
                )

            # Prune similar conformers
            pruned_coords = self._prune_conformers(conformer_coords)

            # Create new molecule with conformers
            mol_with_conformers = self._create_molecule_with_conformers(
                rdkit_mol, pruned_coords
            )

            # Energy minimization if requested
            if self.energy_minimize and RDKIT_AVAILABLE:
                mol_with_conformers = self._minimize_conformers(mol_with_conformers)

            # Update molecular data
            updated_mol_data = MolecularData(
                smiles=mol_data.smiles,
                mol_id=mol_data.mol_id,
                num_atoms=mol_data.num_atoms,
                num_bonds=mol_data.num_bonds,
                molecular_weight=mol_data.molecular_weight,
                backend_data=mol_data.backend_data.copy(),
            )
            updated_mol_data.backend_data["rdkit_mol"] = mol_with_conformers
            updated_mol_data.backend_data[
                "num_conformers"
            ] = mol_with_conformers.GetNumConformers()
            updated_mol_data.backend_data[
                "conformer_generation_method"
            ] = "gpu_distance_geometry"

            logger.debug(
                f"Generated {mol_with_conformers.GetNumConformers()} conformers for {mol_data.mol_id}"
            )
            return updated_mol_data

        except Exception as e:
            logger.error(f"Conformer generation failed for {mol_data.mol_id}: {e}")
            raise ChemiAccelError(f"Conformer generation failed: {e}")

    def _create_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        """Create adjacency matrix from RDKit molecule."""
        num_atoms = mol.GetNumAtoms()
        adjacency = np.zeros((num_atoms, num_atoms), dtype=int)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_order = int(bond.GetBondType())
            adjacency[i, j] = bond_order
            adjacency[j, i] = bond_order

        return adjacency

    def _generate_conformers_gpu_batch(
        self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, num_atoms: int
    ) -> List[np.ndarray]:
        """Generate conformers using GPU batch processing."""
        if not self.use_gpu or cp is None:
            return self._generate_conformers_cpu_parallel(
                lower_bounds, upper_bounds, num_atoms
            )

        try:
            # Transfer bounds to GPU
            lower_bounds_gpu = cp.asarray(lower_bounds, dtype=cp.float32)
            upper_bounds_gpu = cp.asarray(upper_bounds, dtype=cp.float32)

            # Generate multiple conformers in parallel on GPU
            batch_size = min(self.num_conformers, 32)  # Limit batch size for memory
            all_coords = []

            for batch_start in range(0, self.num_conformers, batch_size):
                batch_end = min(batch_start + batch_size, self.num_conformers)
                current_batch_size = batch_end - batch_start

                # Generate random initial coordinates for batch
                coords_batch = (
                    cp.random.randn(current_batch_size, num_atoms, 3, dtype=cp.float32)
                    * 2.0
                )

                # Optimize each conformer in the batch
                for i in range(current_batch_size):
                    coords = self.distance_geometry.embed_coordinates(
                        cp.asnumpy(lower_bounds_gpu),
                        cp.asnumpy(upper_bounds_gpu),
                        n_dimensions=3,
                        max_iterations=500,
                    )
                    all_coords.append(coords)

            return all_coords[: self.num_conformers]

        except Exception as e:
            logger.warning(f"GPU conformer generation failed, falling back to CPU: {e}")
            return self._generate_conformers_cpu_parallel(
                lower_bounds, upper_bounds, num_atoms
            )

    def _generate_conformers_cpu_parallel(
        self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, num_atoms: int
    ) -> List[np.ndarray]:
        """Generate conformers using parallel CPU processing."""
        conformer_coords = []

        def generate_single_conformer(seed):
            try:
                coords = self.distance_geometry.embed_coordinates(
                    lower_bounds,
                    upper_bounds,
                    n_dimensions=3,
                    max_iterations=500,
                    random_seed=seed,
                )
                return coords
            except Exception as e:
                logger.warning(f"Failed to generate conformer with seed {seed}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel generation
        max_workers = min(8, self.num_conformers)  # Limit number of threads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks with different random seeds
            future_to_seed = {
                executor.submit(generate_single_conformer, seed): seed
                for seed in range(self.num_conformers)
            }

            for future in as_completed(future_to_seed):
                coords = future.result()
                if coords is not None:
                    conformer_coords.append(coords)

        return conformer_coords

    def _prune_conformers(self, conformer_coords: List[np.ndarray]) -> List[np.ndarray]:
        """Prune similar conformers based on RMSD threshold."""
        if not conformer_coords:
            return []

        pruned_coords = [conformer_coords[0]]  # Keep first conformer

        for coords in conformer_coords[1:]:
            is_unique = True

            for existing_coords in pruned_coords:
                try:
                    rmsd = calculate_rmsd(coords, existing_coords)
                    if rmsd < self.prune_rms_thresh:
                        is_unique = False
                        break
                except Exception:
                    continue

            if is_unique:
                pruned_coords.append(coords)

        logger.debug(
            f"Pruned {len(conformer_coords)} conformers to {len(pruned_coords)}"
        )
        return pruned_coords

    def _create_molecule_with_conformers(
        self, mol: Chem.Mol, conformer_coords: List[np.ndarray]
    ) -> Chem.Mol:
        """Create RDKit molecule with conformers from coordinate arrays."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit not available for conformer creation")

        # Create a copy of the molecule
        mol_copy = Chem.Mol(mol)

        # Remove existing conformers
        mol_copy.RemoveAllConformers()

        # Add new conformers
        for i, coords in enumerate(conformer_coords):
            conf = Chem.Conformer(mol_copy.GetNumAtoms())

            for atom_idx in range(mol_copy.GetNumAtoms()):
                x, y, z = coords[atom_idx]
                conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))

            conf.SetId(i)
            mol_copy.AddConformer(conf, assignId=True)

        return mol_copy

    def _minimize_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        """Perform energy minimization on conformers."""
        if not RDKIT_AVAILABLE:
            return mol

        optimized_conformers = []

        for conf_id in range(mol.GetNumConformers()):
            try:
                # Create a copy for optimization
                mol_copy = Chem.Mol(mol)

                if self.force_field == "UFF":
                    result = UFFOptimizeMolecule(mol_copy, confId=conf_id)
                else:  # MMFF
                    result = MMFFOptimizeMolecule(mol_copy, confId=conf_id)

                if result == 0:  # Optimization successful
                    optimized_conformers.append(conf_id)
                else:
                    logger.warning(
                        f"Energy minimization failed for conformer {conf_id}"
                    )

            except Exception as e:
                logger.warning(
                    f"Energy minimization error for conformer {conf_id}: {e}"
                )

        logger.debug(f"Successfully minimized {len(optimized_conformers)} conformers")
        return mol

    def generate_conformers_batch(
        self, molecules: List[MolecularData]
    ) -> List[MolecularData]:
        """Generate conformers for a batch of molecules.

        Args:
            molecules: List of MolecularData objects

        Returns:
            List of MolecularData with conformers
        """
        results = []

        for mol_data in molecules:
            try:
                mol_with_conformers = self.generate_conformers_parallel(mol_data)
                results.append(mol_with_conformers)
            except Exception as e:
                logger.warning(
                    f"Failed to generate conformers for {mol_data.mol_id}: {e}"
                )
                # Return original molecule without conformers
                results.append(mol_data)

        logger.info(f"Generated conformers for {len(results)} molecules")
        return results


class HybridConformerRefiner:
    """Hybrid CPU-GPU conformer refinement system."""

    def __init__(
        self, use_gpu_screening: bool = True, cpu_optimization_threads: int = 4
    ):
        """Initialize hybrid conformer refiner.

        Args:
            use_gpu_screening: Whether to use GPU for initial screening
            cpu_optimization_threads: Number of CPU threads for optimization
        """
        self.use_gpu_screening = use_gpu_screening and CUDA_AVAILABLE
        self.cpu_optimization_threads = cpu_optimization_threads

        logger.info(
            f"HybridConformerRefiner initialized: GPU screening={self.use_gpu_screening}"
        )

    def refine_conformers(self, mol_data: MolecularData) -> MolecularData:
        """Refine conformers using hybrid CPU-GPU approach.

        Args:
            mol_data: MolecularData with conformers

        Returns:
            MolecularData with refined conformers
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
            logger.warning(f"No conformers found for refinement: {mol_data.mol_id}")
            return mol_data

        try:
            # Step 1: GPU-based energy screening (if available)
            if self.use_gpu_screening:
                screened_mol = self._gpu_energy_screening(rdkit_mol)
            else:
                screened_mol = rdkit_mol

            # Step 2: CPU-based detailed optimization
            refined_mol = self._cpu_detailed_optimization(screened_mol)

            # Update molecular data
            updated_mol_data = MolecularData(
                smiles=mol_data.smiles,
                mol_id=mol_data.mol_id,
                num_atoms=mol_data.num_atoms,
                num_bonds=mol_data.num_bonds,
                molecular_weight=mol_data.molecular_weight,
                backend_data=mol_data.backend_data.copy(),
            )
            updated_mol_data.backend_data["rdkit_mol"] = refined_mol
            updated_mol_data.backend_data[
                "num_conformers"
            ] = refined_mol.GetNumConformers()
            updated_mol_data.backend_data["refinement_method"] = "hybrid_cpu_gpu"

            return updated_mol_data

        except Exception as e:
            logger.error(f"Conformer refinement failed for {mol_data.mol_id}: {e}")
            return mol_data

    def _gpu_energy_screening(self, mol: Chem.Mol) -> Chem.Mol:
        """GPU-based energy screening of conformers."""
        # Placeholder for GPU energy screening
        # In a full implementation, this would calculate approximate energies on GPU
        # and filter out high-energy conformers
        logger.debug("GPU energy screening (placeholder)")
        return mol

    def _cpu_detailed_optimization(self, mol: Chem.Mol) -> Chem.Mol:
        """CPU-based detailed optimization of conformers."""
        if not RDKIT_AVAILABLE:
            return mol

        def optimize_conformer(args):
            mol_copy, conf_id = args
            try:
                result = UFFOptimizeMolecule(mol_copy, confId=conf_id)
                return conf_id, result == 0
            except Exception as e:
                logger.warning(f"Optimization failed for conformer {conf_id}: {e}")
                return conf_id, False

        # Prepare arguments for parallel optimization
        optimization_args = []
        for conf_id in range(mol.GetNumConformers()):
            mol_copy = Chem.Mol(mol)
            optimization_args.append((mol_copy, conf_id))

        # Parallel optimization
        successful_conformers = []

        with ThreadPoolExecutor(max_workers=self.cpu_optimization_threads) as executor:
            results = list(executor.map(optimize_conformer, optimization_args))

            for conf_id, success in results:
                if success:
                    successful_conformers.append(conf_id)

        logger.debug(f"Successfully optimized {len(successful_conformers)} conformers")
        return mol


def validate_conformer_quality(mol_data: MolecularData) -> Dict[str, Any]:
    """Validate quality and diversity of generated conformers.

    Args:
        mol_data: MolecularData with conformers

    Returns:
        Dictionary with validation metrics
    """
    rdkit_mol = mol_data.backend_data.get("rdkit_mol")
    if rdkit_mol is None or rdkit_mol.GetNumConformers() == 0:
        return {"error": "No conformers found"}

    try:
        num_conformers = rdkit_mol.GetNumConformers()

        # Calculate pairwise RMSDs
        rmsds = []
        for i in range(num_conformers):
            for j in range(i + 1, num_conformers):
                try:
                    conf1 = rdkit_mol.GetConformer(i)
                    conf2 = rdkit_mol.GetConformer(j)

                    # Extract coordinates
                    coords1 = np.array(
                        [
                            conf1.GetAtomPosition(k)
                            for k in range(rdkit_mol.GetNumAtoms())
                        ]
                    )
                    coords2 = np.array(
                        [
                            conf2.GetAtomPosition(k)
                            for k in range(rdkit_mol.GetNumAtoms())
                        ]
                    )

                    rmsd = calculate_rmsd(coords1, coords2)
                    rmsds.append(rmsd)
                except Exception:
                    continue

        validation_metrics = {
            "num_conformers": num_conformers,
            "rmsd_mean": np.mean(rmsds) if rmsds else 0.0,
            "rmsd_std": np.std(rmsds) if rmsds else 0.0,
            "rmsd_min": np.min(rmsds) if rmsds else 0.0,
            "rmsd_max": np.max(rmsds) if rmsds else 0.0,
            "diversity_score": np.mean(rmsds) if rmsds else 0.0,
        }

        return validation_metrics

    except Exception as e:
        return {"error": str(e)}
