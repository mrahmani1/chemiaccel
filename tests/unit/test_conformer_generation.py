"""
Unit tests for conformer generation system.

Tests GPU-accelerated conformer generation, validation, and quality metrics.
"""

import numpy as np
import pytest

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.conformer_generation import (
    GPUConformerGenerator,
    HybridConformerRefiner,
    validate_conformer_quality,
)
from src.chemi_accel.utils.geometry import (
    DistanceGeometry,
    calculate_radius_of_gyration,
    calculate_rmsd,
)


class TestDistanceGeometry:
    """Test distance geometry algorithms."""

    def test_distance_geometry_initialization(self):
        """Test DistanceGeometry initialization."""
        dg = DistanceGeometry(use_gpu=False)
        assert not dg.use_gpu

        if CUDA_AVAILABLE:
            dg_gpu = DistanceGeometry(use_gpu=True)
            assert dg_gpu.use_gpu

    def test_generate_distance_bounds(self):
        """Test distance bounds generation."""
        dg = DistanceGeometry(use_gpu=False)

        # Simple 3-atom linear molecule: A-B-C
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        lower_bounds, upper_bounds = dg.generate_distance_bounds(adjacency)

        assert lower_bounds.shape == (3, 3)
        assert upper_bounds.shape == (3, 3)

        # Check diagonal is zero
        assert np.all(np.diag(lower_bounds) == 0)
        assert np.all(np.diag(upper_bounds) == 0)

        # Check bonded atoms have tight bounds
        assert lower_bounds[0, 1] > 0
        assert upper_bounds[0, 1] > lower_bounds[0, 1]

    def test_embed_coordinates_cpu(self):
        """Test coordinate embedding with CPU."""
        dg = DistanceGeometry(use_gpu=False)

        # Simple 3-atom system
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        lower_bounds, upper_bounds = dg.generate_distance_bounds(adjacency)
        coords = dg.embed_coordinates(
            lower_bounds, upper_bounds, n_dimensions=3, max_iterations=100
        )

        assert coords.shape == (3, 3)
        assert not np.any(np.isnan(coords))
        assert not np.any(np.isinf(coords))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_embed_coordinates_gpu(self):
        """Test coordinate embedding with GPU."""
        dg = DistanceGeometry(use_gpu=True)

        # Simple 3-atom system
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        lower_bounds, upper_bounds = dg.generate_distance_bounds(adjacency)
        coords = dg.embed_coordinates(
            lower_bounds, upper_bounds, n_dimensions=3, max_iterations=100
        )

        assert coords.shape == (3, 3)
        assert not np.any(np.isnan(coords))
        assert not np.any(np.isinf(coords))


class TestGeometryUtilities:
    """Test geometry utility functions."""

    def test_calculate_rmsd(self):
        """Test RMSD calculation."""
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        coords2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        rmsd = calculate_rmsd(coords1, coords2)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

        # Test with different coordinates
        coords3 = np.array([[0.1, 0, 0], [1.1, 0, 0], [0.1, 1, 0]])
        rmsd = calculate_rmsd(coords1, coords3)
        assert rmsd > 0

    def test_calculate_radius_of_gyration(self):
        """Test radius of gyration calculation."""
        # Simple linear molecule
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        masses = np.array([1.0, 1.0, 1.0])

        rg = calculate_radius_of_gyration(coords, masses)
        assert rg > 0
        assert not np.isnan(rg)

        # Test without masses (uniform masses)
        rg_uniform = calculate_radius_of_gyration(coords)
        assert rg_uniform > 0


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestGPUConformerGenerator:
    """Test GPU conformer generation."""

    def setup_method(self):
        """Set up test molecules."""
        # Create simple test molecules
        self.ethane_smiles = "CC"
        self.benzene_smiles = "c1ccccc1"

        # Create MolecularData objects
        ethane_mol = Chem.MolFromSmiles(self.ethane_smiles)
        ethane_mol = Chem.AddHs(ethane_mol)

        self.ethane_mol_data = MolecularData(
            smiles=self.ethane_smiles,
            mol_id="ethane_test",
            num_atoms=ethane_mol.GetNumAtoms(),
            num_bonds=ethane_mol.GetNumBonds(),
            molecular_weight=30.07,
            backend_data={"rdkit_mol": ethane_mol},
        )

    def test_conformer_generator_initialization(self):
        """Test conformer generator initialization."""
        generator = GPUConformerGenerator(
            num_conformers=5, use_gpu=False, energy_minimize=False
        )

        assert generator.num_conformers == 5
        assert not generator.use_gpu
        assert not generator.energy_minimize

    def test_create_adjacency_matrix(self):
        """Test adjacency matrix creation."""
        generator = GPUConformerGenerator(use_gpu=False, energy_minimize=False)

        mol = Chem.MolFromSmiles("CC")
        adjacency = generator._create_adjacency_matrix(mol)

        assert adjacency.shape == (2, 2)
        assert adjacency[0, 1] == 1  # Single bond
        assert adjacency[1, 0] == 1  # Symmetric

    def test_generate_conformers_cpu(self):
        """Test conformer generation with CPU."""
        generator = GPUConformerGenerator(
            num_conformers=3, use_gpu=False, energy_minimize=False
        )

        result = generator.generate_conformers_parallel(self.ethane_mol_data)

        assert result.mol_id == "ethane_test"
        rdkit_mol = result.backend_data["rdkit_mol"]
        assert rdkit_mol.GetNumConformers() > 0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_generate_conformers_gpu(self):
        """Test conformer generation with GPU."""
        generator = GPUConformerGenerator(
            num_conformers=3, use_gpu=True, energy_minimize=False
        )

        result = generator.generate_conformers_parallel(self.ethane_mol_data)

        assert result.mol_id == "ethane_test"
        rdkit_mol = result.backend_data["rdkit_mol"]
        assert rdkit_mol.GetNumConformers() > 0

    def test_prune_conformers(self):
        """Test conformer pruning."""
        generator = GPUConformerGenerator(
            num_conformers=5, prune_rms_thresh=0.1, use_gpu=False, energy_minimize=False
        )

        # Create identical conformers (should be pruned)
        coords1 = np.array([[0, 0, 0], [1, 0, 0]])
        coords2 = np.array([[0, 0, 0], [1, 0, 0]])  # Identical
        coords3 = np.array([[0, 0, 0], [2, 0, 0]])  # Different

        conformer_coords = [coords1, coords2, coords3]
        pruned = generator._prune_conformers(conformer_coords)

        # Should keep first and third (different) conformers
        assert len(pruned) == 2

    def test_generate_conformers_batch(self):
        """Test batch conformer generation."""
        generator = GPUConformerGenerator(
            num_conformers=2, use_gpu=False, energy_minimize=False
        )

        molecules = [self.ethane_mol_data]
        results = generator.generate_conformers_batch(molecules)

        assert len(results) == 1
        assert results[0].mol_id == "ethane_test"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestHybridConformerRefiner:
    """Test hybrid conformer refinement."""

    def setup_method(self):
        """Set up test molecule with conformers."""
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)

        # Generate a conformer
        AllChem.EmbedMolecule(mol)

        self.mol_data = MolecularData(
            smiles="CC",
            mol_id="ethane_refine",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=30.07,
            backend_data={"rdkit_mol": mol},
        )

    def test_refiner_initialization(self):
        """Test refiner initialization."""
        refiner = HybridConformerRefiner(
            use_gpu_screening=False, cpu_optimization_threads=2
        )

        assert not refiner.use_gpu_screening
        assert refiner.cpu_optimization_threads == 2

    def test_refine_conformers(self):
        """Test conformer refinement."""
        refiner = HybridConformerRefiner(
            use_gpu_screening=False, cpu_optimization_threads=1
        )

        result = refiner.refine_conformers(self.mol_data)

        assert result.mol_id == "ethane_refine"
        assert "refinement_method" in result.backend_data


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestConformerValidation:
    """Test conformer quality validation."""

    def setup_method(self):
        """Set up test molecule with multiple conformers."""
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)

        # Generate multiple conformers
        AllChem.EmbedMultipleConfs(mol, numConfs=3)

        self.mol_data = MolecularData(
            smiles="CC",
            mol_id="ethane_validate",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=30.07,
            backend_data={"rdkit_mol": mol},
        )

    def test_validate_conformer_quality(self):
        """Test conformer quality validation."""
        metrics = validate_conformer_quality(self.mol_data)

        assert "num_conformers" in metrics
        assert "rmsd_mean" in metrics
        assert "rmsd_std" in metrics
        assert "diversity_score" in metrics

        assert metrics["num_conformers"] > 0
        assert metrics["rmsd_mean"] >= 0

    def test_validate_no_conformers(self):
        """Test validation with no conformers."""
        mol = Chem.MolFromSmiles("CC")
        mol_data = MolecularData(
            smiles="CC",
            mol_id="no_conformers",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=30.07,
            backend_data={"rdkit_mol": mol},
        )

        metrics = validate_conformer_quality(mol_data)
        assert "error" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
