"""
Unit tests for 3D molecular descriptor calculations.

Tests GPU-accelerated 3D descriptor kernels and calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

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

from src.chemi_accel.utils.geometry import (
    calculate_radius_of_gyration,
    calculate_principal_moments,
    calculate_asphericity,
    calculate_eccentricity
)

if CUDA_AVAILABLE:
    from src.chemi_accel.backends.cuda.descriptors_3d_kernels import (
        CUDA3DDescriptorKernels,
        GPU3DDescriptorCalculator
    )


class TestGeometry3DCalculations:
    """Test 3D geometry calculations."""
    
    def test_radius_of_gyration_linear(self):
        """Test radius of gyration for linear molecule."""
        # Linear molecule: atoms at (0,0,0), (1,0,0), (2,0,0)
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0, 1.0])
        
        rg = calculate_radius_of_gyration(coords, masses)
        
        # For linear molecule with equal masses, RG should be sqrt(2/3) ≈ 0.816
        expected_rg = np.sqrt(2.0/3.0)
        assert rg == pytest.approx(expected_rg, rel=1e-3)
    
    def test_radius_of_gyration_uniform_masses(self):
        """Test radius of gyration with uniform masses."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        rg_with_masses = calculate_radius_of_gyration(coords, np.ones(3))
        rg_without_masses = calculate_radius_of_gyration(coords)
        
        assert rg_with_masses == pytest.approx(rg_without_masses, rel=1e-6)
    
    def test_principal_moments_linear(self):
        """Test principal moments for linear molecule."""
        # Linear molecule along x-axis
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0, 1.0])
        
        eigenvalues, eigenvectors = calculate_principal_moments(coords, masses)
        
        assert len(eigenvalues) == 3
        assert eigenvectors.shape == (3, 3)
        
        # For linear molecule, one eigenvalue should be much smaller (≈0)
        sorted_eigenvals = np.sort(eigenvalues)
        assert sorted_eigenvals[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_principal_moments_symmetric(self):
        """Test principal moments for symmetric molecule."""
        # Square in xy-plane
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=float)
        masses = np.ones(4)
        
        eigenvalues, eigenvectors = calculate_principal_moments(coords, masses)
        
        # Two eigenvalues should be equal for square symmetry
        sorted_eigenvals = np.sort(eigenvalues)
        assert sorted_eigenvals[0] == pytest.approx(sorted_eigenvals[1], rel=1e-3)
    
    def test_asphericity_linear(self):
        """Test asphericity for linear molecule."""
        # Linear molecule
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        masses = np.ones(3)
        
        asphericity = calculate_asphericity(coords, masses)
        
        # Linear molecule should have high asphericity (close to 1)
        assert asphericity > 0.5
        assert asphericity <= 1.0
    
    def test_asphericity_spherical(self):
        """Test asphericity for spherical molecule."""
        # Tetrahedral arrangement (approximately spherical)
        coords = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        masses = np.ones(5)
        
        asphericity = calculate_asphericity(coords, masses)
        
        # Spherical molecule should have low asphericity
        assert asphericity >= 0.0
        assert asphericity < 0.5
    
    def test_eccentricity_linear(self):
        """Test eccentricity for linear molecule."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        masses = np.ones(3)
        
        eccentricity = calculate_eccentricity(coords, masses)
        
        # Linear molecule should have high eccentricity
        assert eccentricity > 0.5
        assert eccentricity <= 1.0
    
    def test_eccentricity_spherical(self):
        """Test eccentricity for spherical molecule."""
        # Tetrahedral arrangement
        coords = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        masses = np.ones(5)
        
        eccentricity = calculate_eccentricity(coords, masses)
        
        # Spherical molecule should have low eccentricity
        assert eccentricity >= 0.0
        assert eccentricity < 0.5


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDA3DDescriptorKernels:
    """Test CUDA 3D descriptor kernels."""
    
    def setup_method(self):
        """Set up test data."""
        # Simple test molecules
        self.coordinates = cp.array([
            # Molecule 1: linear (3 atoms)
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            # Molecule 2: triangular (3 atoms)  
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]
        ], dtype=cp.float32)
        
        self.masses = cp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=cp.float32)
        self.molecule_boundaries = cp.array([3, 6], dtype=cp.int32)
        
        # van der Waals radii
        self.vdw_radii = cp.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2], dtype=cp.float32)
    
    def test_kernel_compilation(self):
        """Test CUDA kernel compilation."""
        try:
            kernels = CUDA3DDescriptorKernels()
            assert len(kernels._kernels) > 0
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_radius_of_gyration_kernel(self):
        """Test radius of gyration CUDA kernel."""
        try:
            kernels = CUDA3DDescriptorKernels()
            
            results = kernels.calculate_radius_of_gyration(
                self.coordinates, self.masses, self.molecule_boundaries
            )
            
            assert len(results) == 2  # Two molecules
            assert cp.all(results >= 0)  # All values should be non-negative
            assert not cp.any(cp.isnan(results))  # No NaN values
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_principal_moments_kernel(self):
        """Test principal moments CUDA kernel."""
        try:
            kernels = CUDA3DDescriptorKernels()
            
            pmi1, pmi2, pmi3 = kernels.calculate_principal_moments(
                self.coordinates, self.masses, self.molecule_boundaries
            )
            
            assert len(pmi1) == 2
            assert len(pmi2) == 2  
            assert len(pmi3) == 2
            
            # All PMI values should be non-negative
            assert cp.all(pmi1 >= 0)
            assert cp.all(pmi2 >= 0)
            assert cp.all(pmi3 >= 0)
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_asphericity_kernel(self):
        """Test asphericity CUDA kernel."""
        try:
            kernels = CUDA3DDescriptorKernels()
            
            # First calculate PMI values
            pmi1, pmi2, pmi3 = kernels.calculate_principal_moments(
                self.coordinates, self.masses, self.molecule_boundaries
            )
            
            # Then calculate asphericity
            asphericity = kernels.calculate_asphericity(pmi1, pmi2, pmi3)
            
            assert len(asphericity) == 2
            assert cp.all(asphericity >= 0)  # Should be non-negative
            assert cp.all(asphericity <= 1)  # Should be <= 1
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_molecular_volume_kernel(self):
        """Test molecular volume CUDA kernel."""
        try:
            kernels = CUDA3DDescriptorKernels()
            
            volumes = kernels.calculate_molecular_volume(
                self.coordinates, self.vdw_radii, self.molecule_boundaries
            )
            
            assert len(volumes) == 2
            assert cp.all(volumes > 0)  # Volumes should be positive
            assert not cp.any(cp.isnan(volumes))
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_geometric_diameter_kernel(self):
        """Test geometric diameter CUDA kernel."""
        try:
            kernels = CUDA3DDescriptorKernels()
            
            diameters = kernels.calculate_geometric_diameter(
                self.coordinates, self.molecule_boundaries
            )
            
            assert len(diameters) == 2
            assert cp.all(diameters >= 0)  # Should be non-negative
            
            # Linear molecule should have diameter = 2.0
            assert diameters[0] == pytest.approx(2.0, rel=1e-3)
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPU3DDescriptorCalculator:
    """Test high-level GPU 3D descriptor calculator."""
    
    def setup_method(self):
        """Set up test molecular data."""
        self.molecular_data = {
            'coordinates': cp.array([
                [0, 0, 0], [1, 0, 0], [2, 0, 0],  # Linear molecule
                [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]  # Triangular molecule
            ], dtype=cp.float32),
            'atomic_numbers': cp.array([6, 6, 6, 6, 6, 6], dtype=cp.int32),  # All carbon
            'molecule_boundaries': cp.array([3, 6], dtype=cp.int32)
        }
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        try:
            calculator = GPU3DDescriptorCalculator()
            assert calculator.vdw_radii[6] == 1.70  # Carbon VdW radius
            assert calculator.atomic_masses[6] == 12.011  # Carbon mass
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_calculate_all_3d_descriptors(self):
        """Test calculation of all 3D descriptors."""
        try:
            calculator = GPU3DDescriptorCalculator()
            
            results = calculator.calculate_3d_descriptors(self.molecular_data)
            
            expected_descriptors = [
                'radius_of_gyration', 'pmi1', 'pmi2', 'pmi3', 
                'asphericity', 'molecular_volume', 'geometric_diameter'
            ]
            
            for desc in expected_descriptors:
                assert desc in results
                assert len(results[desc]) == 2  # Two molecules
                assert not cp.any(cp.isnan(results[desc]))
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise
    
    def test_missing_data_handling(self):
        """Test handling of missing molecular data."""
        try:
            calculator = GPU3DDescriptorCalculator()
            
            # Missing coordinates
            incomplete_data = {
                'atomic_numbers': cp.array([6, 6, 6]),
                'molecule_boundaries': cp.array([3])
            }
            
            results = calculator.calculate_3d_descriptors(incomplete_data)
            assert len(results) == 0  # Should return empty results
            
        except RuntimeError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            else:
                raise


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestIntegrationWith3DDescriptors:
    """Integration tests with RDKit 3D descriptors."""
    
    def setup_method(self):
        """Set up test molecules with 3D coordinates."""
        # Create simple molecules
        self.ethane = Chem.MolFromSmiles("CC")
        self.ethane = Chem.AddHs(self.ethane)
        AllChem.EmbedMolecule(self.ethane)
        
        self.benzene = Chem.MolFromSmiles("c1ccccc1")
        self.benzene = Chem.AddHs(self.benzene)
        AllChem.EmbedMolecule(self.benzene)
    
    def test_rdkit_3d_descriptors_available(self):
        """Test that RDKit 3D descriptors are available."""
        try:
            from rdkit.Chem.Descriptors3D import RadiusOfGyration
            rg = RadiusOfGyration(self.ethane)
            assert rg > 0
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")
    
    def test_compare_radius_of_gyration(self):
        """Compare our RG calculation with RDKit."""
        try:
            from rdkit.Chem.Descriptors3D import RadiusOfGyration
            
            # Get coordinates from RDKit molecule
            conf = self.ethane.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(self.ethane.GetNumAtoms())])
            
            # Calculate using our function
            our_rg = calculate_radius_of_gyration(coords)
            
            # Calculate using RDKit
            rdkit_rg = RadiusOfGyration(self.ethane)
            
            # Should be approximately equal
            assert our_rg == pytest.approx(rdkit_rg, rel=0.1)
            
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")
    
    def test_compare_asphericity(self):
        """Compare asphericity calculation with RDKit."""
        try:
            from rdkit.Chem.Descriptors3D import Asphericity
            
            # Get coordinates from RDKit molecule
            conf = self.ethane.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(self.ethane.GetNumAtoms())])
            
            # Calculate using our function
            our_asphericity = calculate_asphericity(coords)
            
            # Calculate using RDKit
            rdkit_asphericity = Asphericity(self.ethane)
            
            # Should be approximately equal
            assert our_asphericity == pytest.approx(rdkit_asphericity, rel=0.2)
            
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_atom_molecule(self):
        """Test calculations with single atom."""
        coords = np.array([[0, 0, 0]])
        masses = np.array([1.0])
        
        rg = calculate_radius_of_gyration(coords, masses)
        assert rg == 0.0
        
        eigenvalues, _ = calculate_principal_moments(coords, masses)
        assert np.all(eigenvalues == 0.0)
        
        asphericity = calculate_asphericity(coords, masses)
        assert asphericity == 0.0
    
    def test_zero_masses(self):
        """Test calculations with zero masses."""
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        masses = np.array([0.0, 0.0])
        
        rg = calculate_radius_of_gyration(coords, masses)
        assert np.isnan(rg) or rg == 0.0
    
    def test_identical_coordinates(self):
        """Test calculations with identical coordinates."""
        coords = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        masses = np.array([1.0, 1.0, 1.0])
        
        rg = calculate_radius_of_gyration(coords, masses)
        assert rg == 0.0
        
        eigenvalues, _ = calculate_principal_moments(coords, masses)
        assert np.all(eigenvalues == 0.0)


if __name__ == "__main__":
    pytest.main([__file__])