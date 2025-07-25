"""
Tests for geometry handler and conformer ensemble processing.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.qml.geometry_handler import (
    AdvancedConformerGenerator,
    AlignmentMethod,
    ConformerEnsemble,
    ConformerEnsembleProcessor,
    ConformerMethod,
    GeometryMetadata,
    GeometryQuality,
)
from chemi_accel.qml.on_the_fly import GeometryInput


class TestGeometryMetadata:
    """Test GeometryMetadata dataclass."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = GeometryMetadata(
            method=ConformerMethod.ETKDG,
            quality=GeometryQuality.HIGH,
            optimization_steps=100,
            final_energy=-150.5,
            generation_time=2.5,
        )

        assert metadata.method == ConformerMethod.ETKDG
        assert metadata.quality == GeometryQuality.HIGH
        assert metadata.optimization_steps == 100
        assert metadata.final_energy == -150.5
        assert metadata.generation_time == 2.5
        assert metadata.convergence_status == "unknown"

    def test_default_values(self):
        """Test default values in metadata."""
        metadata = GeometryMetadata(
            method=ConformerMethod.UFF, quality=GeometryQuality.MEDIUM
        )

        assert metadata.optimization_steps == 0
        assert metadata.final_energy is None
        assert metadata.rmsd_to_reference is None
        assert metadata.generation_time == 0.0


class TestConformerEnsemble:
    """Test ConformerEnsemble class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-95.0,
            ),
        ]

        ensemble = ConformerEnsemble(
            geometries=geometries,
            ensemble_energy_spread=5.0,
            boltzmann_weights=np.array([0.8, 0.2]),
        )

        assert len(ensemble) == 2
        assert ensemble.ensemble_energy_spread == 5.0
        assert np.array_equal(ensemble.boltzmann_weights, np.array([0.8, 0.2]))
        assert ensemble.reference_geometry is None

    def test_indexing(self):
        """Test ensemble indexing."""
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
        ]

        ensemble = ConformerEnsemble(geometries=geometries)

        assert ensemble[0] is geometries[0]
        assert ensemble[1] is geometries[1]

    def test_weighted_average_geometry(self):
        """Test weighted average geometry calculation."""
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
            GeometryInput(
                coordinates=np.array([[2.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
        ]

        # Equal weights
        ensemble = ConformerEnsemble(geometries=geometries)
        avg_geom = ensemble.get_weighted_average_geometry()

        expected_coords = np.array([[1.0, 0.0, 0.0]])  # Average of (0,0,0) and (2,0,0)
        np.testing.assert_array_almost_equal(avg_geom.coordinates, expected_coords)

        # Custom weights
        ensemble_weighted = ConformerEnsemble(
            geometries=geometries, boltzmann_weights=np.array([0.8, 0.2])
        )
        avg_geom_weighted = ensemble_weighted.get_weighted_average_geometry()

        expected_weighted = np.array([[0.4, 0.0, 0.0]])  # 0.8*0 + 0.2*2 = 0.4
        np.testing.assert_array_almost_equal(
            avg_geom_weighted.coordinates, expected_weighted
        )


class TestAdvancedConformerGenerator:
    """Test AdvancedConformerGenerator class."""

    @patch("chemi_accel.qml.geometry_handler.RDKIT_AVAILABLE", False)
    def test_initialization_no_rdkit(self):
        """Test initialization without RDKit."""
        with pytest.raises(Exception, match="RDKit required"):
            AdvancedConformerGenerator()

    @patch("chemi_accel.qml.geometry_handler.RDKIT_AVAILABLE", True)
    def test_initialization_with_rdkit(self):
        """Test initialization with RDKit available."""
        generator = AdvancedConformerGenerator(
            max_conformers=25,
            energy_threshold=20.0,
            quality=GeometryQuality.HIGH,
            method=ConformerMethod.MMFF,
        )

        assert generator.max_conformers == 25
        assert generator.energy_threshold == 20.0
        assert generator.quality == GeometryQuality.HIGH
        assert generator.method == ConformerMethod.MMFF

    @patch("chemi_accel.qml.geometry_handler.RDKIT_AVAILABLE", True)
    def test_setup_method_parameters(self):
        """Test method parameter setup."""
        generator = AdvancedConformerGenerator(quality=GeometryQuality.ULTRA)
        generator._setup_method_parameters()

        # Should have high quality parameters
        assert generator.etkdg_params.maxAttempts == 5000
        assert generator.etkdg_params.pruneRmsThresh == 0.1

    @patch("chemi_accel.qml.geometry_handler.RDKIT_AVAILABLE", True)
    @patch("chemi_accel.qml.geometry_handler.Chem")
    @patch("chemi_accel.qml.geometry_handler.AllChem")
    def test_generate_etkdg_conformers(self, mock_allchem, mock_chem):
        """Test ETKDG conformer generation."""
        generator = AdvancedConformerGenerator()

        # Mock RDKit components
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetAtoms.return_value = [
            Mock(GetAtomicNum=lambda: 6) for _ in range(3)
        ]

        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.rdmolops.GetFormalCharge.return_value = 0

        # Mock conformer generation
        mock_allchem.EmbedMultipleConfs.return_value = [0, 1, 2]
        mock_allchem.MMFFHasAllMoleculeParams.return_value = True

        # Mock force field
        mock_ff = Mock()
        mock_ff.Minimize.return_value = 0  # Converged
        mock_ff.CalcEnergy.return_value = -120.0
        mock_allchem.MMFFGetMoleculeForceField.return_value = mock_ff
        mock_allchem.MMFFGetMoleculeProperties.return_value = Mock()

        # Mock conformer
        mock_conf = Mock()
        mock_conf.GetAtomPosition.return_value.x = 0.0
        mock_conf.GetAtomPosition.return_value.y = 0.0
        mock_conf.GetAtomPosition.return_value.z = 0.0
        mock_mol.GetConformer.return_value = mock_conf

        geometries = generator._generate_etkdg_conformers(mock_mol)

        assert len(geometries) == 3
        assert all(isinstance(g, GeometryInput) for g in geometries)
        assert all(g.energy == -120.0 for g in geometries)

    @patch("chemi_accel.qml.geometry_handler.RDKIT_AVAILABLE", True)
    @patch("chemi_accel.qml.geometry_handler.Chem")
    @patch("chemi_accel.qml.geometry_handler.AllChem")
    def test_generate_conformer_ensemble(self, mock_allchem, mock_chem):
        """Test complete conformer ensemble generation."""
        generator = AdvancedConformerGenerator(max_conformers=5)

        # Mock successful generation
        with patch.object(generator, "_generate_etkdg_conformers") as mock_gen:
            mock_geometries = [
                GeometryInput(
                    coordinates=np.array([[i, 0.0, 0.0]]),
                    atomic_numbers=np.array([6]),
                    energy=-100.0 - i,
                )
                for i in range(5)
            ]
            mock_gen.return_value = mock_geometries

            mock_mol = Mock()
            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.AddHs.return_value = mock_mol
            mock_chem.RemoveHs.return_value = mock_mol
            mock_chem.MolToSmiles.return_value = "CCO"

            ensemble = generator.generate_conformer_ensemble("CCO")

            assert isinstance(ensemble, ConformerEnsemble)
            assert len(ensemble) <= 5
            assert ensemble.boltzmann_weights is not None
            assert ensemble.diversity_metrics is not None
            assert "method" in ensemble.metadata

    def test_calculate_rmsd(self):
        """Test RMSD calculation."""
        generator = AdvancedConformerGenerator()

        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        # Identical coordinates
        rmsd_same = generator._calculate_rmsd(coords1, coords2)
        assert rmsd_same == 0.0

        # Different coordinates
        rmsd_diff = generator._calculate_rmsd(coords1, coords3)
        assert rmsd_diff > 0.0

        # Different shapes
        coords_wrong = np.array([[0.0, 0.0, 0.0]])
        rmsd_wrong = generator._calculate_rmsd(coords1, coords_wrong)
        assert rmsd_wrong == float("inf")

    def test_remove_similar_conformers(self):
        """Test removal of similar conformers."""
        generator = AdvancedConformerGenerator(rms_threshold=0.5)

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[0.1, 0.0, 0.0]]),  # Very similar
                atomic_numbers=np.array([6]),
                energy=-99.0,
            ),
            GeometryInput(
                coordinates=np.array([[2.0, 0.0, 0.0]]),  # Different
                atomic_numbers=np.array([6]),
                energy=-98.0,
            ),
        ]

        unique_geometries = generator._remove_similar_conformers(geometries)

        # Should remove the similar one
        assert len(unique_geometries) == 2
        assert unique_geometries[0].energy == -100.0  # Lowest energy kept
        assert unique_geometries[1].energy == -98.0  # Different geometry kept

    def test_calculate_boltzmann_weights(self):
        """Test Boltzmann weight calculation."""
        generator = AdvancedConformerGenerator()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-99.0,
            ),
        ]

        weights = generator._calculate_boltzmann_weights(geometries)

        assert len(weights) == 2
        assert np.abs(weights.sum() - 1.0) < 1e-10  # Should sum to 1
        assert weights[0] > weights[1]  # Lower energy should have higher weight

    def test_calculate_boltzmann_weights_no_energies(self):
        """Test Boltzmann weights with no energies."""
        generator = AdvancedConformerGenerator()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
        ]

        weights = generator._calculate_boltzmann_weights(geometries)

        # Should return equal weights
        assert len(weights) == 2
        assert np.allclose(weights, [0.5, 0.5])

    def test_calculate_diversity_metrics(self):
        """Test diversity metrics calculation."""
        generator = AdvancedConformerGenerator()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
            GeometryInput(
                coordinates=np.array([[5.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
            GeometryInput(
                coordinates=np.array([[10.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
        ]

        metrics = generator._calculate_diversity_metrics(geometries)

        assert "diversity" in metrics
        assert "avg_rmsd" in metrics
        assert "max_rmsd" in metrics
        assert "rmsd_std" in metrics

        assert 0 <= metrics["diversity"] <= 1
        assert metrics["avg_rmsd"] > 0
        assert metrics["max_rmsd"] >= metrics["avg_rmsd"]

    def test_post_process_conformers(self):
        """Test conformer post-processing."""
        generator = AdvancedConformerGenerator(max_conformers=3, energy_threshold=5.0)

        # Create geometries with different energies
        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-99.0,  # Within threshold
            ),
            GeometryInput(
                coordinates=np.array([[2.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-90.0,  # Outside threshold (10 kcal/mol difference)
            ),
        ]

        mock_mol = Mock()
        processed = generator._post_process_conformers(geometries, mock_mol)

        # Should filter out high energy conformer
        assert len(processed) == 2
        assert all(geom.energy >= -100.0 for geom in processed)
        assert processed[0].energy == -100.0  # Sorted by energy


class TestConformerEnsembleProcessor:
    """Test ConformerEnsembleProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = ConformerEnsembleProcessor(alignment_method=AlignmentMethod.MCS)

        assert processor.alignment_method == AlignmentMethod.MCS

    def test_align_coordinates(self):
        """Test coordinate alignment using Kabsch algorithm."""
        processor = ConformerEnsembleProcessor()

        # Create two coordinate sets (rotated/translated versions)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atomic_numbers = np.array([6, 6, 6])

        aligned = processor._align_coordinates(coords, ref_coords, atomic_numbers)

        # Should be very close to reference after alignment
        rmsd = processor._calculate_rmsd(aligned, ref_coords)
        assert rmsd < 1e-10

    def test_align_ensemble(self):
        """Test ensemble alignment."""
        processor = ConformerEnsembleProcessor()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6, 1]),
            ),
            GeometryInput(
                coordinates=np.array([[0.1, 0.1, 0.0], [1.1, 0.1, 0.0]]),
                atomic_numbers=np.array([6, 1]),
            ),
        ]

        ensemble = ConformerEnsemble(geometries=geometries)
        aligned_ensemble = processor.align_ensemble(ensemble, reference_idx=0)

        assert isinstance(aligned_ensemble, ConformerEnsemble)
        assert len(aligned_ensemble) == len(ensemble)
        assert aligned_ensemble.metadata.get("aligned") == True
        assert aligned_ensemble.reference_geometry is geometries[0]

    def test_extract_representative_conformers_energy_diversity(self):
        """Test representative conformer extraction using energy/diversity."""
        processor = ConformerEnsembleProcessor()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[5.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-95.0,
            ),
            GeometryInput(
                coordinates=np.array([[0.1, 0.0, 0.0]]),  # Similar to first
                atomic_numbers=np.array([6]),
                energy=-98.0,
            ),
            GeometryInput(
                coordinates=np.array([[10.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-90.0,
            ),
        ]

        ensemble = ConformerEnsemble(geometries=geometries)
        representatives = processor.extract_representative_conformers(
            ensemble, num_representatives=2, method="energy_diversity"
        )

        assert len(representatives) == 2
        assert representatives[0].energy == -100.0  # Lowest energy
        # Second should be most diverse from first

    def test_extract_representative_conformers_boltzmann(self):
        """Test representative conformer extraction using Boltzmann weights."""
        processor = ConformerEnsembleProcessor()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-95.0,
            ),
        ]

        ensemble = ConformerEnsemble(
            geometries=geometries, boltzmann_weights=np.array([0.8, 0.2])
        )

        representatives = processor.extract_representative_conformers(
            ensemble, num_representatives=1, method="boltzmann"
        )

        assert len(representatives) == 1
        assert representatives[0] is geometries[0]  # Highest weight

    @patch("sklearn.cluster.KMeans")
    def test_extract_representative_conformers_clustering(self, mock_kmeans):
        """Test representative conformer extraction using clustering."""
        processor = ConformerEnsembleProcessor()

        # Mock sklearn KMeans
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_kmeans.return_value = mock_kmeans_instance

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[0.1, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-98.0,
            ),
            GeometryInput(
                coordinates=np.array([[5.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-95.0,
            ),
            GeometryInput(
                coordinates=np.array([[5.1, 0.0, 0.0]]),
                atomic_numbers=np.array([6]),
                energy=-93.0,
            ),
        ]

        ensemble = ConformerEnsemble(geometries=geometries)

        with patch("sklearn.cluster", create=True):
            representatives = processor.extract_representative_conformers(
                ensemble, num_representatives=2, method="clustering"
            )

            assert len(representatives) == 2

    def test_calculate_ensemble_properties(self):
        """Test ensemble property calculation."""
        processor = ConformerEnsembleProcessor()

        geometries = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6, 1]),
                energy=-100.0,
            ),
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
                atomic_numbers=np.array([6, 1]),
                energy=-95.0,
            ),
        ]

        ensemble = ConformerEnsemble(
            geometries=geometries,
            ensemble_energy_spread=5.0,
            boltzmann_weights=np.array([0.8, 0.2]),
            diversity_metrics={"diversity": 0.3, "avg_rmsd": 0.5},
        )

        properties = processor.calculate_ensemble_properties(ensemble)

        assert "num_conformers" in properties
        assert "energy_spread" in properties
        assert "energy_stats" in properties
        assert "diversity_metrics" in properties
        assert "boltzmann_stats" in properties
        assert "geometric_stats" in properties

        assert properties["num_conformers"] == 2
        assert properties["energy_spread"] == 5.0
        assert properties["energy_stats"]["min"] == -100.0
        assert properties["energy_stats"]["max"] == -95.0

    def test_get_atomic_mass(self):
        """Test atomic mass lookup."""
        processor = ConformerEnsembleProcessor()

        # Test known elements
        assert processor._get_atomic_mass(1) == 1.008  # Hydrogen
        assert processor._get_atomic_mass(6) == 12.011  # Carbon
        assert processor._get_atomic_mass(8) == 15.999  # Oxygen

        # Test unknown element (should default to carbon)
        assert processor._get_atomic_mass(999) == 12.011

    def test_calculate_com_variation(self):
        """Test center of mass variation calculation."""
        processor = ConformerEnsembleProcessor()

        coords_array = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Conformer 1
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],  # Conformer 2
            ]
        )
        atomic_numbers = np.array([6, 1])  # Carbon and hydrogen

        com_var = processor._calculate_com_variation(coords_array, atomic_numbers)

        assert com_var >= 0  # Should be non-negative
        assert isinstance(com_var, float)

    def test_calculate_rog_variation(self):
        """Test radius of gyration variation calculation."""
        processor = ConformerEnsembleProcessor()

        coords_array = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Conformer 1
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # Conformer 2
            ]
        )
        atomic_numbers = np.array([6, 1])

        rog_var = processor._calculate_rog_variation(coords_array, atomic_numbers)

        assert rog_var >= 0  # Should be non-negative
        assert isinstance(rog_var, float)
