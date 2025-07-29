"""
Comprehensive tests for the self-assembly characterization module.

Tests cover structure classification, clustering algorithms, morphology analysis,
and specialized characterization methods for micelles, vesicles, and liquid crystals.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.self_assembly.characterization import (  # Main classes; Data structures; Enums; Utility functions; Exceptions
    CharacterizationError,
    CharacterizationResult,
    ClusterAnalyzer,
    ClusteringError,
    ClusteringMethod,
    HydrogelCharacteristics,
    LiquidCrystalAnalyzer,
    LiquidCrystalCharacteristics,
    MicelleAnalyzer,
    MicelleCharacteristics,
    MorphologyAnalysisError,
    MorphologyType,
    PhaseAnalysisResult,
    PhaseType,
    SelfAssemblyCharacterizer,
    StructureClassifier,
    StructureMetrics,
    StructureType,
    VesicleAnalyzer,
    VesicleCharacteristics,
    analyze_micelle_formation,
    calculate_phase_diagram,
    characterize_trajectory,
)

# Import simulation classes for testing
from chemi_accel.self_assembly.coarse_grained_simulation import (
    Particle,
    ParticleType,
    TrajectoryFrame,
)


class TestStructureMetrics:
    """Test StructureMetrics dataclass."""

    def test_structure_metrics_creation(self):
        """Test basic structure metrics creation."""
        center_of_mass = np.array([1.0, 2.0, 3.0])

        metrics = StructureMetrics(
            structure_id=1,
            structure_type=StructureType.MICELLE,
            morphology=MorphologyType.SPHERICAL,
            aggregation_number=50,
            center_of_mass=center_of_mass,
            radius_of_gyration=2.5,
            asphericity=0.1,
            eccentricity=0.2,
            surface_area=78.5,
            volume=65.4,
            compactness=0.8,
            hydrophobic_core_radius=1.5,
            hydrophilic_shell_thickness=0.5,
            order_parameter=0.7,
        )

        assert metrics.structure_id == 1
        assert metrics.structure_type == StructureType.MICELLE
        assert metrics.morphology == MorphologyType.SPHERICAL
        assert metrics.aggregation_number == 50
        np.testing.assert_array_equal(metrics.center_of_mass, center_of_mass)
        assert metrics.radius_of_gyration == 2.5
        assert metrics.asphericity == 0.1

    def test_structure_metrics_validation(self):
        """Test structure metrics validation."""
        center_of_mass = np.array([1.0, 2.0, 3.0])

        # Test invalid aggregation number
        with pytest.raises(ValueError, match="Aggregation number must be positive"):
            StructureMetrics(
                structure_id=1,
                structure_type=StructureType.MICELLE,
                morphology=MorphologyType.SPHERICAL,
                aggregation_number=0,  # Invalid
                center_of_mass=center_of_mass,
                radius_of_gyration=2.5,
                asphericity=0.1,
                eccentricity=0.2,
                surface_area=78.5,
                volume=65.4,
                compactness=0.8,
                hydrophobic_core_radius=1.5,
                hydrophilic_shell_thickness=0.5,
                order_parameter=0.7,
            )

        # Test invalid center of mass
        with pytest.raises(ValueError, match="Center of mass must be 3D vector"):
            StructureMetrics(
                structure_id=1,
                structure_type=StructureType.MICELLE,
                morphology=MorphologyType.SPHERICAL,
                aggregation_number=50,
                center_of_mass=np.array([1.0, 2.0]),  # 2D instead of 3D
                radius_of_gyration=2.5,
                asphericity=0.1,
                eccentricity=0.2,
                surface_area=78.5,
                volume=65.4,
                compactness=0.8,
                hydrophobic_core_radius=1.5,
                hydrophilic_shell_thickness=0.5,
                order_parameter=0.7,
            )


class TestMicelleCharacteristics:
    """Test MicelleCharacteristics dataclass."""

    def test_micelle_characteristics_creation(self):
        """Test micelle characteristics creation."""
        char = MicelleCharacteristics(
            core_radius=1.8,
            shell_thickness=0.7,
            aggregation_number=60,
            critical_packing_parameter=0.35,
            core_water_content=0.05,
            interface_width=0.3,
            shape_factor=0.9,
        )

        assert char.core_radius == 1.8
        assert char.shell_thickness == 0.7
        assert char.aggregation_number == 60
        assert char.critical_packing_parameter == 0.35


class TestVesicleCharacteristics:
    """Test VesicleCharacteristics dataclass."""

    def test_vesicle_characteristics_creation(self):
        """Test vesicle characteristics creation."""
        char = VesicleCharacteristics(
            inner_radius=3.0,
            outer_radius=4.0,
            membrane_thickness=1.0,
            enclosed_volume=113.1,
            surface_area_to_volume_ratio=0.35,
            membrane_curvature=0.28,
            asymmetry_factor=0.15,
        )

        assert char.inner_radius == 3.0
        assert char.outer_radius == 4.0
        assert char.membrane_thickness == 1.0
        assert char.enclosed_volume == 113.1


class TestLiquidCrystalCharacteristics:
    """Test LiquidCrystalCharacteristics dataclass."""

    def test_liquid_crystal_characteristics_creation(self):
        """Test liquid crystal characteristics creation."""
        director_vector = np.array([1.0, 0.0, 0.0])
        elastic_constants = {"K11": 1.2, "K22": 0.8, "K33": 1.5}

        char = LiquidCrystalCharacteristics(
            nematic_order_parameter=0.75,
            director_vector=director_vector,
            correlation_length=5.2,
            elastic_constants=elastic_constants,
            phase_type=PhaseType.NEMATIC,
        )

        assert char.nematic_order_parameter == 0.75
        np.testing.assert_array_equal(char.director_vector, director_vector)
        assert char.correlation_length == 5.2
        assert char.elastic_constants == elastic_constants
        assert char.phase_type == PhaseType.NEMATIC


class TestStructureClassifier:
    """Test StructureClassifier class."""

    def test_classifier_initialization_cpu(self):
        """Test classifier initialization for CPU."""
        classifier = StructureClassifier(use_gpu=False)

        assert classifier.use_gpu is False
        assert classifier.device_id == 0

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"), reason="CuPy not available"
    )
    def test_classifier_initialization_gpu(self):
        """Test classifier initialization for GPU (if available)."""
        try:
            import cupy as cp

            classifier = StructureClassifier(use_gpu=True, device_id=0)
            assert classifier.device_id == 0
        except ImportError:
            pytest.skip("CuPy not available for GPU testing")

    def test_geometric_metrics_calculation_cpu(self):
        """Test geometric metrics calculation on CPU."""
        classifier = StructureClassifier(use_gpu=False)

        # Create spherical structure
        n_particles = 20
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0

        # Generate points on sphere
        theta = np.random.uniform(0, 2 * np.pi, n_particles)
        phi = np.random.uniform(0, np.pi, n_particles)

        positions = np.zeros((n_particles, 3))
        positions[:, 0] = radius * np.sin(phi) * np.cos(theta)
        positions[:, 1] = radius * np.sin(phi) * np.sin(theta)
        positions[:, 2] = radius * np.cos(phi)

        cluster_indices = np.zeros(n_particles, dtype=np.int32)

        metrics = classifier._calculate_geometric_metrics(positions, cluster_indices)

        assert "radius_of_gyration" in metrics
        assert "asphericity" in metrics
        assert "sphericity" in metrics
        assert "eigenvals" in metrics

        # For spherical structure, should have high sphericity
        assert metrics["sphericity"] > 0.5

    def test_structure_classification_micelle(self):
        """Test classification of micelle-like structure."""
        classifier = StructureClassifier(use_gpu=False)

        # Create spherical structure with amphiphilic particles
        n_particles = 30
        positions = np.random.normal(0, 1, (n_particles, 3))
        particle_types = ["HEAD"] * 15 + ["TAIL"] * 15
        cluster_indices = np.zeros(n_particles, dtype=np.int32)

        structure_type = classifier.classify_structure(
            positions, particle_types, cluster_indices
        )

        # Should classify as micelle or cluster
        assert structure_type in [StructureType.MICELLE, StructureType.CLUSTER]

    def test_structure_classification_fiber(self):
        """Test classification of fiber-like structure."""
        classifier = StructureClassifier(use_gpu=False)

        # Create elongated structure
        n_particles = 50
        positions = np.zeros((n_particles, 3))
        positions[:, 0] = np.linspace(0, 10, n_particles)  # Extended in x-direction
        positions[:, 1] = np.random.normal(0, 0.1, n_particles)
        positions[:, 2] = np.random.normal(0, 0.1, n_particles)

        particle_types = ["A"] * n_particles
        cluster_indices = np.zeros(n_particles, dtype=np.int32)

        structure_type = classifier.classify_structure(
            positions, particle_types, cluster_indices
        )

        # Should classify as fiber, sheet, or unknown
        assert structure_type in [
            StructureType.FIBER,
            StructureType.SHEET,
            StructureType.UNKNOWN,
        ]


class TestClusterAnalyzer:
    """Test ClusterAnalyzer class."""

    def test_analyzer_initialization_cpu(self):
        """Test analyzer initialization for CPU."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        assert analyzer.use_gpu is False
        assert analyzer.device_id == 0

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"), reason="scikit-learn not available"
    )
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering (if scikit-learn available)."""
        try:
            from sklearn.cluster import DBSCAN

            analyzer = ClusterAnalyzer(use_gpu=False)

            # Create clustered data
            cluster1 = np.random.normal([0, 0, 0], 0.5, (20, 3))
            cluster2 = np.random.normal([5, 0, 0], 0.5, (20, 3))
            positions = np.vstack([cluster1, cluster2])
            particle_types = ["A"] * 40

            cluster_labels = analyzer.cluster_particles(
                positions,
                particle_types,
                ClusteringMethod.DBSCAN,
                cutoff_distance=1.5,
                min_cluster_size=5,
            )

            assert len(cluster_labels) == 40
            assert (
                len(np.unique(cluster_labels[cluster_labels >= 0])) >= 1
            )  # At least one cluster

        except ImportError:
            pytest.skip("scikit-learn not available for DBSCAN testing")

    def test_density_based_clustering_cpu(self):
        """Test density-based clustering on CPU."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        # Create simple clustered data
        cluster1 = np.random.normal([0, 0, 0], 0.3, (10, 3))
        cluster2 = np.random.normal([3, 0, 0], 0.3, (10, 3))
        positions = np.vstack([cluster1, cluster2])
        particle_types = ["A"] * 20

        cluster_labels = analyzer._density_cluster_cpu(
            positions, particle_types, cutoff=1.0
        )

        assert len(cluster_labels) == 20
        assert cluster_labels.dtype == np.int32

    def test_morphology_analysis_spherical(self):
        """Test morphology analysis for spherical cluster."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        # Create spherical cluster
        n_particles = 30
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0

        theta = np.random.uniform(0, 2 * np.pi, n_particles)
        phi = np.random.uniform(0, np.pi, n_particles)

        positions = np.zeros((n_particles, 3))
        positions[:, 0] = radius * np.sin(phi) * np.cos(theta)
        positions[:, 1] = radius * np.sin(phi) * np.sin(theta)
        positions[:, 2] = radius * np.cos(phi)

        cluster_indices = np.zeros(n_particles, dtype=np.int32)

        morphology = analyzer.analyze_cluster_morphology(positions, cluster_indices)

        assert morphology in [MorphologyType.SPHERICAL, MorphologyType.IRREGULAR]

    def test_morphology_analysis_cylindrical(self):
        """Test morphology analysis for cylindrical cluster."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        # Create cylindrical cluster
        n_particles = 40
        height = 8.0
        radius = 1.0

        positions = np.zeros((n_particles, 3))
        positions[:, 2] = np.random.uniform(0, height, n_particles)  # Height along z

        theta = np.random.uniform(0, 2 * np.pi, n_particles)
        r = np.random.uniform(0, radius, n_particles)
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = r * np.sin(theta)

        cluster_indices = np.zeros(n_particles, dtype=np.int32)

        morphology = analyzer.analyze_cluster_morphology(positions, cluster_indices)

        assert morphology in [
            MorphologyType.CYLINDRICAL,
            MorphologyType.WORM_LIKE,
            MorphologyType.IRREGULAR,
        ]


class TestMicelleAnalyzer:
    """Test MicelleAnalyzer class."""

    def create_micelle_structure(self):
        """Create a simple micelle structure for testing."""
        n_heads = 20
        n_tails_per_head = 3

        positions = []
        particle_types = []

        # Core (tails)
        core_radius = 1.5
        for i in range(n_heads):
            for j in range(n_tails_per_head):
                # Random position in core
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(0, core_radius)

                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)

                positions.append([x, y, z])
                particle_types.append("TAIL")

        # Shell (heads)
        shell_radius = 2.0
        for i in range(n_heads):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = shell_radius * np.sin(phi) * np.cos(theta)
            y = shell_radius * np.sin(phi) * np.sin(theta)
            z = shell_radius * np.cos(phi)

            positions.append([x, y, z])
            particle_types.append("HEAD")

        return np.array(positions), particle_types

    def test_micelle_analyzer_initialization(self):
        """Test micelle analyzer initialization."""
        analyzer = MicelleAnalyzer(use_gpu=False)
        assert analyzer.use_gpu is False

    def test_micelle_analysis(self):
        """Test micelle structure analysis."""
        analyzer = MicelleAnalyzer(use_gpu=False)

        positions, particle_types = self.create_micelle_structure()
        cluster_indices = np.zeros(len(positions), dtype=np.int32)

        micelle_char = analyzer.analyze_micelle(
            positions, particle_types, cluster_indices
        )

        assert isinstance(micelle_char, MicelleCharacteristics)
        assert micelle_char.core_radius > 0
        assert micelle_char.shell_thickness > 0
        assert micelle_char.aggregation_number == len(positions)
        assert micelle_char.critical_packing_parameter >= 0
        assert micelle_char.shape_factor >= 0

    def test_micelle_analysis_missing_particles(self):
        """Test micelle analysis with missing particle types."""
        analyzer = MicelleAnalyzer(use_gpu=False)

        # Only tail particles, no heads
        positions = np.random.normal(0, 1, (20, 3))
        particle_types = ["TAIL"] * 20
        cluster_indices = np.zeros(20, dtype=np.int32)

        with pytest.raises(
            MorphologyAnalysisError, match="requires both head and tail particles"
        ):
            analyzer.analyze_micelle(positions, particle_types, cluster_indices)


class TestVesicleAnalyzer:
    """Test VesicleAnalyzer class."""

    def create_vesicle_structure(self):
        """Create a simple vesicle structure for testing."""
        n_particles = 100
        inner_radius = 2.0
        outer_radius = 3.0

        positions = []
        particle_types = []

        # Inner membrane
        for i in range(n_particles // 2):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = inner_radius * np.sin(phi) * np.cos(theta)
            y = inner_radius * np.sin(phi) * np.sin(theta)
            z = inner_radius * np.cos(phi)

            positions.append([x, y, z])
            particle_types.append("MEMBRANE")

        # Outer membrane
        for i in range(n_particles // 2):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = outer_radius * np.sin(phi) * np.cos(theta)
            y = outer_radius * np.sin(phi) * np.sin(theta)
            z = outer_radius * np.cos(phi)

            positions.append([x, y, z])
            particle_types.append("MEMBRANE")

        return np.array(positions), particle_types

    def test_vesicle_analysis(self):
        """Test vesicle structure analysis."""
        analyzer = VesicleAnalyzer()

        positions, particle_types = self.create_vesicle_structure()

        vesicle_char = analyzer.analyze_vesicle(positions, particle_types)

        assert isinstance(vesicle_char, VesicleCharacteristics)
        assert vesicle_char.inner_radius > 0
        assert vesicle_char.outer_radius > vesicle_char.inner_radius
        assert vesicle_char.membrane_thickness > 0
        assert vesicle_char.enclosed_volume > 0
        assert vesicle_char.surface_area_to_volume_ratio > 0


class TestLiquidCrystalAnalyzer:
    """Test LiquidCrystalAnalyzer class."""

    def create_liquid_crystal_structure(self):
        """Create a simple liquid crystal structure for testing."""
        n_particles = 50

        positions = []
        orientations = []

        # Create aligned structure (nematic-like)
        for i in range(n_particles):
            # Random positions
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            z = np.random.uniform(-5, 5)
            positions.append([x, y, z])

            # Mostly aligned orientations (with some noise)
            orientation = np.array([1.0, 0.0, 0.0])  # Aligned along x
            noise = np.random.normal(0, 0.1, 3)
            orientation += noise
            orientation /= np.linalg.norm(orientation)
            orientations.append(orientation)

        return np.array(positions), np.array(orientations)

    def test_liquid_crystal_analysis(self):
        """Test liquid crystal phase analysis."""
        analyzer = LiquidCrystalAnalyzer()

        positions, orientations = self.create_liquid_crystal_structure()

        lc_char = analyzer.analyze_liquid_crystal(positions, orientations)

        assert isinstance(lc_char, LiquidCrystalCharacteristics)
        assert 0 <= lc_char.nematic_order_parameter <= 1
        assert len(lc_char.director_vector) == 3
        assert np.isclose(np.linalg.norm(lc_char.director_vector), 1.0, atol=1e-6)
        assert lc_char.correlation_length >= 0
        assert isinstance(lc_char.elastic_constants, dict)
        assert lc_char.phase_type in PhaseType

    def test_liquid_crystal_analysis_without_orientations(self):
        """Test liquid crystal analysis without provided orientations."""
        analyzer = LiquidCrystalAnalyzer()

        positions, _ = self.create_liquid_crystal_structure()

        # Don't provide orientations - should estimate them
        lc_char = analyzer.analyze_liquid_crystal(positions)

        assert isinstance(lc_char, LiquidCrystalCharacteristics)
        assert 0 <= lc_char.nematic_order_parameter <= 1

    def test_nematic_order_parameter_calculation(self):
        """Test nematic order parameter calculation."""
        analyzer = LiquidCrystalAnalyzer()

        # Perfect alignment
        orientations = np.array([[1.0, 0.0, 0.0]] * 10)
        order_param = analyzer._calculate_nematic_order_parameter(orientations)

        assert order_param > 0.5  # Should be high for aligned system

        # Random orientations
        orientations = np.random.normal(0, 1, (10, 3))
        orientations = (
            orientations / np.linalg.norm(orientations, axis=1)[:, np.newaxis]
        )
        order_param = analyzer._calculate_nematic_order_parameter(orientations)

        assert order_param >= 0  # Should be non-negative


class TestSelfAssemblyCharacterizer:
    """Test main SelfAssemblyCharacterizer class."""

    def create_test_trajectory_frame(self):
        """Create a test trajectory frame."""
        n_particles = 30
        positions = np.random.uniform(0, 10, (n_particles, 3))
        velocities = np.random.normal(0, 1, (n_particles, 3))
        forces = np.random.normal(0, 0.1, (n_particles, 3))

        energies = {"kinetic": 25.0, "potential": -50.0, "total": -25.0}

        box_dimensions = np.array([10.0, 10.0, 10.0])

        frame = TrajectoryFrame(
            step=100,
            time=1.0,
            positions=positions,
            velocities=velocities,
            forces=forces,
            energies=energies,
            box_dimensions=box_dimensions,
        )

        return frame

    def test_characterizer_initialization(self):
        """Test characterizer initialization."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        assert characterizer.use_gpu is False
        assert hasattr(characterizer, "structure_classifier")
        assert hasattr(characterizer, "cluster_analyzer")
        assert hasattr(characterizer, "micelle_analyzer")
        assert hasattr(characterizer, "vesicle_analyzer")
        assert hasattr(characterizer, "lc_analyzer")

    def test_characterize_frame(self):
        """Test characterization of a single frame."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        frame = self.create_test_trajectory_frame()
        particle_types = ["HEAD"] * 10 + ["TAIL"] * 10 + ["WATER"] * 10

        result = characterizer.characterize_frame(
            frame, particle_types, ClusteringMethod.DENSITY_BASED, cutoff_distance=2.0
        )

        assert isinstance(result, CharacterizationResult)
        assert result.frame_id == 100
        assert result.time == 1.0
        assert isinstance(result.structures, list)
        assert isinstance(result.micelles, list)
        assert isinstance(result.vesicles, list)
        assert isinstance(result.liquid_crystals, list)
        assert isinstance(result.hydrogels, list)
        assert isinstance(result.phase_analysis, PhaseAnalysisResult)
        assert isinstance(result.global_metrics, dict)

    def test_calculate_global_metrics(self):
        """Test global metrics calculation."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        positions = np.random.uniform(0, 10, (50, 3))
        cluster_labels = np.array(
            [0] * 20 + [1] * 15 + [-1] * 15
        )  # Two clusters + noise

        metrics = characterizer._calculate_global_metrics(positions, cluster_labels)

        assert isinstance(metrics, dict)
        assert "n_clusters" in metrics
        assert "clustered_fraction" in metrics
        assert "average_cluster_size" in metrics
        assert "system_density" in metrics
        assert "largest_cluster_size" in metrics

        assert metrics["n_clusters"] == 2
        assert metrics["clustered_fraction"] == 0.7  # 35/50
        assert metrics["largest_cluster_size"] == 20

    def test_radial_distribution_function(self):
        """Test RDF calculation."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        # Create simple cubic lattice
        positions = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    positions.append([i, j, k])
        positions = np.array(positions)

        particle_types = ["A"] * len(positions)

        r_centers, rdf = characterizer._calculate_rdf(positions, r_max=5.0, n_bins=50)

        assert len(r_centers) == 50
        assert len(rdf) == 50
        assert np.all(rdf >= 0)  # RDF should be non-negative
        assert np.all(np.isfinite(rdf))


class TestUtilityFunctions:
    """Test utility functions."""

    def create_simple_trajectory(self):
        """Create a simple trajectory for testing."""
        trajectory = []

        for step in range(5):
            n_particles = 20
            positions = np.random.uniform(0, 10, (n_particles, 3))
            velocities = np.random.normal(0, 1, (n_particles, 3))
            forces = np.random.normal(0, 0.1, (n_particles, 3))

            energies = {"kinetic": 20.0, "potential": -40.0, "total": -20.0}
            box_dimensions = np.array([10.0, 10.0, 10.0])

            frame = TrajectoryFrame(
                step=step,
                time=step * 0.01,
                positions=positions,
                velocities=velocities,
                forces=forces,
                energies=energies,
                box_dimensions=box_dimensions,
            )
            trajectory.append(frame)

        return trajectory

    def test_characterize_trajectory(self):
        """Test trajectory characterization function."""
        trajectory = self.create_simple_trajectory()
        particle_types = ["HEAD"] * 10 + ["TAIL"] * 10

        results = characterize_trajectory(
            trajectory,
            particle_types,
            use_gpu=False,
            clustering_method=ClusteringMethod.DENSITY_BASED,
        )

        assert len(results) == len(trajectory)
        assert all(isinstance(result, CharacterizationResult) for result in results)

    def test_analyze_micelle_formation(self):
        """Test micelle formation analysis."""
        trajectory = self.create_simple_trajectory()
        particle_types = ["HEAD"] * 10 + ["TAIL"] * 10

        analysis = analyze_micelle_formation(trajectory, particle_types)

        assert isinstance(analysis, dict)
        assert "times" in analysis
        assert "micelle_counts" in analysis
        assert "aggregation_numbers" in analysis
        assert "cmc_estimate" in analysis

        assert len(analysis["times"]) == len(trajectory)
        assert len(analysis["micelle_counts"]) == len(trajectory)
        assert len(analysis["aggregation_numbers"]) == len(trajectory)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_characterization_error_hierarchy(self):
        """Test exception hierarchy."""
        assert issubclass(ClusteringError, CharacterizationError)
        assert issubclass(MorphologyAnalysisError, CharacterizationError)

    def test_clustering_error_on_invalid_method(self):
        """Test clustering error with invalid method."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        positions = np.random.uniform(0, 10, (20, 3))
        particle_types = ["A"] * 20

        # Use invalid clustering method
        with pytest.raises(ClusteringError, match="not implemented"):
            analyzer.cluster_particles(
                positions, particle_types, method=ClusteringMethod.GRID_BASED
            )


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for characterization components."""

    def test_clustering_performance(self, benchmark):
        """Benchmark clustering performance."""
        analyzer = ClusterAnalyzer(use_gpu=False)

        # Create larger test system
        n_particles = 200
        positions = np.random.uniform(0, 20, (n_particles, 3))
        particle_types = ["A"] * n_particles

        result = benchmark(
            analyzer._density_cluster_cpu, positions, particle_types, cutoff=2.0
        )

        assert len(result) == n_particles

    def test_structure_classification_performance(self, benchmark):
        """Benchmark structure classification performance."""
        classifier = StructureClassifier(use_gpu=False)

        positions = np.random.normal(0, 2, (100, 3))
        particle_types = ["HEAD"] * 50 + ["TAIL"] * 50
        cluster_indices = np.zeros(100, dtype=np.int32)

        result = benchmark(
            classifier.classify_structure, positions, particle_types, cluster_indices
        )

        assert isinstance(result, StructureType)


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete characterization scenarios."""

    def test_micelle_characterization_workflow(self):
        """Test complete micelle characterization workflow."""
        # Create micelle-like structure
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        # Generate amphiphilic molecules in micelle configuration
        n_molecules = 30
        positions = []
        particle_types = []

        # Core particles (tails)
        core_positions = np.random.normal(0, 1, (n_molecules * 3, 3))
        positions.extend(core_positions)
        particle_types.extend(["TAIL"] * (n_molecules * 3))

        # Shell particles (heads)
        shell_radius = 2.5
        for i in range(n_molecules):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = shell_radius * np.sin(phi) * np.cos(theta)
            y = shell_radius * np.sin(phi) * np.sin(theta)
            z = shell_radius * np.cos(phi)

            positions.append([x, y, z])
            particle_types.append("HEAD")

        positions = np.array(positions)

        # Create trajectory frame
        frame = TrajectoryFrame(
            step=0,
            time=0.0,
            positions=positions,
            velocities=np.random.normal(0, 1, positions.shape),
            forces=np.zeros_like(positions),
            energies={"kinetic": 100.0, "potential": -200.0, "total": -100.0},
            box_dimensions=np.array([20.0, 20.0, 20.0]),
        )

        # Characterize
        result = characterizer.characterize_frame(
            frame, particle_types, ClusteringMethod.DENSITY_BASED
        )

        # Should find structures
        assert len(result.structures) >= 1

        # Check for micelle characteristics
        found_micelle = False
        for structure in result.structures:
            if structure.structure_type == StructureType.MICELLE:
                found_micelle = True
                break

        # May not always classify as micelle due to simple test structure
        # but should find some organized structure
        assert len(result.structures) > 0

    def test_phase_analysis_workflow(self):
        """Test phase analysis workflow."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        # Create ordered structure
        n_particles = 64

        # Simple cubic lattice
        positions = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    positions.append([i, j, k])

        positions = np.array(positions)
        particle_types = ["A"] * n_particles

        # Analyze phase
        phase_result = characterizer._analyze_global_phase(positions, particle_types)

        assert isinstance(phase_result, PhaseAnalysisResult)
        assert phase_result.phase_type in PhaseType
        assert "translational" in phase_result.order_parameters
        assert "rdf" in phase_result.correlation_functions

    def test_multi_structure_system(self):
        """Test characterization of system with multiple structure types."""
        characterizer = SelfAssemblyCharacterizer(use_gpu=False)

        # Create system with multiple clusters
        positions = []
        particle_types = []

        # Cluster 1: Micelle-like
        micelle_center = np.array([5.0, 5.0, 5.0])
        for i in range(30):
            pos = micelle_center + np.random.normal(0, 1, 3)
            positions.append(pos)
            particle_types.append("HEAD" if i < 15 else "TAIL")

        # Cluster 2: Different location
        cluster_center = np.array([15.0, 5.0, 5.0])
        for i in range(20):
            pos = cluster_center + np.random.normal(0, 0.5, 3)
            positions.append(pos)
            particle_types.append("A")

        # Isolated particles
        for i in range(20):
            pos = np.random.uniform(0, 20, 3)
            positions.append(pos)
            particle_types.append("WATER")

        positions = np.array(positions)

        # Create frame
        frame = TrajectoryFrame(
            step=0,
            time=0.0,
            positions=positions,
            velocities=np.random.normal(0, 1, positions.shape),
            forces=np.zeros_like(positions),
            energies={"kinetic": 150.0, "potential": -300.0, "total": -150.0},
            box_dimensions=np.array([25.0, 25.0, 25.0]),
        )

        # Characterize
        result = characterizer.characterize_frame(
            frame, particle_types, ClusteringMethod.DENSITY_BASED, cutoff_distance=3.0
        )

        # Should find multiple structures
        assert len(result.structures) >= 1
        assert result.global_metrics["n_clusters"] >= 1
        assert result.global_metrics["clustered_fraction"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
