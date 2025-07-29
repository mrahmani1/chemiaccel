"""
Unit tests for crystal structure prediction module.
"""

from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.materials.crystal_structure_prediction import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    AtomicPosition,
    ConvergenceError,
    CrystalGraphConv,
    CrystalPredictionConfig,
    CrystalStructure,
    CrystalStructurePredictionError,
    CrystalSystemType,
    CSPMethod,
    CSPPrediction,
    GPUAcceleratedCSPEngine,
    LatticeEnergyError,
    LatticeEnergyMethod,
    LatticeEnergyPredictor,
    PackingMotifGNN,
    PackingMotifType,
    StructureGenerationError,
    SymmetryError,
    SymmetryHandling,
    UnitCellGenerator,
    UnitCellParameters,
    analyze_polymorphs,
    create_csp_config,
    predict_crystal_structure,
)


@pytest.fixture
def sample_unit_cell():
    """Create sample unit cell parameters."""
    return UnitCellParameters(a=10.0, b=12.0, c=8.0, alpha=90.0, beta=90.0, gamma=90.0)


@pytest.fixture
def sample_atomic_positions():
    """Create sample atomic positions."""
    return [
        AtomicPosition(element="C", x=0.25, y=0.25, z=0.25),
        AtomicPosition(element="C", x=0.75, y=0.75, z=0.75),
        AtomicPosition(element="H", x=0.1, y=0.1, z=0.1),
        AtomicPosition(element="H", x=0.9, y=0.9, z=0.9),
    ]


@pytest.fixture
def sample_crystal_structure(sample_unit_cell, sample_atomic_positions):
    """Create sample crystal structure."""
    return CrystalStructure(
        unit_cell=sample_unit_cell,
        atomic_positions=sample_atomic_positions,
        space_group=14,
        formula="C2H2",
        lattice_energy=-100.0,
        density=1.2,
        packing_motif=PackingMotifType.LAYERED,
    )


@pytest.fixture
def sample_csp_config():
    """Create sample CSP configuration."""
    return CrystalPredictionConfig(
        method=CSPMethod.HYBRID_APPROACH,
        max_structures=10,
        use_gpu=False,
        energy_cutoff=25.0,
        temperature=298.0,
        max_iterations=100,
    )


@pytest.fixture
def sample_csp_prediction(sample_crystal_structure):
    """Create sample CSP prediction."""
    return CSPPrediction(
        predicted_structures=[sample_crystal_structure],
        energy_landscape={"structure_0_C2H2": -100.0},
        polymorphs=[sample_crystal_structure],
        most_stable_structure=sample_crystal_structure,
        prediction_confidence=0.85,
        computational_cost=12.5,
        method_used=CSPMethod.HYBRID_APPROACH,
        convergence_achieved=True,
    )


class TestUnitCellParameters:
    """Test UnitCellParameters class."""

    def test_unit_cell_creation(self, sample_unit_cell):
        """Test unit cell creation and volume calculation."""
        uc = sample_unit_cell

        assert uc.a == 10.0
        assert uc.b == 12.0
        assert uc.c == 8.0
        assert uc.alpha == 90.0
        assert uc.beta == 90.0
        assert uc.gamma == 90.0
        assert uc.volume == 960.0  # 10 * 12 * 8
        assert uc.crystal_system == CrystalSystemType.ORTHORHOMBIC

    def test_cubic_system_detection(self):
        """Test cubic crystal system detection."""
        uc = UnitCellParameters(
            a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0
        )
        assert uc.crystal_system == CrystalSystemType.CUBIC

    def test_tetragonal_system_detection(self):
        """Test tetragonal crystal system detection."""
        uc = UnitCellParameters(
            a=10.0, b=10.0, c=8.0, alpha=90.0, beta=90.0, gamma=90.0
        )
        assert uc.crystal_system == CrystalSystemType.TETRAGONAL

    def test_hexagonal_system_detection(self):
        """Test hexagonal crystal system detection."""
        uc = UnitCellParameters(
            a=10.0, b=10.0, c=8.0, alpha=90.0, beta=90.0, gamma=120.0
        )
        assert uc.crystal_system == CrystalSystemType.HEXAGONAL

    def test_monoclinic_system_detection(self):
        """Test monoclinic crystal system detection."""
        uc = UnitCellParameters(
            a=10.0, b=12.0, c=8.0, alpha=90.0, beta=105.0, gamma=90.0
        )
        assert uc.crystal_system == CrystalSystemType.MONOCLINIC

    def test_triclinic_system_detection(self):
        """Test triclinic crystal system detection."""
        uc = UnitCellParameters(
            a=10.0, b=12.0, c=8.0, alpha=85.0, beta=95.0, gamma=105.0
        )
        assert uc.crystal_system == CrystalSystemType.TRICLINIC

    def test_volume_calculation_non_orthogonal(self):
        """Test volume calculation for non-orthogonal cell."""
        uc = UnitCellParameters(a=5.0, b=6.0, c=7.0, alpha=80.0, beta=90.0, gamma=110.0)

        # Volume should be positive and reasonable
        assert uc.volume > 0
        assert 180 < uc.volume < 220  # Approximate expected range


class TestAtomicPosition:
    """Test AtomicPosition class."""

    def test_atomic_position_creation(self):
        """Test atomic position creation."""
        pos = AtomicPosition(
            element="C", x=0.25, y=0.5, z=0.75, occupancy=1.0, thermal_factor=0.02
        )

        assert pos.element == "C"
        assert pos.x == 0.25
        assert pos.y == 0.5
        assert pos.z == 0.75
        assert pos.occupancy == 1.0
        assert pos.thermal_factor == 0.02

    def test_default_values(self):
        """Test default values for atomic position."""
        pos = AtomicPosition(element="N", x=0.1, y=0.2, z=0.3)

        assert pos.occupancy == 1.0
        assert pos.thermal_factor == 0.0


class TestCrystalStructure:
    """Test CrystalStructure class."""

    def test_crystal_structure_creation(self, sample_crystal_structure):
        """Test crystal structure creation."""
        structure = sample_crystal_structure

        assert structure.formula == "C2H2"
        assert structure.space_group == 14
        assert structure.lattice_energy == -100.0
        assert structure.density == 1.2
        assert structure.packing_motif == PackingMotifType.LAYERED
        assert len(structure.atomic_positions) == 4

    def test_density_calculation(self, sample_unit_cell):
        """Test density calculation."""
        positions = [
            AtomicPosition(element="C", x=0.25, y=0.25, z=0.25),
            AtomicPosition(element="C", x=0.75, y=0.75, z=0.75),
        ]

        structure = CrystalStructure(
            unit_cell=sample_unit_cell,
            atomic_positions=positions,
            space_group=1,
            formula="C2",
        )

        # Density should be calculated automatically
        assert structure.density > 0
        assert structure.packing_efficiency is not None
        assert 0 <= structure.packing_efficiency <= 1

    def test_packing_efficiency_calculation(self, sample_crystal_structure):
        """Test packing efficiency calculation."""
        structure = sample_crystal_structure

        assert structure.packing_efficiency is not None
        assert 0 <= structure.packing_efficiency <= 1


class TestCrystalPredictionConfig:
    """Test CrystalPredictionConfig class."""

    def test_config_creation(self, sample_csp_config):
        """Test configuration creation."""
        config = sample_csp_config

        assert config.method == CSPMethod.HYBRID_APPROACH
        assert config.max_structures == 10
        assert config.use_gpu == False
        assert config.energy_cutoff == 25.0
        assert config.temperature == 298.0
        assert config.max_iterations == 100

    def test_default_config(self):
        """Test default configuration values."""
        config = CrystalPredictionConfig()

        assert config.method == CSPMethod.HYBRID_APPROACH
        assert config.max_structures == 100
        assert config.use_gpu == True
        assert config.energy_cutoff == 50.0
        assert config.temperature == 300.0
        assert config.pressure == 1.0


class TestCSPPrediction:
    """Test CSPPrediction class."""

    def test_prediction_creation(self, sample_csp_prediction):
        """Test CSP prediction creation."""
        prediction = sample_csp_prediction

        assert len(prediction.predicted_structures) == 1
        assert len(prediction.energy_landscape) == 1
        assert len(prediction.polymorphs) == 1
        assert prediction.most_stable_structure is not None
        assert prediction.prediction_confidence == 0.85
        assert prediction.computational_cost == 12.5
        assert prediction.method_used == CSPMethod.HYBRID_APPROACH
        assert prediction.convergence_achieved == True


class TestGPUAcceleratedCSPEngine:
    """Test GPUAcceleratedCSPEngine class."""

    def test_engine_creation(self, sample_csp_config):
        """Test CSP engine creation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        assert engine.config == sample_csp_config
        assert engine.use_gpu == False  # Based on config
        assert isinstance(engine.prediction_cache, dict)
        assert isinstance(engine.performance_stats, dict)

    def test_default_engine_creation(self):
        """Test engine creation with default config."""
        engine = GPUAcceleratedCSPEngine()

        assert engine.config is not None
        assert isinstance(engine.config, CrystalPredictionConfig)

    def test_molecular_features_generation(self, sample_csp_config):
        """Test molecular features generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        features = engine._generate_molecular_features("C6H6")

        assert isinstance(features, np.ndarray)
        assert len(features) == 256
        assert features[100] == 2  # Number of different elements (C, H)
        assert features[101] == 12  # Total atoms (6C + 6H)

    def test_molecular_formula_parsing(self, sample_csp_config):
        """Test molecular formula parsing."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Test simple formula
        elements = engine._parse_molecular_formula("C6H6")
        assert elements == {"C": 6, "H": 6}

        # Test complex formula
        elements = engine._parse_molecular_formula("CaCl2")
        assert elements == {"Ca": 1, "Cl": 2}

        # Test formula with no numbers
        elements = engine._parse_molecular_formula("NaCl")
        assert elements == {"Na": 1, "Cl": 1}

    def test_atomic_number_lookup(self, sample_csp_config):
        """Test atomic number lookup."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        assert engine._get_atomic_number("H") == 1
        assert engine._get_atomic_number("C") == 6
        assert engine._get_atomic_number("N") == 7
        assert engine._get_atomic_number("O") == 8
        assert engine._get_atomic_number("Unknown") == 1  # Default

    def test_random_unit_cell_generation(self, sample_csp_config):
        """Test random unit cell generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        unit_cell = engine._generate_random_unit_cell()

        assert isinstance(unit_cell, UnitCellParameters)
        assert 5.0 <= unit_cell.a <= 20.0
        assert 5.0 <= unit_cell.b <= 20.0
        assert 5.0 <= unit_cell.c <= 20.0
        assert 70.0 <= unit_cell.alpha <= 110.0
        assert 70.0 <= unit_cell.beta <= 110.0
        assert 70.0 <= unit_cell.gamma <= 110.0

    def test_atomic_positions_generation(self, sample_csp_config, sample_unit_cell):
        """Test atomic positions generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        positions = engine._generate_atomic_positions("C2H4", sample_unit_cell)

        assert len(positions) == 6  # 2 C + 4 H
        assert all(isinstance(pos, AtomicPosition) for pos in positions)
        assert all(0.05 <= pos.x <= 0.95 for pos in positions)
        assert all(0.05 <= pos.y <= 0.95 for pos in positions)
        assert all(0.05 <= pos.z <= 0.95 for pos in positions)

    def test_lennard_jones_energy_calculation(
        self, sample_csp_config, sample_crystal_structure
    ):
        """Test Lennard-Jones energy calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        energy = engine._calculate_lj_energy(sample_crystal_structure)

        assert isinstance(energy, float)
        # Energy could be positive or negative depending on distances

    def test_coulomb_energy_calculation(
        self, sample_csp_config, sample_crystal_structure
    ):
        """Test Coulomb energy calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        energy = engine._calculate_coulomb_energy(sample_crystal_structure)

        assert isinstance(energy, float)

    def test_structure_energy_calculation(
        self, sample_csp_config, sample_crystal_structure
    ):
        """Test total structure energy calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        energy = engine._calculate_structure_energy(sample_crystal_structure)

        assert isinstance(energy, float)

    @patch("chemi_accel.materials.crystal_structure_prediction._SCIPY_AVAILABLE", True)
    def test_energy_minimization(self, sample_csp_config, sample_crystal_structure):
        """Test lattice energy minimization."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Mock the minimize function to avoid actual optimization
        with patch(
            "chemi_accel.materials.crystal_structure_prediction.minimize"
        ) as mock_minimize:
            mock_result = Mock()
            mock_result.success = True
            mock_result.fun = -120.0
            mock_result.x = [9.5, 11.5, 7.5, 89.0, 91.0, 90.0]
            mock_minimize.return_value = mock_result

            optimized = engine._minimize_lattice_energy(sample_crystal_structure)

            assert isinstance(optimized, CrystalStructure)
            assert optimized.lattice_energy == -120.0
            assert optimized.stability_score is not None

    def test_polymorph_identification(self, sample_csp_config):
        """Test polymorph identification."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create similar structures (polymorphs)
        structure1 = CrystalStructure(
            unit_cell=UnitCellParameters(
                a=10.0, b=12.0, c=8.0, alpha=90, beta=90, gamma=90
            ),
            atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
            space_group=14,
            formula="C",
        )

        structure2 = CrystalStructure(
            unit_cell=UnitCellParameters(
                a=10.5, b=12.5, c=8.5, alpha=90, beta=90, gamma=90
            ),
            atomic_positions=[AtomicPosition(element="C", x=0.3, y=0.3, z=0.3)],
            space_group=15,
            formula="C",
        )

        polymorphs = engine._identify_polymorphs([structure1, structure2])

        assert len(polymorphs) >= 1
        assert all(isinstance(p, CrystalStructure) for p in polymorphs)

    def test_structure_ranking(self, sample_csp_config):
        """Test structure ranking by stability."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create structures with different energies
        structures = []
        for i, energy in enumerate([-100, -80, -120, -90]):
            structure = CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                lattice_energy=energy,
            )
            structures.append(structure)

        ranked = engine._rank_structures_by_stability(structures)

        # Should be sorted by energy (most stable first)
        energies = [s.lattice_energy for s in ranked]
        assert energies == sorted(energies)
        assert energies[0] == -120  # Most stable

    def test_prediction_confidence_calculation(self, sample_csp_config):
        """Test prediction confidence calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create structures with energy gap
        structures = [
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                lattice_energy=-100,
            ),
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                lattice_energy=-80,
            ),
        ]

        confidence = engine._calculate_prediction_confidence(structures)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_structural_features_extraction(
        self, sample_csp_config, sample_crystal_structure
    ):
        """Test structural features extraction."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        features = engine._extract_structural_features(sample_crystal_structure)

        assert isinstance(features, np.ndarray)
        assert len(features) == 1024
        assert features[0] == 10.0  # unit cell a parameter
        assert features[1] == 12.0  # unit cell b parameter
        assert features[2] == 8.0  # unit cell c parameter

    def test_minimum_interatomic_distance(
        self, sample_csp_config, sample_crystal_structure
    ):
        """Test minimum interatomic distance calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        min_dist = engine._calculate_minimum_interatomic_distance(
            sample_crystal_structure
        )

        assert isinstance(min_dist, float)
        assert min_dist > 0

    def test_structure_validation(self, sample_csp_config, sample_crystal_structure):
        """Test structure validation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        validation_report = engine.validate_structure(sample_crystal_structure)

        assert isinstance(validation_report, dict)
        assert "valid" in validation_report
        assert "warnings" in validation_report
        assert "errors" in validation_report
        assert "checks_performed" in validation_report
        assert isinstance(validation_report["valid"], bool)
        assert isinstance(validation_report["warnings"], list)
        assert isinstance(validation_report["errors"], list)
        assert isinstance(validation_report["checks_performed"], list)

    def test_invalid_structure_validation(self, sample_csp_config):
        """Test validation of invalid structure."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create invalid structure with extreme parameters
        invalid_structure = CrystalStructure(
            unit_cell=UnitCellParameters(
                a=0.5, b=0.5, c=0.5, alpha=90, beta=90, gamma=90
            ),  # Too small
            atomic_positions=[],  # No atoms
            space_group=1,
            formula="C",
        )

        validation_report = engine.validate_structure(invalid_structure)

        assert validation_report["valid"] == False
        assert len(validation_report["errors"]) > 0


class TestStructureGeneration:
    """Test structure generation methods."""

    def test_random_structure_generation(self, sample_csp_config):
        """Test random structure generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        structures = engine._generate_structures_random("C6H6", np.zeros(256), None)

        assert len(structures) == sample_csp_config.max_structures
        assert all(isinstance(s, CrystalStructure) for s in structures)
        assert all(s.formula == "C6H6" for s in structures)

    def test_energy_minimization_generation(self, sample_csp_config):
        """Test energy minimization structure generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Mock the minimize function
        with patch.object(engine, "_minimize_lattice_energy", side_effect=lambda x: x):
            structures = engine._generate_structures_energy_min(
                "C2H2", np.zeros(256), None
            )

            assert len(structures) == sample_csp_config.max_structures
            assert all(isinstance(s, CrystalStructure) for s in structures)

    def test_hybrid_structure_generation(self, sample_csp_config):
        """Test hybrid structure generation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Mock individual generation methods
        with patch.object(
            engine, "_generate_structures_gnn", return_value=[Mock()]
        ), patch.object(
            engine, "_generate_structures_generative", return_value=[Mock()]
        ), patch.object(
            engine, "_generate_structures_energy_min", return_value=[Mock()]
        ):
            structures = engine._generate_structures_hybrid("C2H2", np.zeros(256), None)

            assert len(structures) <= sample_csp_config.max_structures
            assert all(structures)  # All should be Mock objects


class TestMotifGeneration:
    """Test packing motif-specific generation."""

    def test_motif_unit_cell_generation(self, sample_csp_config):
        """Test unit cell generation for specific motifs."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Test cubic motif
        cubic_cell = engine._generate_unit_cell_for_motif(PackingMotifType.CLOSE_PACKED)
        assert isinstance(cubic_cell, UnitCellParameters)

        # Test layered motif
        layered_cell = engine._generate_unit_cell_for_motif(PackingMotifType.LAYERED)
        assert isinstance(layered_cell, UnitCellParameters)
        assert layered_cell.c > layered_cell.a  # Layered should have large c parameter

        # Test herringbone motif
        herringbone_cell = engine._generate_unit_cell_for_motif(
            PackingMotifType.HERRINGBONE
        )
        assert isinstance(herringbone_cell, UnitCellParameters)

    def test_motif_positions_generation(self, sample_csp_config, sample_unit_cell):
        """Test atomic positions generation for specific motifs."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Test layered positions
        layered_positions = engine._generate_positions_for_motif(
            "C2H4", sample_unit_cell, PackingMotifType.LAYERED
        )
        assert len(layered_positions) == 6  # 2C + 4H
        assert all(isinstance(pos, AtomicPosition) for pos in layered_positions)

        # Test herringbone positions
        herringbone_positions = engine._generate_positions_for_motif(
            "C2H4", sample_unit_cell, PackingMotifType.HERRINGBONE
        )
        assert len(herringbone_positions) == 6
        assert all(isinstance(pos, AtomicPosition) for pos in herringbone_positions)

    def test_space_group_suggestion(self, sample_csp_config):
        """Test space group suggestion for motifs."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        cubic_sg = engine._suggest_space_group_for_motif(PackingMotifType.CLOSE_PACKED)
        assert isinstance(cubic_sg, int)
        assert 1 <= cubic_sg <= 230

        layered_sg = engine._suggest_space_group_for_motif(PackingMotifType.LAYERED)
        assert isinstance(layered_sg, int)
        assert 1 <= layered_sg <= 230


class TestGeneticAlgorithm:
    """Test genetic algorithm components."""

    def test_tournament_selection(self, sample_csp_config):
        """Test tournament selection."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create mock population
        population = [Mock() for _ in range(10)]
        fitness_scores = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.95]

        selected = engine._tournament_selection(population, fitness_scores)

        assert selected in population

    def test_structure_crossover(self, sample_csp_config):
        """Test structure crossover operation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        parent1 = CrystalStructure(
            unit_cell=UnitCellParameters(
                a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
            ),
            atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
            space_group=1,
            formula="C",
        )

        parent2 = CrystalStructure(
            unit_cell=UnitCellParameters(
                a=12.0, b=12.0, c=12.0, alpha=90, beta=90, gamma=90
            ),
            atomic_positions=[AtomicPosition(element="C", x=0.75, y=0.75, z=0.75)],
            space_group=1,
            formula="C",
        )

        child = engine._crossover_structures(parent1, parent2)

        assert isinstance(child, CrystalStructure)
        assert child.formula == "C"
        # Child parameters should be between parents
        assert 10.0 <= child.unit_cell.a <= 12.0

    def test_structure_mutation(self, sample_csp_config, sample_crystal_structure):
        """Test structure mutation operation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        mutated = engine._mutate_structure(sample_crystal_structure)

        assert isinstance(mutated, CrystalStructure)
        assert mutated.formula == sample_crystal_structure.formula
        # Parameters should be slightly different
        assert (
            mutated.unit_cell.a != sample_crystal_structure.unit_cell.a
            or mutated.unit_cell.b != sample_crystal_structure.unit_cell.b
            or mutated.unit_cell.c != sample_crystal_structure.unit_cell.c
        )


class TestLandscapeAnalysis:
    """Test polymorphic landscape analysis."""

    def test_landscape_exploration(self, sample_csp_config):
        """Test polymorphic landscape exploration."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Mock the prediction method
        mock_structures = [
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                lattice_energy=-100 - i * 5,  # Different energies
            )
            for i in range(5)
        ]

        mock_prediction = CSPPrediction(
            predicted_structures=mock_structures,
            energy_landscape={},
            polymorphs=mock_structures,
            most_stable_structure=mock_structures[0],
            prediction_confidence=0.8,
            computational_cost=10.0,
            method_used=CSPMethod.HYBRID_APPROACH,
            convergence_achieved=True,
        )

        with patch.object(
            engine, "predict_crystal_structure", return_value=mock_prediction
        ):
            landscape = engine.explore_polymorphic_landscape("C", energy_window=20.0)

            assert isinstance(landscape, dict)
            assert "total_structures" in landscape
            assert "structures_in_window" in landscape
            assert "global_minimum_energy" in landscape
            assert "polymorphs" in landscape
            assert "crystal_systems" in landscape
            assert "stability_ranking" in landscape

    def test_polymorph_family_analysis(self, sample_csp_config):
        """Test polymorph family analysis."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Create structures that should be grouped as polymorphs
        structures = [
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                lattice_energy=-100,
            ),
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.1, b=10.1, c=10.1, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.3, y=0.3, z=0.3)],
                space_group=2,
                formula="C",
                lattice_energy=-95,
            ),
        ]

        analysis = engine._analyze_polymorph_families(structures)

        assert isinstance(analysis, dict)
        assert "num_families" in analysis
        assert "family_statistics" in analysis
        assert "total_polymorphs" in analysis

    def test_crystal_system_analysis(self, sample_csp_config):
        """Test crystal system distribution analysis."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        structures = [
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
            ),
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=12.0, c=8.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
            ),
        ]

        analysis = engine._analyze_crystal_systems(structures)

        assert isinstance(analysis, dict)
        assert "cubic" in analysis
        assert "orthorhombic" in analysis
        assert analysis["cubic"] == 1
        assert analysis["orthorhombic"] == 1

    def test_structure_diversity_calculation(self, sample_csp_config):
        """Test structure diversity calculation."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        structures = [
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.25, y=0.25, z=0.25)],
                space_group=1,
                formula="C",
                density=2.0,
            ),
            CrystalStructure(
                unit_cell=UnitCellParameters(
                    a=15.0, b=12.0, c=8.0, alpha=90, beta=90, gamma=90
                ),
                atomic_positions=[AtomicPosition(element="C", x=0.75, y=0.75, z=0.75)],
                space_group=14,
                formula="C",
                density=1.5,
            ),
        ]

        diversity = engine._calculate_structure_diversity(structures)

        assert isinstance(diversity, dict)
        assert "diversity_score" in diversity
        assert "uniqueness_score" in diversity
        assert 0.0 <= diversity["diversity_score"] <= 1.0
        assert 0.0 <= diversity["uniqueness_score"] <= 1.0


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_batch_prediction(self, sample_csp_config):
        """Test batch crystal structure prediction."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        formulas = ["C2H2", "C6H6", "NaCl"]

        # Mock individual predictions
        with patch.object(engine, "predict_crystal_structure") as mock_predict:
            mock_predict.return_value = Mock(spec=CSPPrediction)

            results = engine.batch_csp_prediction(formulas)

            assert len(results) == 3
            assert all(formula in results for formula in formulas)
            assert mock_predict.call_count == 3

    def test_batch_prediction_with_failures(self, sample_csp_config):
        """Test batch prediction with some failures."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        formulas = ["C2H2", "InvalidFormula", "C6H6"]

        def mock_predict(formula, features=None):
            if formula == "InvalidFormula":
                raise CrystalStructurePredictionError("Invalid formula")
            return Mock(spec=CSPPrediction)

        with patch.object(
            engine, "predict_crystal_structure", side_effect=mock_predict
        ):
            results = engine.batch_csp_prediction(formulas)

            assert len(results) == 3
            assert all(formula in results for formula in formulas)
            # Failed prediction should have empty structures
            assert len(results["InvalidFormula"].predicted_structures) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_predict_crystal_structure_function(self):
        """Test predict_crystal_structure utility function."""
        with patch(
            "chemi_accel.materials.crystal_structure_prediction.GPUAcceleratedCSPEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine.predict_crystal_structure.return_value = Mock(
                spec=CSPPrediction
            )
            mock_engine_class.return_value = mock_engine

            result = predict_crystal_structure("C6H6")

            assert result is not None
            mock_engine_class.assert_called_once()
            mock_engine.predict_crystal_structure.assert_called_once_with(
                "C6H6", None, None
            )

    def test_create_csp_config_function(self):
        """Test create_csp_config utility function."""
        config = create_csp_config(
            method=CSPMethod.GENETIC_ALGORITHM,
            max_structures=50,
            use_gpu=False,
            energy_cutoff=30.0,
        )

        assert isinstance(config, CrystalPredictionConfig)
        assert config.method == CSPMethod.GENETIC_ALGORITHM
        assert config.max_structures == 50
        assert config.use_gpu == False
        assert config.energy_cutoff == 30.0

    def test_analyze_polymorphs_function(self, sample_crystal_structure):
        """Test analyze_polymorphs utility function."""
        structures = [sample_crystal_structure] * 3  # Same structure repeated

        polymorph_groups = analyze_polymorphs(structures, tolerance=0.1)

        assert isinstance(polymorph_groups, list)
        assert len(polymorph_groups) >= 1
        assert all(isinstance(group, list) for group in polymorph_groups)


class TestNeuralNetworks:
    """Test neural network components (if available)."""

    @pytest.mark.skipif(
        not hasattr(CrystalGraphConv, "__bases__")
        or CrystalGraphConv.__bases__ == (object,),
        reason="PyTorch not available",
    )
    def test_crystal_graph_conv_creation(self):
        """Test CrystalGraphConv creation."""
        try:
            conv = CrystalGraphConv(in_channels=92, out_channels=128)
            assert conv.in_channels == 92
            assert conv.out_channels == 128
        except ImportError:
            pytest.skip("PyTorch not available")

    @pytest.mark.skipif(
        not hasattr(PackingMotifGNN, "__bases__")
        or PackingMotifGNN.__bases__ == (object,),
        reason="PyTorch not available",
    )
    def test_packing_motif_gnn_creation(self):
        """Test PackingMotifGNN creation."""
        try:
            gnn = PackingMotifGNN(
                node_features=92,
                hidden_dim=128,
                num_layers=3,
                num_motif_types=len(PackingMotifType),
            )
            assert hasattr(gnn, "atom_embedding")
            assert hasattr(gnn, "conv_layers")
            assert hasattr(gnn, "classifier")
        except ImportError:
            pytest.skip("PyTorch not available")

    @pytest.mark.skipif(
        not hasattr(UnitCellGenerator, "__bases__")
        or UnitCellGenerator.__bases__ == (object,),
        reason="PyTorch not available",
    )
    def test_unit_cell_generator_creation(self):
        """Test UnitCellGenerator creation."""
        try:
            generator = UnitCellGenerator(
                molecular_features=256, hidden_dim=512, latent_dim=128
            )
            assert hasattr(generator, "encoder")
            assert hasattr(generator, "decoder")
            assert hasattr(generator, "crystal_system_classifier")
        except ImportError:
            pytest.skip("PyTorch not available")

    @pytest.mark.skipif(
        not hasattr(LatticeEnergyPredictor, "__bases__")
        or LatticeEnergyPredictor.__bases__ == (object,),
        reason="PyTorch not available",
    )
    def test_lattice_energy_predictor_creation(self):
        """Test LatticeEnergyPredictor creation."""
        try:
            predictor = LatticeEnergyPredictor(
                structural_features=1024, hidden_dim=512, num_layers=4
            )
            assert hasattr(predictor, "network")
            assert hasattr(predictor, "uncertainty_head")
        except ImportError:
            pytest.skip("PyTorch not available")


class TestExceptions:
    """Test exception classes."""

    def test_crystal_structure_prediction_error(self):
        """Test CrystalStructurePredictionError."""
        with pytest.raises(CrystalStructurePredictionError):
            raise CrystalStructurePredictionError("Test error")

    def test_lattice_energy_error(self):
        """Test LatticeEnergyError."""
        with pytest.raises(LatticeEnergyError):
            raise LatticeEnergyError("Energy calculation failed")

    def test_structure_generation_error(self):
        """Test StructureGenerationError."""
        with pytest.raises(StructureGenerationError):
            raise StructureGenerationError("Structure generation failed")

    def test_symmetry_error(self):
        """Test SymmetryError."""
        with pytest.raises(SymmetryError):
            raise SymmetryError("Symmetry handling failed")

    def test_convergence_error(self):
        """Test ConvergenceError."""
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Optimization did not converge")


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_prediction(self, sample_csp_config):
        """Test complete prediction workflow."""
        # Use small config for faster testing
        test_config = CrystalPredictionConfig(
            method=CSPMethod.RANDOM_SEARCH,
            max_structures=3,
            use_gpu=False,
            max_iterations=10,
        )

        engine = GPUAcceleratedCSPEngine(test_config)

        try:
            prediction = engine.predict_crystal_structure("C2")

            assert isinstance(prediction, CSPPrediction)
            assert len(prediction.predicted_structures) <= test_config.max_structures
            assert all(
                isinstance(s, CrystalStructure) for s in prediction.predicted_structures
            )
            assert all(s.formula == "C2" for s in prediction.predicted_structures)
            assert prediction.method_used == CSPMethod.RANDOM_SEARCH

        except Exception as e:
            # Integration test might fail due to missing dependencies
            # but shouldn't crash with unhandled exceptions
            assert isinstance(e, (CrystalStructurePredictionError, ImportError))

    def test_prediction_with_caching(self, sample_csp_config):
        """Test prediction caching mechanism."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        # Mock the actual prediction to avoid computation
        mock_prediction = Mock(spec=CSPPrediction)

        with patch.object(
            engine, "_generate_candidate_structures", return_value=[]
        ), patch.object(engine, "_optimize_structures", return_value=[]), patch.object(
            engine, "_identify_polymorphs", return_value=[]
        ), patch.object(
            engine, "_rank_structures_by_stability", return_value=[]
        ), patch.object(
            engine, "_calculate_prediction_confidence", return_value=0.8
        ):
            # First prediction should be computed
            prediction1 = engine.predict_crystal_structure("C6H6")
            cache_size_1 = len(engine.prediction_cache)

            # Second prediction with same input should use cache
            prediction2 = engine.predict_crystal_structure("C6H6")
            cache_size_2 = len(engine.prediction_cache)

            assert cache_size_1 == cache_size_2  # Cache size shouldn't increase
            # Both predictions should be the same object (from cache)
            assert prediction1 is prediction2

    def test_performance_tracking(self, sample_csp_config):
        """Test performance statistics tracking."""
        engine = GPUAcceleratedCSPEngine(sample_csp_config)

        initial_total = engine.performance_stats["total_predictions"]
        initial_successful = engine.performance_stats["successful_predictions"]

        # Mock prediction to succeed
        with patch.object(
            engine, "_generate_candidate_structures", return_value=[]
        ), patch.object(engine, "_optimize_structures", return_value=[]), patch.object(
            engine, "_identify_polymorphs", return_value=[]
        ), patch.object(
            engine, "_rank_structures_by_stability", return_value=[]
        ), patch.object(
            engine, "_calculate_prediction_confidence", return_value=0.8
        ):
            try:
                engine.predict_crystal_structure("C2H2")

                assert (
                    engine.performance_stats["total_predictions"] == initial_total + 1
                )
                assert (
                    engine.performance_stats["successful_predictions"]
                    == initial_successful + 1
                )

            except:
                # If prediction fails, only total should increase
                assert (
                    engine.performance_stats["total_predictions"] == initial_total + 1
                )
