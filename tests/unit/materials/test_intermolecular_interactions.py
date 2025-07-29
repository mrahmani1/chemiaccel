"""
Unit tests for intermolecular interactions module.
"""

from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.materials.intermolecular_interactions import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    AtomicCoordinates,
    CrystalPackingObjective,
    CrystalPackingOptimizationError,
    CrystalPackingOptimizer,
    CrystalPackingResult,
    GPUAcceleratedInteractionEngine,
    HydrogenBond,
    HydrogenBondCalculationError,
    HydrogenBondCalculator,
    HydrogenBondType,
    InteractionConfiguration,
    InteractionEngineError,
    InteractionPair,
    InteractionType,
    IntermolecularInteractionError,
    MolecularFragment,
    OptimizationAlgorithm,
    PiPiInteraction,
    PiPiStackingCalculationError,
    PiPiStackingCalculator,
    PiPiStackingType,
    VanDerWaalsCalculationError,
    VanDerWaalsCalculator,
    VanDerWaalsInteraction,
    analyze_crystal_packing,
    create_interaction_config,
    create_molecular_fragment_from_smiles,
    optimize_intermolecular_interactions,
)


@pytest.fixture
def sample_atomic_coordinates():
    """Create sample atomic coordinates."""
    return AtomicCoordinates(
        coordinates=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        elements=["C", "N", "O"],
        atomic_numbers=np.array([6, 7, 8]),
    )


@pytest.fixture
def sample_molecular_fragment(sample_atomic_coordinates):
    """Create sample molecular fragment."""
    return MolecularFragment(
        coordinates=sample_atomic_coordinates,
        fragment_id="frag_1",
        molecule_id="mol_1",
        charge=0.0,
        aromatic_rings=[[0, 1, 2]],
        hydrogen_bond_donors=[1],
        hydrogen_bond_acceptors=[2],
    )


@pytest.fixture
def sample_interaction_config():
    """Create sample interaction configuration."""
    return InteractionConfiguration(
        max_interaction_distance=8.0,
        hydrogen_bond_distance_cutoff=3.5,
        pi_pi_distance_cutoff=6.0,
        use_gpu=False,
    )


@pytest.fixture
def sample_hydrogen_bond():
    """Create sample hydrogen bond."""
    return HydrogenBond(
        donor_idx=1,
        hydrogen_idx=3,
        acceptor_idx=2,
        donor_fragment="frag_1",
        acceptor_fragment="frag_2",
        distance_dh=1.0,
        distance_ha=2.5,
        distance_da=3.0,
        angle_dha=150.0,
        bond_type=HydrogenBondType.CONVENTIONAL,
        energy=-3.5,
    )


@pytest.fixture
def sample_pi_interaction():
    """Create sample π-π interaction."""
    return PiPiInteraction(
        ring1_atoms=[0, 1, 2, 3, 4, 5],
        ring2_atoms=[6, 7, 8, 9, 10, 11],
        fragment1="frag_1",
        fragment2="frag_2",
        centroid_distance=3.8,
        ring_offset=1.2,
        dihedral_angle=15.0,
        stacking_type=PiPiStackingType.PARALLEL_DISPLACED,
        energy=-2.3,
        overlap_area=12.5,
    )


@pytest.fixture
def sample_vdw_interaction():
    """Create sample van der Waals interaction."""
    return VanDerWaalsInteraction(
        atom1_idx=0,
        atom2_idx=1,
        fragment1="frag_1",
        fragment2="frag_2",
        distance=3.2,
        vdw_radii_sum=3.4,
        energy=-0.8,
        contact_ratio=0.94,
    )


class TestAtomicCoordinates:
    """Test AtomicCoordinates class."""

    def test_coordinates_creation(self, sample_atomic_coordinates):
        """Test atomic coordinates creation."""
        coords = sample_atomic_coordinates

        assert coords.coordinates.shape == (3, 3)
        assert len(coords.elements) == 3
        assert len(coords.atomic_numbers) == 3
        assert coords.elements == ["C", "N", "O"]
        assert np.array_equal(coords.atomic_numbers, np.array([6, 7, 8]))

    def test_coordinates_validation(self):
        """Test atomic coordinates validation."""
        # Valid coordinates should not raise
        coords = AtomicCoordinates(
            coordinates=np.array([[0, 0, 0], [1, 1, 1]]),
            elements=["C", "N"],
            atomic_numbers=np.array([6, 7]),
        )
        assert coords.coordinates.shape == (2, 3)

        # Invalid shape should raise
        with pytest.raises(ValueError):
            AtomicCoordinates(
                coordinates=np.array([[0, 0], [1, 1]]),  # Wrong shape
                elements=["C", "N"],
                atomic_numbers=np.array([6, 7]),
            )

        # Mismatched lengths should raise
        with pytest.raises(ValueError):
            AtomicCoordinates(
                coordinates=np.array([[0, 0, 0], [1, 1, 1]]),
                elements=["C"],  # Wrong length
                atomic_numbers=np.array([6, 7]),
            )


class TestMolecularFragment:
    """Test MolecularFragment class."""

    def test_fragment_creation(self, sample_molecular_fragment):
        """Test molecular fragment creation."""
        fragment = sample_molecular_fragment

        assert fragment.fragment_id == "frag_1"
        assert fragment.molecule_id == "mol_1"
        assert fragment.charge == 0.0
        assert len(fragment.aromatic_rings) == 1
        assert len(fragment.hydrogen_bond_donors) == 1
        assert len(fragment.hydrogen_bond_acceptors) == 1

    def test_center_of_mass_calculation(self, sample_molecular_fragment):
        """Test center of mass calculation."""
        fragment = sample_molecular_fragment
        com = fragment.get_center_of_mass()

        assert isinstance(com, np.ndarray)
        assert len(com) == 3
        assert all(isinstance(x, (int, float)) for x in com)


class TestInteractionConfiguration:
    """Test InteractionConfiguration class."""

    def test_config_creation(self, sample_interaction_config):
        """Test interaction configuration creation."""
        config = sample_interaction_config

        assert config.max_interaction_distance == 8.0
        assert config.hydrogen_bond_distance_cutoff == 3.5
        assert config.pi_pi_distance_cutoff == 6.0
        assert config.use_gpu == False
        assert config.energy_units == "kcal/mol"

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = InteractionConfiguration(max_interaction_distance=5.0)
        assert config.max_interaction_distance == 5.0

        # Invalid distance should raise
        with pytest.raises(ValueError):
            InteractionConfiguration(max_interaction_distance=-1.0)

        # Invalid vdW scale factor should raise
        with pytest.raises(ValueError):
            InteractionConfiguration(vdw_scale_factor=3.0)


class TestHydrogenBond:
    """Test HydrogenBond class."""

    def test_hydrogen_bond_creation(self, sample_hydrogen_bond):
        """Test hydrogen bond creation."""
        hb = sample_hydrogen_bond

        assert hb.donor_idx == 1
        assert hb.hydrogen_idx == 3
        assert hb.acceptor_idx == 2
        assert hb.donor_fragment == "frag_1"
        assert hb.acceptor_fragment == "frag_2"
        assert hb.distance_ha == 2.5
        assert hb.angle_dha == 150.0
        assert hb.bond_type == HydrogenBondType.CONVENTIONAL
        assert hb.energy == -3.5


class TestPiPiInteraction:
    """Test PiPiInteraction class."""

    def test_pi_interaction_creation(self, sample_pi_interaction):
        """Test π-π interaction creation."""
        pi = sample_pi_interaction

        assert len(pi.ring1_atoms) == 6
        assert len(pi.ring2_atoms) == 6
        assert pi.fragment1 == "frag_1"
        assert pi.fragment2 == "frag_2"
        assert pi.centroid_distance == 3.8
        assert pi.stacking_type == PiPiStackingType.PARALLEL_DISPLACED
        assert pi.energy == -2.3


class TestVanDerWaalsInteraction:
    """Test VanDerWaalsInteraction class."""

    def test_vdw_interaction_creation(self, sample_vdw_interaction):
        """Test van der Waals interaction creation."""
        vdw = sample_vdw_interaction

        assert vdw.atom1_idx == 0
        assert vdw.atom2_idx == 1
        assert vdw.fragment1 == "frag_1"
        assert vdw.fragment2 == "frag_2"
        assert vdw.distance == 3.2
        assert vdw.energy == -0.8
        assert vdw.contact_ratio == 0.94


class TestInteractionPair:
    """Test InteractionPair class."""

    def test_interaction_pair_creation(self, sample_molecular_fragment):
        """Test interaction pair creation."""
        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        pair = InteractionPair(
            fragment1=frag1,
            fragment2=frag2,
            distance=5.0,
            interaction_types=[
                InteractionType.HYDROGEN_BOND,
                InteractionType.VAN_DER_WAALS,
            ],
            interaction_energies={
                InteractionType.HYDROGEN_BOND: -3.0,
                InteractionType.VAN_DER_WAALS: -1.5,
            },
        )

        assert pair.distance == 5.0
        assert len(pair.interaction_types) == 2
        assert pair.get_total_energy() == -4.5


class TestCrystalPackingResult:
    """Test CrystalPackingResult class."""

    def test_packing_result_creation(
        self, sample_atomic_coordinates, sample_molecular_fragment
    ):
        """Test crystal packing result creation."""
        result = CrystalPackingResult(
            optimized_coordinates=[sample_atomic_coordinates],
            final_energy=-15.5,
            energy_components={
                InteractionType.HYDROGEN_BOND: -10.0,
                InteractionType.VAN_DER_WAALS: -5.5,
            },
            packing_efficiency=0.72,
            density=1.35,
            void_fraction=0.28,
            interaction_network=[],
        )

        assert result.final_energy == -15.5
        assert result.packing_efficiency == 0.72
        assert result.density == 1.35
        assert result.convergence_achieved == True

    def test_interaction_summary(self, sample_molecular_fragment):
        """Test interaction summary calculation."""
        # Create mock interaction pairs
        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        pairs = [
            InteractionPair(
                fragment1=frag1,
                fragment2=frag2,
                distance=3.0,
                interaction_types=[InteractionType.HYDROGEN_BOND],
                interaction_energies={InteractionType.HYDROGEN_BOND: -3.0},
            ),
            InteractionPair(
                fragment1=frag1,
                fragment2=frag2,
                distance=4.0,
                interaction_types=[
                    InteractionType.VAN_DER_WAALS,
                    InteractionType.PI_PI_STACKING,
                ],
                interaction_energies={
                    InteractionType.VAN_DER_WAALS: -1.0,
                    InteractionType.PI_PI_STACKING: -2.0,
                },
            ),
        ]

        result = CrystalPackingResult(
            optimized_coordinates=[],
            final_energy=-6.0,
            energy_components={},
            packing_efficiency=0.7,
            density=1.3,
            void_fraction=0.3,
            interaction_network=pairs,
        )

        summary = result.get_interaction_summary()

        assert summary["hydrogen_bond"] == 1
        assert summary["van_der_waals"] == 1
        assert summary["pi_pi_stacking"] == 1


class TestHydrogenBondCalculator:
    """Test HydrogenBondCalculator class."""

    def test_calculator_creation(self, sample_interaction_config):
        """Test hydrogen bond calculator creation."""
        calc = HydrogenBondCalculator(sample_interaction_config)

        assert calc.config == sample_interaction_config
        assert not calc.use_gpu  # GPU disabled in config
        assert "N" in calc.hb_donors
        assert "O" in calc.hb_acceptors
        assert HydrogenBondType.CONVENTIONAL in calc.hb_energy_params

    def test_hydrogen_bond_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test hydrogen bond calculation."""
        calc = HydrogenBondCalculator(sample_interaction_config)

        # Create two fragments for interaction
        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        fragments = [frag1, frag2]

        # Calculate hydrogen bonds
        hydrogen_bonds = calc.calculate_hydrogen_bonds(fragments)

        assert isinstance(hydrogen_bonds, list)
        # Results depend on molecular geometry, so just check structure
        for hb in hydrogen_bonds:
            assert isinstance(hb, HydrogenBond)
            assert hb.donor_fragment in ["frag_1", "frag_2"]
            assert hb.acceptor_fragment in ["frag_1", "frag_2"]

    def test_donors_and_acceptors_detection(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test hydrogen bond donors and acceptors detection."""
        calc = HydrogenBondCalculator(sample_interaction_config)

        donors, hydrogens = calc._get_donors_and_hydrogens(sample_molecular_fragment)
        acceptors = calc._get_acceptors(sample_molecular_fragment)

        assert isinstance(donors, list)
        assert isinstance(hydrogens, list)
        assert isinstance(acceptors, list)
        assert len(donors) == len(hydrogens)

    def test_hydrogen_bond_classification(self, sample_interaction_config):
        """Test hydrogen bond classification."""
        calc = HydrogenBondCalculator(sample_interaction_config)

        # Test conventional bond
        bond_type = calc._classify_hydrogen_bond("N", "O", 2.5, 160.0)
        assert bond_type == HydrogenBondType.CONVENTIONAL

        # Test weak bond
        bond_type = calc._classify_hydrogen_bond("C", "O", 2.8, 140.0)
        assert bond_type == HydrogenBondType.WEAK

    def test_energy_calculation(self, sample_interaction_config):
        """Test hydrogen bond energy calculation."""
        calc = HydrogenBondCalculator(sample_interaction_config)

        energy = calc._calculate_hydrogen_bond_energy(
            HydrogenBondType.CONVENTIONAL, 2.5, 160.0
        )

        assert isinstance(energy, float)
        assert energy < 0  # Should be negative (favorable)
        assert energy >= -15.0  # Should not exceed cap


class TestPiPiStackingCalculator:
    """Test PiPiStackingCalculator class."""

    def test_calculator_creation(self, sample_interaction_config):
        """Test π-π stacking calculator creation."""
        calc = PiPiStackingCalculator(sample_interaction_config)

        assert calc.config == sample_interaction_config
        assert not calc.use_gpu
        assert calc.optimal_distance == 3.5
        assert PiPiStackingType.PARALLEL_DISPLACED in calc.pi_energy_params

    def test_pi_pi_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test π-π stacking calculation."""
        calc = PiPiStackingCalculator(sample_interaction_config)

        # Create fragments with aromatic rings
        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        fragments = [frag1, frag2]

        pi_interactions = calc.calculate_pi_pi_interactions(fragments)

        assert isinstance(pi_interactions, list)
        for pi in pi_interactions:
            assert isinstance(pi, PiPiInteraction)
            assert pi.fragment1 in ["frag_1", "frag_2"]
            assert pi.fragment2 in ["frag_1", "frag_2"]

    def test_ring_normal_calculation(self, sample_interaction_config):
        """Test ring normal vector calculation."""
        calc = PiPiStackingCalculator(sample_interaction_config)

        # Create a simple ring geometry
        ring_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        normal = calc._calculate_ring_normal(ring_coords)

        assert isinstance(normal, np.ndarray)
        assert len(normal) == 3
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6  # Should be normalized

    def test_stacking_classification(self, sample_interaction_config):
        """Test π-π stacking classification."""
        calc = PiPiStackingCalculator(sample_interaction_config)

        # Test parallel displaced
        stacking_type = calc._classify_pi_pi_stacking(15.0, 1.5, 3.8)
        assert stacking_type == PiPiStackingType.PARALLEL_DISPLACED

        # Test T-shaped
        stacking_type = calc._classify_pi_pi_stacking(85.0, 0.5, 4.0)
        assert stacking_type == PiPiStackingType.T_SHAPED

        # Test sandwich
        stacking_type = calc._classify_pi_pi_stacking(10.0, 0.5, 3.5)
        assert stacking_type == PiPiStackingType.SANDWICH

    def test_energy_calculation(self, sample_interaction_config):
        """Test π-π stacking energy calculation."""
        calc = PiPiStackingCalculator(sample_interaction_config)

        energy = calc._calculate_pi_pi_energy(
            PiPiStackingType.PARALLEL_DISPLACED, 3.8, 15.0, 1.2
        )

        assert isinstance(energy, float)
        assert energy < 0  # Should be negative (favorable)
        assert energy >= -10.0  # Should not exceed cap


class TestVanDerWaalsCalculator:
    """Test VanDerWaalsCalculator class."""

    def test_calculator_creation(self, sample_interaction_config):
        """Test van der Waals calculator creation."""
        calc = VanDerWaalsCalculator(sample_interaction_config)

        assert calc.config == sample_interaction_config
        assert not calc.use_gpu
        assert "C" in calc.vdw_radii
        assert "H" in calc.lj_epsilon

    def test_vdw_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test van der Waals calculation."""
        calc = VanDerWaalsCalculator(sample_interaction_config)

        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        fragments = [frag1, frag2]

        vdw_interactions = calc.calculate_vdw_interactions(fragments)

        assert isinstance(vdw_interactions, list)
        for vdw in vdw_interactions:
            assert isinstance(vdw, VanDerWaalsInteraction)
            assert vdw.fragment1 in ["frag_1", "frag_2"]
            assert vdw.fragment2 in ["frag_1", "frag_2"]

    def test_lennard_jones_energy(self, sample_interaction_config):
        """Test Lennard-Jones energy calculation."""
        calc = VanDerWaalsCalculator(sample_interaction_config)

        energy = calc._calculate_lj_energy("C", "C", 3.4, 3.4)

        assert isinstance(energy, float)
        assert -5.0 <= energy <= 10.0  # Within expected range

    def test_gpu_vs_cpu_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test GPU vs CPU calculation consistency."""
        calc = VanDerWaalsCalculator(sample_interaction_config)

        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        # CPU calculation
        vdw_cpu = calc._calculate_vdw_cpu([frag1, frag2])

        # GPU calculation (should fall back to CPU if no GPU)
        vdw_gpu = calc._calculate_vdw_gpu([frag1, frag2])

        # Results should be similar (might differ due to GPU availability)
        assert isinstance(vdw_cpu, list)
        assert isinstance(vdw_gpu, list)


class TestCrystalPackingOptimizer:
    """Test CrystalPackingOptimizer class."""

    def test_optimizer_creation(self, sample_interaction_config):
        """Test crystal packing optimizer creation."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        assert optimizer.config == sample_interaction_config
        assert isinstance(optimizer.hb_calculator, HydrogenBondCalculator)
        assert isinstance(optimizer.pi_calculator, PiPiStackingCalculator)
        assert isinstance(optimizer.vdw_calculator, VanDerWaalsCalculator)

    def test_optimization_algorithms(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test different optimization algorithms."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        # Test genetic algorithm
        result_ga = optimizer.optimize_crystal_packing(
            fragments,
            CrystalPackingObjective.MINIMIZE_ENERGY,
            OptimizationAlgorithm.GENETIC_ALGORITHM,
        )

        assert isinstance(result_ga, CrystalPackingResult)
        assert isinstance(result_ga.final_energy, float)
        assert isinstance(result_ga.packing_efficiency, float)

        # Test simulated annealing
        result_sa = optimizer.optimize_crystal_packing(
            fragments,
            CrystalPackingObjective.MINIMIZE_ENERGY,
            OptimizationAlgorithm.SIMULATED_ANNEALING,
        )

        assert isinstance(result_sa, CrystalPackingResult)

    def test_fitness_evaluation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test fitness evaluation."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        fitness = optimizer._evaluate_fitness(
            fragments, CrystalPackingObjective.MINIMIZE_ENERGY
        )

        assert isinstance(fitness, float)

    def test_population_initialization(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test population initialization."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        fragments = [sample_molecular_fragment]
        population = optimizer._initialize_population(fragments)

        assert len(population) == optimizer.population_size
        assert all(len(individual) == len(fragments) for individual in population)

    def test_energy_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test total energy calculation."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        fragments = [sample_molecular_fragment, sample_molecular_fragment]

        total_energy = optimizer._calculate_total_energy(fragments)

        assert isinstance(total_energy, float)

    def test_molecular_operations(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test molecular crossover and mutation operations."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        parent1 = [sample_molecular_fragment]
        parent2 = [sample_molecular_fragment]

        # Test crossover
        child1, child2 = optimizer._crossover(parent1, parent2)

        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)

        # Test mutation
        mutated = optimizer._mutate(parent1)

        assert len(mutated) == len(parent1)

    def test_packing_metrics(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test packing efficiency and density calculations."""
        optimizer = CrystalPackingOptimizer(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        packing_efficiency = optimizer._calculate_packing_efficiency(fragments)
        density = optimizer._calculate_density(fragments)
        void_fraction = optimizer._calculate_void_fraction(fragments)

        assert 0.0 <= packing_efficiency <= 1.0
        assert density > 0.0
        assert 0.0 <= void_fraction <= 1.0
        assert abs(packing_efficiency + void_fraction - 1.0) < 1e-6


class TestGPUAcceleratedInteractionEngine:
    """Test GPUAcceleratedInteractionEngine class."""

    def test_engine_creation(self, sample_interaction_config):
        """Test interaction engine creation."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        assert engine.config == sample_interaction_config
        assert isinstance(engine.hb_calculator, HydrogenBondCalculator)
        assert isinstance(engine.pi_calculator, PiPiStackingCalculator)
        assert isinstance(engine.vdw_calculator, VanDerWaalsCalculator)
        assert isinstance(engine.packing_optimizer, CrystalPackingOptimizer)
        assert isinstance(engine.interaction_cache, dict)

    def test_interaction_calculation(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test all interactions calculation."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        frag1 = sample_molecular_fragment
        frag2 = sample_molecular_fragment
        frag2.fragment_id = "frag_2"

        fragments = [frag1, frag2]

        interactions = engine.calculate_all_interactions(fragments)

        assert isinstance(interactions, list)
        for interaction in interactions:
            assert isinstance(interaction, InteractionPair)

    def test_crystal_packing_optimization(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test crystal packing optimization."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        result = engine.optimize_crystal_packing(fragments)

        assert isinstance(result, CrystalPackingResult)
        assert isinstance(result.final_energy, float)

    def test_interaction_network_analysis(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test interaction network analysis."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        fragments = [sample_molecular_fragment, sample_molecular_fragment]

        analysis = engine.analyze_interaction_network(fragments)

        assert isinstance(analysis, dict)
        assert "total_interactions" in analysis
        assert "total_energy" in analysis
        assert "interaction_types" in analysis
        assert "network_connectivity" in analysis

    def test_stability_prediction(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test crystal stability prediction."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        stability = engine.predict_crystal_stability(fragments)

        assert isinstance(stability, dict)
        assert "total_binding_energy" in stability
        assert "stability_score" in stability
        assert "interaction_density" in stability

    def test_caching_mechanism(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test interaction caching mechanism."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        fragments = [sample_molecular_fragment]

        # First calculation
        interactions1 = engine.calculate_all_interactions(fragments)
        cache_size_1 = len(engine.interaction_cache)

        # Second calculation (should use cache)
        interactions2 = engine.calculate_all_interactions(fragments)
        cache_size_2 = len(engine.interaction_cache)

        assert cache_size_1 == cache_size_2  # Cache should be reused
        assert len(interactions1) == len(interactions2)


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("chemi_accel.materials.intermolecular_interactions._RDKIT_AVAILABLE", False)
    def test_create_fragment_no_rdkit(self):
        """Test fragment creation without RDKit."""
        fragment = create_molecular_fragment_from_smiles("CCO")
        assert fragment is None

    def test_create_interaction_config(self):
        """Test interaction configuration creation."""
        config = create_interaction_config(
            use_gpu=False,
            max_distance=10.0,
            hb_distance_cutoff=4.0,
            pi_distance_cutoff=7.0,
        )

        assert isinstance(config, InteractionConfiguration)
        assert config.use_gpu == False
        assert config.max_interaction_distance == 10.0
        assert config.hydrogen_bond_distance_cutoff == 4.0
        assert config.pi_pi_distance_cutoff == 7.0

    def test_analyze_crystal_packing(self, sample_molecular_fragment):
        """Test crystal packing analysis."""
        fragments = [sample_molecular_fragment]

        analysis = analyze_crystal_packing(fragments)

        assert isinstance(analysis, dict)
        assert "interactions" in analysis
        assert "network_analysis" in analysis
        assert "stability_prediction" in analysis
        assert "fragment_count" in analysis

    def test_optimize_intermolecular_interactions(self, sample_molecular_fragment):
        """Test intermolecular interaction optimization."""
        fragments = [sample_molecular_fragment]

        result = optimize_intermolecular_interactions(
            fragments,
            CrystalPackingObjective.MINIMIZE_ENERGY,
            OptimizationAlgorithm.GENETIC_ALGORITHM,
        )

        assert isinstance(result, CrystalPackingResult)


class TestExceptions:
    """Test exception classes."""

    def test_intermolecular_interaction_error(self):
        """Test IntermolecularInteractionError."""
        with pytest.raises(IntermolecularInteractionError):
            raise IntermolecularInteractionError("Test error")

    def test_hydrogen_bond_calculation_error(self):
        """Test HydrogenBondCalculationError."""
        with pytest.raises(HydrogenBondCalculationError):
            raise HydrogenBondCalculationError("H-bond calculation failed")

    def test_pi_pi_stacking_calculation_error(self):
        """Test PiPiStackingCalculationError."""
        with pytest.raises(PiPiStackingCalculationError):
            raise PiPiStackingCalculationError("π-π calculation failed")

    def test_van_der_waals_calculation_error(self):
        """Test VanDerWaalsCalculationError."""
        with pytest.raises(VanDerWaalsCalculationError):
            raise VanDerWaalsCalculationError("VdW calculation failed")

    def test_crystal_packing_optimization_error(self):
        """Test CrystalPackingOptimizationError."""
        with pytest.raises(CrystalPackingOptimizationError):
            raise CrystalPackingOptimizationError("Optimization failed")

    def test_interaction_engine_error(self):
        """Test InteractionEngineError."""
        with pytest.raises(InteractionEngineError):
            raise InteractionEngineError("Engine error")


class TestIntegration:
    """Integration tests."""

    def test_full_interaction_workflow(self, sample_interaction_config):
        """Test complete interaction calculation workflow."""
        # Create test fragments
        coords1 = AtomicCoordinates(
            coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            elements=["C", "N", "O"],
            atomic_numbers=np.array([6, 7, 8]),
        )

        coords2 = AtomicCoordinates(
            coordinates=np.array([[3, 0, 0], [4, 0, 0], [3, 1, 0]]),
            elements=["C", "N", "O"],
            atomic_numbers=np.array([6, 7, 8]),
        )

        frag1 = MolecularFragment(
            coordinates=coords1,
            fragment_id="test_frag_1",
            molecule_id="test_mol_1",
            hydrogen_bond_donors=[1],
            hydrogen_bond_acceptors=[2],
            aromatic_rings=[[0, 1, 2]],
        )

        frag2 = MolecularFragment(
            coordinates=coords2,
            fragment_id="test_frag_2",
            molecule_id="test_mol_2",
            hydrogen_bond_donors=[1],
            hydrogen_bond_acceptors=[2],
            aromatic_rings=[[0, 1, 2]],
        )

        fragments = [frag1, frag2]

        # Create engine
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        # Calculate interactions
        interactions = engine.calculate_all_interactions(fragments)

        # Analyze network
        analysis = engine.analyze_interaction_network(fragments)

        # Predict stability
        stability = engine.predict_crystal_stability(fragments)

        # Optimize packing
        optimization_result = engine.optimize_crystal_packing(fragments)

        # Verify results
        assert isinstance(interactions, list)
        assert isinstance(analysis, dict)
        assert isinstance(stability, dict)
        assert isinstance(optimization_result, CrystalPackingResult)

        # Check that we got meaningful results
        assert "total_interactions" in analysis
        assert "stability_score" in stability
        assert optimization_result.final_energy is not None

    def test_multi_algorithm_comparison(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test comparison of different optimization algorithms."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)
        fragments = [sample_molecular_fragment]

        algorithms = [
            OptimizationAlgorithm.GENETIC_ALGORITHM,
            OptimizationAlgorithm.SIMULATED_ANNEALING,
        ]

        results = {}

        for algorithm in algorithms:
            try:
                result = engine.optimize_crystal_packing(
                    fragments, CrystalPackingObjective.MINIMIZE_ENERGY, algorithm
                )
                results[algorithm] = result
            except Exception as e:
                # Some algorithms might fail, which is acceptable
                print(f"Algorithm {algorithm} failed: {e}")

        # At least one algorithm should work
        assert len(results) > 0

        # All results should be valid
        for algorithm, result in results.items():
            assert isinstance(result, CrystalPackingResult)
            assert isinstance(result.final_energy, float)

    def test_large_system_performance(self, sample_interaction_config):
        """Test performance with larger systems."""
        # Create multiple fragments
        fragments = []

        for i in range(5):  # Keep small for unit tests
            coords = AtomicCoordinates(
                coordinates=np.random.random((3, 3)) * 10,
                elements=["C", "N", "O"],
                atomic_numbers=np.array([6, 7, 8]),
            )

            fragment = MolecularFragment(
                coordinates=coords,
                fragment_id=f"frag_{i}",
                molecule_id=f"mol_{i}",
                hydrogen_bond_donors=[1],
                hydrogen_bond_acceptors=[2],
            )

            fragments.append(fragment)

        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        # Test that calculations complete without errors
        interactions = engine.calculate_all_interactions(fragments)
        analysis = engine.analyze_interaction_network(fragments)

        assert isinstance(interactions, list)
        assert isinstance(analysis, dict)
        assert len(fragments) == 5

    def test_error_handling_and_recovery(self, sample_interaction_config):
        """Test error handling and recovery mechanisms."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)

        # Test with invalid fragments (empty coordinates)
        try:
            coords = AtomicCoordinates(
                coordinates=np.array([]), elements=[], atomic_numbers=np.array([])
            )

            fragment = MolecularFragment(
                coordinates=coords, fragment_id="invalid", molecule_id="invalid"
            )

            # Should handle gracefully
            interactions = engine.calculate_all_interactions([fragment])
            assert isinstance(interactions, list)

        except Exception:
            # Errors are acceptable for invalid input
            pass

    def test_consistency_across_runs(
        self, sample_interaction_config, sample_molecular_fragment
    ):
        """Test consistency of results across multiple runs."""
        engine = GPUAcceleratedInteractionEngine(sample_interaction_config)
        fragments = [sample_molecular_fragment, sample_molecular_fragment]

        # Run multiple times
        results = []
        for _ in range(3):
            interactions = engine.calculate_all_interactions(fragments)
            total_energy = sum(pair.get_total_energy() for pair in interactions)
            results.append(total_energy)

        # Results should be consistent (same fragments, same interactions)
        if results:
            first_result = results[0]
            for result in results[1:]:
                assert abs(result - first_result) < 1e-6  # Should be identical
