"""
Comprehensive tests for the multi-scale integration module.

Tests cover scale definitions, data transfer between scales, coupling mechanisms,
and complete multi-scale simulation workflows.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.self_assembly.multiscale_integration import (  # Enums; Data structures; Models; Utility functions; Exceptions
    AtomisticModel,
    CouplingConfiguration,
    CouplingMethod,
    DataTransferEngine,
    DataTransferError,
    DataTransferMethod,
    MesoscaleModel,
    MolecularModel,
    MultiScaleIntegrationError,
    MultiScaleIntegrator,
    MultiScaleSystem,
    ScaleCouplingError,
    ScaleData,
    ScaleParameters,
    ScaleType,
    TransferResult,
    ValidationError,
    ValidationLevel,
    create_hierarchical_system,
    run_multiscale_workflow,
    validate_multiscale_consistency,
)


class TestScaleParameters:
    """Test ScaleParameters dataclass."""

    def test_scale_parameters_creation(self):
        """Test basic scale parameters creation."""
        params = ScaleParameters(
            scale_type=ScaleType.ATOMISTIC,
            spatial_resolution=0.1,
            temporal_resolution=0.001,
            system_size=(10.0, 10.0, 10.0),
            particle_count=1000,
            description="All-atom MD simulation",
        )

        assert params.scale_type == ScaleType.ATOMISTIC
        assert params.spatial_resolution == 0.1
        assert params.temporal_resolution == 0.001
        assert params.system_size == (10.0, 10.0, 10.0)
        assert params.particle_count == 1000
        assert params.description == "All-atom MD simulation"

    def test_scale_parameters_validation(self):
        """Test scale parameters validation."""
        with pytest.raises(ValueError, match="Spatial resolution must be positive"):
            ScaleParameters(
                scale_type=ScaleType.MOLECULAR,
                spatial_resolution=0.0,
                temporal_resolution=0.01,
                system_size=(5.0, 5.0, 5.0),
                particle_count=100,
            )

        with pytest.raises(ValueError, match="Temporal resolution must be positive"):
            ScaleParameters(
                scale_type=ScaleType.MOLECULAR,
                spatial_resolution=0.5,
                temporal_resolution=-0.01,
                system_size=(5.0, 5.0, 5.0),
                particle_count=100,
            )

        with pytest.raises(ValueError, match="Particle count must be positive"):
            ScaleParameters(
                scale_type=ScaleType.MOLECULAR,
                spatial_resolution=0.5,
                temporal_resolution=0.01,
                system_size=(5.0, 5.0, 5.0),
                particle_count=0,
            )


class TestCouplingConfiguration:
    """Test CouplingConfiguration dataclass."""

    def test_coupling_configuration_creation(self):
        """Test coupling configuration creation."""
        config = CouplingConfiguration(
            source_scale=ScaleType.ATOMISTIC,
            target_scale=ScaleType.MOLECULAR,
            coupling_method=CouplingMethod.HIERARCHICAL,
            data_transfer_method=DataTransferMethod.STATISTICAL_COARSE_GRAINING,
            transfer_frequency=100,
            validation_level=ValidationLevel.COMPREHENSIVE,
        )

        assert config.source_scale == ScaleType.ATOMISTIC
        assert config.target_scale == ScaleType.MOLECULAR
        assert config.coupling_method == CouplingMethod.HIERARCHICAL
        assert (
            config.data_transfer_method
            == DataTransferMethod.STATISTICAL_COARSE_GRAINING
        )
        assert config.transfer_frequency == 100
        assert config.validation_level == ValidationLevel.COMPREHENSIVE

    def test_coupling_configuration_validation(self):
        """Test coupling configuration validation."""
        with pytest.raises(ValueError, match="Transfer frequency must be positive"):
            CouplingConfiguration(
                source_scale=ScaleType.ATOMISTIC,
                target_scale=ScaleType.MOLECULAR,
                coupling_method=CouplingMethod.HIERARCHICAL,
                data_transfer_method=DataTransferMethod.DIRECT_MAPPING,
                transfer_frequency=0,
            )


class TestMultiScaleSystem:
    """Test MultiScaleSystem dataclass."""

    def create_test_multiscale_system(self):
        """Create a test multi-scale system."""
        scales = [
            ScaleParameters(
                scale_type=ScaleType.ATOMISTIC,
                spatial_resolution=0.1,
                temporal_resolution=0.001,
                system_size=(10.0, 10.0, 10.0),
                particle_count=1000,
            ),
            ScaleParameters(
                scale_type=ScaleType.MOLECULAR,
                spatial_resolution=0.5,
                temporal_resolution=0.01,
                system_size=(10.0, 10.0, 10.0),
                particle_count=200,
            ),
        ]

        couplings = [
            CouplingConfiguration(
                source_scale=ScaleType.ATOMISTIC,
                target_scale=ScaleType.MOLECULAR,
                coupling_method=CouplingMethod.HIERARCHICAL,
                data_transfer_method=DataTransferMethod.DIRECT_MAPPING,
            )
        ]

        return MultiScaleSystem(
            name="Test System",
            description="Test multi-scale system",
            scales=scales,
            couplings=couplings,
        )

    def test_multiscale_system_creation(self):
        """Test multi-scale system creation."""
        system = self.create_test_multiscale_system()

        assert system.name == "Test System"
        assert system.description == "Test multi-scale system"
        assert len(system.scales) == 2
        assert len(system.couplings) == 1
        assert len(system.global_parameters) == 0

    def test_multiscale_system_validation_empty_scales(self):
        """Test validation with empty scales."""
        with pytest.raises(ValueError, match="At least one scale must be defined"):
            MultiScaleSystem(
                name="Empty System",
                description="System with no scales",
                scales=[],
                couplings=[],
            )

    def test_multiscale_system_validation_duplicate_scales(self):
        """Test validation with duplicate scale types."""
        scales = [
            ScaleParameters(
                scale_type=ScaleType.ATOMISTIC,
                spatial_resolution=0.1,
                temporal_resolution=0.001,
                system_size=(10.0, 10.0, 10.0),
                particle_count=1000,
            ),
            ScaleParameters(
                scale_type=ScaleType.ATOMISTIC,  # Duplicate
                spatial_resolution=0.2,
                temporal_resolution=0.002,
                system_size=(10.0, 10.0, 10.0),
                particle_count=500,
            ),
        ]

        with pytest.raises(ValueError, match="Duplicate scale types found"):
            MultiScaleSystem(
                name="Duplicate System",
                description="System with duplicate scales",
                scales=scales,
                couplings=[],
            )

    def test_multiscale_system_validation_invalid_coupling(self):
        """Test validation with invalid coupling references."""
        scales = [
            ScaleParameters(
                scale_type=ScaleType.ATOMISTIC,
                spatial_resolution=0.1,
                temporal_resolution=0.001,
                system_size=(10.0, 10.0, 10.0),
                particle_count=1000,
            )
        ]

        couplings = [
            CouplingConfiguration(
                source_scale=ScaleType.ATOMISTIC,
                target_scale=ScaleType.MOLECULAR,  # Not in scales
                coupling_method=CouplingMethod.HIERARCHICAL,
                data_transfer_method=DataTransferMethod.DIRECT_MAPPING,
            )
        ]

        with pytest.raises(ValueError, match="Target scale .* not found"):
            MultiScaleSystem(
                name="Invalid Coupling System",
                description="System with invalid coupling",
                scales=scales,
                couplings=couplings,
            )


class TestScaleData:
    """Test ScaleData dataclass."""

    def test_scale_data_creation(self):
        """Test scale data creation."""
        positions = np.random.uniform(-5, 5, (100, 3))
        velocities = np.random.normal(0, 1, (100, 3))
        forces = np.random.normal(0, 0.1, (100, 3))
        energies = {"kinetic": 150.0, "potential": -200.0, "total": -50.0}
        properties = {"temperature": 298.15, "pressure": 1.0}

        data = ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=positions,
            velocities=velocities,
            forces=forces,
            energies=energies,
            properties=properties,
        )

        assert data.scale_type == ScaleType.ATOMISTIC
        np.testing.assert_array_equal(data.positions, positions)
        np.testing.assert_array_equal(data.velocities, velocities)
        np.testing.assert_array_equal(data.forces, forces)
        assert data.energies == energies
        assert data.properties == properties

    def test_scale_data_validation(self):
        """Test scale data validation."""
        # Invalid positions shape
        with pytest.raises(ValueError, match="Positions must be Nx3 array"):
            ScaleData(
                scale_type=ScaleType.ATOMISTIC,
                positions=np.array([[1, 2]]),  # Missing z coordinate
            )

        # Mismatched velocities shape
        positions = np.random.uniform(-1, 1, (50, 3))
        velocities = np.random.normal(0, 1, (40, 3))  # Wrong number of particles

        with pytest.raises(ValueError, match="Velocities must match positions shape"):
            ScaleData(
                scale_type=ScaleType.ATOMISTIC,
                positions=positions,
                velocities=velocities,
            )

        # Mismatched forces shape
        forces = np.random.normal(0, 0.1, (30, 3))  # Wrong number of particles

        with pytest.raises(ValueError, match="Forces must match positions shape"):
            ScaleData(
                scale_type=ScaleType.ATOMISTIC, positions=positions, forces=forces
            )


class TestAtomisticModel:
    """Test AtomisticModel class."""

    def test_atomistic_model_initialization(self):
        """Test atomistic model initialization."""
        model = AtomisticModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.universe is None
        assert model.current_step == 0
        assert model.timestep == 1.0

    @pytest.mark.skipif(True, reason="MDAnalysis not available in test environment")
    def test_atomistic_model_with_files(self):
        """Test atomistic model with topology/trajectory files."""
        # This would require actual files
        pass

    def test_atomistic_model_initialize(self):
        """Test atomistic model initialization with data."""
        model = AtomisticModel(use_gpu=False)

        positions = np.random.uniform(-5, 5, (50, 3))
        initial_data = ScaleData(scale_type=ScaleType.ATOMISTIC, positions=positions)

        model.initialize(initial_data)

        assert model.current_step == 0
        # Universe creation is mocked/simplified in tests

    def test_atomistic_model_step(self):
        """Test atomistic model step."""
        model = AtomisticModel(use_gpu=False)

        positions = np.random.uniform(-2, 2, (10, 3))
        initial_data = ScaleData(scale_type=ScaleType.ATOMISTIC, positions=positions)

        model.initialize(initial_data)

        # Mock universe for testing
        mock_universe = Mock()
        mock_atoms = Mock()
        mock_atoms.positions = positions * 10  # nm to Angstrom
        mock_universe.atoms = mock_atoms
        model.universe = mock_universe

        result = model.step(5)

        assert isinstance(result, ScaleData)
        assert result.scale_type == ScaleType.ATOMISTIC
        assert result.positions.shape == (10, 3)
        assert "radius_of_gyration" in result.properties
        assert result.metadata["step"] == 5

    def test_atomistic_model_get_current_state(self):
        """Test getting current atomistic state."""
        model = AtomisticModel(use_gpu=False)

        positions = np.random.uniform(-1, 1, (5, 3))

        # Mock universe
        mock_universe = Mock()
        mock_atoms = Mock()
        mock_atoms.positions = positions * 10  # nm to Angstrom
        mock_universe.atoms = mock_atoms
        model.universe = mock_universe

        state = model.get_current_state()

        assert isinstance(state, ScaleData)
        assert state.scale_type == ScaleType.ATOMISTIC
        assert state.positions.shape == (5, 3)


class TestMolecularModel:
    """Test MolecularModel class."""

    def test_molecular_model_initialization(self):
        """Test molecular model initialization."""
        model = MolecularModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.calculator is not None
        assert model.molecules == []
        assert model.current_properties == {}

    def test_molecular_model_initialize(self):
        """Test molecular model initialization with data."""
        model = MolecularModel(use_gpu=False)

        positions = np.random.uniform(-3, 3, (20, 3))
        initial_data = ScaleData(scale_type=ScaleType.MOLECULAR, positions=positions)

        model.initialize(initial_data)

        assert len(model.molecules) == 20
        assert all(mol == "O" for mol in model.molecules)  # Simplified representation

    @patch("chemi_accel.self_assembly.multiscale_integration.Calculator")
    def test_molecular_model_step(self, mock_calculator_class):
        """Test molecular model step with mocked calculator."""
        # Mock calculator results
        mock_calculator = Mock()
        mock_results = Mock()
        mock_results.columns = ["molecular_weight", "logp", "tpsa"]
        mock_results.__getitem__ = Mock(
            side_effect=lambda key: Mock(
                mean=lambda: {"molecular_weight": 18.0, "logp": 0.0, "tpsa": 0.0}[key]
            )
        )
        mock_calculator.pandas.return_value = mock_results
        mock_calculator_class.return_value = mock_calculator

        model = MolecularModel(use_gpu=False)
        model.calculator = mock_calculator

        # Initialize with molecules
        positions = np.random.uniform(-2, 2, (10, 3))
        initial_data = ScaleData(scale_type=ScaleType.MOLECULAR, positions=positions)
        model.initialize(initial_data)

        result = model.step(3)

        assert isinstance(result, ScaleData)
        assert result.scale_type == ScaleType.MOLECULAR
        assert result.positions.shape[0] == 10
        assert result.positions.shape[1] == 3
        assert "molecular_weight" in result.properties
        assert "n_molecules" in result.properties

    def test_molecular_model_calculate_interaction_parameters(self):
        """Test interaction parameter calculation."""
        model = MolecularModel(use_gpu=False)

        # Set some molecular properties
        model.current_properties = {"molecular_weight": 50.0, "logp": 2.0}

        params = model.calculate_interaction_parameters()

        assert "repulsion_strength" in params
        assert "attractive_strength" in params
        assert "particle_size" in params
        assert params["repulsion_strength"] > 25.0  # Should increase with logp
        assert params["particle_size"] == 1.0  # 50/50


class TestMesoscaleModel:
    """Test MesoscaleModel class."""

    def test_mesoscale_model_initialization(self):
        """Test mesoscale model initialization."""
        model = MesoscaleModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.simulation_engine is None
        assert model.current_trajectory == []
        assert model.characterizer is not None

    @patch("chemi_accel.self_assembly.multiscale_integration.create_amphiphilic_system")
    @patch(
        "chemi_accel.self_assembly.multiscale_integration.CoarseGrainedSimulationEngine"
    )
    def test_mesoscale_model_initialize(self, mock_engine_class, mock_create_system):
        """Test mesoscale model initialization with data."""
        # Mock system creation
        mock_particles = [Mock() for _ in range(10)]
        mock_sim_box = Mock()
        mock_interaction_params = Mock()
        mock_create_system.return_value = (
            mock_particles,
            mock_sim_box,
            mock_interaction_params,
        )

        # Mock simulation engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        model = MesoscaleModel(use_gpu=False)

        positions = np.random.uniform(-5, 5, (50, 3))
        initial_data = ScaleData(scale_type=ScaleType.MESOSCALE, positions=positions)

        model.initialize(initial_data)

        assert model.simulation_engine is not None
        mock_create_system.assert_called_once()
        mock_engine_class.assert_called_once()

    def test_mesoscale_model_step_without_engine(self):
        """Test mesoscale model step without engine."""
        model = MesoscaleModel(use_gpu=False)

        with pytest.raises(ValueError, match="Simulation engine not initialized"):
            model.step(1)

    @patch("chemi_accel.self_assembly.multiscale_integration.characterize_trajectory")
    def test_mesoscale_model_step_with_engine(self, mock_characterize):
        """Test mesoscale model step with mocked engine."""
        # Mock characterization
        mock_structure = Mock()
        mock_structure.aggregation_number = 25
        mock_structure.radius_of_gyration = 2.0
        mock_characterize.return_value = {"structures": [mock_structure]}

        # Mock simulation engine
        mock_engine = Mock()
        mock_state = Mock()
        mock_state.positions = np.random.uniform(-3, 3, (15, 3))
        mock_state.velocities = np.random.normal(0, 1, (15, 3))
        mock_state.forces = np.random.normal(0, 0.1, (15, 3))
        mock_state.energies = {"kinetic": 100.0, "potential": -150.0, "total": -50.0}
        mock_state.step = 5
        mock_state.time = 0.05
        mock_engine.get_current_state.return_value = mock_state
        mock_engine._simulation_step = Mock()
        mock_engine.state = Mock()
        mock_engine.state.step = 0

        model = MesoscaleModel(use_gpu=False)
        model.simulation_engine = mock_engine
        model.current_trajectory = [Mock()]  # Add trajectory for characterization

        result = model.step(2)

        assert isinstance(result, ScaleData)
        assert result.scale_type == ScaleType.MESOSCALE
        assert result.positions.shape == (15, 3)
        assert result.velocities.shape == (15, 3)
        assert result.forces.shape == (15, 3)
        assert "kinetic" in result.energies
        assert result.metadata["step"] == 5

        # Verify simulation engine was stepped
        assert mock_engine._simulation_step.call_count == 2


class TestDataTransferEngine:
    """Test DataTransferEngine class."""

    def test_data_transfer_engine_initialization(self):
        """Test data transfer engine initialization."""
        engine = DataTransferEngine(use_gpu=False)

        assert engine.use_gpu is False
        assert len(engine.transfer_functions) > 0
        assert (ScaleType.ATOMISTIC, ScaleType.MOLECULAR) in engine.transfer_functions
        assert (ScaleType.MOLECULAR, ScaleType.MESOSCALE) in engine.transfer_functions

    def test_atomistic_to_molecular_transfer(self):
        """Test atomistic to molecular scale transfer."""
        engine = DataTransferEngine(use_gpu=False)

        # Create atomistic data
        positions = np.random.uniform(-2, 2, (30, 3))
        source_data = ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=positions,
            properties={"box_size": [4, 4, 4]},
        )

        result = engine.transfer_data(
            source_data,
            ScaleType.MOLECULAR,
            DataTransferMethod.DIRECT_MAPPING,
            atoms_per_molecule=3,
        )

        assert result.success
        assert result.source_scale == ScaleType.ATOMISTIC
        assert result.target_scale == ScaleType.MOLECULAR
        assert result.transferred_data.scale_type == ScaleType.MOLECULAR
        assert result.transferred_data.positions.shape[0] == 10  # 30/3
        assert result.transferred_data.positions.shape[1] == 3
        assert "n_molecules" in result.transferred_data.properties

    def test_molecular_to_mesoscale_transfer(self):
        """Test molecular to mesoscale transfer."""
        engine = DataTransferEngine(use_gpu=False)

        # Create molecular data
        positions = np.random.uniform(-3, 3, (25, 3))
        source_data = ScaleData(scale_type=ScaleType.MOLECULAR, positions=positions)

        result = engine.transfer_data(
            source_data,
            ScaleType.MESOSCALE,
            DataTransferMethod.DIRECT_MAPPING,
            molecules_per_particle=5,
        )

        assert result.success
        assert result.source_scale == ScaleType.MOLECULAR
        assert result.target_scale == ScaleType.MESOSCALE
        assert result.transferred_data.scale_type == ScaleType.MESOSCALE
        assert result.transferred_data.positions.shape[0] == 5  # 25/5
        assert result.transferred_data.velocities is not None
        assert "n_cg_particles" in result.transferred_data.properties

    def test_mesoscale_to_molecular_transfer(self):
        """Test mesoscale to molecular transfer."""
        engine = DataTransferEngine(use_gpu=False)

        # Create mesoscale data
        positions = np.random.uniform(-2, 2, (10, 3))
        source_data = ScaleData(scale_type=ScaleType.MESOSCALE, positions=positions)

        result = engine.transfer_data(
            source_data,
            ScaleType.MOLECULAR,
            DataTransferMethod.INTERPOLATION,
            expansion_factor=3,
        )

        assert result.success
        assert result.source_scale == ScaleType.MESOSCALE
        assert result.target_scale == ScaleType.MOLECULAR
        assert result.transferred_data.scale_type == ScaleType.MOLECULAR
        assert result.transferred_data.positions.shape[0] == 30  # 10*3
        assert "reconstruction_ratio" in result.transferred_data.properties

    def test_molecular_to_atomistic_transfer(self):
        """Test molecular to atomistic transfer."""
        engine = DataTransferEngine(use_gpu=False)

        # Create molecular data
        positions = np.random.uniform(-1, 1, (8, 3))
        source_data = ScaleData(scale_type=ScaleType.MOLECULAR, positions=positions)

        result = engine.transfer_data(
            source_data,
            ScaleType.ATOMISTIC,
            DataTransferMethod.MACHINE_LEARNING,
            atoms_per_molecule=4,
        )

        assert result.success
        assert result.source_scale == ScaleType.MOLECULAR
        assert result.target_scale == ScaleType.ATOMISTIC
        assert result.transferred_data.scale_type == ScaleType.ATOMISTIC
        assert result.transferred_data.positions.shape[0] == 32  # 8*4
        assert "expansion_ratio" in result.transferred_data.properties

    def test_invalid_transfer_path(self):
        """Test invalid transfer path."""
        engine = DataTransferEngine(use_gpu=False)

        source_data = ScaleData(
            scale_type=ScaleType.QUANTUM,  # Not supported
            positions=np.array([[0, 0, 0]]),
        )

        result = engine.transfer_data(
            source_data,
            ScaleType.CONTINUUM,  # Not supported
            DataTransferMethod.DIRECT_MAPPING,
        )

        assert not result.success
        assert len(result.warnings) > 0

    def test_transfer_validation(self):
        """Test transfer validation."""
        engine = DataTransferEngine(use_gpu=False)

        # Create source data with known properties
        source_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        source_data = ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=source_positions,
            energies={"kinetic": 10.0, "potential": -5.0},
        )

        # Create target data
        target_positions = np.array([[0.5, 0.5, 0]])  # Different but related
        target_data = ScaleData(
            scale_type=ScaleType.MOLECULAR,
            positions=target_positions,
            energies={"kinetic": 8.0, "potential": -4.0},
        )

        error_metrics = engine._validate_transfer(source_data, target_data)

        assert "center_of_mass_error" in error_metrics
        assert "size_scale_error" in error_metrics
        assert "kinetic_energy_error" in error_metrics
        assert "potential_energy_error" in error_metrics

        # Check that errors are computed
        assert error_metrics["center_of_mass_error"] >= 0
        assert error_metrics["kinetic_energy_error"] == 2.0  # |10-8|
        assert error_metrics["potential_energy_error"] == 1.0  # |-5-(-4)|


class TestMultiScaleIntegrator:
    """Test MultiScaleIntegrator class."""

    def create_test_system(self):
        """Create a test multi-scale system."""
        scales = [
            ScaleParameters(
                scale_type=ScaleType.ATOMISTIC,
                spatial_resolution=0.1,
                temporal_resolution=0.001,
                system_size=(5.0, 5.0, 5.0),
                particle_count=50,
            ),
            ScaleParameters(
                scale_type=ScaleType.MOLECULAR,
                spatial_resolution=0.5,
                temporal_resolution=0.01,
                system_size=(5.0, 5.0, 5.0),
                particle_count=10,
            ),
        ]

        couplings = [
            CouplingConfiguration(
                source_scale=ScaleType.ATOMISTIC,
                target_scale=ScaleType.MOLECULAR,
                coupling_method=CouplingMethod.HIERARCHICAL,
                data_transfer_method=DataTransferMethod.DIRECT_MAPPING,
                transfer_frequency=5,
            )
        ]

        return MultiScaleSystem(
            name="Test Integration System",
            description="System for testing multi-scale integration",
            scales=scales,
            couplings=couplings,
        )

    def test_multiscale_integrator_initialization(self):
        """Test multi-scale integrator initialization."""
        system = self.create_test_system()
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        assert integrator.system == system
        assert integrator.use_gpu is False
        assert integrator.parallel_execution is False
        assert (
            len(integrator.scale_models) >= 1
        )  # Should have models for available scales
        assert integrator.data_transfer_engine is not None

    def test_initialize_system(self):
        """Test system initialization."""
        system = self.create_test_system()
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Create initial conditions
        atomistic_positions = np.random.uniform(-2, 2, (50, 3))
        molecular_positions = np.random.uniform(-2, 2, (10, 3))

        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(ScaleType.ATOMISTIC, atomistic_positions),
            ScaleType.MOLECULAR: ScaleData(ScaleType.MOLECULAR, molecular_positions),
        }

        integrator.initialize_system(initial_conditions)

        assert len(integrator.current_data) == 2
        assert ScaleType.ATOMISTIC in integrator.current_data
        assert ScaleType.MOLECULAR in integrator.current_data

    @patch.object(AtomisticModel, "step")
    @patch.object(MolecularModel, "step")
    def test_sequential_step(self, mock_mol_step, mock_atom_step):
        """Test sequential step execution."""
        # Setup mocks
        mock_atom_step.return_value = ScaleData(
            ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (5, 3))
        )
        mock_mol_step.return_value = ScaleData(
            ScaleType.MOLECULAR, np.random.uniform(-1, 1, (2, 3))
        )

        system = self.create_test_system()
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Initialize with minimal data
        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(ScaleType.ATOMISTIC, np.array([[0, 0, 0]])),
            ScaleType.MOLECULAR: ScaleData(ScaleType.MOLECULAR, np.array([[0, 0, 0]])),
        }
        integrator.initialize_system(initial_conditions)

        step_results = integrator._sequential_step()

        assert len(step_results) >= 1
        for scale_type, data in step_results.items():
            assert isinstance(data, ScaleData)
            assert data.scale_type == scale_type

    @patch.object(DataTransferEngine, "transfer_data")
    def test_perform_coupling(self, mock_transfer):
        """Test coupling between scales."""
        # Setup mock transfer
        mock_transfer_result = TransferResult(
            source_scale=ScaleType.ATOMISTIC,
            target_scale=ScaleType.MOLECULAR,
            transfer_method=DataTransferMethod.DIRECT_MAPPING,
            success=True,
            transferred_data=ScaleData(ScaleType.MOLECULAR, np.array([[0, 0, 0]])),
            transfer_time=0.1,
            error_metrics={"test_error": 0.05},
        )
        mock_transfer.return_value = mock_transfer_result

        system = self.create_test_system()
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Setup current data
        integrator.current_data = {
            ScaleType.ATOMISTIC: ScaleData(
                ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (10, 3))
            ),
            ScaleType.MOLECULAR: ScaleData(
                ScaleType.MOLECULAR, np.random.uniform(-1, 1, (3, 3))
            ),
        }

        # Test coupling at transfer frequency
        coupling_frequency = {(ScaleType.ATOMISTIC, ScaleType.MOLECULAR): 5}
        coupling_results = integrator._perform_coupling(5, coupling_frequency)

        assert len(coupling_results) == 1
        assert coupling_results[0].success
        assert len(integrator.coupling_history) == 1

        # Verify transfer was called
        mock_transfer.assert_called_once()

    @patch.object(MultiScaleIntegrator, "_sequential_step")
    @patch.object(MultiScaleIntegrator, "_perform_coupling")
    def test_run_coupled_simulation(self, mock_coupling, mock_step):
        """Test complete coupled simulation."""
        # Setup mocks
        mock_step.return_value = {
            ScaleType.ATOMISTIC: ScaleData(
                ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (5, 3))
            ),
            ScaleType.MOLECULAR: ScaleData(
                ScaleType.MOLECULAR, np.random.uniform(-1, 1, (2, 3))
            ),
        }
        mock_coupling.return_value = []

        system = self.create_test_system()
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Initialize system
        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(ScaleType.ATOMISTIC, np.array([[0, 0, 0]])),
            ScaleType.MOLECULAR: ScaleData(ScaleType.MOLECULAR, np.array([[0, 0, 0]])),
        }
        integrator.initialize_system(initial_conditions)

        # Run simulation
        results = integrator.run_coupled_simulation(total_steps=10, output_frequency=5)

        assert "trajectories" in results
        assert "coupling_history" in results
        assert "performance_metrics" in results
        assert "total_time" in results["performance_metrics"]
        assert "steps_per_second" in results["performance_metrics"]

        # Verify methods were called
        assert mock_step.call_count == 10
        assert (
            mock_coupling.call_count == 9
        )  # Steps 1-9 (step 0 doesn't perform coupling)

    def test_analyze_coupling_quality_empty_history(self):
        """Test coupling quality analysis with empty history."""
        system = self.create_test_system()
        integrator = MultiScaleIntegrator(system, use_gpu=False)

        analysis = integrator.analyze_coupling_quality()

        assert analysis["transfer_success_rate"] == 0.0
        assert analysis["average_transfer_time"] == 0.0
        assert len(analysis["coupling_errors"]) == 0

    def test_analyze_coupling_quality_with_history(self):
        """Test coupling quality analysis with coupling history."""
        system = self.create_test_system()
        integrator = MultiScaleIntegrator(system, use_gpu=False)

        # Add mock coupling history
        for i in range(5):
            transfer_result = TransferResult(
                source_scale=ScaleType.ATOMISTIC,
                target_scale=ScaleType.MOLECULAR,
                transfer_method=DataTransferMethod.DIRECT_MAPPING,
                success=i < 4,  # 4 out of 5 successful
                transferred_data=ScaleData(ScaleType.MOLECULAR, np.array([[0, 0, 0]])),
                transfer_time=0.1 + i * 0.05,
                error_metrics={"test_error": i * 0.1},
            )

            integrator.coupling_history.append(
                {"step": i * 5, "coupling": Mock(), "result": transfer_result}
            )

        analysis = integrator.analyze_coupling_quality()

        assert analysis["transfer_success_rate"] == 0.8  # 4/5
        assert (
            analysis["average_transfer_time"] == 0.2
        )  # Mean of [0.1, 0.15, 0.2, 0.25, 0.3]
        assert len(analysis["coupling_errors"]) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_hierarchical_system(self):
        """Test hierarchical system creation."""
        system = create_hierarchical_system(
            n_atoms=500, n_molecules=100, n_cg_particles=25, box_size=8.0
        )

        assert system.name == "Hierarchical Self-Assembly System"
        assert len(system.scales) == 3
        assert len(system.couplings) == 2

        # Check scale types
        scale_types = [scale.scale_type for scale in system.scales]
        assert ScaleType.ATOMISTIC in scale_types
        assert ScaleType.MOLECULAR in scale_types
        assert ScaleType.MESOSCALE in scale_types

        # Check particle counts
        for scale in system.scales:
            if scale.scale_type == ScaleType.ATOMISTIC:
                assert scale.particle_count == 500
            elif scale.scale_type == ScaleType.MOLECULAR:
                assert scale.particle_count == 100
            elif scale.scale_type == ScaleType.MESOSCALE:
                assert scale.particle_count == 25

        # Check couplings
        coupling_pairs = [(c.source_scale, c.target_scale) for c in system.couplings]
        assert (ScaleType.ATOMISTIC, ScaleType.MOLECULAR) in coupling_pairs
        assert (ScaleType.MOLECULAR, ScaleType.MESOSCALE) in coupling_pairs

    @patch("chemi_accel.self_assembly.multiscale_integration.MultiScaleIntegrator")
    def test_run_multiscale_workflow(self, mock_integrator_class):
        """Test complete multi-scale workflow."""
        # Setup mock integrator
        mock_integrator = Mock()
        mock_results = {
            "trajectories": {ScaleType.ATOMISTIC: [], ScaleType.MOLECULAR: []},
            "coupling_history": [],
            "performance_metrics": {"total_time": 10.0},
        }
        mock_integrator.run_coupled_simulation.return_value = mock_results
        mock_integrator.analyze_coupling_quality.return_value = {
            "transfer_success_rate": 0.95
        }
        mock_integrator_class.return_value = mock_integrator

        # Create test system
        system = create_hierarchical_system()

        # Run workflow
        results = run_multiscale_workflow(system=system, total_steps=50, use_gpu=False)

        assert "trajectories" in results
        assert "coupling_history" in results
        assert "performance_metrics" in results
        assert "coupling_analysis" in results
        assert results["coupling_analysis"]["transfer_success_rate"] == 0.95

        # Verify integrator was used correctly
        mock_integrator_class.assert_called_once()
        mock_integrator.initialize_system.assert_called_once()
        mock_integrator.run_coupled_simulation.assert_called_once()
        mock_integrator.analyze_coupling_quality.assert_called_once()

    def test_validate_multiscale_consistency(self):
        """Test multi-scale consistency validation."""
        # Create mock results
        trajectories = {
            ScaleType.ATOMISTIC: [
                {
                    "data": ScaleData(
                        ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (100, 3))
                    )
                },
                {
                    "data": ScaleData(
                        ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (100, 3))
                    )
                },
            ],
            ScaleType.MOLECULAR: [
                {
                    "data": ScaleData(
                        ScaleType.MOLECULAR, np.random.uniform(-1, 1, (20, 3))
                    )
                },
                {
                    "data": ScaleData(
                        ScaleType.MOLECULAR, np.random.uniform(-1, 1, (20, 3))
                    )
                },
            ],
        }

        coupling_history = [
            {"error_metrics": {"energy_error": 0.05}},
            {"error_metrics": {"energy_error": 0.03}},
        ]

        coupling_analysis = {"transfer_success_rate": 0.95}

        results = {
            "trajectories": trajectories,
            "coupling_history": coupling_history,
            "coupling_analysis": coupling_analysis,
        }

        validation = validate_multiscale_consistency(results, tolerance=0.1)

        assert validation["mass_conservation"] is True  # Particle counts consistent
        assert (
            validation["energy_conservation"] is True
        )  # Energy errors within tolerance
        assert validation["scale_coherence"] is True  # High success rate

        # Test with poor coupling quality
        results["coupling_analysis"]["transfer_success_rate"] = 0.5
        validation_poor = validate_multiscale_consistency(results, tolerance=0.1)
        assert validation_poor["scale_coherence"] is False


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        assert issubclass(ScaleCouplingError, MultiScaleIntegrationError)
        assert issubclass(DataTransferError, MultiScaleIntegrationError)
        assert issubclass(ValidationError, MultiScaleIntegrationError)

        # Test that they can be raised
        with pytest.raises(MultiScaleIntegrationError):
            raise ScaleCouplingError("Test coupling error")

        with pytest.raises(MultiScaleIntegrationError):
            raise DataTransferError("Test transfer error")

        with pytest.raises(MultiScaleIntegrationError):
            raise ValidationError("Test validation error")

    def test_data_transfer_error_handling(self):
        """Test data transfer error handling."""
        engine = DataTransferEngine(use_gpu=False)

        # Test with unsupported transfer method
        source_data = ScaleData(
            scale_type=ScaleType.ATOMISTIC, positions=np.array([[0, 0, 0]])
        )

        # Mock an internal error in transfer function
        with patch.object(
            engine, "_atomistic_to_molecular", side_effect=Exception("Internal error")
        ):
            result = engine.transfer_data(
                source_data, ScaleType.MOLECULAR, DataTransferMethod.DIRECT_MAPPING
            )

            assert not result.success
            assert len(result.warnings) > 0
            assert "Internal error" in str(result.warnings[0])

    def test_model_initialization_errors(self):
        """Test model initialization error handling."""
        # Test molecular model with invalid calculator
        with patch(
            "chemi_accel.self_assembly.multiscale_integration.Calculator",
            side_effect=Exception("Calculator error"),
        ):
            # Should not raise exception, but handle gracefully
            model = MolecularModel(use_gpu=False)
            assert model is not None

    def test_mesoscale_model_step_error(self):
        """Test mesoscale model step error handling."""
        model = MesoscaleModel(use_gpu=False)

        # Try to step without initialization
        with pytest.raises(ValueError, match="Simulation engine not initialized"):
            model.step(1)

    def test_multiscale_system_validation_errors(self):
        """Test comprehensive multi-scale system validation."""
        # Test all validation scenarios

        # Empty scales
        with pytest.raises(ValueError, match="At least one scale must be defined"):
            MultiScaleSystem("Empty", "Empty system", [], [])

        # Duplicate scales
        duplicate_scales = [
            ScaleParameters(ScaleType.ATOMISTIC, 0.1, 0.001, (1, 1, 1), 10),
            ScaleParameters(ScaleType.ATOMISTIC, 0.2, 0.002, (1, 1, 1), 20),
        ]
        with pytest.raises(ValueError, match="Duplicate scale types"):
            MultiScaleSystem("Duplicate", "Duplicate scales", duplicate_scales, [])

        # Invalid coupling source
        valid_scale = ScaleParameters(ScaleType.MOLECULAR, 0.5, 0.01, (1, 1, 1), 10)
        invalid_coupling = CouplingConfiguration(
            ScaleType.ATOMISTIC,  # Not in scales
            ScaleType.MOLECULAR,
            CouplingMethod.HIERARCHICAL,
            DataTransferMethod.DIRECT_MAPPING,
        )
        with pytest.raises(ValueError, match="Source scale .* not found"):
            MultiScaleSystem(
                "Invalid", "Invalid coupling", [valid_scale], [invalid_coupling]
            )


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete multi-scale workflows."""

    @patch("chemi_accel.self_assembly.multiscale_integration.create_amphiphilic_system")
    @patch.object(AtomisticModel, "step")
    @patch.object(MolecularModel, "step")
    @patch.object(MesoscaleModel, "step")
    def test_complete_multiscale_simulation(
        self, mock_meso_step, mock_mol_step, mock_atom_step, mock_create_system
    ):
        """Test complete multi-scale simulation workflow."""
        # Setup mocks
        mock_create_system.return_value = ([Mock()], Mock(), Mock())

        mock_atom_step.return_value = ScaleData(
            ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (20, 3))
        )
        mock_mol_step.return_value = ScaleData(
            ScaleType.MOLECULAR, np.random.uniform(-1, 1, (5, 3))
        )
        mock_meso_step.return_value = ScaleData(
            ScaleType.MESOSCALE, np.random.uniform(-1, 1, (3, 3))
        )

        # Create and run workflow
        system = create_hierarchical_system(n_atoms=20, n_molecules=5, n_cg_particles=3)

        results = run_multiscale_workflow(system=system, total_steps=10, use_gpu=False)

        # Verify results structure
        assert "trajectories" in results
        assert "coupling_history" in results
        assert "performance_metrics" in results
        assert "coupling_analysis" in results

        # Check that all scales were simulated
        assert ScaleType.ATOMISTIC in results["trajectories"]
        assert ScaleType.MOLECULAR in results["trajectories"]
        assert ScaleType.MESOSCALE in results["trajectories"]

        # Verify models were stepped
        assert mock_atom_step.call_count > 0
        assert mock_mol_step.call_count > 0
        assert mock_meso_step.call_count > 0

    @patch.object(DataTransferEngine, "transfer_data")
    def test_coupling_between_all_scales(self, mock_transfer):
        """Test coupling between all scales in hierarchy."""

        # Setup successful transfers
        def mock_transfer_func(source_data, target_scale, method, **kwargs):
            if target_scale == ScaleType.MOLECULAR:
                positions = np.random.uniform(-1, 1, (5, 3))
            elif target_scale == ScaleType.MESOSCALE:
                positions = np.random.uniform(-1, 1, (2, 3))
            else:
                positions = np.random.uniform(-1, 1, (10, 3))

            return TransferResult(
                source_scale=source_data.scale_type,
                target_scale=target_scale,
                transfer_method=method,
                success=True,
                transferred_data=ScaleData(target_scale, positions),
                transfer_time=0.1,
                error_metrics={},
            )

        mock_transfer.side_effect = mock_transfer_func

        # Create system with frequent coupling
        system = create_hierarchical_system()
        for coupling in system.couplings:
            coupling.transfer_frequency = 2  # Very frequent

        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Initialize with all scales
        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(
                ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (20, 3))
            ),
            ScaleType.MOLECULAR: ScaleData(
                ScaleType.MOLECULAR, np.random.uniform(-1, 1, (5, 3))
            ),
            ScaleType.MESOSCALE: ScaleData(
                ScaleType.MESOSCALE, np.random.uniform(-1, 1, (2, 3))
            ),
        }
        integrator.initialize_system(initial_conditions)

        # Run simulation
        results = integrator.run_coupled_simulation(total_steps=6, output_frequency=2)

        # Should have performed multiple couplings
        assert len(results["coupling_history"]) > 0

        # Verify transfer was called
        assert mock_transfer.call_count > 0

        # Check coupling quality
        coupling_analysis = integrator.analyze_coupling_quality()
        assert coupling_analysis["transfer_success_rate"] == 1.0  # All successful


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for multi-scale integration."""

    def test_data_transfer_performance(self, benchmark):
        """Benchmark data transfer performance."""
        engine = DataTransferEngine(use_gpu=False)

        # Create large atomistic dataset
        positions = np.random.uniform(-5, 5, (1000, 3))
        source_data = ScaleData(scale_type=ScaleType.ATOMISTIC, positions=positions)

        result = benchmark(
            engine.transfer_data,
            source_data,
            ScaleType.MOLECULAR,
            DataTransferMethod.DIRECT_MAPPING,
            atoms_per_molecule=5,
        )

        assert result.success
        assert result.transferred_data.positions.shape[0] == 200  # 1000/5

    @patch.object(AtomisticModel, "step")
    @patch.object(MolecularModel, "step")
    def test_multiscale_step_performance(
        self, mock_mol_step, mock_atom_step, benchmark
    ):
        """Benchmark multi-scale step performance."""
        # Setup mocks with realistic data
        mock_atom_step.return_value = ScaleData(
            ScaleType.ATOMISTIC, np.random.uniform(-2, 2, (100, 3))
        )
        mock_mol_step.return_value = ScaleData(
            ScaleType.MOLECULAR, np.random.uniform(-2, 2, (20, 3))
        )

        system = create_hierarchical_system(
            n_atoms=100, n_molecules=20, n_cg_particles=5
        )
        integrator = MultiScaleIntegrator(
            system, use_gpu=False, parallel_execution=False
        )

        # Initialize system
        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(
                ScaleType.ATOMISTIC, np.random.uniform(-1, 1, (100, 3))
            ),
            ScaleType.MOLECULAR: ScaleData(
                ScaleType.MOLECULAR, np.random.uniform(-1, 1, (20, 3))
            ),
        }
        integrator.initialize_system(initial_conditions)

        result = benchmark(integrator._sequential_step)

        assert len(result) >= 1
        for scale_type, data in result.items():
            assert isinstance(data, ScaleData)


if __name__ == "__main__":
    pytest.main([__file__])
