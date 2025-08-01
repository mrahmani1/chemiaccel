"""
Integration tests for mixture module.

Tests the complete workflow from mixture creation to visualization.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.mixture import (
    ComponentData,
    DynamicProcessIntegrator,
    InteractionModeler,
    MixtureCalculator,
    MixtureData,
    MixtureInverseDesigner,
    MixtureVisualizer,
    PhaseBehaviorPredictor,
)
from chemi_accel.mixture.dynamic_processes import ProcessConditions


class TestMixtureIntegration:
    """Integration tests for complete mixture workflows."""

    def test_basic_mixture_analysis_workflow(self):
        """Test complete mixture analysis workflow."""
        # Create mixture calculator
        calc = MixtureCalculator(backend="cpu")

        # Create a simple binary mixture
        components = [("CCO", 0.6), ("O", 0.4)]  # Ethanol-water
        mixture = calc.create_mixture(components, temperature=298.15)

        assert mixture.n_components == 2
        assert mixture.temperature == 298.15

        # Calculate basic properties
        properties = calc.calculate_mixture_properties(mixture)
        assert isinstance(properties, dict)
        assert "molecular_weight_avg" in properties
        assert properties["molecular_weight_avg"] > 0

    @patch("rdkit.Chem.MolFromSmiles")
    def test_interaction_analysis_workflow(self, mock_mol_from_smiles):
        """Test molecular interaction analysis workflow."""
        # Mock RDKit molecules
        mock_mol = Mock()
        mock_mol_from_smiles.return_value = mock_mol

        calc = MixtureCalculator(backend="cpu")

        # Create mixture
        components = [("CCO", 0.5), ("O", 0.5)]
        mixture = calc.create_mixture(components)

        # Mock interaction calculations
        with patch.object(
            calc.interaction_modeler, "calculate_interactions"
        ) as mock_calc:
            mock_calc.return_value = []

            # Calculate interactions
            interactions = calc.model_interactions(mixture)

            mock_calc.assert_called_once()
            assert isinstance(interactions, list)

    def test_phase_behavior_prediction_workflow(self):
        """Test phase behavior prediction workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Create mixture
        components = [("CCO", 0.7), ("O", 0.3)]
        mixture = calc.create_mixture(components)

        # Mock phase behavior calculations
        with patch.object(
            calc.phase_predictor, "predict_phase_equilibrium"
        ) as mock_predict:
            from chemi_accel.mixture.phase_behavior import PhaseEquilibrium

            mock_result = PhaseEquilibrium(
                temperature=298.15,
                pressure=101325.0,
                phases=[{"type": "liquid", "composition": [0.7, 0.3]}],
                stability={"overall": 0.9},
            )
            mock_predict.return_value = mock_result

            # Predict phase behavior
            phase_result = calc.predict_phase_behavior(mixture)

            assert phase_result.temperature == 298.15
            assert phase_result.n_phases == 1
            assert phase_result.is_stable

    @patch("scipy.optimize.differential_evolution")
    def test_inverse_design_workflow(self, mock_optimization):
        """Test inverse design optimization workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Mock optimization result
        mock_result = Mock()
        mock_result.x = np.array([0.6, 0.4])
        mock_result.fun = 0.1
        mock_result.nit = 50
        mock_result.success = True
        mock_optimization.return_value = mock_result

        # Define target properties
        target_properties = {"molecular_weight_avg": 30.0}
        component_library = ["CCO", "O"]

        # Run inverse design
        optimal_composition = calc.design_optimal_mixture(
            target_properties, component_library
        )

        assert optimal_composition is not None
        assert len(optimal_composition.composition) == 2
        assert abs(np.sum(optimal_composition.composition) - 1.0) < 1e-6

    def test_dynamic_process_simulation_workflow(self):
        """Test dynamic process simulation workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Create initial mixture
        components = [("CCO", 0.8), ("O", 0.2)]
        mixture = calc.create_mixture(components)

        # Define process conditions
        conditions = ProcessConditions()
        conditions.temperature_profile = lambda t: 298.15 + 10 * t  # Heating
        conditions.pressure_profile = lambda t: 101325.0  # Constant pressure

        # Run simulation with mocked process models
        with patch.object(calc.process_integrator, "simulate_process") as mock_simulate:
            from chemi_accel.mixture.dynamic_processes import FormulationProcess

            mock_result = FormulationProcess(
                process_type="crystallization",
                time_points=np.array([0, 1, 2]),
                mixture_evolution=[mixture, mixture, mixture],
                property_evolution=[{}, {}, {}],
                microstructure_evolution=[{}, {}, {}],
                process_conditions=conditions,
            )
            mock_simulate.return_value = mock_result

            result = calc.integrate_dynamic_processes(
                mixture, conditions, ["crystallization"]
            )

            assert result.process_type == "crystallization"
            assert len(result.time_points) == 3

    def test_visualization_workflow(self):
        """Test visualization generation workflow."""
        calc = MixtureCalculator(backend="cpu")
        visualizer = MixtureVisualizer()

        # Create mixture
        components = [("CCO", 0.5), ("O", 0.5)]
        mixture = calc.create_mixture(components)

        # Create mock phase equilibrium data
        from chemi_accel.mixture.phase_behavior import PhaseEquilibrium

        phase_data = {
            "data_points": [
                {"temperature": 298.15, "pressure": 101325.0, "n_phases": 1}
            ],
            "temperature_range": (250, 350),
            "pressure_range": (1e5, 2e5),
        }

        phase_equilibrium = PhaseEquilibrium(
            temperature=298.15,
            pressure=101325.0,
            phases=[{"type": "liquid"}],
            stability={"overall": 0.9},
            phase_diagram_data=phase_data,
        )

        # Create phase diagram visualization
        with patch("scipy.interpolate.griddata", return_value=np.ones((50, 50))):
            viz_config = visualizer.create_phase_diagram(phase_equilibrium)

            assert viz_config["type"] == "2d_phase_diagram"
            assert "data" in viz_config
            assert "layout" in viz_config

    def test_comprehensive_mixture_characterization(self):
        """Test comprehensive mixture characterization workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Create complex mixture
        components = [
            ("CCO", 0.4),  # Ethanol
            ("O", 0.3),  # Water
            ("CCCCCC", 0.3),  # Hexane
        ]
        mixture = calc.create_mixture(components, temperature=298.15, pressure=101325.0)

        # Step 1: Calculate basic properties
        properties = calc.calculate_mixture_properties(mixture)
        assert "molecular_weight_avg" in properties

        # Step 2: Analyze interactions (mocked)
        with patch("rdkit.Chem.MolFromSmiles", return_value=Mock()):
            with patch.object(
                calc.interaction_modeler, "calculate_interactions"
            ) as mock_interactions:
                mock_interactions.return_value = []
                interactions = calc.model_interactions(mixture)
                assert isinstance(interactions, list)

        # Step 3: Predict phase behavior (mocked)
        with patch.object(
            calc.phase_predictor, "predict_phase_equilibrium"
        ) as mock_phase:
            from chemi_accel.mixture.phase_behavior import PhaseEquilibrium

            mock_phase.return_value = PhaseEquilibrium(
                temperature=298.15,
                pressure=101325.0,
                phases=[{"type": "liquid"}],
                stability={"overall": 0.8},
            )
            phase_behavior = calc.predict_phase_behavior(mixture)
            assert phase_behavior.is_stable

        # Step 4: Create visualization dashboard
        visualizer = MixtureVisualizer()
        dashboard = visualizer.create_dashboard(
            mixture=mixture, phase_equilibrium=phase_behavior, interactions=interactions
        )

        assert dashboard["type"] == "interactive_dashboard"
        assert len(dashboard["panels"]) > 0

    def test_mixture_optimization_with_constraints(self):
        """Test mixture optimization with various constraints."""
        calc = MixtureCalculator(backend="cpu")

        # Define optimization problem
        target_properties = {"molecular_weight_avg": 50.0, "density_mixture": 0.8}

        component_library = ["CCO", "O", "CCCCCC"]

        constraints = {
            "composition_bounds": {
                "CCO": (0.1, 0.8),  # Ethanol between 10-80%
                "O": (0.1, 0.6),  # Water between 10-60%
            },
            "property_bounds": {"molecular_weight_avg": (20.0, 80.0)},
        }

        # Mock optimization
        with patch("scipy.optimize.differential_evolution") as mock_opt:
            mock_result = Mock()
            mock_result.x = np.array([0.5, 0.3, 0.2])
            mock_result.fun = 0.05
            mock_result.nit = 30
            mock_result.success = True
            mock_opt.return_value = mock_result

            optimal_composition = calc.design_optimal_mixture(
                target_properties, component_library, constraints
            )

            assert optimal_composition.feasibility_score > 0
            assert len(optimal_composition.components) == 3

    def test_process_optimization_workflow(self):
        """Test dynamic process optimization workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Create mixture
        components = [("CCO", 0.6), ("O", 0.4)]
        mixture = calc.create_mixture(components)

        # Define target properties for process
        target_properties = {
            "crystallization_crystal_fraction": 0.3,
            "mixing_homogeneity": 0.9,
        }

        # Define variable process conditions
        variable_conditions = {
            "temperature": (280.0, 320.0),
            "shear_rate": (0.0, 100.0),
        }

        # Mock process optimization
        with patch.object(
            calc.process_integrator, "optimize_process_conditions"
        ) as mock_opt:
            from chemi_accel.mixture.dynamic_processes import (
                FormulationProcess,
                ProcessConditions,
            )

            optimal_conditions = ProcessConditions()
            optimal_conditions.temperature_profile = lambda t: 300.0

            mock_result = FormulationProcess(
                process_type="crystallization_mixing",
                time_points=np.array([0, 1]),
                mixture_evolution=[mixture, mixture],
                property_evolution=[{}, target_properties],
                microstructure_evolution=[{}, {}],
                process_conditions=optimal_conditions,
            )

            mock_opt.return_value = (optimal_conditions, mock_result)

            (
                optimal_conditions,
                simulation_result,
            ) = calc.process_integrator.optimize_process_conditions(
                mixture,
                target_properties,
                ["crystallization", "mixing"],
                variable_conditions,
            )

            assert simulation_result.final_properties == target_properties

    def test_error_handling_in_workflows(self):
        """Test error handling throughout mixture workflows."""
        calc = MixtureCalculator(backend="cpu")

        # Test invalid SMILES handling
        with pytest.raises(Exception):  # Should raise some kind of error
            invalid_components = [("INVALID_SMILES", 1.0)]
            mixture = calc.create_mixture(invalid_components)

            # Try to calculate interactions with invalid SMILES
            with patch("rdkit.Chem.MolFromSmiles", return_value=None):
                calc.model_interactions(mixture)

    def test_property_landscape_generation(self):
        """Test property landscape generation for visualization."""
        calc = MixtureCalculator(backend="cpu")
        visualizer = MixtureVisualizer()

        # Generate mixture data for different compositions
        mixture_data = []
        for x1 in [0.2, 0.5, 0.8]:
            x2 = 1.0 - x1
            components = [("CCO", x1), ("O", x2)]
            mixture = calc.create_mixture(components)

            # Add mock property data
            mixture.mixture_properties["test_property"] = x1 * 100 + x2 * 50
            mixture_data.append(mixture)

        assert len(mixture_data) == 3

        # Create property landscape
        with patch("scipy.interpolate.griddata", return_value=np.random.rand(10)):
            landscape = visualizer.create_property_landscape(
                mixture_data, "test_property", resolution=10
            )

            assert landscape.property_name == "test_property"
            assert len(landscape.components) == 2
            assert landscape.composition_grid.shape[1] >= 2

    def test_multi_objective_optimization_workflow(self):
        """Test multi-objective optimization workflow."""
        calc = MixtureCalculator(backend="cpu")

        # Define multiple objectives
        from chemi_accel.mixture.inverse_design import (
            CostObjective,
            PropertyTargetObjective,
            StabilityObjective,
        )

        objectives = [
            PropertyTargetObjective({"molecular_weight_avg": 40.0}),
            StabilityObjective(),
            CostObjective({"CCO": 1.0, "O": 0.5, "CCCCCC": 2.0}),
        ]

        component_library = ["CCO", "O", "CCCCCC"]

        # Mock multi-objective optimization
        with patch(
            "chemi_accel.mixture.inverse_design.pymoo_minimize"
        ) as mock_minimize:
            mock_result = Mock()
            mock_result.X = np.array([[0.5, 0.3, 0.2]])
            mock_result.F = np.array([[0.1, 0.8, 1.5]])
            mock_minimize.return_value = mock_result

            pareto_solutions = calc.inverse_designer.design_pareto_optimal_mixtures(
                objectives, component_library, n_solutions=10
            )

            # Should return list of optimal compositions
            assert isinstance(pareto_solutions, list)
