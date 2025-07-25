"""
Unit tests for multi-objective optimization framework.

This module tests the MPOptimizer and related classes for correctness,
performance, and GPU acceleration capabilities.
"""

import time

import numpy as np
import pytest

from src.chemi_accel.ai.multi_objective_optimization import (
    CrowdingDistance,
    HypervolumeCalculator,
    MPOptimizer,
    MultiObjectiveOptimizationError,
    MultiObjectiveResult,
    NonDominatedSorting,
    ObjectiveConfig,
    ParetoFrontResult,
    ScalarizationFunctions,
    create_drug_discovery_objectives,
    create_polymer_objectives,
    run_multi_objective_optimization,
)


class TestObjectiveConfig:
    """Test ObjectiveConfig dataclass."""

    def test_default_values(self):
        """Test default values for ObjectiveConfig."""
        config = ObjectiveConfig(name="test_objective")

        assert config.name == "test_objective"
        assert config.minimize is True
        assert config.weight == 1.0
        assert config.target is None
        assert config.bounds is None
        assert config.normalization == "none"

    def test_custom_values(self):
        """Test custom values for ObjectiveConfig."""
        config = ObjectiveConfig(
            name="molecular_weight",
            minimize=False,
            weight=2.0,
            target=350.0,
            bounds=(100.0, 500.0),
            normalization="minmax",
        )

        assert config.name == "molecular_weight"
        assert config.minimize is False
        assert config.weight == 2.0
        assert config.target == 350.0
        assert config.bounds == (100.0, 500.0)
        assert config.normalization == "minmax"


class TestNonDominatedSorting:
    """Test non-dominated sorting algorithms."""

    def setup_method(self):
        """Set up test data."""
        self.sorter_cpu = NonDominatedSorting(use_gpu=False)
        self.sorter_gpu = NonDominatedSorting(use_gpu=True)

        # Simple 2D test case
        self.objectives_2d = np.array(
            [
                [1.0, 5.0],  # Pareto optimal (best in obj2)
                [2.0, 3.0],  # Pareto optimal
                [3.0, 2.0],  # Pareto optimal
                [4.0, 4.0],  # Dominated by [2.0, 3.0]
                [5.0, 1.0],  # Pareto optimal (best in obj1)
            ]
        )

        # 3D test case
        self.objectives_3d = np.array(
            [
                [1.0, 1.0, 5.0],  # Pareto optimal
                [2.0, 2.0, 4.0],  # Dominated
                [3.0, 1.0, 3.0],  # Pareto optimal
                [1.0, 3.0, 3.0],  # Pareto optimal
                [4.0, 4.0, 2.0],  # Dominated
                [2.0, 1.0, 1.0],  # Pareto optimal
            ]
        )

    def test_cpu_sorting_2d(self):
        """Test CPU non-dominated sorting for 2D objectives."""
        fronts = self.sorter_cpu.fast_non_dominated_sort(self.objectives_2d)

        assert len(fronts) >= 1
        assert len(fronts[0]) == 4  # Four Pareto optimal solutions

        # Check that Pareto optimal solutions are correct
        pareto_indices = set(fronts[0])
        expected_pareto = {0, 1, 2, 4}  # Indices of Pareto optimal solutions
        assert pareto_indices == expected_pareto

    def test_cpu_sorting_3d(self):
        """Test CPU non-dominated sorting for 3D objectives."""
        fronts = self.sorter_cpu.fast_non_dominated_sort(self.objectives_3d)

        assert len(fronts) >= 1
        assert len(fronts[0]) == 4  # Four Pareto optimal solutions

        # Check that Pareto optimal solutions are correct
        pareto_indices = set(fronts[0])
        expected_pareto = {0, 2, 3, 5}  # Indices of Pareto optimal solutions
        assert pareto_indices == expected_pareto

    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU implementations give same results."""
        fronts_cpu = self.sorter_cpu.fast_non_dominated_sort(self.objectives_2d)
        fronts_gpu = self.sorter_gpu.fast_non_dominated_sort(self.objectives_2d)

        assert len(fronts_cpu) == len(fronts_gpu)

        for i, (front_cpu, front_gpu) in enumerate(zip(fronts_cpu, fronts_gpu)):
            assert set(front_cpu) == set(
                front_gpu
            ), f"Front {i} differs between CPU and GPU"

    def test_dominance_function(self):
        """Test the dominance comparison function."""
        obj1 = np.array([1.0, 2.0])
        obj2 = np.array([2.0, 3.0])
        obj3 = np.array([1.0, 3.0])
        obj4 = np.array([2.0, 2.0])

        # obj1 dominates obj2
        assert self.sorter_cpu._dominates(obj1, obj2)

        # obj1 does not dominate obj3 (equal in first objective)
        assert not self.sorter_cpu._dominates(obj1, obj3)

        # obj1 does not dominate obj4 (worse in second objective)
        assert not self.sorter_cpu._dominates(obj1, obj4)

    def test_empty_objectives(self):
        """Test handling of empty objectives array."""
        empty_objectives = np.array([]).reshape(0, 2)
        fronts = self.sorter_cpu.fast_non_dominated_sort(empty_objectives)

        assert len(fronts) == 0

    def test_single_objective(self):
        """Test handling of single objective."""
        single_obj = np.array([[1.0, 2.0]])
        fronts = self.sorter_cpu.fast_non_dominated_sort(single_obj)

        assert len(fronts) == 1
        assert len(fronts[0]) == 1
        assert fronts[0][0] == 0


class TestCrowdingDistance:
    """Test crowding distance calculation."""

    def setup_method(self):
        """Set up test data."""
        self.crowding_cpu = CrowdingDistance(use_gpu=False)
        self.crowding_gpu = CrowdingDistance(use_gpu=True)

        # Test objectives for crowding distance
        self.objectives = np.array(
            [
                [1.0, 5.0],
                [2.0, 4.0],
                [3.0, 3.0],
                [4.0, 2.0],
                [5.0, 1.0],
            ]
        )

        self.front_indices = np.array([0, 1, 2, 3, 4])

    def test_cpu_crowding_distance(self):
        """Test CPU crowding distance calculation."""
        distances = self.crowding_cpu.calculate_crowding_distance(
            self.objectives, self.front_indices
        )

        assert len(distances) == len(self.front_indices)

        # Boundary points should have infinite distance
        assert distances[0] == float("inf")
        assert distances[-1] == float("inf")

        # Interior points should have finite positive distances
        for i in range(1, len(distances) - 1):
            assert distances[i] > 0
            assert distances[i] != float("inf")

    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU implementations give same results."""
        distances_cpu = self.crowding_cpu.calculate_crowding_distance(
            self.objectives, self.front_indices
        )
        distances_gpu = self.crowding_gpu.calculate_crowding_distance(
            self.objectives, self.front_indices
        )

        np.testing.assert_array_almost_equal(distances_cpu, distances_gpu, decimal=6)

    def test_small_front(self):
        """Test crowding distance for small fronts."""
        small_front = np.array([0, 1])
        distances = self.crowding_cpu.calculate_crowding_distance(
            self.objectives, small_front
        )

        # All points in small fronts should have infinite distance
        assert all(d == float("inf") for d in distances)

    def test_single_point_front(self):
        """Test crowding distance for single point front."""
        single_front = np.array([0])
        distances = self.crowding_cpu.calculate_crowding_distance(
            self.objectives, single_front
        )

        assert len(distances) == 1
        assert distances[0] == float("inf")


class TestHypervolumeCalculator:
    """Test hypervolume calculation."""

    def setup_method(self):
        """Set up test data."""
        self.hv_calculator = HypervolumeCalculator()

        # Simple 2D Pareto front
        self.objectives_2d = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )

        self.pareto_indices_2d = np.array([0, 1, 2, 3])

    def test_hypervolume_calculation_2d(self):
        """Test hypervolume calculation for 2D objectives."""
        hypervolume = self.hv_calculator.calculate_hypervolume(
            self.objectives_2d, self.pareto_indices_2d
        )

        assert hypervolume > 0
        assert isinstance(hypervolume, float)

    def test_hypervolume_with_reference_point(self):
        """Test hypervolume calculation with custom reference point."""
        reference_point = np.array([5.0, 5.0])
        hv_calculator = HypervolumeCalculator(reference_point=reference_point)

        hypervolume = hv_calculator.calculate_hypervolume(
            self.objectives_2d, self.pareto_indices_2d
        )

        assert hypervolume > 0

    def test_empty_pareto_front(self):
        """Test hypervolume calculation with empty Pareto front."""
        empty_indices = np.array([])
        hypervolume = self.hv_calculator.calculate_hypervolume(
            self.objectives_2d, empty_indices
        )

        assert hypervolume == 0.0

    def test_single_point_pareto_front(self):
        """Test hypervolume calculation with single point Pareto front."""
        single_index = np.array([0])
        hypervolume = self.hv_calculator.calculate_hypervolume(
            self.objectives_2d, single_index
        )

        assert hypervolume > 0


class TestScalarizationFunctions:
    """Test scalarization functions."""

    def setup_method(self):
        """Set up test data."""
        self.scalarization = ScalarizationFunctions()

        self.objectives = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 3.0],
                [3.0, 3.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        self.weights = np.array([0.5, 0.3, 0.2])
        self.ideal_point = np.array([0.5, 0.5, 0.5])

    def test_weighted_sum(self):
        """Test weighted sum scalarization."""
        scores = self.scalarization.weighted_sum(self.objectives, self.weights)

        assert len(scores) == len(self.objectives)
        assert all(isinstance(score, (int, float)) for score in scores)

        # Check calculation for first point
        expected_first = 1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2
        assert abs(scores[0] - expected_first) < 1e-10

    def test_tchebycheff(self):
        """Test Tchebycheff scalarization."""
        scores = self.scalarization.tchebycheff(
            self.objectives, self.weights, self.ideal_point
        )

        assert len(scores) == len(self.objectives)
        assert all(score >= 0 for score in scores)

    def test_achievement_scalarizing_function(self):
        """Test achievement scalarizing function."""
        reference_point = np.array([1.5, 1.5, 1.5])
        scores = self.scalarization.achievement_scalarizing_function(
            self.objectives, self.weights, reference_point
        )

        assert len(scores) == len(self.objectives)
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_penalty_boundary_intersection(self):
        """Test penalty boundary intersection scalarization."""
        scores = self.scalarization.penalty_boundary_intersection(
            self.objectives, self.weights, self.ideal_point
        )

        assert len(scores) == len(self.objectives)
        assert all(score >= 0 for score in scores)


class TestMPOptimizer:
    """Test the main MPOptimizer class."""

    def setup_method(self):
        """Set up test data."""
        self.objectives = [
            ObjectiveConfig(name="obj1", minimize=True, weight=1.0),
            ObjectiveConfig(name="obj2", minimize=True, weight=1.0),
        ]

        self.optimizer = MPOptimizer(objectives=self.objectives, use_gpu=False)

        self.molecules = ["mol1", "mol2", "mol3", "mol4", "mol5"]
        self.properties = {
            "obj1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "obj2": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert len(self.optimizer.objectives) == 2
        assert self.optimizer.use_gpu is False
        assert len(self.optimizer.optimization_history) == 0

    def test_objective_validation(self):
        """Test objective validation."""
        # Test empty objectives
        with pytest.raises(MultiObjectiveOptimizationError):
            MPOptimizer(objectives=[])

        # Test invalid objective type
        with pytest.raises(MultiObjectiveOptimizationError):
            MPOptimizer(objectives=["invalid"])

    def test_optimize(self):
        """Test the main optimize method."""
        result = self.optimizer.optimize(self.molecules, self.properties)

        assert isinstance(result, MultiObjectiveResult)
        assert len(result.molecules) == len(self.molecules)
        assert result.objective_values.shape == (
            len(self.molecules),
            len(self.objectives),
        )
        assert isinstance(result.pareto_front, ParetoFrontResult)
        assert result.scalarized_scores is not None
        assert result.diversity_metrics is not None
        assert result.convergence_metrics is not None

    def test_extract_objective_values(self):
        """Test objective value extraction."""
        objective_values = self.optimizer._extract_objective_values(self.properties)

        assert objective_values.shape == (len(self.molecules), len(self.objectives))

        # Check that values are correctly extracted and transformed
        np.testing.assert_array_equal(objective_values[:, 0], self.properties["obj1"])
        np.testing.assert_array_equal(objective_values[:, 1], self.properties["obj2"])

    def test_normalize_objectives(self):
        """Test objective normalization."""
        objective_values = self.optimizer._extract_objective_values(self.properties)

        # Test minmax normalization
        self.optimizer.objectives[0].normalization = "minmax"
        normalized = self.optimizer._normalize_objectives(objective_values)

        # First objective should be normalized to [0, 1]
        assert np.min(normalized[:, 0]) == 0.0
        assert np.max(normalized[:, 0]) == 1.0

        # Second objective should remain unchanged
        np.testing.assert_array_equal(normalized[:, 1], objective_values[:, 1])

    def test_missing_property(self):
        """Test handling of missing properties."""
        incomplete_properties = {"obj1": np.array([1.0, 2.0, 3.0])}

        with pytest.raises(MultiObjectiveOptimizationError):
            self.optimizer.optimize(self.molecules, incomplete_properties)

    def test_scalarization_guidance(self):
        """Test scalarization guidance for generative models."""
        objective_values = np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
                [3.0, 3.0],
            ]
        )

        # Test weighted sum
        scores = self.optimizer.get_scalarization_guidance(
            objective_values, method="weighted_sum"
        )
        assert len(scores) == len(objective_values)

        # Test Tchebycheff
        scores = self.optimizer.get_scalarization_guidance(
            objective_values, method="tchebycheff"
        )
        assert len(scores) == len(objective_values)

        # Test unknown method
        with pytest.raises(MultiObjectiveOptimizationError):
            self.optimizer.get_scalarization_guidance(
                objective_values, method="unknown"
            )

    def test_population_size_limit(self):
        """Test population size limiting."""
        result = self.optimizer.optimize(
            self.molecules, self.properties, population_size=3
        )

        assert len(result.molecules) == 3
        assert result.objective_values.shape[0] == 3

    def test_optimization_summary(self):
        """Test optimization summary generation."""
        # Test empty history
        summary = self.optimizer.get_optimization_summary()
        assert "message" in summary

        # Test with optimization results
        self.optimizer.optimize(self.molecules, self.properties)
        summary = self.optimizer.get_optimization_summary()

        assert "n_objectives" in summary
        assert "n_molecules" in summary
        assert "n_pareto_solutions" in summary
        assert "hypervolume" in summary
        assert "computation_time" in summary

    def test_export_pareto_front(self, tmp_path):
        """Test Pareto front export functionality."""
        result = self.optimizer.optimize(self.molecules, self.properties)

        # Test CSV export
        csv_path = tmp_path / "pareto_front.csv"
        self.optimizer.export_pareto_front(result, str(csv_path))
        assert csv_path.exists()

        # Test JSON export
        json_path = tmp_path / "pareto_front.json"
        self.optimizer.export_pareto_front(result, str(json_path))
        assert json_path.exists()

        # Test unsupported format
        with pytest.raises(MultiObjectiveOptimizationError):
            self.optimizer.export_pareto_front(result, "test.txt")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_drug_discovery_objectives(self):
        """Test drug discovery objectives creation."""
        objectives = create_drug_discovery_objectives()

        assert len(objectives) == 4
        assert all(isinstance(obj, ObjectiveConfig) for obj in objectives)

        objective_names = [obj.name for obj in objectives]
        expected_names = ["molecular_weight", "logp", "tpsa", "synthetic_accessibility"]
        assert objective_names == expected_names

    def test_create_polymer_objectives(self):
        """Test polymer objectives creation."""
        objectives = create_polymer_objectives()

        assert len(objectives) == 3
        assert all(isinstance(obj, ObjectiveConfig) for obj in objectives)

        objective_names = [obj.name for obj in objectives]
        expected_names = [
            "glass_transition_temp",
            "youngs_modulus",
            "synthetic_accessibility",
        ]
        assert objective_names == expected_names

    def test_run_multi_objective_optimization(self):
        """Test the complete optimization workflow function."""
        molecules = ["mol1", "mol2", "mol3", "mol4"]
        properties = {
            "molecular_weight": np.array([200.0, 300.0, 400.0, 500.0]),
            "logp": np.array([1.0, 2.0, 3.0, 4.0]),
        }

        objectives = [
            ObjectiveConfig(name="molecular_weight", minimize=True),
            ObjectiveConfig(name="logp", minimize=True),
        ]

        result = run_multi_objective_optimization(
            molecules, properties, objectives, use_gpu=False
        )

        assert isinstance(result, MultiObjectiveResult)
        assert len(result.molecules) == len(molecules)


class TestPerformance:
    """Performance tests for large-scale multi-objective problems."""

    def test_large_scale_performance(self):
        """Test performance with large datasets."""
        n_molecules = 10000
        n_objectives = 5

        # Generate synthetic data
        molecules = [f"mol_{i}" for i in range(n_molecules)]
        properties = {}

        for i in range(n_objectives):
            properties[f"obj_{i}"] = np.random.rand(n_molecules)

        objectives = [
            ObjectiveConfig(name=f"obj_{i}", minimize=True) for i in range(n_objectives)
        ]

        # Test CPU performance
        start_time = time.time()
        optimizer_cpu = MPOptimizer(objectives=objectives, use_gpu=False)
        result_cpu = optimizer_cpu.optimize(molecules, properties)
        cpu_time = time.time() - start_time

        # Test GPU performance (if available)
        start_time = time.time()
        optimizer_gpu = MPOptimizer(objectives=objectives, use_gpu=True)
        result_gpu = optimizer_gpu.optimize(molecules, properties)
        gpu_time = time.time() - start_time

        print(f"CPU time: {cpu_time:.2f}s")
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # Verify results are consistent
        assert len(result_cpu.pareto_front.pareto_indices) > 0
        assert len(result_gpu.pareto_front.pareto_indices) > 0

        # GPU should be faster or at least comparable
        assert gpu_time <= cpu_time * 2.0  # Allow some overhead

    def test_scalability_with_objectives(self):
        """Test scalability with increasing number of objectives."""
        n_molecules = 1000

        for n_objectives in [2, 5, 10, 20]:
            molecules = [f"mol_{i}" for i in range(n_molecules)]
            properties = {}

            for i in range(n_objectives):
                properties[f"obj_{i}"] = np.random.rand(n_molecules)

            objectives = [
                ObjectiveConfig(name=f"obj_{i}", minimize=True)
                for i in range(n_objectives)
            ]

            start_time = time.time()
            optimizer = MPOptimizer(objectives=objectives, use_gpu=False)
            result = optimizer.optimize(molecules, properties)
            computation_time = time.time() - start_time

            print(
                f"Objectives: {n_objectives}, Time: {computation_time:.2f}s, "
                f"Pareto solutions: {len(result.pareto_front.pareto_indices)}"
            )

            assert len(result.pareto_front.pareto_indices) > 0
            assert computation_time < 60.0  # Should complete within 1 minute

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Large dataset
        n_molecules = 50000
        molecules = [f"mol_{i}" for i in range(n_molecules)]
        properties = {
            "obj1": np.random.rand(n_molecules),
            "obj2": np.random.rand(n_molecules),
            "obj3": np.random.rand(n_molecules),
        }

        objectives = [
            ObjectiveConfig(name="obj1", minimize=True),
            ObjectiveConfig(name="obj2", minimize=True),
            ObjectiveConfig(name="obj3", minimize=True),
        ]

        optimizer = MPOptimizer(objectives=objectives, use_gpu=False)
        result = optimizer.optimize(molecules, properties)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Memory per molecule: {memory_increase / n_molecules * 1024:.2f} KB")

        # Memory usage should be reasonable
        assert memory_increase < 2000  # Less than 2GB increase
        assert len(result.pareto_front.pareto_indices) > 0


if __name__ == "__main__":
    pytest.main([__file__])
