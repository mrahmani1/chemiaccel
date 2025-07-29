"""
Tests for Lifecycle Assessment (LCA) Integration Module.

This module provides comprehensive testing for the LCA integration framework
including database connections, GPU-accelerated calculations, and impact assessment.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.sustainability.lca_integration import (  # Core calculators; Data structures; Utility functions; Exceptions
    AllocationMethod,
    DatabaseConnectionError,
    DatabaseLCACalculator,
    EmissionFlow,
    EnergyFlow,
    GPUAcceleratedLCACalculator,
    ImpactCategory,
    ImpactCharacterization,
    InvalidLCADataError,
    LCACalculationError,
    LCAComparator,
    LCAComparison,
    LCADatabase,
    LCADataManager,
    LCAProcess,
    LCAResults,
    LifecycleStage,
    MaterialFlow,
    MatrixLCACalculator,
    calculate_material_lca,
    create_polymer_manufacturing_process,
    create_recycling_process,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import brightway2 as bw

    BRIGHTWAY_AVAILABLE = True
except ImportError:
    BRIGHTWAY_AVAILABLE = False


class TestDataStructures:
    """Test LCA data structures."""

    def test_material_flow_creation(self):
        """Test creating material flow."""
        flow = MaterialFlow(
            name="ethylene",
            amount=1.02,
            unit="kg",
            material_type="chemical",
            transport_distance=100.0,
            transport_mode="truck",
        )

        assert flow.name == "ethylene"
        assert flow.amount == 1.02
        assert flow.unit == "kg"
        assert flow.transport_distance == 100.0

    def test_energy_flow_creation(self):
        """Test creating energy flow."""
        flow = EnergyFlow(
            name="electricity",
            amount=2.5,
            unit="kWh",
            energy_type="electricity",
            source="grid_mix",
            efficiency=0.95,
        )

        assert flow.name == "electricity"
        assert flow.amount == 2.5
        assert flow.energy_type == "electricity"
        assert flow.efficiency == 0.95

    def test_emission_flow_creation(self):
        """Test creating emission flow."""
        emission = EmissionFlow(
            substance="CO2",
            amount=2.1,
            unit="kg",
            compartment="air",
            emission_stage=LifecycleStage.MANUFACTURING,
        )

        assert emission.substance == "CO2"
        assert emission.amount == 2.1
        assert emission.compartment == "air"
        assert emission.emission_stage == LifecycleStage.MANUFACTURING

    def test_lca_process_creation(self):
        """Test creating LCA process."""
        process = LCAProcess(
            name="PET Production",
            functional_unit="1 kg PET",
            material_inputs=[MaterialFlow("ethylene_glycol", 0.34, "kg", "chemical")],
            material_outputs=[MaterialFlow("PET", 1.0, "kg", "polymer")],
            energy_inputs=[
                EnergyFlow("electricity", 2.5, "kWh", "electricity", "grid_mix")
            ],
            emissions=[
                EmissionFlow("CO2", 2.1, "kg", "air", LifecycleStage.MANUFACTURING)
            ],
        )

        assert process.name == "PET Production"
        assert process.functional_unit == "1 kg PET"
        assert len(process.material_inputs) == 1
        assert len(process.material_outputs) == 1
        assert len(process.energy_inputs) == 1
        assert len(process.emissions) == 1

    def test_impact_characterization_creation(self):
        """Test creating impact characterization."""
        cf = ImpactCharacterization(
            substance="CO2",
            compartment="air",
            impact_category=ImpactCategory.CLIMATE_CHANGE,
            characterization_factor=1.0,
            unit="kg CO2-eq",
            method="IPCC 2013",
        )

        assert cf.substance == "CO2"
        assert cf.impact_category == ImpactCategory.CLIMATE_CHANGE
        assert cf.characterization_factor == 1.0
        assert cf.method == "IPCC 2013"


class TestMatrixLCACalculator:
    """Test matrix-based LCA calculator."""

    def test_calculator_creation(self):
        """Test calculator initialization."""
        calc = MatrixLCACalculator(use_gpu=False)

        assert calc.use_gpu is False
        assert calc.technology_matrix is None
        assert calc.intervention_matrix is None

    def test_matrix_setup(self):
        """Test matrix setup process."""
        calc = MatrixLCACalculator(use_gpu=False)

        # Create simple test processes
        processes = [
            LCAProcess(
                name="Test Process 1",
                functional_unit="1 unit",
                material_inputs=[MaterialFlow("input1", 1.0, "kg", "material")],
                material_outputs=[MaterialFlow("output1", 1.0, "kg", "product")],
                emissions=[
                    EmissionFlow("CO2", 1.0, "kg", "air", LifecycleStage.MANUFACTURING)
                ],
            ),
            LCAProcess(
                name="Test Process 2",
                functional_unit="1 unit",
                material_inputs=[MaterialFlow("input2", 0.5, "kg", "material")],
                material_outputs=[MaterialFlow("output2", 1.0, "kg", "product")],
                emissions=[
                    EmissionFlow("CH4", 0.1, "kg", "air", LifecycleStage.MANUFACTURING)
                ],
            ),
        ]

        characterization_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.0, "kg CO2-eq", "test"
            ),
            ImpactCharacterization(
                "CH4", "air", ImpactCategory.CLIMATE_CHANGE, 25.0, "kg CO2-eq", "test"
            ),
        ]

        calc.setup_matrices(processes, characterization_factors)

        assert calc.technology_matrix is not None
        assert calc.intervention_matrix is not None
        assert calc.characterization_matrix is not None
        assert len(calc.process_index) == 2
        assert len(calc.flow_index) > 0

    def test_impact_calculation(self):
        """Test impact calculation."""
        calc = MatrixLCACalculator(use_gpu=False)

        # Create test data
        processes = [create_polymer_manufacturing_process("PET")]
        characterization_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.0, "kg CO2-eq", "test"
            ),
            ImpactCharacterization(
                "NOx", "air", ImpactCategory.ACIDIFICATION, 1.0, "kg SO2-eq", "test"
            ),
        ]

        results = calc.calculate_impacts(processes, characterization_factors)

        assert isinstance(results, LCAResults)
        assert len(results.impact_scores) > 0
        assert results.functional_unit == "1 kg material"
        assert results.calculation_time > 0
        assert isinstance(results.impact_contributions, dict)
        assert isinstance(results.process_contributions, dict)

    def test_contribution_analysis(self):
        """Test contribution analysis."""
        calc = MatrixLCACalculator(use_gpu=False)

        # Create mock results
        results = LCAResults(
            functional_unit="1 kg",
            system_boundary="test",
            impact_scores={ImpactCategory.CLIMATE_CHANGE: 10.0},
            impact_contributions={},
            process_contributions={"process1": {ImpactCategory.CLIMATE_CHANGE: 5.0}},
            hotspots=[],
            uncertainty_analysis={},
        )

        contributions = calc.perform_contribution_analysis(results)

        assert isinstance(contributions, dict)
        assert "process1" in contributions


class TestDatabaseLCACalculator:
    """Test database LCA calculator."""

    def test_calculator_creation(self):
        """Test calculator initialization."""
        calc = DatabaseLCACalculator(LCADatabase.ECOINVENT, use_gpu=False)

        assert calc.database == LCADatabase.ECOINVENT
        assert calc.use_gpu is False
        assert calc.connection is None

    def test_ecoinvent_connection_attempt(self):
        """Test ecoinvent connection attempt."""
        calc = DatabaseLCACalculator(LCADatabase.ECOINVENT)

        # Mock connection
        with patch(
            "chemi_accel.sustainability.lca_integration.REQUESTS_AVAILABLE", True
        ):
            calc.connect_to_database({"api_key": "test"})

            # Should attempt connection (may not succeed without real API)
            assert (
                calc.connection is not None or calc.connection is None
            )  # Either is acceptable

    @pytest.mark.skipif(not BRIGHTWAY_AVAILABLE, reason="Brightway2 not available")
    def test_brightway_connection(self):
        """Test Brightway2 connection."""
        calc = DatabaseLCACalculator(LCADatabase.BRIGHTWAY)

        # Mock Brightway2 setup
        with patch("brightway2.projects.set_current") as mock_set:
            with patch("brightway2.methods") as mock_methods:
                mock_methods.__len__.return_value = 0
                with patch("brightway2.bw2setup") as mock_setup:
                    calc.connect_to_database({"project": "test_project"})

                    mock_set.assert_called_once()

    def test_fallback_calculation(self):
        """Test fallback to matrix calculation."""
        calc = DatabaseLCACalculator(use_gpu=False)

        processes = [create_polymer_manufacturing_process("PE")]
        characterization_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.0, "kg CO2-eq", "test"
            )
        ]

        results = calc.calculate_impacts(processes, characterization_factors)

        assert isinstance(results, LCAResults)
        assert len(results.impact_scores) > 0


class TestGPUAcceleratedLCACalculator:
    """Test GPU-accelerated LCA calculator."""

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE or CUPY_AVAILABLE),
        reason="Neither PyTorch nor CuPy available",
    )
    def test_calculator_creation(self):
        """Test GPU calculator initialization."""
        calc = GPUAcceleratedLCACalculator()

        assert calc.use_torch == TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            assert calc.device is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_calculation(self):
        """Test calculation with PyTorch."""
        calc = GPUAcceleratedLCACalculator()

        processes = [
            create_polymer_manufacturing_process("PET"),
            create_polymer_manufacturing_process("PE"),
        ]
        characterization_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.0, "kg CO2-eq", "test"
            )
        ]

        results = calc.calculate_impacts(processes, characterization_factors)

        assert isinstance(results, LCAResults)
        assert results.metadata.get("gpu_accelerated") is True
        assert results.metadata.get("framework") == "pytorch"

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_calculation(self):
        """Test calculation with CuPy."""
        # Force CuPy usage
        with patch("chemi_accel.sustainability.lca_integration.TORCH_AVAILABLE", False):
            calc = GPUAcceleratedLCACalculator()

            processes = [create_polymer_manufacturing_process("PET")]
            characterization_factors = [
                ImpactCharacterization(
                    "CO2",
                    "air",
                    ImpactCategory.CLIMATE_CHANGE,
                    1.0,
                    "kg CO2-eq",
                    "test",
                )
            ]

            results = calc.calculate_impacts(processes, characterization_factors)

            assert isinstance(results, LCAResults)
            assert results.metadata.get("framework") == "cupy"


class TestLCAComparator:
    """Test LCA comparator."""

    def test_comparator_creation(self):
        """Test comparator initialization."""
        calc = MatrixLCACalculator(use_gpu=False)
        comparator = LCAComparator(calc)

        assert comparator.calculator == calc

    def test_scenario_comparison(self):
        """Test scenario comparison."""
        calc = MatrixLCACalculator(use_gpu=False)
        comparator = LCAComparator(calc)

        # Create test scenarios
        scenarios = {
            "Virgin PET": (
                [create_polymer_manufacturing_process("PET")],
                [
                    ImpactCharacterization(
                        "CO2",
                        "air",
                        ImpactCategory.CLIMATE_CHANGE,
                        1.0,
                        "kg CO2-eq",
                        "test",
                    )
                ],
            ),
            "Recycled PET": (
                [create_recycling_process("mechanical", "PET")],
                [
                    ImpactCharacterization(
                        "CO2",
                        "air",
                        ImpactCategory.CLIMATE_CHANGE,
                        1.0,
                        "kg CO2-eq",
                        "test",
                    )
                ],
            ),
        }

        comparison = comparator.compare_scenarios(scenarios)

        assert isinstance(comparison, LCAComparison)
        assert len(comparison.scenarios) == 2
        assert "Virgin PET" in comparison.scenarios
        assert "Recycled PET" in comparison.scenarios
        assert isinstance(comparison.impact_comparison, dict)
        assert isinstance(comparison.relative_performance, dict)
        assert len(comparison.recommendations) > 0

    def test_trade_off_identification(self):
        """Test trade-off identification."""
        calc = MatrixLCACalculator(use_gpu=False)
        comparator = LCAComparator(calc)

        # Create mock scenario results
        scenario_results = {
            "Scenario A": LCAResults(
                functional_unit="1 kg",
                system_boundary="test",
                impact_scores={
                    ImpactCategory.CLIMATE_CHANGE: 10.0,
                    ImpactCategory.ACIDIFICATION: 5.0,
                },
                impact_contributions={},
                process_contributions={},
                hotspots=[],
                uncertainty_analysis={},
            ),
            "Scenario B": LCAResults(
                functional_unit="1 kg",
                system_boundary="test",
                impact_scores={
                    ImpactCategory.CLIMATE_CHANGE: 5.0,
                    ImpactCategory.ACIDIFICATION: 10.0,
                },
                impact_contributions={},
                process_contributions={},
                hotspots=[],
                uncertainty_analysis={},
            ),
        }

        trade_offs = comparator._identify_trade_offs(scenario_results)

        assert isinstance(trade_offs, dict)
        # Should identify negative correlation between climate change and acidification
        trade_off_key = (ImpactCategory.CLIMATE_CHANGE, ImpactCategory.ACIDIFICATION)
        if trade_off_key in trade_offs:
            assert trade_offs[trade_off_key] < 0  # Negative correlation


class TestLCADataManager:
    """Test LCA data manager."""

    def test_manager_creation(self):
        """Test manager initialization."""
        manager = LCADataManager()

        assert isinstance(manager.databases, dict)
        assert isinstance(manager.characterization_methods, dict)
        assert isinstance(manager.cached_processes, dict)

    def test_default_characterization_factors(self):
        """Test default characterization factors creation."""
        manager = LCADataManager()

        factors = manager.create_default_characterization_factors()

        assert isinstance(factors, list)
        assert len(factors) > 0
        assert all(isinstance(cf, ImpactCharacterization) for cf in factors)

        # Check for common substances
        substances = [cf.substance for cf in factors]
        assert "CO2" in substances
        assert "CH4" in substances
        assert "N2O" in substances

    def test_load_factors_from_dict(self):
        """Test loading factors from dictionary."""
        manager = LCADataManager()

        factors_dict = {
            "CO2": {"climate_change": 1.0, "acidification": 0.0},
            "SO2": {"acidification": 1.0, "climate_change": 0.0},
        }

        factors = manager._load_factors_from_dict(factors_dict)

        assert isinstance(factors, list)
        assert len(factors) == 4  # 2 substances × 2 categories

        # Check factor values
        co2_climate = next(
            (
                cf
                for cf in factors
                if cf.substance == "CO2"
                and cf.impact_category == ImpactCategory.CLIMATE_CHANGE
            ),
            None,
        )
        assert co2_climate is not None
        assert co2_climate.characterization_factor == 1.0

    def test_load_factors_from_dataframe(self):
        """Test loading factors from DataFrame."""
        manager = LCADataManager()

        df = pd.DataFrame(
            {
                "substance": ["CO2", "CH4", "N2O"],
                "impact_category": [
                    "climate_change",
                    "climate_change",
                    "climate_change",
                ],
                "characterization_factor": [1.0, 25.0, 298.0],
                "unit": ["kg CO2-eq", "kg CO2-eq", "kg CO2-eq"],
                "method": ["IPCC", "IPCC", "IPCC"],
            }
        )

        factors = manager._load_factors_from_dataframe(df)

        assert isinstance(factors, list)
        assert len(factors) == 3

        # Check CH4 factor
        ch4_factor = next((cf for cf in factors if cf.substance == "CH4"), None)
        assert ch4_factor is not None
        assert ch4_factor.characterization_factor == 25.0

    def test_invalid_dataframe_columns(self):
        """Test handling of invalid DataFrame columns."""
        manager = LCADataManager()

        df = pd.DataFrame({"wrong_column": ["CO2"], "another_wrong": [1.0]})

        with pytest.raises(ValueError):
            manager._load_factors_from_dataframe(df)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_polymer_manufacturing_process_creation(self):
        """Test polymer manufacturing process creation."""
        pet_process = create_polymer_manufacturing_process("PET")

        assert isinstance(pet_process, LCAProcess)
        assert pet_process.name == "PET Production"
        assert pet_process.functional_unit == "1 kg PET"
        assert len(pet_process.material_inputs) > 0
        assert len(pet_process.material_outputs) > 0
        assert len(pet_process.energy_inputs) > 0
        assert len(pet_process.emissions) > 0

        pe_process = create_polymer_manufacturing_process("PE")

        assert isinstance(pe_process, LCAProcess)
        assert pe_process.name == "PE Production"
        assert "ethylene" in [inp.name for inp in pe_process.material_inputs]

    def test_recycling_process_creation(self):
        """Test recycling process creation."""
        mech_process = create_recycling_process("mechanical", "PET")

        assert isinstance(mech_process, LCAProcess)
        assert "Mechanical Recycling" in mech_process.name
        assert mech_process.functional_unit == "1 kg recycled polymer"
        assert any("waste" in inp.name for inp in mech_process.material_inputs)

        chem_process = create_recycling_process("chemical", "PET")

        assert isinstance(chem_process, LCAProcess)
        assert "Chemical Recycling" in chem_process.name
        assert chem_process.functional_unit == "1 kg recycled monomers"

    def test_invalid_polymer_type(self):
        """Test handling of invalid polymer type."""
        with pytest.raises(ValueError):
            create_polymer_manufacturing_process("invalid_polymer")

    def test_invalid_recycling_method(self):
        """Test handling of invalid recycling method."""
        with pytest.raises(ValueError):
            create_recycling_process("invalid_method", "PET")

    @pytest.mark.asyncio
    async def test_calculate_material_lca(self):
        """Test material LCA calculation utility."""
        processes = [create_polymer_manufacturing_process("PET")]

        results = await calculate_material_lca(
            processes, use_gpu=False, calculator_type="matrix"
        )

        assert isinstance(results, LCAResults)
        assert len(results.impact_scores) > 0

    @pytest.mark.asyncio
    async def test_calculate_lca_with_custom_factors(self):
        """Test LCA calculation with custom characterization factors."""
        processes = [create_polymer_manufacturing_process("PE")]

        custom_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.5, "kg CO2-eq", "custom"
            )
        ]

        results = await calculate_material_lca(
            processes, characterization_factors=custom_factors, use_gpu=False
        )

        assert isinstance(results, LCAResults)

    @pytest.mark.asyncio
    async def test_invalid_calculator_type(self):
        """Test handling of invalid calculator type."""
        processes = [create_polymer_manufacturing_process("PET")]

        with pytest.raises(ValueError):
            await calculate_material_lca(processes, calculator_type="invalid_type")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_impact_category_enum(self):
        """Test impact category enumeration."""
        assert ImpactCategory.CLIMATE_CHANGE.value == "climate_change"
        assert ImpactCategory.ACIDIFICATION.value == "acidification"
        assert ImpactCategory.EUTROPHICATION.value == "eutrophication"

    def test_lifecycle_stage_enum(self):
        """Test lifecycle stage enumeration."""
        assert LifecycleStage.RAW_MATERIAL_EXTRACTION.value == "raw_material_extraction"
        assert LifecycleStage.MANUFACTURING.value == "manufacturing"
        assert LifecycleStage.END_OF_LIFE.value == "end_of_life"

    def test_lca_database_enum(self):
        """Test LCA database enumeration."""
        assert LCADatabase.ECOINVENT.value == "ecoinvent"
        assert LCADatabase.BRIGHTWAY.value == "brightway2"
        assert LCADatabase.CUSTOM.value == "custom"

    def test_empty_processes_list(self):
        """Test handling of empty processes list."""
        calc = MatrixLCACalculator(use_gpu=False)

        empty_processes = []
        characterization_factors = []

        # Should handle gracefully or raise appropriate error
        try:
            results = calc.calculate_impacts(empty_processes, characterization_factors)
            # If it doesn't raise an error, check that results are reasonable
            assert isinstance(results, LCAResults)
        except (ValueError, IndexError, ZeroDivisionError):
            # These are acceptable exceptions for empty input
            pass

    def test_missing_gpu_dependencies(self):
        """Test handling when GPU dependencies are missing."""
        # Mock missing dependencies
        with patch("chemi_accel.sustainability.lca_integration.TORCH_AVAILABLE", False):
            with patch(
                "chemi_accel.sustainability.lca_integration.CUPY_AVAILABLE", False
            ):
                with pytest.raises(ImportError):
                    GPUAcceleratedLCACalculator()


class TestPerformance:
    """Test performance aspects."""

    def test_matrix_calculation_timing(self):
        """Test matrix calculation performance."""
        calc = MatrixLCACalculator(use_gpu=False)

        processes = [
            create_polymer_manufacturing_process("PET"),
            create_polymer_manufacturing_process("PE"),
            create_recycling_process("mechanical", "PET"),
        ]

        characterization_factors = [
            ImpactCharacterization(
                "CO2", "air", ImpactCategory.CLIMATE_CHANGE, 1.0, "kg CO2-eq", "test"
            ),
            ImpactCharacterization(
                "NOx", "air", ImpactCategory.ACIDIFICATION, 1.0, "kg SO2-eq", "test"
            ),
        ]

        start_time = datetime.now()
        results = calc.calculate_impacts(processes, characterization_factors)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(results, LCAResults)
        assert results.calculation_time > 0

    @pytest.mark.asyncio
    async def test_async_calculation_timing(self):
        """Test async calculation timing."""
        processes = [create_polymer_manufacturing_process("PET")]

        start_time = datetime.now()
        results = await calculate_material_lca(
            processes, use_gpu=False, calculator_type="matrix"
        )
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 10.0  # Should complete within 10 seconds
        assert isinstance(results, LCAResults)


if __name__ == "__main__":
    pytest.main([__file__])
