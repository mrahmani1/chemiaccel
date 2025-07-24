"""
Tests for Automated Experiment Planning and Instruction Generation

This module contains comprehensive tests for the autonomous experiment planning
functionality including synthesis and characterization instruction generation.
"""

import json
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import List, Dict, Any

from chemi_accel.autonomous.experiment_planner import (
    ExperimentPlanner,
    SynthesisInstruction,
    CharacterizationInstruction,
    InstructionSet,
    ProtocolStep
)
from chemi_accel.autonomous.schemas import SchemaValidator


class TestExperimentPlanner:
    """Test cases for ExperimentPlanner class."""
    
    @pytest.fixture
    def sample_lab_config(self) -> Dict[str, Any]:
        """Sample laboratory configuration for testing."""
        return {
            "liquid_handler": {
                "type": "tecan",
                "model": "Freedom EVO 200",
                "software_version": "3.7"
            },
            "synthesis_platforms": ["manual", "automated"],
            "characterization_equipment": ["NMR", "LCMS", "HPLC", "IR"],
            "available_reagents": ["standard_organic"],
            "safety_protocols": ["fume_hood", "ppe_standard"]
        }
    
    @pytest.fixture
    def experiment_planner(self, sample_lab_config) -> ExperimentPlanner:
        """Create ExperimentPlanner instance for testing."""
        return ExperimentPlanner(lab_config=sample_lab_config)
    
    @pytest.fixture
    def sample_molecules(self) -> List[str]:
        """Sample molecule SMILES for testing."""
        return [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC"  # Triethylamine
        ]
    
    def test_initialization(self, sample_lab_config):
        """Test ExperimentPlanner initialization."""
        planner = ExperimentPlanner(lab_config=sample_lab_config)
        
        assert planner.lab_config == sample_lab_config
        assert isinstance(planner.validator, SchemaValidator)
        assert planner.reagent_database is not None
        assert planner.synthesis_templates is not None
    
    def test_default_initialization(self):
        """Test ExperimentPlanner with default configuration."""
        planner = ExperimentPlanner()
        
        assert planner.lab_config is not None
        assert "liquid_handler" in planner.lab_config
        assert "synthesis_platforms" in planner.lab_config
    
    def test_generate_synthesis_instructions(self, experiment_planner, sample_molecules):
        """Test synthesis instruction generation."""
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=sample_molecules[:2],
            synthesis_strategy="automated"
        )
        
        assert len(instructions) == 2
        assert all(isinstance(instr, SynthesisInstruction) for instr in instructions)
        
        # Check first instruction structure
        first_instruction = instructions[0]
        assert first_instruction.synthesis_id is not None
        assert first_instruction.target_compound["smiles"] == sample_molecules[0]
        assert len(first_instruction.reagents) > 0
        assert len(first_instruction.procedure_steps) > 0
        assert first_instruction.estimated_duration is not None
        assert first_instruction.success_probability is not None
    
    def test_generate_synthesis_instructions_invalid_smiles(self, experiment_planner):
        """Test synthesis instruction generation with invalid SMILES."""
        invalid_molecules = ["INVALID_SMILES", "CCO", "ALSO_INVALID"]
        
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=invalid_molecules
        )
        
        # Should only generate instruction for valid SMILES (CCO)
        assert len(instructions) == 1
        assert instructions[0].target_compound["smiles"] == "CCO"
    
    def test_generate_characterization_instructions(self, experiment_planner):
        """Test characterization instruction generation."""
        sample_ids = ["sample_1", "sample_2", "sample_3"]
        expected_compounds = ["CCO", "CC(=O)O", "c1ccccc1"]
        
        instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=sample_ids,
            expected_compounds=expected_compounds,
            characterization_scope="full"
        )
        
        assert len(instructions) == 3
        assert all(isinstance(instr, CharacterizationInstruction) for instr in instructions)
        
        # Check first instruction
        first_instruction = instructions[0]
        assert first_instruction.characterization_id is not None
        assert first_instruction.sample_id == "sample_1"
        assert len(first_instruction.techniques) > 0
        assert first_instruction.estimated_duration is not None
    
    def test_characterization_scope_variations(self, experiment_planner):
        """Test different characterization scopes."""
        sample_ids = ["sample_1"]
        expected_compounds = ["CCO"]
        
        # Test minimal scope
        minimal_instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=sample_ids,
            expected_compounds=expected_compounds,
            characterization_scope="minimal"
        )
        
        # Test standard scope
        standard_instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=sample_ids,
            expected_compounds=expected_compounds,
            characterization_scope="standard"
        )
        
        # Test full scope
        full_instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=sample_ids,
            expected_compounds=expected_compounds,
            characterization_scope="full"
        )
        
        # Full scope should have more techniques than standard, standard more than minimal
        assert len(full_instructions[0].techniques) >= len(standard_instructions[0].techniques)
        assert len(standard_instructions[0].techniques) >= len(minimal_instructions[0].techniques)
    
    def test_generate_liquid_handler_protocol(self, experiment_planner, sample_molecules):
        """Test liquid handler protocol generation."""
        # First generate synthesis instructions
        synthesis_instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=sample_molecules[:2]
        )
        
        # Generate liquid handler protocol
        protocol = experiment_planner.generate_liquid_handler_protocol(
            synthesis_instructions=synthesis_instructions,
            platform_type="tecan"
        )
        
        # Validate protocol structure
        assert "protocol_id" in protocol
        assert "protocol_name" in protocol
        assert "version" in protocol
        assert "platform" in protocol
        assert "labware" in protocol
        assert "operations" in protocol
        assert "quality_control" in protocol
        
        # Validate platform information
        assert protocol["platform"]["type"] == "tecan"
        
        # Validate operations exist
        assert len(protocol["operations"]) > 0
        
        # Validate against schema
        validator = SchemaValidator()
        assert validator.validate_liquid_handler(protocol)
    
    def test_create_complete_instruction_set(self, experiment_planner, sample_molecules):
        """Test complete instruction set creation."""
        instruction_set = experiment_planner.create_complete_instruction_set(
            target_molecules=sample_molecules[:2],
            experiment_name="Test_Experiment"
        )
        
        assert isinstance(instruction_set, InstructionSet)
        assert instruction_set.experiment_id is not None
        assert len(instruction_set.target_molecules) == 2
        assert len(instruction_set.synthesis_instructions) == 2
        assert len(instruction_set.characterization_instructions) == 2
        assert len(instruction_set.liquid_handler_protocols) > 0
        assert instruction_set.timeline is not None
        assert instruction_set.resource_requirements is not None
        assert instruction_set.success_criteria is not None
    
    def test_export_instructions_json(self, experiment_planner, sample_molecules, tmp_path):
        """Test JSON export functionality."""
        instruction_set = experiment_planner.create_complete_instruction_set(
            target_molecules=sample_molecules[:1],
            experiment_name="Export_Test"
        )
        
        output_file = tmp_path / "test_instructions.json"
        experiment_planner.export_instructions_json(instruction_set, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file, 'r') as f:
            exported_data = json.load(f)
        
        assert "experiment_id" in exported_data
        assert "target_molecules" in exported_data
        assert "synthesis_instructions" in exported_data
        assert "characterization_instructions" in exported_data
        assert "generated_timestamp" in exported_data
        
        # Verify data integrity
        assert len(exported_data["target_molecules"]) == 1
        assert len(exported_data["synthesis_instructions"]) == 1
    
    def test_protocol_step_creation(self):
        """Test ProtocolStep dataclass functionality."""
        step = ProtocolStep(
            step=1,
            action="Test action",
            duration=30.0,
            temperature=25.0,
            parameters={"param1": "value1"},
            safety_check=True,
            monitoring=["temperature", "pressure"]
        )
        
        assert step.step == 1
        assert step.action == "Test action"
        assert step.duration == 30.0
        assert step.temperature == 25.0
        assert step.parameters["param1"] == "value1"
        assert step.safety_check is True
        assert "temperature" in step.monitoring
    
    def test_synthesis_instruction_structure(self, experiment_planner, sample_molecules):
        """Test SynthesisInstruction structure and content."""
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=[sample_molecules[0]]
        )
        
        instruction = instructions[0]
        
        # Verify required fields
        assert instruction.synthesis_id is not None
        assert instruction.target_compound is not None
        assert instruction.reagents is not None
        assert instruction.procedure_steps is not None
        
        # Verify target compound structure
        assert "smiles" in instruction.target_compound
        assert "molecular_weight" in instruction.target_compound
        
        # Verify reagent structure
        assert len(instruction.reagents) > 0
        reagent = instruction.reagents[0]
        assert "name" in reagent
        assert "amount" in reagent
        assert "unit" in reagent
        
        # Verify procedure steps
        assert len(instruction.procedure_steps) > 0
        step = instruction.procedure_steps[0]
        assert isinstance(step, ProtocolStep)
        assert step.step is not None
        assert step.action is not None
    
    def test_characterization_instruction_structure(self, experiment_planner):
        """Test CharacterizationInstruction structure."""
        instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=["test_sample"],
            expected_compounds=["CCO"],
            characterization_scope="standard"
        )
        
        instruction = instructions[0]
        
        # Verify required fields
        assert instruction.characterization_id is not None
        assert instruction.sample_id == "test_sample"
        assert instruction.techniques is not None
        assert instruction.sample_preparation is not None
        
        # Verify techniques structure
        assert len(instruction.techniques) > 0
        technique = instruction.techniques[0]
        assert "technique" in technique
        assert "priority" in technique
        
        # Verify sample preparation
        assert "amount_needed" in instruction.sample_preparation
        assert "unit" in instruction.sample_preparation
    
    def test_optimization_criteria_handling(self, experiment_planner, sample_molecules):
        """Test handling of optimization criteria."""
        optimization_criteria = {
            "yield_priority": "high",
            "time_priority": "medium",
            "cost_priority": "low"
        }
        
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=[sample_molecules[0]],
            optimization_criteria=optimization_criteria
        )
        
        # Should still generate valid instructions
        assert len(instructions) == 1
        assert instructions[0].synthesis_id is not None
    
    def test_resource_requirement_calculation(self, experiment_planner, sample_molecules):
        """Test resource requirement calculation."""
        instruction_set = experiment_planner.create_complete_instruction_set(
            target_molecules=sample_molecules[:2],
            experiment_name="Resource_Test"
        )
        
        resources = instruction_set.resource_requirements
        
        assert "unique_reagents" in resources
        assert "characterization_techniques" in resources
        assert "estimated_cost_usd" in resources
        assert "equipment_time_hours" in resources
        
        # Verify reasonable values
        assert resources["unique_reagents"] >= 0
        assert resources["estimated_cost_usd"] >= 0
        assert isinstance(resources["characterization_techniques"], list)
    
    def test_timeline_generation(self, experiment_planner, sample_molecules):
        """Test experiment timeline generation."""
        instruction_set = experiment_planner.create_complete_instruction_set(
            target_molecules=sample_molecules[:2],
            experiment_name="Timeline_Test"
        )
        
        timeline = instruction_set.timeline
        
        assert "total_duration_hours" in timeline
        assert "synthesis_phase_hours" in timeline
        assert "characterization_phase_hours" in timeline
        assert "parallel_processing" in timeline
        
        # Verify reasonable duration values
        assert timeline["total_duration_hours"] > 0
        assert timeline["synthesis_phase_hours"] >= 0
        assert timeline["characterization_phase_hours"] >= 0
    
    def test_success_criteria_definition(self, experiment_planner, sample_molecules):
        """Test success criteria definition."""
        instruction_set = experiment_planner.create_complete_instruction_set(
            target_molecules=sample_molecules[:3],
            experiment_name="Success_Test"
        )
        
        criteria = instruction_set.success_criteria
        
        assert "minimum_yield" in criteria
        assert "minimum_purity" in criteria
        assert "identity_confirmation_required" in criteria
        assert "target_molecules_count" in criteria
        
        # Verify values
        assert 0 <= criteria["minimum_yield"] <= 1
        assert 0 <= criteria["minimum_purity"] <= 1
        assert criteria["target_molecules_count"] == 3
    
    @pytest.mark.parametrize("platform_type", ["tecan", "hamilton", "beckman"])
    def test_liquid_handler_platform_types(self, experiment_planner, sample_molecules, platform_type):
        """Test liquid handler protocol generation for different platforms."""
        synthesis_instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=[sample_molecules[0]]
        )
        
        protocol = experiment_planner.generate_liquid_handler_protocol(
            synthesis_instructions=synthesis_instructions,
            platform_type=platform_type
        )
        
        assert protocol["platform"]["type"] == platform_type
        
        # Should validate against schema regardless of platform
        validator = SchemaValidator()
        assert validator.validate_liquid_handler(protocol)
    
    def test_empty_molecule_list_handling(self, experiment_planner):
        """Test handling of empty molecule lists."""
        # Empty synthesis instructions
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=[]
        )
        assert len(instructions) == 0
        
        # Empty characterization instructions
        char_instructions = experiment_planner.generate_characterization_instructions(
            sample_ids=[],
            expected_compounds=[]
        )
        assert len(char_instructions) == 0
    
    def test_large_molecule_batch(self, experiment_planner):
        """Test handling of large molecule batches."""
        # Generate a larger list of molecules
        large_molecule_list = ["CCO"] * 10  # 10 copies of ethanol for testing
        
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=large_molecule_list
        )
        
        assert len(instructions) == 10
        assert all(instr.target_compound["smiles"] == "CCO" for instr in instructions)
    
    def test_error_recovery_in_instruction_generation(self, experiment_planner):
        """Test error recovery during instruction generation."""
        # Mix of valid and invalid molecules
        mixed_molecules = ["CCO", "INVALID", "CC(=O)O", "ALSO_INVALID", "c1ccccc1"]
        
        instructions = experiment_planner.generate_synthesis_instructions(
            target_molecules=mixed_molecules
        )
        
        # Should generate instructions only for valid molecules
        assert len(instructions) == 3  # CCO, CC(=O)O, c1ccccc1
        valid_smiles = [instr.target_compound["smiles"] for instr in instructions]
        assert "CCO" in valid_smiles
        assert "CC(=O)O" in valid_smiles
        assert "c1ccccc1" in valid_smiles


class TestSchemaValidation:
    """Test cases for schema validation functionality."""
    
    @pytest.fixture
    def validator(self) -> SchemaValidator:
        """Create SchemaValidator instance."""
        return SchemaValidator()
    
    def test_liquid_handler_schema_validation(self, validator):
        """Test liquid handler instruction validation."""
        valid_instruction = {
            "protocol_id": "test_123",
            "protocol_name": "Test Protocol",
            "platform": {
                "type": "tecan",
                "model": "Freedom EVO"
            },
            "operations": [
                {
                    "step": 1,
                    "operation": "aspirate"
                }
            ]
        }
        
        assert validator.validate_liquid_handler(valid_instruction)
        
        # Test invalid instruction (missing required field)
        invalid_instruction = {
            "protocol_name": "Test Protocol",
            # Missing protocol_id
            "platform": {
                "type": "tecan",
                "model": "Freedom EVO"
            },
            "operations": []
        }
        
        assert not validator.validate_liquid_handler(invalid_instruction)
    
    def test_synthesis_schema_validation(self, validator):
        """Test synthesis instruction validation."""
        valid_instruction = {
            "synthesis_id": "synth_123",
            "target_compound": {
                "smiles": "CCO"
            },
            "reagents": [
                {
                    "name": "test_reagent",
                    "amount": 1.0,
                    "unit": "g"
                }
            ],
            "procedure_steps": [
                {
                    "step": 1,
                    "action": "test_action"
                }
            ]
        }
        
        assert validator.validate_synthesis(valid_instruction)
    
    def test_characterization_schema_validation(self, validator):
        """Test characterization instruction validation."""
        valid_instruction = {
            "characterization_id": "char_123",
            "sample_id": "sample_1",
            "techniques": [
                {
                    "technique": "NMR",
                    "priority": 1
                }
            ]
        }
        
        assert validator.validate_characterization(valid_instruction)