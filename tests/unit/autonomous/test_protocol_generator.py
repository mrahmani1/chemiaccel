"""
Tests for Custom Protocol Generation

This module contains tests for the protocol generator functionality including
lab-specific adapters and custom protocol templates.
"""

import json
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from chemi_accel.autonomous.experiment_planner import (
    CharacterizationInstruction,
    ProtocolStep,
    SynthesisInstruction,
)
from chemi_accel.autonomous.protocol_generator import (
    CustomProtocol,
    CustomProtocolAdapter,
    HamiltonAdapter,
    ProtocolGenerator,
    ProtocolTemplate,
    TecanAdapter,
    create_hamilton_protocol_template,
    create_tecan_protocol_template,
)


class TestProtocolTemplate:
    """Test cases for ProtocolTemplate dataclass."""

    def test_protocol_template_creation(self):
        """Test ProtocolTemplate creation and structure."""
        template = ProtocolTemplate(
            template_id="test_123",
            name="Test Template",
            description="Test description",
            equipment_type="tecan",
            version="1.0.0",
        )

        assert template.template_id == "test_123"
        assert template.name == "Test Template"
        assert template.description == "Test description"
        assert template.equipment_type == "tecan"
        assert template.version == "1.0.0"
        assert template.parameters == {}
        assert template.validation_rules == []
        assert template.created_date is not None

    def test_protocol_template_with_parameters(self):
        """Test ProtocolTemplate with parameters and validation rules."""
        parameters = {"param1": "value1", "param2": 123}
        validation_rules = [{"type": "range", "min": 0, "max": 100}]

        template = ProtocolTemplate(
            template_id="test_456",
            name="Parameterized Template",
            description="Template with parameters",
            equipment_type="hamilton",
            version="2.0.0",
            parameters=parameters,
            validation_rules=validation_rules,
        )

        assert template.parameters == parameters
        assert template.validation_rules == validation_rules


class TestCustomProtocol:
    """Test cases for CustomProtocol dataclass."""

    def test_custom_protocol_creation(self):
        """Test CustomProtocol creation and structure."""
        equipment_config = {"type": "tecan", "model": "Freedom EVO"}
        custom_parameters = {"batch_size": 96, "tip_type": "standard"}

        protocol = CustomProtocol(
            protocol_id="custom_123",
            template_id="template_456",
            lab_name="Test Lab",
            equipment_config=equipment_config,
            custom_parameters=custom_parameters,
            instruction_mapping={},
            safety_overrides={},
        )

        assert protocol.protocol_id == "custom_123"
        assert protocol.template_id == "template_456"
        assert protocol.lab_name == "Test Lab"
        assert protocol.equipment_config == equipment_config
        assert protocol.custom_parameters == custom_parameters
        assert protocol.validation_enabled is True


class TestTecanAdapter:
    """Test cases for TecanAdapter class."""

    @pytest.fixture
    def tecan_adapter(self) -> TecanAdapter:
        """Create TecanAdapter instance."""
        return TecanAdapter()

    @pytest.fixture
    def sample_synthesis_instruction(self) -> SynthesisInstruction:
        """Create sample synthesis instruction."""
        return SynthesisInstruction(
            synthesis_id="synth_123",
            target_compound={"smiles": "CCO", "name": "ethanol"},
            reagents=[
                {
                    "name": "sodium_borohydride",
                    "smiles": "[BH4-].[Na+]",
                    "amount": 1.0,
                    "unit": "g",
                }
            ],
            solvents=[{"name": "ethanol", "volume": 50.0, "unit": "mL"}],
            reaction_conditions={"temperature": 25.0, "pressure": 1.0},
            procedure_steps=[ProtocolStep(step=1, action="Add reagent", duration=10.0)],
            workup={"extraction": ["water"]},
            safety_requirements={"fume_hood": True},
        )

    @pytest.fixture
    def sample_lab_config(self) -> Dict[str, Any]:
        """Sample lab configuration."""
        return {
            "max_tip_capacity": 1000,
            "deck_positions": 8,
            "available_channels": 8,
            "liquid_classes": ["aqueous", "organic"],
        }

    def test_adapt_synthesis_instruction(
        self, tecan_adapter, sample_synthesis_instruction, sample_lab_config
    ):
        """Test Tecan synthesis instruction adaptation."""
        adapted = tecan_adapter.adapt_synthesis_instruction(
            sample_synthesis_instruction, sample_lab_config
        )

        assert adapted["protocol_type"] == "tecan_evo"
        assert "deck_layout" in adapted
        assert "liquid_classes" in adapted
        assert "worklist" in adapted
        assert "error_handling" in adapted

        # Verify deck layout
        deck_layout = adapted["deck_layout"]
        assert len(deck_layout) > 0
        assert any(item["labware_name"] == "reagent_plate" for item in deck_layout)

        # Verify liquid classes
        liquid_classes = adapted["liquid_classes"]
        assert "sodium_borohydride" in liquid_classes
        assert "ethanol" in liquid_classes

        # Verify worklist
        worklist = adapted["worklist"]
        assert len(worklist) > 0
        assert all("step" in item for item in worklist)

    def test_adapt_characterization_instruction(self, tecan_adapter, sample_lab_config):
        """Test Tecan characterization instruction adaptation."""
        char_instruction = CharacterizationInstruction(
            characterization_id="char_123",
            sample_id="sample_1",
            techniques=[{"technique": "NMR", "priority": 1}],
            sample_preparation={"amount_needed": 5.0},
            data_processing={"automated_analysis": True},
            quality_metrics={"purity_target": 0.95},
        )

        adapted = tecan_adapter.adapt_characterization_instruction(
            char_instruction, sample_lab_config
        )

        assert "sample_prep_protocol" in adapted
        assert "dilution_series" in adapted
        assert "plate_layout" in adapted

        # Verify dilution series
        dilution_series = adapted["dilution_series"]
        assert len(dilution_series) > 0
        assert all("dilution_factor" in item for item in dilution_series)

    def test_validate_equipment_compatibility(self, tecan_adapter, sample_lab_config):
        """Test Tecan equipment compatibility validation."""
        # Test with valid configuration
        valid_instructions = [
            {"worklist": [{"step": 1}, {"step": 2}]},
            {"deck_layout": [{"position": "P1"}, {"position": "P2"}]},
        ]

        issues = tecan_adapter.validate_equipment_compatibility(
            valid_instructions, sample_lab_config
        )
        assert len(issues) == 0

        # Test with invalid configuration (too many operations)
        sample_lab_config["max_tip_capacity"] = 1  # Very low capacity

        issues = tecan_adapter.validate_equipment_compatibility(
            valid_instructions, sample_lab_config
        )
        assert len(issues) > 0
        assert any("tip capacity" in issue.lower() for issue in issues)


class TestHamiltonAdapter:
    """Test cases for HamiltonAdapter class."""

    @pytest.fixture
    def hamilton_adapter(self) -> HamiltonAdapter:
        """Create HamiltonAdapter instance."""
        return HamiltonAdapter()

    def test_adapt_synthesis_instruction(self, hamilton_adapter, sample_lab_config):
        """Test Hamilton synthesis instruction adaptation."""
        synthesis_instruction = SynthesisInstruction(
            synthesis_id="synth_456",
            target_compound={"smiles": "CCO"},
            reagents=[{"name": "test_reagent", "amount": 1.0, "unit": "g"}],
            solvents=[{"name": "ethanol", "volume": 50.0, "unit": "mL"}],
            reaction_conditions={},
            procedure_steps=[ProtocolStep(step=1, action="Test")],
            workup={},
            safety_requirements={},
        )

        adapted = hamilton_adapter.adapt_synthesis_instruction(
            synthesis_instruction, sample_lab_config
        )

        assert adapted["protocol_type"] == "hamilton_microlab"
        assert "method_file" in adapted
        assert "deck_template" in adapted
        assert "liquid_handling_steps" in adapted
        assert "error_recovery" in adapted

        # Verify error recovery settings
        error_recovery = adapted["error_recovery"]
        assert error_recovery["clot_detection"] is True
        assert error_recovery["pressure_monitoring"] is True

    def test_validate_equipment_compatibility(
        self, hamilton_adapter, sample_lab_config
    ):
        """Test Hamilton equipment compatibility validation."""
        # Test with valid configuration
        valid_instructions = [{"liquid_handling_steps": [{"step": 1}, {"step": 2}]}]

        issues = hamilton_adapter.validate_equipment_compatibility(
            valid_instructions, sample_lab_config
        )
        assert len(issues) == 0

        # Test with invalid configuration (too many channels needed)
        sample_lab_config["available_channels"] = 1  # Very low capacity
        invalid_instructions = [
            {"liquid_handling_steps": [{"step": i} for i in range(10)]}
        ]

        issues = hamilton_adapter.validate_equipment_compatibility(
            invalid_instructions, sample_lab_config
        )
        assert len(issues) > 0
        assert any("channel" in issue.lower() for issue in issues)


class TestCustomProtocolAdapter:
    """Test cases for CustomProtocolAdapter class."""

    def test_custom_adapter_with_functions(self):
        """Test CustomProtocolAdapter with custom functions."""

        def mock_synthesis_adapter(instruction, config):
            return {"custom_field": "synthesis_value"}

        def mock_characterization_adapter(instruction, config):
            return {"custom_field": "characterization_value"}

        def mock_validator(instructions, config):
            return ["custom_validation_issue"]

        custom_functions = {
            "synthesis_adapter": mock_synthesis_adapter,
            "characterization_adapter": mock_characterization_adapter,
            "validator": mock_validator,
        }

        adapter = CustomProtocolAdapter(custom_functions)

        # Test synthesis adaptation
        synthesis_result = adapter.adapt_synthesis_instruction(Mock(), {})
        assert synthesis_result["custom_field"] == "synthesis_value"

        # Test characterization adaptation
        char_result = adapter.adapt_characterization_instruction(Mock(), {})
        assert char_result["custom_field"] == "characterization_value"

        # Test validation
        validation_result = adapter.validate_equipment_compatibility([], {})
        assert validation_result == ["custom_validation_issue"]

    def test_custom_adapter_without_functions(self):
        """Test CustomProtocolAdapter without custom functions."""
        adapter = CustomProtocolAdapter({})

        # Should return empty results when functions not provided
        assert adapter.adapt_synthesis_instruction(Mock(), {}) == {}
        assert adapter.adapt_characterization_instruction(Mock(), {}) == {}
        assert adapter.validate_equipment_compatibility([], {}) == []


class TestProtocolGenerator:
    """Test cases for ProtocolGenerator class."""

    @pytest.fixture
    def protocol_generator(self) -> ProtocolGenerator:
        """Create ProtocolGenerator instance."""
        return ProtocolGenerator()

    @pytest.fixture
    def sample_template(self) -> ProtocolTemplate:
        """Create sample protocol template."""
        return ProtocolTemplate(
            template_id="template_123",
            name="Test Template",
            description="Sample template for testing",
            equipment_type="tecan",
            version="1.0.0",
            parameters={"required": ["deck_positions"], "optional": ["tip_capacity"]},
            validation_rules=[
                {
                    "type": "parameter_range",
                    "parameter": "deck_positions",
                    "min": 1,
                    "max": 32,
                }
            ],
        )

    def test_initialization(self):
        """Test ProtocolGenerator initialization."""
        generator = ProtocolGenerator()

        assert isinstance(generator.templates, dict)
        assert isinstance(generator.adapters, dict)
        assert "tecan" in generator.adapters
        assert "hamilton" in generator.adapters

    def test_register_adapter(self, protocol_generator):
        """Test custom adapter registration."""
        custom_adapter = CustomProtocolAdapter({})
        protocol_generator.register_adapter("custom", custom_adapter)

        assert "custom" in protocol_generator.adapters
        assert protocol_generator.adapters["custom"] == custom_adapter

    def test_create_protocol_template(self, protocol_generator):
        """Test protocol template creation."""
        parameters = {"param1": "value1"}
        validation_rules = [{"type": "test"}]

        template = protocol_generator.create_protocol_template(
            name="Test Template",
            description="Test description",
            equipment_type="tecan",
            parameters=parameters,
            validation_rules=validation_rules,
        )

        assert template.name == "Test Template"
        assert template.description == "Test description"
        assert template.equipment_type == "tecan"
        assert template.parameters == parameters
        assert template.validation_rules == validation_rules
        assert template.template_id in protocol_generator.templates

    def test_save_and_load_template(
        self, protocol_generator, sample_template, tmp_path
    ):
        """Test template saving and loading."""
        template_file = tmp_path / "test_template.json"

        # Save template
        protocol_generator.save_template(sample_template, str(template_file))

        # Verify file exists
        assert template_file.exists()

        # Load template
        loaded_template = protocol_generator.load_template(str(template_file))

        assert loaded_template.template_id == sample_template.template_id
        assert loaded_template.name == sample_template.name
        assert loaded_template.description == sample_template.description
        assert loaded_template.equipment_type == sample_template.equipment_type

    def test_create_custom_protocol(self, protocol_generator, sample_template):
        """Test custom protocol creation."""
        # First add template to generator
        protocol_generator.templates[sample_template.template_id] = sample_template

        equipment_config = {"deck_positions": 8, "tip_capacity": 1000}
        custom_parameters = {"batch_size": 96}

        custom_protocol = protocol_generator.create_custom_protocol(
            template_id=sample_template.template_id,
            lab_name="Test Lab",
            equipment_config=equipment_config,
            custom_parameters=custom_parameters,
        )

        assert custom_protocol.template_id == sample_template.template_id
        assert custom_protocol.lab_name == "Test Lab"
        assert custom_protocol.equipment_config == equipment_config
        assert custom_protocol.custom_parameters == custom_parameters

    def test_create_custom_protocol_invalid_template(self, protocol_generator):
        """Test custom protocol creation with invalid template."""
        with pytest.raises(ValueError, match="Template .* not found"):
            protocol_generator.create_custom_protocol(
                template_id="nonexistent_template",
                lab_name="Test Lab",
                equipment_config={},
            )

    def test_generate_lab_specific_instructions(
        self, protocol_generator, sample_template
    ):
        """Test lab-specific instruction generation."""
        # Setup template and protocol
        protocol_generator.templates[sample_template.template_id] = sample_template

        custom_protocol = CustomProtocol(
            protocol_id="protocol_123",
            template_id=sample_template.template_id,
            lab_name="Test Lab",
            equipment_config={"deck_positions": 8},
            custom_parameters={},
            instruction_mapping={},
            safety_overrides={},
        )

        # Create sample instructions
        synthesis_instructions = [
            SynthesisInstruction(
                synthesis_id="synth_1",
                target_compound={"smiles": "CCO"},
                reagents=[{"name": "reagent1", "amount": 1.0, "unit": "g"}],
                solvents=[{"name": "ethanol", "volume": 50.0, "unit": "mL"}],
                reaction_conditions={},
                procedure_steps=[ProtocolStep(step=1, action="Test")],
                workup={},
                safety_requirements={},
            )
        ]

        characterization_instructions = [
            CharacterizationInstruction(
                characterization_id="char_1",
                sample_id="sample_1",
                techniques=[{"technique": "NMR", "priority": 1}],
                sample_preparation={},
                data_processing={},
                quality_metrics={},
            )
        ]

        # Generate lab-specific instructions
        lab_instructions = protocol_generator.generate_lab_specific_instructions(
            synthesis_instructions, characterization_instructions, custom_protocol
        )

        assert lab_instructions["protocol_id"] == custom_protocol.protocol_id
        assert lab_instructions["lab_name"] == custom_protocol.lab_name
        assert lab_instructions["equipment_type"] == sample_template.equipment_type
        assert "synthesis_protocols" in lab_instructions
        assert "characterization_protocols" in lab_instructions
        assert "compatibility_issues" in lab_instructions
        assert "validation_passed" in lab_instructions

    def test_validate_custom_protocol(self, protocol_generator, sample_template):
        """Test custom protocol validation."""
        # Add template to generator
        protocol_generator.templates[sample_template.template_id] = sample_template

        # Test valid protocol
        valid_protocol = CustomProtocol(
            protocol_id="protocol_123",
            template_id=sample_template.template_id,
            lab_name="Test Lab",
            equipment_config={},
            custom_parameters={"deck_positions": 8},  # Required parameter
            instruction_mapping={},
            safety_overrides={},
        )

        issues = protocol_generator.validate_custom_protocol(valid_protocol)
        assert len(issues) == 0

        # Test invalid protocol (missing required parameter)
        invalid_protocol = CustomProtocol(
            protocol_id="protocol_456",
            template_id=sample_template.template_id,
            lab_name="Test Lab",
            equipment_config={},
            custom_parameters={},  # Missing required parameter
            instruction_mapping={},
            safety_overrides={},
        )

        issues = protocol_generator.validate_custom_protocol(invalid_protocol)
        assert len(issues) > 0
        assert any("deck_positions" in issue for issue in issues)

    def test_export_protocol_documentation(
        self, protocol_generator, sample_template, tmp_path
    ):
        """Test protocol documentation export."""
        # Setup template and protocol
        protocol_generator.templates[sample_template.template_id] = sample_template

        custom_protocol = CustomProtocol(
            protocol_id="protocol_123",
            template_id=sample_template.template_id,
            lab_name="Test Lab",
            equipment_config={"deck_positions": 8},
            custom_parameters={"deck_positions": 8},
            instruction_mapping={},
            safety_overrides={},
        )

        # Export documentation
        doc_file = tmp_path / "protocol_doc.json"
        protocol_generator.export_protocol_documentation(custom_protocol, str(doc_file))

        # Verify file exists and has correct structure
        assert doc_file.exists()

        with open(doc_file) as f:
            doc_data = json.load(f)

        assert "protocol_overview" in doc_data
        assert "equipment_configuration" in doc_data
        assert "validation_status" in doc_data

        # Verify protocol overview
        overview = doc_data["protocol_overview"]
        assert overview["protocol_id"] == custom_protocol.protocol_id
        assert overview["lab_name"] == custom_protocol.lab_name


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_tecan_protocol_template(self):
        """Test Tecan protocol template factory function."""
        template = create_tecan_protocol_template()

        assert template.name == "Tecan Standard Synthesis"
        assert template.equipment_type == "tecan"
        assert template.version == "1.0.0"
        assert "required" in template.parameters
        assert "deck_positions" in template.parameters["required"]
        assert len(template.validation_rules) > 0

    def test_create_hamilton_protocol_template(self):
        """Test Hamilton protocol template factory function."""
        template = create_hamilton_protocol_template()

        assert template.name == "Hamilton ML Star Synthesis"
        assert template.equipment_type == "hamilton"
        assert template.version == "1.0.0"
        assert "required" in template.parameters
        assert "channel_count" in template.parameters["required"]
        assert len(template.validation_rules) > 0


class TestIntegrationScenarios:
    """Integration test scenarios for protocol generation."""

    def test_end_to_end_protocol_generation(self, tmp_path):
        """Test complete end-to-end protocol generation workflow."""
        # Create protocol generator
        generator = ProtocolGenerator()

        # Create and save template
        template = create_tecan_protocol_template()
        template_file = tmp_path / "tecan_template.json"
        generator.save_template(template, str(template_file))

        # Load template
        loaded_template = generator.load_template(str(template_file))

        # Create custom protocol
        custom_protocol = generator.create_custom_protocol(
            template_id=loaded_template.template_id,
            lab_name="Integration Test Lab",
            equipment_config={"deck_positions": 16, "tip_capacity": 2000},
            custom_parameters={"deck_positions": 16},
        )

        # Validate protocol
        validation_issues = generator.validate_custom_protocol(custom_protocol)
        assert len(validation_issues) == 0

        # Create sample instructions
        synthesis_instructions = [
            SynthesisInstruction(
                synthesis_id="integration_synth",
                target_compound={"smiles": "CCO"},
                reagents=[{"name": "test_reagent", "amount": 1.0, "unit": "g"}],
                solvents=[{"name": "ethanol", "volume": 50.0, "unit": "mL"}],
                reaction_conditions={},
                procedure_steps=[],
                workup={},
                safety_requirements={},
            )
        ]

        characterization_instructions = [
            CharacterizationInstruction(
                characterization_id="integration_char",
                sample_id="integration_sample",
                techniques=[{"technique": "NMR", "priority": 1}],
                sample_preparation={},
                data_processing={},
                quality_metrics={},
            )
        ]

        # Generate lab-specific instructions
        lab_instructions = generator.generate_lab_specific_instructions(
            synthesis_instructions, characterization_instructions, custom_protocol
        )

        assert lab_instructions["validation_passed"] is True
        assert len(lab_instructions["synthesis_protocols"]) == 1
        assert len(lab_instructions["characterization_protocols"]) == 1

        # Export documentation
        doc_file = tmp_path / "integration_doc.json"
        generator.export_protocol_documentation(custom_protocol, str(doc_file))

        # Verify documentation was created
        assert doc_file.exists()

    def test_multi_platform_protocol_generation(self):
        """Test protocol generation for multiple platforms."""
        generator = ProtocolGenerator()

        # Create templates for different platforms
        tecan_template = create_tecan_protocol_template()
        hamilton_template = create_hamilton_protocol_template()

        generator.templates[tecan_template.template_id] = tecan_template
        generator.templates[hamilton_template.template_id] = hamilton_template

        # Create protocols for both platforms
        tecan_protocol = generator.create_custom_protocol(
            template_id=tecan_template.template_id,
            lab_name="Tecan Lab",
            equipment_config={"deck_positions": 8, "tip_capacity": 1000},
            custom_parameters={"deck_positions": 8},
        )

        hamilton_protocol = generator.create_custom_protocol(
            template_id=hamilton_template.template_id,
            lab_name="Hamilton Lab",
            equipment_config={"channel_count": 8},
            custom_parameters={"channel_count": 8},
        )

        # Both should validate successfully
        tecan_issues = generator.validate_custom_protocol(tecan_protocol)
        hamilton_issues = generator.validate_custom_protocol(hamilton_protocol)

        assert len(tecan_issues) == 0
        assert len(hamilton_issues) == 0
