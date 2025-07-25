"""
Custom Protocol Generation for Lab-Specific Automation

This module provides interfaces for defining and generating custom instruction sets
based on lab-specific automation protocols and equipment configurations.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .experiment_planner import CharacterizationInstruction, SynthesisInstruction
from .schemas import SchemaValidator


@dataclass
class ProtocolTemplate:
    """Template for lab-specific protocols."""

    template_id: str
    name: str
    description: str
    equipment_type: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    transformation_functions: Dict[str, str] = field(default_factory=dict)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CustomProtocol:
    """Custom protocol instance for specific laboratory."""

    protocol_id: str
    template_id: str
    lab_name: str
    equipment_config: Dict[str, Any]
    custom_parameters: Dict[str, Any]
    instruction_mapping: Dict[str, Any]
    safety_overrides: Dict[str, Any]
    validation_enabled: bool = True


class LabSpecificAdapter(ABC):
    """Abstract base class for lab-specific protocol adapters."""

    @abstractmethod
    def adapt_synthesis_instruction(
        self, instruction: SynthesisInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt synthesis instruction for specific lab equipment."""
        pass

    @abstractmethod
    def adapt_characterization_instruction(
        self, instruction: CharacterizationInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt characterization instruction for specific lab equipment."""
        pass

    @abstractmethod
    def validate_equipment_compatibility(
        self, instructions: List[Dict[str, Any]], lab_config: Dict[str, Any]
    ) -> List[str]:
        """Validate compatibility with lab equipment."""
        pass


class TecanAdapter(LabSpecificAdapter):
    """Adapter for Tecan liquid handling platforms."""

    def adapt_synthesis_instruction(
        self, instruction: SynthesisInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt synthesis instruction for Tecan platform."""
        tecan_protocol = {
            "protocol_type": "tecan_evo",
            "deck_layout": self._generate_tecan_deck_layout(instruction, lab_config),
            "liquid_classes": self._define_tecan_liquid_classes(instruction),
            "worklist": self._generate_tecan_worklist(instruction),
            "error_handling": {
                "liquid_level_detection": True,
                "air_gap": 5.0,
                "blow_out": True,
            },
        }
        return tecan_protocol

    def adapt_characterization_instruction(
        self, instruction: CharacterizationInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt characterization for Tecan sample preparation."""
        return {
            "sample_prep_protocol": self._generate_sample_prep_protocol(instruction),
            "dilution_series": self._generate_dilution_series(instruction),
            "plate_layout": self._generate_plate_layout(instruction),
        }

    def validate_equipment_compatibility(
        self, instructions: List[Dict[str, Any]], lab_config: Dict[str, Any]
    ) -> List[str]:
        """Validate Tecan compatibility."""
        issues = []

        # Check tip capacity
        total_operations = sum(len(instr.get("worklist", [])) for instr in instructions)
        if total_operations > lab_config.get("max_tip_capacity", 1000):
            issues.append("Insufficient tip capacity for protocol")

        # Check deck space
        required_positions = sum(
            len(instr.get("deck_layout", [])) for instr in instructions
        )
        if required_positions > lab_config.get("deck_positions", 8):
            issues.append("Insufficient deck positions for labware")

        return issues

    def _generate_tecan_deck_layout(
        self, instruction: SynthesisInstruction, lab_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate Tecan-specific deck layout."""
        layout = [
            {
                "position": "P1",
                "labware_type": "96_well_plate",
                "labware_name": "reagent_plate",
                "contents": "reagents",
            },
            {
                "position": "P2",
                "labware_type": "96_well_plate",
                "labware_name": "reaction_plate",
                "contents": "reactions",
            },
            {
                "position": "P3",
                "labware_type": "tip_rack_300",
                "labware_name": "tips_300ul",
                "contents": "tips",
            },
        ]
        return layout

    def _define_tecan_liquid_classes(
        self, instruction: SynthesisInstruction
    ) -> Dict[str, Any]:
        """Define liquid classes for reagents and solvents."""
        liquid_classes = {}

        for reagent in instruction.reagents:
            liquid_classes[reagent["name"]] = {
                "aspirate_speed": 100,
                "dispense_speed": 150,
                "air_gap": 5.0,
                "mix_volume": 20.0,
                "viscosity_class": "aqueous",
            }

        for solvent in instruction.solvents:
            liquid_classes[solvent["name"]] = {
                "aspirate_speed": 200,
                "dispense_speed": 200,
                "air_gap": 10.0,
                "mix_volume": 50.0,
                "viscosity_class": "organic",
            }

        return liquid_classes

    def _generate_tecan_worklist(
        self, instruction: SynthesisInstruction
    ) -> List[Dict[str, Any]]:
        """Generate Tecan worklist commands."""
        worklist = []
        step = 1

        for i, reagent in enumerate(instruction.reagents):
            worklist.extend(
                [
                    {
                        "step": step,
                        "command": "A",  # Aspirate
                        "rack": "P1",
                        "position": i + 1,
                        "volume": reagent["amount"] * 1000,  # Convert to μL
                        "liquid_class": reagent["name"],
                    },
                    {
                        "step": step + 1,
                        "command": "D",  # Dispense
                        "rack": "P2",
                        "position": 1,
                        "volume": reagent["amount"] * 1000,
                        "liquid_class": reagent["name"],
                    },
                ]
            )
            step += 2

        return worklist

    def _generate_sample_prep_protocol(
        self, instruction: CharacterizationInstruction
    ) -> Dict[str, Any]:
        """Generate sample preparation protocol."""
        return {
            "dilution_factor": 10,
            "final_volume": 200.0,
            "solvent": "MeCN",
            "mixing_cycles": 5,
        }

    def _generate_dilution_series(
        self, instruction: CharacterizationInstruction
    ) -> List[Dict[str, Any]]:
        """Generate dilution series for analysis."""
        dilutions = []
        for i, factor in enumerate([1, 10, 100, 1000]):
            dilutions.append(
                {
                    "well": f"A{i+1}",
                    "dilution_factor": factor,
                    "final_concentration": 1.0 / factor,
                }
            )
        return dilutions

    def _generate_plate_layout(
        self, instruction: CharacterizationInstruction
    ) -> Dict[str, Any]:
        """Generate plate layout for characterization."""
        return {
            "sample_wells": ["A1", "A2", "A3"],
            "standard_wells": ["B1", "B2", "B3"],
            "blank_wells": ["C1", "C2", "C3"],
            "qc_wells": ["D1", "D2", "D3"],
        }


class HamiltonAdapter(LabSpecificAdapter):
    """Adapter for Hamilton liquid handling platforms."""

    def adapt_synthesis_instruction(
        self, instruction: SynthesisInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt synthesis instruction for Hamilton platform."""
        hamilton_protocol = {
            "protocol_type": "hamilton_microlab",
            "method_file": f"synthesis_{instruction.synthesis_id}.hsl",
            "deck_template": "synthesis_template.lay",
            "liquid_handling_steps": self._generate_hamilton_steps(instruction),
            "error_recovery": {
                "clot_detection": True,
                "pressure_monitoring": True,
                "recovery_method": "retry_with_new_tip",
            },
        }
        return hamilton_protocol

    def adapt_characterization_instruction(
        self, instruction: CharacterizationInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt characterization for Hamilton platform."""
        return {
            "method_file": f"char_{instruction.characterization_id}.hsl",
            "sample_tracking": True,
            "barcode_integration": True,
        }

    def validate_equipment_compatibility(
        self, instructions: List[Dict[str, Any]], lab_config: Dict[str, Any]
    ) -> List[str]:
        """Validate Hamilton compatibility."""
        issues = []

        # Check channel requirements
        max_channels_needed = max(
            len(instr.get("liquid_handling_steps", [])) for instr in instructions
        )
        if max_channels_needed > lab_config.get("available_channels", 8):
            issues.append("Insufficient pipetting channels available")

        return issues

    def _generate_hamilton_steps(
        self, instruction: SynthesisInstruction
    ) -> List[Dict[str, Any]]:
        """Generate Hamilton-specific liquid handling steps."""
        steps = []

        for i, reagent in enumerate(instruction.reagents):
            steps.append(
                {
                    "step_type": "liquid_transfer",
                    "source_labware": "reagent_plate",
                    "source_position": i + 1,
                    "target_labware": "reaction_plate",
                    "target_position": 1,
                    "volume": reagent["amount"] * 1000,
                    "tip_type": "300ul",
                    "mixing": {"enabled": True, "cycles": 3, "volume": 50.0},
                }
            )

        return steps


class CustomProtocolAdapter(LabSpecificAdapter):
    """Generic adapter for custom lab protocols."""

    def __init__(self, custom_functions: Dict[str, Callable]):
        """Initialize with custom transformation functions."""
        self.custom_functions = custom_functions

    def adapt_synthesis_instruction(
        self, instruction: SynthesisInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply custom adaptation function."""
        if "synthesis_adapter" in self.custom_functions:
            return self.custom_functions["synthesis_adapter"](instruction, lab_config)
        return {}

    def adapt_characterization_instruction(
        self, instruction: CharacterizationInstruction, lab_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply custom characterization adapter."""
        if "characterization_adapter" in self.custom_functions:
            return self.custom_functions["characterization_adapter"](
                instruction, lab_config
            )
        return {}

    def validate_equipment_compatibility(
        self, instructions: List[Dict[str, Any]], lab_config: Dict[str, Any]
    ) -> List[str]:
        """Apply custom validation function."""
        if "validator" in self.custom_functions:
            return self.custom_functions["validator"](instructions, lab_config)
        return []


class ProtocolGenerator:
    """
    Main protocol generator for creating custom lab-specific instruction sets.

    Provides interfaces for defining and generating custom protocols based on
    lab-specific automation equipment and procedures.
    """

    def __init__(self, template_directory: Optional[str] = None):
        """
        Initialize protocol generator.

        Args:
            template_directory: Directory containing protocol templates
        """
        self.template_directory = (
            Path(template_directory) if template_directory else None
        )
        self.templates: Dict[str, ProtocolTemplate] = {}
        self.adapters: Dict[str, LabSpecificAdapter] = {
            "tecan": TecanAdapter(),
            "hamilton": HamiltonAdapter(),
        }
        self.validator = SchemaValidator()

        if self.template_directory and self.template_directory.exists():
            self._load_templates()

    def register_adapter(self, name: str, adapter: LabSpecificAdapter) -> None:
        """Register a custom lab adapter."""
        self.adapters[name] = adapter

    def create_protocol_template(
        self,
        name: str,
        description: str,
        equipment_type: str,
        parameters: Dict[str, Any],
        validation_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> ProtocolTemplate:
        """
        Create a new protocol template.

        Args:
            name: Template name
            description: Template description
            equipment_type: Type of equipment (tecan, hamilton, custom)
            parameters: Template parameters
            validation_rules: Optional validation rules

        Returns:
            Created ProtocolTemplate
        """
        template_id = str(uuid.uuid4())

        template = ProtocolTemplate(
            template_id=template_id,
            name=name,
            description=description,
            equipment_type=equipment_type,
            version="1.0.0",
            parameters=parameters,
            validation_rules=validation_rules or [],
        )

        self.templates[template_id] = template
        return template

    def save_template(self, template: ProtocolTemplate, file_path: str) -> None:
        """Save protocol template to file."""
        template_data = {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "equipment_type": template.equipment_type,
            "version": template.version,
            "parameters": template.parameters,
            "validation_rules": template.validation_rules,
            "transformation_functions": template.transformation_functions,
            "created_date": template.created_date,
        }

        with open(file_path, "w") as f:
            json.dump(template_data, f, indent=2)

    def load_template(self, file_path: str) -> ProtocolTemplate:
        """Load protocol template from file."""
        with open(file_path) as f:
            template_data = json.load(f)

        template = ProtocolTemplate(
            template_id=template_data["template_id"],
            name=template_data["name"],
            description=template_data["description"],
            equipment_type=template_data["equipment_type"],
            version=template_data["version"],
            parameters=template_data["parameters"],
            validation_rules=template_data.get("validation_rules", []),
            transformation_functions=template_data.get("transformation_functions", {}),
            created_date=template_data.get("created_date", datetime.now().isoformat()),
        )

        self.templates[template.template_id] = template
        return template

    def create_custom_protocol(
        self,
        template_id: str,
        lab_name: str,
        equipment_config: Dict[str, Any],
        custom_parameters: Optional[Dict[str, Any]] = None,
        instruction_mapping: Optional[Dict[str, Any]] = None,
    ) -> CustomProtocol:
        """
        Create custom protocol instance from template.

        Args:
            template_id: ID of the template to use
            lab_name: Name of the laboratory
            equipment_config: Equipment configuration
            custom_parameters: Custom parameters for this lab
            instruction_mapping: Custom instruction mappings

        Returns:
            Created CustomProtocol
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        protocol_id = str(uuid.uuid4())

        protocol = CustomProtocol(
            protocol_id=protocol_id,
            template_id=template_id,
            lab_name=lab_name,
            equipment_config=equipment_config,
            custom_parameters=custom_parameters or {},
            instruction_mapping=instruction_mapping or {},
            safety_overrides={},
        )

        return protocol

    def generate_lab_specific_instructions(
        self,
        synthesis_instructions: List[SynthesisInstruction],
        characterization_instructions: List[CharacterizationInstruction],
        custom_protocol: CustomProtocol,
    ) -> Dict[str, Any]:
        """
        Generate lab-specific instructions using custom protocol.

        Args:
            synthesis_instructions: List of synthesis instructions
            characterization_instructions: List of characterization instructions
            custom_protocol: Custom protocol configuration

        Returns:
            Lab-specific instruction set
        """
        template = self.templates[custom_protocol.template_id]
        equipment_type = template.equipment_type

        if equipment_type not in self.adapters:
            raise ValueError(
                f"No adapter available for equipment type: {equipment_type}"
            )

        adapter = self.adapters[equipment_type]

        # Adapt synthesis instructions
        adapted_synthesis = []
        for instruction in synthesis_instructions:
            adapted = adapter.adapt_synthesis_instruction(
                instruction, custom_protocol.equipment_config
            )
            # Apply custom mappings and overrides
            adapted = self._apply_custom_mappings(adapted, custom_protocol)
            adapted_synthesis.append(adapted)

        # Adapt characterization instructions
        adapted_characterization = []
        for instruction in characterization_instructions:
            adapted = adapter.adapt_characterization_instruction(
                instruction, custom_protocol.equipment_config
            )
            adapted = self._apply_custom_mappings(adapted, custom_protocol)
            adapted_characterization.append(adapted)

        # Validate compatibility
        all_instructions = adapted_synthesis + adapted_characterization
        compatibility_issues = adapter.validate_equipment_compatibility(
            all_instructions, custom_protocol.equipment_config
        )

        # Generate final instruction set
        lab_instructions = {
            "protocol_id": custom_protocol.protocol_id,
            "lab_name": custom_protocol.lab_name,
            "template_id": custom_protocol.template_id,
            "equipment_type": equipment_type,
            "generated_timestamp": datetime.now().isoformat(),
            "synthesis_protocols": adapted_synthesis,
            "characterization_protocols": adapted_characterization,
            "compatibility_issues": compatibility_issues,
            "equipment_configuration": custom_protocol.equipment_config,
            "validation_passed": len(compatibility_issues) == 0,
        }

        return lab_instructions

    def validate_custom_protocol(self, custom_protocol: CustomProtocol) -> List[str]:
        """
        Validate custom protocol configuration.

        Args:
            custom_protocol: Protocol to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Check if template exists
        if custom_protocol.template_id not in self.templates:
            issues.append(f"Template {custom_protocol.template_id} not found")
            return issues

        template = self.templates[custom_protocol.template_id]

        # Validate required parameters
        required_params = template.parameters.get("required", [])
        for param in required_params:
            if param not in custom_protocol.custom_parameters:
                issues.append(f"Required parameter '{param}' missing")

        # Apply validation rules
        for rule in template.validation_rules:
            validation_result = self._apply_validation_rule(rule, custom_protocol)
            if not validation_result["passed"]:
                issues.append(validation_result["message"])

        return issues

    def export_protocol_documentation(
        self, custom_protocol: CustomProtocol, output_path: str
    ) -> None:
        """
        Export protocol documentation for laboratory use.

        Args:
            custom_protocol: Protocol to document
            output_path: Output file path
        """
        template = self.templates[custom_protocol.template_id]

        documentation = {
            "protocol_overview": {
                "protocol_id": custom_protocol.protocol_id,
                "lab_name": custom_protocol.lab_name,
                "template_name": template.name,
                "description": template.description,
                "equipment_type": template.equipment_type,
                "created_date": datetime.now().isoformat(),
            },
            "equipment_configuration": custom_protocol.equipment_config,
            "custom_parameters": custom_protocol.custom_parameters,
            "instruction_mappings": custom_protocol.instruction_mapping,
            "safety_considerations": custom_protocol.safety_overrides,
            "validation_status": {
                "issues": self.validate_custom_protocol(custom_protocol),
                "validated": len(self.validate_custom_protocol(custom_protocol)) == 0,
            },
        }

        with open(output_path, "w") as f:
            json.dump(documentation, f, indent=2)

    # Private helper methods

    def _load_templates(self) -> None:
        """Load templates from template directory."""
        if not self.template_directory:
            return

        for template_file in self.template_directory.glob("*.json"):
            try:
                self.load_template(str(template_file))
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")

    def _apply_custom_mappings(
        self, instruction: Dict[str, Any], custom_protocol: CustomProtocol
    ) -> Dict[str, Any]:
        """Apply custom instruction mappings."""
        mapped_instruction = instruction.copy()

        for key, mapping in custom_protocol.instruction_mapping.items():
            if key in mapped_instruction:
                mapped_instruction[key] = mapping.get(
                    mapped_instruction[key], mapped_instruction[key]
                )

        # Apply custom parameters
        for key, value in custom_protocol.custom_parameters.items():
            if key in mapped_instruction:
                mapped_instruction[key] = value

        return mapped_instruction

    def _apply_validation_rule(
        self, rule: Dict[str, Any], custom_protocol: CustomProtocol
    ) -> Dict[str, Any]:
        """Apply a single validation rule."""
        rule_type = rule.get("type")

        if rule_type == "parameter_range":
            param_name = rule["parameter"]
            min_val = rule.get("min")
            max_val = rule.get("max")

            if param_name in custom_protocol.custom_parameters:
                value = custom_protocol.custom_parameters[param_name]
                if min_val is not None and value < min_val:
                    return {
                        "passed": False,
                        "message": f"Parameter {param_name} below minimum ({min_val})",
                    }
                if max_val is not None and value > max_val:
                    return {
                        "passed": False,
                        "message": f"Parameter {param_name} above maximum ({max_val})",
                    }

        elif rule_type == "equipment_capability":
            capability = rule["capability"]
            required = rule.get("required", True)

            equipment_capabilities = custom_protocol.equipment_config.get(
                "capabilities", []
            )
            has_capability = capability in equipment_capabilities

            if required and not has_capability:
                return {
                    "passed": False,
                    "message": f"Required equipment capability '{capability}' not available",
                }

        return {"passed": True, "message": ""}


# Example usage and factory functions


def create_tecan_protocol_template() -> ProtocolTemplate:
    """Create a standard Tecan protocol template."""
    return ProtocolTemplate(
        template_id=str(uuid.uuid4()),
        name="Tecan Standard Synthesis",
        description="Standard liquid handling protocol for Tecan platforms",
        equipment_type="tecan",
        version="1.0.0",
        parameters={
            "required": ["deck_positions", "tip_capacity"],
            "optional": ["liquid_classes", "error_handling"],
        },
        validation_rules=[
            {
                "type": "parameter_range",
                "parameter": "deck_positions",
                "min": 1,
                "max": 32,
            },
            {
                "type": "equipment_capability",
                "capability": "liquid_level_detection",
                "required": True,
            },
        ],
    )


def create_hamilton_protocol_template() -> ProtocolTemplate:
    """Create a standard Hamilton protocol template."""
    return ProtocolTemplate(
        template_id=str(uuid.uuid4()),
        name="Hamilton ML Star Synthesis",
        description="Standard protocol for Hamilton ML Star platforms",
        equipment_type="hamilton",
        version="1.0.0",
        parameters={
            "required": ["channel_count", "method_directory"],
            "optional": ["barcode_integration", "error_recovery"],
        },
        validation_rules=[
            {
                "type": "parameter_range",
                "parameter": "channel_count",
                "min": 1,
                "max": 16,
            },
            {
                "type": "equipment_capability",
                "capability": "clot_detection",
                "required": False,
            },
        ],
    )
