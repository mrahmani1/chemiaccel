"""
JSON Schemas for Robotic Laboratory Platform Instructions

This module defines standardized JSON schemas for generating machine-readable
instructions for various robotic laboratory platforms including liquid handlers,
synthesis robots, and characterization equipment.
"""

import json
from typing import Dict, Any
from jsonschema import validate, ValidationError

# Liquid Handler Instruction Schema
LIQUID_HANDLER_SCHEMA = {
    "type": "object",
    "properties": {
        "protocol_id": {"type": "string"},
        "protocol_name": {"type": "string"},
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "platform": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["tecan", "hamilton", "beckman", "agilent", "custom"]},
                "model": {"type": "string"},
                "software_version": {"type": "string"}
            },
            "required": ["type", "model"]
        },
        "labware": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["plate", "tube_rack", "reservoir", "tip_rack", "vial_rack"]},
                    "format": {"type": "string", "enum": ["96", "384", "1536", "24", "48", "custom"]},
                    "position": {"type": "string"},
                    "contents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "well": {"type": "string"},
                                "compound_id": {"type": "string"},
                                "smiles": {"type": "string"},
                                "concentration": {"type": "number"},
                                "volume": {"type": "number"},
                                "solvent": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["id", "type", "position"]
            }
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "operation": {"type": "string", "enum": ["aspirate", "dispense", "mix", "wash", "tip_pickup", "tip_drop"]},
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "volume": {"type": "number"},
                    "speed": {"type": "number"},
                    "mix_cycles": {"type": "integer"},
                    "parameters": {"type": "object"}
                },
                "required": ["step", "operation"]
            }
        },
        "quality_control": {
            "type": "object",
            "properties": {
                "volume_verification": {"type": "boolean"},
                "liquid_level_detection": {"type": "boolean"},
                "barcode_scanning": {"type": "boolean"},
                "error_handling": {"type": "string", "enum": ["stop", "continue", "retry"]}
            }
        }
    },
    "required": ["protocol_id", "protocol_name", "platform", "operations"]
}

# Synthesis Instruction Schema
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "synthesis_id": {"type": "string"},
        "target_compound": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "name": {"type": "string"},
                "target_yield": {"type": "number"},
                "purity_target": {"type": "number"}
            },
            "required": ["smiles"]
        },
        "reaction_conditions": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "pressure": {"type": "number"},
                "atmosphere": {"type": "string", "enum": ["air", "nitrogen", "argon", "vacuum"]},
                "reaction_time": {"type": "number"},
                "stirring_speed": {"type": "number"}
            }
        },
        "reagents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "smiles": {"type": "string"},
                    "cas_number": {"type": "string"},
                    "amount": {"type": "number"},
                    "unit": {"type": "string", "enum": ["mg", "g", "kg", "μL", "mL", "L", "mmol", "mol"]},
                    "equivalent": {"type": "number"},
                    "addition_order": {"type": "integer"},
                    "addition_rate": {"type": "string"},
                    "safety_notes": {"type": "string"}
                },
                "required": ["name", "amount", "unit"]
            }
        },
        "solvents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "volume": {"type": "number"},
                    "unit": {"type": "string", "enum": ["μL", "mL", "L"]},
                    "purity": {"type": "string"},
                    "drying_agent": {"type": "string"}
                }
            }
        },
        "procedure_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "action": {"type": "string"},
                    "duration": {"type": "number"},
                    "temperature": {"type": "number"},
                    "monitoring": {"type": "array", "items": {"type": "string"}},
                    "safety_check": {"type": "boolean"}
                },
                "required": ["step", "action"]
            }
        },
        "workup": {
            "type": "object",
            "properties": {
                "extraction": {"type": "array", "items": {"type": "string"}},
                "washing": {"type": "array", "items": {"type": "string"}},
                "drying": {"type": "string"},
                "purification": {"type": "string", "enum": ["column", "recrystallization", "distillation", "none"]}
            }
        },
        "safety_requirements": {
            "type": "object",
            "properties": {
                "fume_hood": {"type": "boolean"},
                "ppe_required": {"type": "array", "items": {"type": "string"}},
                "waste_disposal": {"type": "string"},
                "emergency_procedures": {"type": "string"}
            }
        }
    },
    "required": ["synthesis_id", "target_compound", "reagents", "procedure_steps"]
}

# Characterization Instruction Schema
CHARACTERIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "characterization_id": {"type": "string"},
        "sample_id": {"type": "string"},
        "techniques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "technique": {"type": "string", "enum": ["NMR", "LCMS", "GCMS", "HPLC", "IR", "UV-Vis", "melting_point", "elemental_analysis"]},
                    "priority": {"type": "integer"},
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "solvent": {"type": "string"},
                            "concentration": {"type": "number"},
                            "temperature": {"type": "number"},
                            "acquisition_time": {"type": "number"},
                            "resolution": {"type": "string"},
                            "custom_settings": {"type": "object"}
                        }
                    },
                    "expected_results": {
                        "type": "object",
                        "properties": {
                            "molecular_weight": {"type": "number"},
                            "retention_time": {"type": "number"},
                            "characteristic_peaks": {"type": "array", "items": {"type": "number"}},
                            "purity_threshold": {"type": "number"}
                        }
                    }
                },
                "required": ["technique", "priority"]
            }
        },
        "sample_preparation": {
            "type": "object",
            "properties": {
                "amount_needed": {"type": "number"},
                "unit": {"type": "string"},
                "preparation_steps": {"type": "array", "items": {"type": "string"}},
                "storage_conditions": {"type": "string"}
            }
        },
        "data_processing": {
            "type": "object",
            "properties": {
                "automated_analysis": {"type": "boolean"},
                "peak_integration": {"type": "boolean"},
                "structure_elucidation": {"type": "boolean"},
                "report_generation": {"type": "boolean"}
            }
        },
        "quality_metrics": {
            "type": "object",
            "properties": {
                "purity_target": {"type": "number"},
                "identity_confirmation": {"type": "boolean"},
                "quantitative_analysis": {"type": "boolean"}
            }
        }
    },
    "required": ["characterization_id", "sample_id", "techniques"]
}

# Reaction Conditions Schema
REACTION_CONDITIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "temperature_profile": {
            "type": "object",
            "properties": {
                "initial_temp": {"type": "number"},
                "target_temp": {"type": "number"},
                "heating_rate": {"type": "number"},
                "hold_time": {"type": "number"},
                "cooling_rate": {"type": "number"}
            }
        },
        "pressure_control": {
            "type": "object",
            "properties": {
                "initial_pressure": {"type": "number"},
                "target_pressure": {"type": "number"},
                "pressure_unit": {"type": "string", "enum": ["atm", "bar", "psi", "torr"]}
            }
        },
        "atmosphere_control": {
            "type": "object",
            "properties": {
                "gas_type": {"type": "string"},
                "flow_rate": {"type": "number"},
                "purity": {"type": "string"}
            }
        },
        "mixing_parameters": {
            "type": "object",
            "properties": {
                "stirring_speed": {"type": "number"},
                "mixing_type": {"type": "string", "enum": ["magnetic", "overhead", "orbital", "vortex"]},
                "mixing_pattern": {"type": "string"}
            }
        },
        "monitoring": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "frequency": {"type": "number"},
                    "unit": {"type": "string"},
                    "alarm_limits": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"}
                        }
                    }
                }
            }
        }
    }
}

class SchemaValidator:
    """Utility class for validating instruction JSON against schemas."""
    
    @staticmethod
    def validate_liquid_handler(instruction: Dict[str, Any]) -> bool:
        """Validate liquid handler instruction against schema."""
        try:
            validate(instance=instruction, schema=LIQUID_HANDLER_SCHEMA)
            return True
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            return False
    
    @staticmethod
    def validate_synthesis(instruction: Dict[str, Any]) -> bool:
        """Validate synthesis instruction against schema."""
        try:
            validate(instance=instruction, schema=SYNTHESIS_SCHEMA)
            return True
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            return False
    
    @staticmethod
    def validate_characterization(instruction: Dict[str, Any]) -> bool:
        """Validate characterization instruction against schema."""
        try:
            validate(instance=instruction, schema=CHARACTERIZATION_SCHEMA)
            return True
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            return False
    
    @staticmethod
    def validate_reaction_conditions(conditions: Dict[str, Any]) -> bool:
        """Validate reaction conditions against schema."""
        try:
            validate(instance=conditions, schema=REACTION_CONDITIONS_SCHEMA)
            return True
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            return False