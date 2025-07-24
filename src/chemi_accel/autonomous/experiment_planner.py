"""
Automated Experiment Planning and Instruction Generation

This module translates AI-selected candidate molecules into standardized,
machine-readable instructions for robotic synthesis and characterization platforms.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from rdkit import Chem
from rdkit.Chem import Descriptors

from .schemas import SchemaValidator


@dataclass
class ProtocolStep:
    """Represents a single step in an experimental protocol."""
    step: int
    action: str
    duration: Optional[float] = None
    temperature: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    safety_check: bool = False
    monitoring: Optional[List[str]] = None


@dataclass
class SynthesisInstruction:
    """Complete synthesis instruction for a target compound."""
    synthesis_id: str
    target_compound: Dict[str, Any]
    reagents: List[Dict[str, Any]]
    solvents: List[Dict[str, Any]]
    reaction_conditions: Dict[str, Any]
    procedure_steps: List[ProtocolStep]
    workup: Dict[str, Any]
    safety_requirements: Dict[str, Any]
    estimated_duration: Optional[float] = None
    success_probability: Optional[float] = None


@dataclass
class CharacterizationInstruction:
    """Complete characterization instruction for a synthesized compound."""
    characterization_id: str
    sample_id: str
    techniques: List[Dict[str, Any]]
    sample_preparation: Dict[str, Any]
    data_processing: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    estimated_duration: Optional[float] = None


@dataclass
class InstructionSet:
    """Complete set of instructions for autonomous experimentation."""
    experiment_id: str
    target_molecules: List[str]
    synthesis_instructions: List[SynthesisInstruction]
    characterization_instructions: List[CharacterizationInstruction]
    liquid_handler_protocols: List[Dict[str, Any]]
    timeline: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_criteria: Dict[str, Any]


class ExperimentPlanner:
    """
    Main class for automated experiment planning and instruction generation.
    
    Translates AI-selected candidate molecules into standardized instructions
    for robotic laboratory platforms.
    """
    
    def __init__(self, lab_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experiment planner.
        
        Args:
            lab_config: Laboratory-specific configuration including available
                       equipment, reagents, and protocols.
        """
        self.lab_config = lab_config or self._get_default_lab_config()
        self.validator = SchemaValidator()
        self.reagent_database = self._load_reagent_database()
        self.synthesis_templates = self._load_synthesis_templates()
        
    def generate_synthesis_instructions(
        self, 
        target_molecules: List[str],
        synthesis_strategy: str = "automated",
        optimization_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SynthesisInstruction]:
        """
        Generate synthesis instructions for target molecules.
        
        Args:
            target_molecules: List of SMILES strings for target compounds
            synthesis_strategy: Strategy for synthesis planning
            optimization_criteria: Criteria for optimizing synthesis conditions
            
        Returns:
            List of SynthesisInstruction objects
        """
        instructions = []
        
        for smiles in target_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Generate synthesis instruction for this molecule
                synthesis_id = str(uuid.uuid4())
                
                # Analyze target compound properties
                target_compound = self._analyze_target_compound(smiles, mol)
                
                # Plan synthesis route
                synthesis_route = self._plan_synthesis_route(mol, synthesis_strategy)
                
                # Generate reagent list
                reagents = self._generate_reagent_list(synthesis_route)
                
                # Determine solvents
                solvents = self._select_solvents(synthesis_route, mol)
                
                # Set reaction conditions
                reaction_conditions = self._determine_reaction_conditions(
                    synthesis_route, optimization_criteria
                )
                
                # Generate procedure steps
                procedure_steps = self._generate_procedure_steps(synthesis_route)
                
                # Plan workup procedures
                workup = self._plan_workup_procedures(synthesis_route)
                
                # Assess safety requirements
                safety_requirements = self._assess_safety_requirements(reagents, solvents)
                
                # Estimate duration and success probability
                estimated_duration = self._estimate_synthesis_duration(procedure_steps)
                success_probability = self._estimate_success_probability(synthesis_route)
                
                instruction = SynthesisInstruction(
                    synthesis_id=synthesis_id,
                    target_compound=target_compound,
                    reagents=reagents,
                    solvents=solvents,
                    reaction_conditions=reaction_conditions,
                    procedure_steps=procedure_steps,
                    workup=workup,
                    safety_requirements=safety_requirements,
                    estimated_duration=estimated_duration,
                    success_probability=success_probability
                )
                
                instructions.append(instruction)
                
            except Exception as e:
                print(f"Error generating synthesis instruction for {smiles}: {e}")
                continue
                
        return instructions
    
    def generate_characterization_instructions(
        self,
        sample_ids: List[str],
        expected_compounds: List[str],
        characterization_scope: str = "full"
    ) -> List[CharacterizationInstruction]:
        """
        Generate characterization instructions for synthesized compounds.
        
        Args:
            sample_ids: List of sample identifiers
            expected_compounds: List of expected compound SMILES
            characterization_scope: Scope of characterization ("minimal", "standard", "full")
            
        Returns:
            List of CharacterizationInstruction objects
        """
        instructions = []
        
        for i, sample_id in enumerate(sample_ids):
            try:
                characterization_id = str(uuid.uuid4())
                expected_smiles = expected_compounds[i] if i < len(expected_compounds) else None
                
                # Determine characterization techniques
                techniques = self._select_characterization_techniques(
                    expected_smiles, characterization_scope
                )
                
                # Plan sample preparation
                sample_preparation = self._plan_sample_preparation(techniques)
                
                # Configure data processing
                data_processing = self._configure_data_processing(techniques)
                
                # Set quality metrics
                quality_metrics = self._set_quality_metrics(expected_smiles)
                
                # Estimate duration
                estimated_duration = self._estimate_characterization_duration(techniques)
                
                instruction = CharacterizationInstruction(
                    characterization_id=characterization_id,
                    sample_id=sample_id,
                    techniques=techniques,
                    sample_preparation=sample_preparation,
                    data_processing=data_processing,
                    quality_metrics=quality_metrics,
                    estimated_duration=estimated_duration
                )
                
                instructions.append(instruction)
                
            except Exception as e:
                print(f"Error generating characterization instruction for {sample_id}: {e}")
                continue
                
        return instructions
    
    def generate_liquid_handler_protocol(
        self,
        synthesis_instructions: List[SynthesisInstruction],
        platform_type: str = "tecan"
    ) -> Dict[str, Any]:
        """
        Generate liquid handler protocol for automated synthesis.
        
        Args:
            synthesis_instructions: List of synthesis instructions
            platform_type: Type of liquid handling platform
            
        Returns:
            Complete liquid handler protocol as dictionary
        """
        protocol_id = str(uuid.uuid4())
        
        # Initialize protocol structure
        protocol = {
            "protocol_id": protocol_id,
            "protocol_name": f"Automated_Synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "type": platform_type,
                "model": self.lab_config.get("liquid_handler", {}).get("model", "Unknown"),
                "software_version": self.lab_config.get("liquid_handler", {}).get("software_version", "1.0")
            }
        }
        
        # Generate labware configuration
        labware = self._generate_labware_configuration(synthesis_instructions)
        protocol["labware"] = labware
        
        # Generate liquid handling operations
        operations = self._generate_liquid_handling_operations(synthesis_instructions, labware)
        protocol["operations"] = operations
        
        # Configure quality control
        protocol["quality_control"] = {
            "volume_verification": True,
            "liquid_level_detection": True,
            "barcode_scanning": True,
            "error_handling": "retry"
        }
        
        # Validate protocol against schema
        if not self.validator.validate_liquid_handler(protocol):
            raise ValueError("Generated liquid handler protocol failed validation")
            
        return protocol
    
    def create_complete_instruction_set(
        self,
        target_molecules: List[str],
        experiment_name: str,
        optimization_criteria: Optional[Dict[str, Any]] = None
    ) -> InstructionSet:
        """
        Create a complete instruction set for autonomous experimentation.
        
        Args:
            target_molecules: List of target molecule SMILES
            experiment_name: Name for the experiment
            optimization_criteria: Optimization criteria for synthesis
            
        Returns:
            Complete InstructionSet object
        """
        experiment_id = str(uuid.uuid4())
        
        # Generate synthesis instructions
        synthesis_instructions = self.generate_synthesis_instructions(
            target_molecules, optimization_criteria=optimization_criteria
        )
        
        # Generate sample IDs for characterization
        sample_ids = [f"sample_{i+1}_{instr.synthesis_id[:8]}" 
                     for i, instr in enumerate(synthesis_instructions)]
        
        # Generate characterization instructions
        characterization_instructions = self.generate_characterization_instructions(
            sample_ids, target_molecules
        )
        
        # Generate liquid handler protocols
        liquid_handler_protocols = []
        if synthesis_instructions:
            protocol = self.generate_liquid_handler_protocol(synthesis_instructions)
            liquid_handler_protocols.append(protocol)
        
        # Create experiment timeline
        timeline = self._create_experiment_timeline(
            synthesis_instructions, characterization_instructions
        )
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(
            synthesis_instructions, characterization_instructions
        )
        
        # Define success criteria
        success_criteria = self._define_success_criteria(target_molecules)
        
        return InstructionSet(
            experiment_id=experiment_id,
            target_molecules=target_molecules,
            synthesis_instructions=synthesis_instructions,
            characterization_instructions=characterization_instructions,
            liquid_handler_protocols=liquid_handler_protocols,
            timeline=timeline,
            resource_requirements=resource_requirements,
            success_criteria=success_criteria
        )
    
    def export_instructions_json(
        self, 
        instruction_set: InstructionSet, 
        output_path: str
    ) -> None:
        """
        Export complete instruction set to JSON file.
        
        Args:
            instruction_set: Complete instruction set
            output_path: Path for output JSON file
        """
        # Convert dataclasses to dictionaries
        export_data = {
            "experiment_id": instruction_set.experiment_id,
            "target_molecules": instruction_set.target_molecules,
            "synthesis_instructions": [asdict(instr) for instr in instruction_set.synthesis_instructions],
            "characterization_instructions": [asdict(instr) for instr in instruction_set.characterization_instructions],
            "liquid_handler_protocols": instruction_set.liquid_handler_protocols,
            "timeline": instruction_set.timeline,
            "resource_requirements": instruction_set.resource_requirements,
            "success_criteria": instruction_set.success_criteria,
            "generated_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    # Private helper methods
    
    def _get_default_lab_config(self) -> Dict[str, Any]:
        """Get default laboratory configuration."""
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
    
    def _load_reagent_database(self) -> Dict[str, Any]:
        """Load reagent database with properties and availability."""
        # This would typically load from an external database
        return {
            "common_reagents": [
                {"name": "sodium_borohydride", "cas": "16940-66-2", "hazards": ["H261", "H301"]},
                {"name": "lithium_aluminum_hydride", "cas": "16853-85-3", "hazards": ["H260", "H314"]},
                {"name": "palladium_catalyst", "cas": "7440-05-3", "hazards": ["H317", "H334"]}
            ]
        }
    
    def _load_synthesis_templates(self) -> Dict[str, Any]:
        """Load synthesis template library."""
        return {
            "reduction": {
                "reagents": ["reducing_agent", "solvent"],
                "conditions": {"temperature": 0, "atmosphere": "nitrogen"}
            },
            "coupling": {
                "reagents": ["coupling_partner", "catalyst", "base"],
                "conditions": {"temperature": 80, "atmosphere": "nitrogen"}
            }
        }
    
    def _analyze_target_compound(self, smiles: str, mol: Chem.Mol) -> Dict[str, Any]:
        """Analyze target compound properties."""
        return {
            "smiles": smiles,
            "name": f"compound_{smiles[:10]}",
            "molecular_weight": Descriptors.MolWt(mol),
            "target_yield": 0.8,
            "purity_target": 0.95
        }
    
    def _plan_synthesis_route(self, mol: Chem.Mol, strategy: str) -> Dict[str, Any]:
        """Plan synthesis route for target molecule."""
        return {
            "route_type": "direct",
            "steps": 1,
            "complexity": "low",
            "template": "reduction"
        }
    
    def _generate_reagent_list(self, synthesis_route: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate list of required reagents."""
        return [
            {
                "name": "sodium_borohydride",
                "smiles": "[BH4-].[Na+]",
                "cas_number": "16940-66-2",
                "amount": 1.2,
                "unit": "g",
                "equivalent": 2.0,
                "addition_order": 2,
                "addition_rate": "slow",
                "safety_notes": "Water reactive - add under nitrogen"
            }
        ]
    
    def _select_solvents(self, synthesis_route: Dict[str, Any], mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Select appropriate solvents for the reaction."""
        return [
            {
                "name": "ethanol",
                "volume": 50.0,
                "unit": "mL",
                "purity": "anhydrous",
                "drying_agent": "molecular_sieves"
            }
        ]
    
    def _determine_reaction_conditions(
        self, 
        synthesis_route: Dict[str, Any], 
        optimization_criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine optimal reaction conditions."""
        return {
            "temperature": 0.0,
            "pressure": 1.0,
            "atmosphere": "nitrogen",
            "reaction_time": 2.0,
            "stirring_speed": 300.0
        }
    
    def _generate_procedure_steps(self, synthesis_route: Dict[str, Any]) -> List[ProtocolStep]:
        """Generate detailed procedure steps."""
        return [
            ProtocolStep(
                step=1,
                action="Setup reaction vessel under nitrogen atmosphere",
                duration=5.0,
                temperature=25.0,
                safety_check=True
            ),
            ProtocolStep(
                step=2,
                action="Add substrate solution dropwise",
                duration=15.0,
                temperature=0.0,
                monitoring=["temperature", "stirring"]
            ),
            ProtocolStep(
                step=3,
                action="Add reducing agent slowly",
                duration=30.0,
                temperature=0.0,
                monitoring=["gas_evolution", "temperature"]
            )
        ]
    
    def _plan_workup_procedures(self, synthesis_route: Dict[str, Any]) -> Dict[str, Any]:
        """Plan workup and purification procedures."""
        return {
            "extraction": ["water", "ethyl_acetate"],
            "washing": ["brine"],
            "drying": "Na2SO4",
            "purification": "column"
        }
    
    def _assess_safety_requirements(
        self, 
        reagents: List[Dict[str, Any]], 
        solvents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess safety requirements for the synthesis."""
        return {
            "fume_hood": True,
            "ppe_required": ["safety_glasses", "lab_coat", "gloves"],
            "waste_disposal": "organic_waste",
            "emergency_procedures": "Have water and fire extinguisher nearby"
        }
    
    def _estimate_synthesis_duration(self, procedure_steps: List[ProtocolStep]) -> float:
        """Estimate total synthesis duration."""
        return sum(step.duration or 0 for step in procedure_steps) + 60.0  # Add workup time
    
    def _estimate_success_probability(self, synthesis_route: Dict[str, Any]) -> float:
        """Estimate probability of synthesis success."""
        # Simple heuristic - would be more sophisticated in practice
        complexity_factors = {
            "low": 0.9,
            "medium": 0.7,
            "high": 0.5
        }
        return complexity_factors.get(synthesis_route.get("complexity", "medium"), 0.7)
    
    def _select_characterization_techniques(
        self, 
        expected_smiles: Optional[str], 
        scope: str
    ) -> List[Dict[str, Any]]:
        """Select appropriate characterization techniques."""
        techniques = []
        
        if scope in ["standard", "full"]:
            techniques.extend([
                {
                    "technique": "NMR",
                    "priority": 1,
                    "parameters": {
                        "solvent": "CDCl3",
                        "temperature": 25.0,
                        "acquisition_time": 60.0
                    }
                },
                {
                    "technique": "LCMS",
                    "priority": 2,
                    "parameters": {
                        "solvent": "MeCN/H2O",
                        "acquisition_time": 30.0
                    }
                }
            ])
        
        if scope == "full":
            techniques.extend([
                {
                    "technique": "IR",
                    "priority": 3,
                    "parameters": {
                        "resolution": "4cm-1",
                        "acquisition_time": 5.0
                    }
                },
                {
                    "technique": "elemental_analysis",
                    "priority": 4,
                    "parameters": {}
                }
            ])
        
        return techniques
    
    def _plan_sample_preparation(self, techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan sample preparation for characterization."""
        total_amount = sum(self._get_sample_amount_needed(tech) for tech in techniques)
        
        return {
            "amount_needed": total_amount,
            "unit": "mg",
            "preparation_steps": [
                "Dry sample under vacuum",
                "Weigh accurately",
                "Prepare solutions as needed"
            ],
            "storage_conditions": "dry, inert atmosphere"
        }
    
    def _get_sample_amount_needed(self, technique: Dict[str, Any]) -> float:
        """Get sample amount needed for a technique."""
        amounts = {
            "NMR": 5.0,
            "LCMS": 1.0,
            "IR": 2.0,
            "elemental_analysis": 3.0
        }
        return amounts.get(technique["technique"], 1.0)
    
    def _configure_data_processing(self, techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure automated data processing."""
        return {
            "automated_analysis": True,
            "peak_integration": True,
            "structure_elucidation": "NMR" in [t["technique"] for t in techniques],
            "report_generation": True
        }
    
    def _set_quality_metrics(self, expected_smiles: Optional[str]) -> Dict[str, Any]:
        """Set quality metrics for characterization."""
        return {
            "purity_target": 0.95,
            "identity_confirmation": expected_smiles is not None,
            "quantitative_analysis": True
        }
    
    def _estimate_characterization_duration(self, techniques: List[Dict[str, Any]]) -> float:
        """Estimate total characterization duration."""
        durations = {
            "NMR": 60.0,
            "LCMS": 30.0,
            "IR": 5.0,
            "elemental_analysis": 120.0
        }
        return sum(durations.get(tech["technique"], 30.0) for tech in techniques)
    
    def _generate_labware_configuration(
        self, 
        synthesis_instructions: List[SynthesisInstruction]
    ) -> List[Dict[str, Any]]:
        """Generate labware configuration for liquid handler."""
        labware = [
            {
                "id": "reagent_plate_1",
                "type": "plate",
                "format": "96",
                "position": "P1",
                "contents": []
            },
            {
                "id": "reaction_plate_1",
                "type": "plate",
                "format": "96",
                "position": "P2",
                "contents": []
            }
        ]
        
        # Add reagent information
        for i, instruction in enumerate(synthesis_instructions):
            for j, reagent in enumerate(instruction.reagents):
                well = f"{chr(65 + i)}{j+1}"
                labware[0]["contents"].append({
                    "well": well,
                    "compound_id": reagent["name"],
                    "smiles": reagent.get("smiles", ""),
                    "concentration": 1.0,
                    "volume": 100.0,
                    "solvent": "DMSO"
                })
        
        return labware
    
    def _generate_liquid_handling_operations(
        self, 
        synthesis_instructions: List[SynthesisInstruction],
        labware: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate liquid handling operations."""
        operations = []
        step = 1
        
        for i, instruction in enumerate(synthesis_instructions):
            target_well = f"{chr(65 + i)}1"
            
            # Add reagents sequentially
            for j, reagent in enumerate(instruction.reagents):
                source_well = f"{chr(65 + i)}{j+1}"
                
                operations.extend([
                    {
                        "step": step,
                        "operation": "tip_pickup",
                        "parameters": {"tip_type": "standard"}
                    },
                    {
                        "step": step + 1,
                        "operation": "aspirate",
                        "source": f"reagent_plate_1:{source_well}",
                        "volume": 10.0,
                        "speed": 5.0
                    },
                    {
                        "step": step + 2,
                        "operation": "dispense",
                        "destination": f"reaction_plate_1:{target_well}",
                        "volume": 10.0,
                        "speed": 5.0
                    },
                    {
                        "step": step + 3,
                        "operation": "tip_drop"
                    }
                ])
                step += 4
        
        return operations
    
    def _create_experiment_timeline(
        self,
        synthesis_instructions: List[SynthesisInstruction],
        characterization_instructions: List[CharacterizationInstruction]
    ) -> Dict[str, Any]:
        """Create experiment timeline."""
        total_synthesis_time = sum(
            instr.estimated_duration or 0 for instr in synthesis_instructions
        )
        total_characterization_time = sum(
            instr.estimated_duration or 0 for instr in characterization_instructions
        )
        
        return {
            "total_duration_hours": (total_synthesis_time + total_characterization_time) / 60.0,
            "synthesis_phase_hours": total_synthesis_time / 60.0,
            "characterization_phase_hours": total_characterization_time / 60.0,
            "parallel_processing": len(synthesis_instructions) > 1
        }
    
    def _calculate_resource_requirements(
        self,
        synthesis_instructions: List[SynthesisInstruction],
        characterization_instructions: List[CharacterizationInstruction]
    ) -> Dict[str, Any]:
        """Calculate resource requirements."""
        all_reagents = []
        for instr in synthesis_instructions:
            all_reagents.extend(instr.reagents)
        
        all_techniques = []
        for instr in characterization_instructions:
            all_techniques.extend([t["technique"] for t in instr.techniques])
        
        return {
            "unique_reagents": len(set(r["name"] for r in all_reagents)),
            "characterization_techniques": list(set(all_techniques)),
            "estimated_cost_usd": len(synthesis_instructions) * 100.0,  # Rough estimate
            "equipment_time_hours": {
                "liquid_handler": 2.0,
                "synthesis_platform": 4.0,
                "characterization": 3.0
            }
        }
    
    def _define_success_criteria(self, target_molecules: List[str]) -> Dict[str, Any]:
        """Define success criteria for the experiment."""
        return {
            "minimum_yield": 0.5,
            "minimum_purity": 0.9,
            "identity_confirmation_required": True,
            "acceptable_failure_rate": 0.2,
            "target_molecules_count": len(target_molecules)
        }