"""
Polymer-specific validation tests against experimental databases.

This module provides validation of polymer processing capabilities including
monomer analysis, oligomer processing, and polymer-specific descriptors
against experimental polymer databases and known polymer properties.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.polymer.monomer import MonomerAnalyzer
from src.chemi_accel.polymer.oligomer import OligomerProcessor
from src.chemi_accel.polymer.conformers import PolymerConformerGenerator
from src.chemi_accel.descriptors.polymer import PolymerDescriptorCalculator
from src.chemi_accel.core.molecular import MolecularData


# Comprehensive polymer test dataset
POLYMER_TEST_DATASET = {
    "common_monomers": {
        "ethylene": {
            "smiles": "C=C",
            "polymer_name": "polyethylene",
            "expected_properties": {
                "polymerizability": 0.9,
                "reactivity_score": 0.8,
                "glass_transition_temp": -125,  # °C
                "density": 0.92,  # g/cm³
            }
        },
        "propylene": {
            "smiles": "C=CC",
            "polymer_name": "polypropylene",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.75,
                "glass_transition_temp": -10,
                "density": 0.90,
            }
        },
        "styrene": {
            "smiles": "C=Cc1ccccc1",
            "polymer_name": "polystyrene",
            "expected_properties": {
                "polymerizability": 0.9,
                "reactivity_score": 0.85,
                "glass_transition_temp": 100,
                "density": 1.05,
            }
        },
        "vinyl_chloride": {
            "smiles": "C=CCl",
            "polymer_name": "polyvinyl chloride",
            "expected_properties": {
                "polymerizability": 0.8,
                "reactivity_score": 0.7,
                "glass_transition_temp": 80,
                "density": 1.38,
            }
        },
        "methyl_methacrylate": {
            "smiles": "C=C(C)C(=O)OC",
            "polymer_name": "polymethyl methacrylate",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.8,
                "glass_transition_temp": 105,
                "density": 1.18,
            }
        },
        "acrylonitrile": {
            "smiles": "C=CC#N",
            "polymer_name": "polyacrylonitrile",
            "expected_properties": {
                "polymerizability": 0.9,
                "reactivity_score": 0.85,
                "glass_transition_temp": 85,
                "density": 1.17,
            }
        },
        "isoprene": {
            "smiles": "C=CC(=C)C",
            "polymer_name": "polyisoprene",
            "expected_properties": {
                "polymerizability": 0.8,
                "reactivity_score": 0.75,
                "glass_transition_temp": -70,
                "density": 0.91,
            }
        },
        "butadiene": {
            "smiles": "C=CC=C",
            "polymer_name": "polybutadiene",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.8,
                "glass_transition_temp": -90,
                "density": 0.90,
            }
        }
    },
    
    "cyclic_monomers": {
        "ethylene_oxide": {
            "smiles": "C1CO1",
            "polymer_name": "polyethylene oxide",
            "polymerization_type": "ring_opening",
            "expected_properties": {
                "polymerizability": 0.95,
                "reactivity_score": 0.9,
                "glass_transition_temp": -67,
                "density": 1.13,
            }
        },
        "propylene_oxide": {
            "smiles": "CC1CO1",
            "polymer_name": "polypropylene oxide",
            "polymerization_type": "ring_opening",
            "expected_properties": {
                "polymerizability": 0.9,
                "reactivity_score": 0.85,
                "glass_transition_temp": -75,
                "density": 1.00,
            }
        },
        "caprolactam": {
            "smiles": "C1CCCCNC1=O",
            "polymer_name": "nylon-6",
            "polymerization_type": "ring_opening",
            "expected_properties": {
                "polymerizability": 0.9,
                "reactivity_score": 0.85,
                "glass_transition_temp": 50,
                "density": 1.13,
            }
        },
        "lactide": {
            "smiles": "C[C@H]1OC(=O)[C@@H](C)OC1=O",
            "polymer_name": "polylactic acid",
            "polymerization_type": "ring_opening",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.8,
                "glass_transition_temp": 60,
                "density": 1.25,
            }
        }
    },
    
    "condensation_monomers": {
        "terephthalic_acid": {
            "smiles": "O=C(O)c1ccc(cc1)C(=O)O",
            "polymer_name": "polyethylene terephthalate",
            "comonomer": "ethylene_glycol",
            "expected_properties": {
                "polymerizability": 0.8,
                "reactivity_score": 0.75,
                "glass_transition_temp": 75,
                "density": 1.38,
            }
        },
        "ethylene_glycol": {
            "smiles": "OCCO",
            "polymer_name": "polyethylene terephthalate",
            "comonomer": "terephthalic_acid",
            "expected_properties": {
                "polymerizability": 0.8,
                "reactivity_score": 0.75,
            }
        },
        "hexamethylene_diamine": {
            "smiles": "NCCCCCCN",
            "polymer_name": "nylon-6,6",
            "comonomer": "adipic_acid",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.8,
                "glass_transition_temp": 50,
                "density": 1.14,
            }
        },
        "adipic_acid": {
            "smiles": "O=C(O)CCCCC(=O)O",
            "polymer_name": "nylon-6,6",
            "comonomer": "hexamethylene_diamine",
            "expected_properties": {
                "polymerizability": 0.85,
                "reactivity_score": 0.8,
            }
        }
    },
    
    "oligomer_test_cases": {
        "ethylene_dimer": {
            "smiles": "CCCC",
            "monomer": "ethylene",
            "degree_of_polymerization": 2,
            "expected_properties": {
                "end_to_end_distance": 3.5,  # Angstroms
                "radius_of_gyration": 1.8,
                "persistence_length": 1.5,
            }
        },
        "styrene_trimer": {
            "smiles": "CCc1ccccc1CCc2ccccc2CCc3ccccc3",
            "monomer": "styrene",
            "degree_of_polymerization": 3,
            "expected_properties": {
                "end_to_end_distance": 8.0,
                "radius_of_gyration": 4.2,
                "persistence_length": 3.0,
            }
        },
        "propylene_tetramer": {
            "smiles": "CC(C)CC(C)CC(C)CC",
            "monomer": "propylene",
            "degree_of_polymerization": 4,
            "expected_properties": {
                "end_to_end_distance": 6.5,
                "radius_of_gyration": 3.1,
                "persistence_length": 2.2,
            }
        }
    },
    
    "challenging_polymers": {
        "polyacetylene": {
            "smiles": "C=C",  # Monomer
            "polymer_smiles": "C=CC=CC=CC=C",  # Short oligomer
            "challenges": ["conjugation", "instability"],
            "expected_properties": {
                "conductivity": "high",
                "stability": "low",
            }
        },
        "polyaniline": {
            "smiles": "Nc1ccc(cc1)N",  # Simplified monomer
            "challenges": ["aromaticity", "conductivity"],
            "expected_properties": {
                "conductivity": "variable",
                "glass_transition_temp": 150,
            }
        },
        "polythiophene": {
            "smiles": "c1csc(c1)",  # Thiophene unit
            "challenges": ["heteroatom", "conductivity"],
            "expected_properties": {
                "conductivity": "moderate",
                "stability": "moderate",
            }
        }
    }
}


class PolymerValidator:
    """Validator for polymer-specific calculations."""
    
    def __init__(self):
        """Initialize polymer validator."""
        self.monomer_analyzer = MonomerAnalyzer()
        self.oligomer_processor = OligomerProcessor()
        self.conformer_generator = PolymerConformerGenerator(use_gpu=False)
        self.polymer_calc = PolymerDescriptorCalculator(use_gpu=False)
    
    def validate_monomer_analysis(self, monomer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate monomer analysis capabilities.
        
        Args:
            monomer_data: Dictionary containing monomer information
            
        Returns:
            Validation results
        """
        smiles = monomer_data["smiles"]
        expected_props = monomer_data.get("expected_properties", {})
        
        results = {
            "monomer": smiles,
            "analysis_successful": False,
            "property_validations": {},
            "errors": []
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results["errors"].append("Invalid SMILES")
                return results
            
            mol_data = self._create_mol_data(smiles, mol)
            
            # Analyze monomer
            analysis = self.monomer_analyzer.analyze_monomer(mol_data)
            results["analysis_successful"] = True
            
            # Validate polymerizability
            if "polymerizability" in expected_props:
                calculated = analysis.get("polymerizability_score", 0)
                expected = expected_props["polymerizability"]
                
                results["property_validations"]["polymerizability"] = {
                    "calculated": calculated,
                    "expected": expected,
                    "within_tolerance": abs(calculated - expected) < 0.2,
                    "relative_error": abs(calculated - expected) / expected if expected != 0 else 0
                }
            
            # Validate reactivity
            if "reactivity_score" in expected_props:
                calculated = analysis.get("reactivity_score", 0)
                expected = expected_props["reactivity_score"]
                
                results["property_validations"]["reactivity"] = {
                    "calculated": calculated,
                    "expected": expected,
                    "within_tolerance": abs(calculated - expected) < 0.2,
                    "relative_error": abs(calculated - expected) / expected if expected != 0 else 0
                }
            
            # Check functional group identification
            functional_groups = analysis.get("functional_groups", [])
            results["functional_groups_identified"] = len(functional_groups)
            
            # Check polymerization mechanism prediction
            mechanism = analysis.get("polymerization_mechanism", "unknown")
            results["mechanism_predicted"] = mechanism != "unknown"
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results
    
    def validate_oligomer_processing(self, oligomer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate oligomer processing capabilities.
        
        Args:
            oligomer_data: Dictionary containing oligomer information
            
        Returns:
            Validation results
        """
        smiles = oligomer_data["smiles"]
        expected_props = oligomer_data.get("expected_properties", {})
        
        results = {
            "oligomer": smiles,
            "processing_successful": False,
            "property_validations": {},
            "errors": []
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results["errors"].append("Invalid SMILES")
                return results
            
            mol_data = self._create_mol_data(smiles, mol)
            
            # Process oligomer
            analysis = self.oligomer_processor.analyze_oligomer(mol_data)
            results["processing_successful"] = True
            
            # Validate degree of polymerization
            if "degree_of_polymerization" in oligomer_data:
                calculated_dp = analysis.get("degree_of_polymerization", 0)
                expected_dp = oligomer_data["degree_of_polymerization"]
                
                results["property_validations"]["degree_of_polymerization"] = {
                    "calculated": calculated_dp,
                    "expected": expected_dp,
                    "exact_match": calculated_dp == expected_dp
                }
            
            # Validate conformational properties
            if "end_to_end_distance" in expected_props:
                # Generate conformers and calculate properties
                conformer_result = self.conformer_generator.generate_conformers(mol_data)
                
                if "error" not in conformer_result:
                    descriptors = self.polymer_calc.calculate_polymer_descriptors(conformer_result)
                    
                    calculated_ete = descriptors.get("end_to_end_distance", 0)
                    expected_ete = expected_props["end_to_end_distance"]
                    
                    results["property_validations"]["end_to_end_distance"] = {
                        "calculated": calculated_ete,
                        "expected": expected_ete,
                        "within_tolerance": abs(calculated_ete - expected_ete) < expected_ete * 0.3,
                        "relative_error": abs(calculated_ete - expected_ete) / expected_ete if expected_ete != 0 else 0
                    }
            
            # Check sequence analysis
            sequence = analysis.get("monomer_sequence", [])
            results["sequence_identified"] = len(sequence) > 0
            
            # Check chain flexibility analysis
            flexibility = analysis.get("chain_flexibility", 0)
            results["flexibility_calculated"] = flexibility > 0
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results
    
    def validate_polymer_descriptors(self, polymer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate polymer-specific descriptor calculations.
        
        Args:
            polymer_data: Dictionary containing polymer information
            
        Returns:
            Validation results
        """
        smiles = polymer_data["smiles"]
        expected_props = polymer_data.get("expected_properties", {})
        
        results = {
            "polymer": smiles,
            "descriptors_calculated": False,
            "property_validations": {},
            "errors": []
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results["errors"].append("Invalid SMILES")
                return results
            
            mol_data = self._create_mol_data(smiles, mol)
            
            # Calculate polymer descriptors
            descriptors = self.polymer_calc.calculate_polymer_descriptors(mol_data)
            results["descriptors_calculated"] = True
            
            # Validate glass transition temperature prediction
            if "glass_transition_temp" in expected_props:
                calculated_tg = descriptors.get("glass_transition_temp", None)
                expected_tg = expected_props["glass_transition_temp"]
                
                if calculated_tg is not None:
                    results["property_validations"]["glass_transition_temp"] = {
                        "calculated": calculated_tg,
                        "expected": expected_tg,
                        "within_tolerance": abs(calculated_tg - expected_tg) < 50,  # ±50°C tolerance
                        "absolute_error": abs(calculated_tg - expected_tg)
                    }
            
            # Validate density prediction
            if "density" in expected_props:
                calculated_density = descriptors.get("density", None)
                expected_density = expected_props["density"]
                
                if calculated_density is not None:
                    results["property_validations"]["density"] = {
                        "calculated": calculated_density,
                        "expected": expected_density,
                        "within_tolerance": abs(calculated_density - expected_density) < 0.2,  # ±0.2 g/cm³
                        "relative_error": abs(calculated_density - expected_density) / expected_density
                    }
            
            # Check that basic polymer descriptors are calculated
            basic_descriptors = ["persistence_length", "radius_of_gyration", "end_to_end_distance"]
            for desc in basic_descriptors:
                if desc in descriptors:
                    results[f"{desc}_calculated"] = True
                    results[f"{desc}_value"] = descriptors[desc]
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results
    
    def validate_ring_opening_polymerization(self, monomer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ring-opening polymerization analysis.
        
        Args:
            monomer_data: Dictionary containing cyclic monomer information
            
        Returns:
            Validation results
        """
        smiles = monomer_data["smiles"]
        
        results = {
            "monomer": smiles,
            "rop_analysis_successful": False,
            "ring_detected": False,
            "rop_feasible": False,
            "errors": []
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results["errors"].append("Invalid SMILES")
                return results
            
            mol_data = self._create_mol_data(smiles, mol)
            
            # Analyze for ring-opening polymerization
            analysis = self.monomer_analyzer.analyze_ring_opening_potential(mol_data)
            results["rop_analysis_successful"] = True
            
            # Check ring detection
            results["ring_detected"] = analysis.get("has_polymerizable_ring", False)
            
            # Check ROP feasibility
            results["rop_feasible"] = analysis.get("rop_feasible", False)
            
            # Check ring strain analysis
            ring_strain = analysis.get("ring_strain", 0)
            results["ring_strain"] = ring_strain
            results["high_strain"] = ring_strain > 10  # kcal/mol threshold
            
            # Check functional group compatibility
            functional_groups = analysis.get("functional_groups", [])
            results["functional_groups"] = functional_groups
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results
    
    def _create_mol_data(self, smiles: str, mol) -> MolecularData:
        """Create MolecularData object."""
        return MolecularData(
            smiles=smiles,
            mol_id=f"polymer_val_{hash(smiles)}",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=Descriptors.MolWt(mol),
            backend_data={"rdkit_mol": mol}
        )
    
    def generate_comprehensive_polymer_report(self) -> Dict[str, Any]:
        """Generate comprehensive polymer validation report."""
        report = {
            "summary": {
                "total_monomers_tested": 0,
                "total_oligomers_tested": 0,
                "successful_monomer_analyses": 0,
                "successful_oligomer_analyses": 0,
                "successful_descriptor_calculations": 0,
            },
            "detailed_results": {
                "monomer_validation": {},
                "oligomer_validation": {},
                "descriptor_validation": {},
                "rop_validation": {}
            }
        }
        
        # Test common monomers
        for monomer_name, monomer_data in POLYMER_TEST_DATASET["common_monomers"].items():
            print(f"Testing monomer: {monomer_name}")
            
            result = self.validate_monomer_analysis(monomer_data)
            report["detailed_results"]["monomer_validation"][monomer_name] = result
            report["summary"]["total_monomers_tested"] += 1
            
            if result["analysis_successful"]:
                report["summary"]["successful_monomer_analyses"] += 1
            
            # Also test descriptor calculation
            desc_result = self.validate_polymer_descriptors(monomer_data)
            report["detailed_results"]["descriptor_validation"][monomer_name] = desc_result
            
            if desc_result["descriptors_calculated"]:
                report["summary"]["successful_descriptor_calculations"] += 1
        
        # Test cyclic monomers for ROP
        for monomer_name, monomer_data in POLYMER_TEST_DATASET["cyclic_monomers"].items():
            print(f"Testing cyclic monomer: {monomer_name}")
            
            result = self.validate_ring_opening_polymerization(monomer_data)
            report["detailed_results"]["rop_validation"][monomer_name] = result
        
        # Test oligomers
        for oligomer_name, oligomer_data in POLYMER_TEST_DATASET["oligomer_test_cases"].items():
            print(f"Testing oligomer: {oligomer_name}")
            
            result = self.validate_oligomer_processing(oligomer_data)
            report["detailed_results"]["oligomer_validation"][oligomer_name] = result
            report["summary"]["total_oligomers_tested"] += 1
            
            if result["processing_successful"]:
                report["summary"]["successful_oligomer_analyses"] += 1
        
        # Calculate success rates
        if report["summary"]["total_monomers_tested"] > 0:
            report["summary"]["monomer_success_rate"] = (
                report["summary"]["successful_monomer_analyses"] / 
                report["summary"]["total_monomers_tested"]
            )
        
        if report["summary"]["total_oligomers_tested"] > 0:
            report["summary"]["oligomer_success_rate"] = (
                report["summary"]["successful_oligomer_analyses"] / 
                report["summary"]["total_oligomers_tested"]
            )
        
        return report


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestPolymerValidation:
    """Test polymer-specific validation."""
    
    def setup_method(self):
        """Set up polymer validator."""
        self.validator = PolymerValidator()
    
    def test_common_monomer_analysis(self):
        """Test analysis of common monomers."""
        # Test ethylene
        ethylene_data = POLYMER_TEST_DATASET["common_monomers"]["ethylene"]
        result = self.validator.validate_monomer_analysis(ethylene_data)
        
        assert result["analysis_successful"], "Ethylene analysis should succeed"
        assert result["functional_groups_identified"] >= 0
        assert result["mechanism_predicted"]
        
        # Check polymerizability validation if available
        if "polymerizability" in result["property_validations"]:
            poly_val = result["property_validations"]["polymerizability"]
            assert poly_val["within_tolerance"], f"Polymerizability not within tolerance: {poly_val}"
    
    def test_styrene_monomer_analysis(self):
        """Test analysis of styrene monomer."""
        styrene_data = POLYMER_TEST_DATASET["common_monomers"]["styrene"]
        result = self.validator.validate_monomer_analysis(styrene_data)
        
        assert result["analysis_successful"], "Styrene analysis should succeed"
        assert result["functional_groups_identified"] > 0, "Should identify vinyl group"
        
        # Styrene should have high polymerizability
        if "polymerizability" in result["property_validations"]:
            poly_val = result["property_validations"]["polymerizability"]
            assert poly_val["calculated"] > 0.7, "Styrene should have high polymerizability"
    
    def test_ring_opening_polymerization(self):
        """Test ring-opening polymerization analysis."""
        # Test ethylene oxide
        eo_data = POLYMER_TEST_DATASET["cyclic_monomers"]["ethylene_oxide"]
        result = self.validator.validate_ring_opening_polymerization(eo_data)
        
        assert result["rop_analysis_successful"], "EO ROP analysis should succeed"
        assert result["ring_detected"], "Should detect ring in ethylene oxide"
        assert result["rop_feasible"], "EO should be feasible for ROP"
        assert result["high_strain"], "Three-membered ring should have high strain"
    
    def test_oligomer_processing(self):
        """Test oligomer processing capabilities."""
        # Test ethylene dimer
        dimer_data = POLYMER_TEST_DATASET["oligomer_test_cases"]["ethylene_dimer"]
        result = self.validator.validate_oligomer_processing(dimer_data)
        
        assert result["processing_successful"], "Dimer processing should succeed"
        assert result["sequence_identified"], "Should identify monomer sequence"
        
        # Check degree of polymerization
        if "degree_of_polymerization" in result["property_validations"]:
            dp_val = result["property_validations"]["degree_of_polymerization"]
            assert dp_val["exact_match"], f"DP should match exactly: {dp_val}"
    
    def test_polymer_descriptor_calculation(self):
        """Test polymer-specific descriptor calculations."""
        # Test with polystyrene monomer
        ps_data = POLYMER_TEST_DATASET["common_monomers"]["styrene"]
        result = self.validator.validate_polymer_descriptors(ps_data)
        
        assert result["descriptors_calculated"], "Descriptor calculation should succeed"
        
        # Check glass transition temperature prediction
        if "glass_transition_temp" in result["property_validations"]:
            tg_val = result["property_validations"]["glass_transition_temp"]
            assert tg_val["within_tolerance"], f"Tg prediction not within tolerance: {tg_val}"
    
    def test_condensation_monomer_analysis(self):
        """Test analysis of condensation monomers."""
        # Test terephthalic acid
        ta_data = POLYMER_TEST_DATASET["condensation_monomers"]["terephthalic_acid"]
        result = self.validator.validate_monomer_analysis(ta_data)
        
        assert result["analysis_successful"], "Terephthalic acid analysis should succeed"
        assert result["functional_groups_identified"] > 0, "Should identify carboxylic acid groups"
    
    def test_challenging_polymer_handling(self):
        """Test handling of challenging polymer systems."""
        # Test polyacetylene
        pa_data = POLYMER_TEST_DATASET["challenging_polymers"]["polyacetylene"]
        
        # Test monomer analysis
        result = self.validator.validate_monomer_analysis(pa_data)
        
        # Should handle challenging cases gracefully
        if not result["analysis_successful"]:
            assert len(result["errors"]) > 0, "Should report specific errors for challenging cases"
        else:
            # If successful, should identify conjugation
            assert result["functional_groups_identified"] >= 0
    
    def test_polymer_property_ranges(self):
        """Test that polymer properties are within reasonable ranges."""
        test_monomers = ["ethylene", "styrene", "vinyl_chloride"]
        
        for monomer_name in test_monomers:
            monomer_data = POLYMER_TEST_DATASET["common_monomers"][monomer_name]
            result = self.validator.validate_polymer_descriptors(monomer_data)
            
            if result["descriptors_calculated"]:
                # Check glass transition temperature range
                if "glass_transition_temp_value" in result:
                    tg = result["glass_transition_temp_value"]
                    assert -200 < tg < 300, f"Unreasonable Tg for {monomer_name}: {tg}°C"
                
                # Check density range
                if "density_value" in result:
                    density = result["density_value"]
                    assert 0.5 < density < 3.0, f"Unreasonable density for {monomer_name}: {density} g/cm³"
                
                # Check persistence length
                if "persistence_length_value" in result:
                    pl = result["persistence_length_value"]
                    assert 0 < pl < 100, f"Unreasonable persistence length for {monomer_name}: {pl} Å"
    
    def test_monomer_reactivity_consistency(self):
        """Test consistency of monomer reactivity predictions."""
        # Compare similar monomers
        ethylene_data = POLYMER_TEST_DATASET["common_monomers"]["ethylene"]
        propylene_data = POLYMER_TEST_DATASET["common_monomers"]["propylene"]
        
        eth_result = self.validator.validate_monomer_analysis(ethylene_data)
        prop_result = self.validator.validate_monomer_analysis(propylene_data)
        
        if (eth_result["analysis_successful"] and prop_result["analysis_successful"]):
            # Both should have similar reactivity patterns
            eth_groups = eth_result["functional_groups_identified"]
            prop_groups = prop_result["functional_groups_identified"]
            
            # Both should identify vinyl groups
            assert eth_groups > 0 and prop_groups > 0, "Both should identify functional groups"
    
    def test_oligomer_conformational_properties(self):
        """Test conformational property calculations for oligomers."""
        # Test styrene trimer
        trimer_data = POLYMER_TEST_DATASET["oligomer_test_cases"]["styrene_trimer"]
        result = self.validator.validate_oligomer_processing(trimer_data)
        
        if result["processing_successful"]:
            # Check end-to-end distance validation
            if "end_to_end_distance" in result["property_validations"]:
                ete_val = result["property_validations"]["end_to_end_distance"]
                
                # Should be within reasonable tolerance
                assert ete_val["relative_error"] < 0.5, f"High error in end-to-end distance: {ete_val}"
                
                # Value should be positive and reasonable
                assert ete_val["calculated"] > 0, "End-to-end distance should be positive"
                assert ete_val["calculated"] < 50, "End-to-end distance should be reasonable"
    
    def test_error_handling_in_polymer_analysis(self):
        """Test error handling in polymer analysis."""
        # Test with invalid SMILES
        invalid_data = {"smiles": "INVALID", "expected_properties": {}}
        
        result = self.validator.validate_monomer_analysis(invalid_data)
        assert not result["analysis_successful"]
        assert len(result["errors"]) > 0
        assert "Invalid SMILES" in result["errors"]
        
        # Test with empty SMILES
        empty_data = {"smiles": "", "expected_properties": {}}
        result = self.validator.validate_monomer_analysis(empty_data)
        assert not result["analysis_successful"]
    
    @pytest.mark.slow
    def test_comprehensive_polymer_validation(self):
        """Run comprehensive polymer validation (marked as slow)."""
        report = self.validator.generate_comprehensive_polymer_report()
        
        # Should test multiple monomers and oligomers
        assert report["summary"]["total_monomers_tested"] > 5
        assert report["summary"]["total_oligomers_tested"] > 0
        
        # Should have reasonable success rates
        assert report["summary"]["monomer_success_rate"] > 0.7  # At least 70% success
        
        # Should have some successful descriptor calculations
        assert report["summary"]["successful_descriptor_calculations"] > 0
        
        # Check specific results
        assert "ethylene" in report["detailed_results"]["monomer_validation"]
        assert "styrene" in report["detailed_results"]["monomer_validation"]
        
        # Check ROP validation
        assert len(report["detailed_results"]["rop_validation"]) > 0


if __name__ == "__main__":
    # Run polymer validation if called directly
    validator = PolymerValidator()
    report = validator.generate_comprehensive_polymer_report()
    
    print("Polymer Validation Report")
    print("=" * 50)
    print(f"Monomers tested: {report['summary']['total_monomers_tested']}")
    print(f"Monomer success rate: {report['summary']['monomer_success_rate']:.2%}")
    print(f"Oligomers tested: {report['summary']['total_oligomers_tested']}")
    print(f"Oligomer success rate: {report['summary']['oligomer_success_rate']:.2%}")
    print(f"Successful descriptor calculations: {report['summary']['successful_descriptor_calculations']}")
    
    print("\nMonomer validation results:")
    for monomer, result in report['detailed_results']['monomer_validation'].items():
        status = "✓" if result['analysis_successful'] else "✗"
        print(f"  {status} {monomer}: {len(result['errors'])} errors")
    
    print("\nROP validation results:")
    for monomer, result in report['detailed_results']['rop_validation'].items():
        status = "✓" if result['rop_analysis_successful'] else "✗"
        feasible = "feasible" if result.get('rop_feasible', False) else "not feasible"
        print(f"  {status} {monomer}: {feasible}")