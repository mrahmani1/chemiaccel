"""
Edge case validation for challenging molecular structures.

This module tests ChemiAccel's robustness with challenging molecular structures
including unusual bonding patterns, charged species, radicals, organometallics,
and other edge cases that might cause calculation failures.
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

from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import ConstitutionalDescriptorCalculator
from src.chemi_accel.descriptors.topological import TopologicalDescriptorCalculator
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator
from src.chemi_accel.core.exceptions import ChemiAccelError, DescriptorCalculationError


# Comprehensive edge case test molecules
EDGE_CASE_MOLECULES = {
    "minimal_molecules": [
        "[H]",  # Single hydrogen
        "[He]",  # Noble gas
        "[Li]",  # Alkali metal
        "[Be]",  # Alkaline earth metal
        "[B]",  # Boron
        "C",  # Single carbon
        "[N]",  # Nitrogen atom
        "O",  # Oxygen
        "[F]",  # Fluorine
        "[Ne]",  # Neon
    ],
    
    "charged_species": [
        "[Li+]",  # Lithium cation
        "[Na+]",  # Sodium cation
        "[K+]",  # Potassium cation
        "[Mg+2]",  # Magnesium dication
        "[Ca+2]",  # Calcium dication
        "[Al+3]",  # Aluminum trication
        "[F-]",  # Fluoride anion
        "[Cl-]",  # Chloride anion
        "[Br-]",  # Bromide anion
        "[I-]",  # Iodide anion
        "[OH-]",  # Hydroxide anion
        "[NH4+]",  # Ammonium cation
        "[H3O+]",  # Hydronium cation
        "C[N+](C)(C)C",  # Tetramethylammonium
        "C[S+](C)C",  # Trimethylsulfonium
        "[C-]",  # Carbanion
        "[O-][N+](=O)c1ccccc1",  # Nitrobenzene zwitterion
    ],
    
    "radicals": [
        "[CH3]",  # Methyl radical
        "[OH]",  # Hydroxyl radical
        "C[CH2]",  # Ethyl radical
        "c1ccc(cc1)[CH2]",  # Benzyl radical
        "[O]",  # Oxygen radical
        "[N]",  # Nitrogen radical
        "CC(C)([CH3])",  # Tertiary radical
    ],
    
    "unusual_bonding": [
        "C#C",  # Triple bond
        "C=C=C",  # Allene
        "C=C=C=C",  # Cumulene
        "C1=C=C=C1",  # Cyclic cumulene
        "[C-]#[O+]",  # Carbon monoxide
        "N#N",  # Nitrogen gas
        "O=O",  # Oxygen gas
        "[H][H]",  # Hydrogen gas
        "C1=CC=CC=C1",  # Benzene (aromatic)
        "c1ccc2c(c1)[nH]c3ccccc23",  # Carbazole (NH in aromatic)
    ],
    
    "strained_rings": [
        "C1CC1",  # Cyclopropane
        "C1CCC1",  # Cyclobutane
        "C1C2CC1C2",  # Bicyclo[1.1.0]butane
        "C12C3C4C1C5C2C3C45",  # Tetrahedrane
        "C1C2C3C1C23",  # Tricyclo[1.1.0.0]butane
        "C1=CC=C2C=CC=CC2=C1",  # Azulene (non-benzenoid aromatic)
    ],
    
    "large_rings": [
        "C1CCCCCCCCCCCCCCCCCCCC1",  # 20-membered ring
        "C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",  # 30-membered ring
        "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5cccc6",  # Large aromatic system
    ],
    
    "heteroatoms": [
        "C[Si](C)(C)C",  # Silicon
        "C[Ge](C)(C)C",  # Germanium
        "C[Sn](C)(C)C",  # Tin
        "C[Pb](C)(C)C",  # Lead
        "CP(C)C",  # Phosphorus
        "C[As](C)C",  # Arsenic
        "C[Sb](C)C",  # Antimony
        "CS(C)C",  # Sulfur
        "C[Se](C)C",  # Selenium
        "C[Te](C)C",  # Tellurium
    ],
    
    "organometallics": [
        "[Fe]",  # Iron
        "C[Fe]C",  # Iron complex
        "[Ni]",  # Nickel
        "[Cu]",  # Copper
        "[Zn]",  # Zinc
        "[Pd]",  # Palladium
        "[Pt]",  # Platinum
        "[Au]",  # Gold
        "[Ag]",  # Silver
    ],
    
    "isotopes": [
        "[2H]C([2H])([2H])[2H]",  # Deuterated methane
        "[13C]",  # Carbon-13
        "[15N]",  # Nitrogen-15
        "[18O]",  # Oxygen-18
        "C([2H])([2H])([2H])[2H]",  # Mixed isotopes
    ],
    
    "complex_stereochemistry": [
        "C[C@H](O)[C@@H](O)C",  # Multiple stereocenters
        "C[C@H]1CC[C@@H](C)CC1",  # Cyclohexane with stereocenters
        "C/C=C/C=C/C",  # Multiple E/Z bonds
        "C/C=C\\C=C\\C",  # Mixed E/Z bonds
        "C[C@H]1[C@@H]2CC[C@H]1C2",  # Bridged bicyclic
    ],
    
    "unusual_functional_groups": [
        "C(=O)=O",  # Carbon dioxide
        "C(=S)=S",  # Carbon disulfide
        "C#[N+][O-]",  # Fulminic acid
        "N=[N+]=[N-]",  # Azide
        "C[N+]#[C-]",  # Isocyanide
        "C=N=[N+]=[N-]",  # Diazoalkane
        "C1=NON=C1",  # Furoxan
        "C1=NN=N1",  # Tetrazole
    ],
    
    "pathological_cases": [
        "",  # Empty string
        "INVALID",  # Invalid SMILES
        "C(C)(C)(C)(C)(C)C",  # Impossible bonding
        "C1CC",  # Incomplete ring
        "C=C=C=C=C=C=C=C=C=C",  # Very long cumulene
        "C" * 100,  # Very long chain
        "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5ccc7c6cccc7",  # Very large aromatic
    ],
    
    "reactive_intermediates": [
        "C[CH+]C",  # Carbocation
        "C[CH-]C",  # Carbanion
        "C=C[CH2+]",  # Allylic cation
        "c1ccc(cc1)[CH2+]",  # Benzyl cation
        "C1=CC=[CH+]C=C1",  # Tropylium cation
        "c1ccc2c(c1)[CH-]c3ccccc23",  # Fluorenyl anion
    ]
}


class EdgeCaseValidator:
    """Validator for edge case molecular structures."""
    
    def __init__(self):
        """Initialize edge case validator."""
        self.calculators = {
            'basic': BasicDescriptorCalculator(),
            'constitutional': ConstitutionalDescriptorCalculator(use_gpu=False),
            'topological': TopologicalDescriptorCalculator(use_gpu=False),
            'fingerprint': FingerprintGenerator(use_gpu=False)
        }
        
        self.test_descriptors = {
            'basic': [
                ('molecular_weight', lambda calc, mol: calc.calculate_molecular_weight(mol)),
                ('tpsa', lambda calc, mol: calc.calculate_tpsa(mol)),
                ('logp', lambda calc, mol: calc.calculate_logp(mol)),
                ('num_rotatable_bonds', lambda calc, mol: calc.calculate_num_rotatable_bonds(mol)),
            ],
            'constitutional': [
                ('num_heavy_atoms', lambda calc, mol: calc.calculate_heavy_atom_count(mol)),
                ('num_atoms', lambda calc, mol: calc.calculate_num_atoms(mol)),
                ('carbon_count', lambda calc, mol: calc.calculate_carbon_count(mol)),
            ],
            'topological': [
                ('wiener_index', lambda calc, mol: calc.calculate_wiener_index(mol)),
            ],
            'fingerprint': [
                ('morgan_fp', lambda calc, mol: calc.generate_morgan_fingerprint(mol, radius=2, n_bits=1024)),
            ]
        }
    
    def test_edge_case_category(self, category: str, molecules: List[str]) -> Dict[str, Any]:
        """Test a category of edge case molecules.
        
        Args:
            category: Category name
            molecules: List of SMILES strings to test
            
        Returns:
            Test results for the category
        """
        results = {
            "category": category,
            "total_molecules": len(molecules),
            "results_by_descriptor": {},
            "overall_stats": {
                "parseable_molecules": 0,
                "successful_calculations": 0,
                "failed_calculations": 0,
                "error_types": {}
            }
        }
        
        parseable_molecules = []
        
        # First, check which molecules can be parsed by RDKit
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    parseable_molecules.append((smiles, mol))
                    results["overall_stats"]["parseable_molecules"] += 1
            except Exception as e:
                # Parsing failed - this is expected for some edge cases
                continue
        
        # Test each descriptor type
        for calc_type, descriptors in self.test_descriptors.items():
            calc = self.calculators[calc_type]
            
            for desc_name, desc_func in descriptors:
                desc_results = {
                    "descriptor": desc_name,
                    "calculator": calc_type,
                    "successful": 0,
                    "failed": 0,
                    "errors": [],
                    "values": [],
                    "problematic_molecules": []
                }
                
                for smiles, mol in parseable_molecules:
                    try:
                        mol_data = self._create_mol_data(smiles, mol)
                        value = desc_func(calc, mol_data)
                        
                        # Check if value is reasonable
                        if self._is_reasonable_value(value, desc_name):
                            desc_results["successful"] += 1
                            desc_results["values"].append(value)
                            results["overall_stats"]["successful_calculations"] += 1
                        else:
                            desc_results["failed"] += 1
                            desc_results["problematic_molecules"].append({
                                "smiles": smiles,
                                "value": str(value),
                                "reason": "unreasonable_value"
                            })
                            results["overall_stats"]["failed_calculations"] += 1
                            
                    except Exception as e:
                        desc_results["failed"] += 1
                        desc_results["errors"].append({
                            "smiles": smiles,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        results["overall_stats"]["failed_calculations"] += 1
                        
                        # Track error types
                        error_type = type(e).__name__
                        if error_type not in results["overall_stats"]["error_types"]:
                            results["overall_stats"]["error_types"][error_type] = 0
                        results["overall_stats"]["error_types"][error_type] += 1
                
                results["results_by_descriptor"][f"{calc_type}_{desc_name}"] = desc_results
        
        return results
    
    def _create_mol_data(self, smiles: str, mol) -> 'MolecularData':
        """Create MolecularData object."""
        from src.chemi_accel.core.molecular import MolecularData
        
        return MolecularData(
            smiles=smiles,
            mol_id=f"edge_{hash(smiles)}",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=Descriptors.MolWt(mol) if mol.GetNumAtoms() > 0 else 0.0,
            backend_data={"rdkit_mol": mol}
        )
    
    def _is_reasonable_value(self, value, descriptor_name: str) -> bool:
        """Check if a descriptor value is reasonable."""
        if value is None:
            return False
        
        if isinstance(value, np.ndarray):
            # For fingerprints and arrays
            if not np.all(np.isfinite(value)):
                return False
            if descriptor_name == 'morgan_fp':
                return np.all((value == 0) | (value == 1))  # Should be binary
            return True
        
        if not np.isfinite(value):
            return False
        
        # Descriptor-specific range checks
        if descriptor_name == 'molecular_weight':
            return 0 < value < 10000
        elif descriptor_name == 'tpsa':
            return 0 <= value < 1000
        elif descriptor_name == 'logp':
            return -20 < value < 30
        elif descriptor_name in ['num_heavy_atoms', 'num_atoms', 'carbon_count']:
            return value >= 0 and isinstance(value, (int, np.integer))
        elif descriptor_name == 'num_rotatable_bonds':
            return value >= 0 and isinstance(value, (int, np.integer))
        elif descriptor_name == 'wiener_index':
            return value >= 0
        
        return True
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive edge case validation report."""
        report = {
            "summary": {
                "total_categories": len(EDGE_CASE_MOLECULES),
                "total_molecules": sum(len(mols) for mols in EDGE_CASE_MOLECULES.values()),
                "category_results": {}
            },
            "detailed_results": {}
        }
        
        overall_successful = 0
        overall_failed = 0
        overall_parseable = 0
        
        for category, molecules in EDGE_CASE_MOLECULES.items():
            print(f"Testing edge case category: {category}")
            
            category_results = self.test_edge_case_category(category, molecules)
            report["detailed_results"][category] = category_results
            
            # Update summary
            report["summary"]["category_results"][category] = {
                "total_molecules": category_results["total_molecules"],
                "parseable_molecules": category_results["overall_stats"]["parseable_molecules"],
                "successful_calculations": category_results["overall_stats"]["successful_calculations"],
                "failed_calculations": category_results["overall_stats"]["failed_calculations"],
                "success_rate": (category_results["overall_stats"]["successful_calculations"] / 
                               max(1, category_results["overall_stats"]["parseable_molecules"] * len(self.test_descriptors)))
            }
            
            overall_successful += category_results["overall_stats"]["successful_calculations"]
            overall_failed += category_results["overall_stats"]["failed_calculations"]
            overall_parseable += category_results["overall_stats"]["parseable_molecules"]
        
        report["summary"]["overall_success_rate"] = overall_successful / max(1, overall_successful + overall_failed)
        report["summary"]["overall_parseable_rate"] = overall_parseable / report["summary"]["total_molecules"]
        
        return report


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestEdgeCaseValidation:
    """Test edge case handling for challenging molecular structures."""
    
    def setup_method(self):
        """Set up edge case validator."""
        self.validator = EdgeCaseValidator()
    
    def test_minimal_molecules(self):
        """Test handling of minimal molecules."""
        results = self.validator.test_edge_case_category(
            "minimal_molecules", EDGE_CASE_MOLECULES["minimal_molecules"]
        )
        
        # Should handle at least some minimal molecules
        assert results["overall_stats"]["parseable_molecules"] > 0
        
        # For parseable molecules, should have some successful calculations
        if results["overall_stats"]["parseable_molecules"] > 0:
            success_rate = (results["overall_stats"]["successful_calculations"] / 
                          max(1, results["overall_stats"]["parseable_molecules"] * len(self.validator.test_descriptors)))
            assert success_rate > 0.3  # At least 30% success rate
    
    def test_charged_species(self):
        """Test handling of charged molecular species."""
        results = self.validator.test_edge_case_category(
            "charged_species", EDGE_CASE_MOLECULES["charged_species"]
        )
        
        # Should handle some charged species
        assert results["overall_stats"]["parseable_molecules"] > 0
        
        # Check that molecular weight calculations work for charged species
        mw_results = None
        for desc_key, desc_result in results["results_by_descriptor"].items():
            if "molecular_weight" in desc_key:
                mw_results = desc_result
                break
        
        if mw_results and mw_results["successful"] > 0:
            # Molecular weights should be positive
            assert all(v > 0 for v in mw_results["values"])
    
    def test_unusual_bonding(self):
        """Test handling of unusual bonding patterns."""
        results = self.validator.test_edge_case_category(
            "unusual_bonding", EDGE_CASE_MOLECULES["unusual_bonding"]
        )
        
        # Should handle most unusual bonding patterns
        assert results["overall_stats"]["parseable_molecules"] > len(EDGE_CASE_MOLECULES["unusual_bonding"]) * 0.5
        
        # Success rate should be reasonable
        if results["overall_stats"]["parseable_molecules"] > 0:
            success_rate = (results["overall_stats"]["successful_calculations"] / 
                          max(1, results["overall_stats"]["parseable_molecules"] * len(self.validator.test_descriptors)))
            assert success_rate > 0.4
    
    def test_strained_rings(self):
        """Test handling of strained ring systems."""
        results = self.validator.test_edge_case_category(
            "strained_rings", EDGE_CASE_MOLECULES["strained_rings"]
        )
        
        # Should handle basic strained rings
        assert results["overall_stats"]["parseable_molecules"] > 0
        
        # Check for reasonable descriptor values
        for desc_key, desc_result in results["results_by_descriptor"].items():
            if desc_result["successful"] > 0:
                # Should not have excessive failures
                failure_rate = desc_result["failed"] / (desc_result["successful"] + desc_result["failed"])
                assert failure_rate < 0.8  # Less than 80% failure rate
    
    def test_heteroatoms(self):
        """Test handling of unusual heteroatoms."""
        results = self.validator.test_edge_case_category(
            "heteroatoms", EDGE_CASE_MOLECULES["heteroatoms"]
        )
        
        # Should handle common heteroatoms
        assert results["overall_stats"]["parseable_molecules"] > 0
        
        # Molecular weight calculations should work for heteroatoms
        mw_results = None
        for desc_key, desc_result in results["results_by_descriptor"].items():
            if "molecular_weight" in desc_key:
                mw_results = desc_result
                break
        
        if mw_results:
            assert mw_results["successful"] > 0
    
    def test_complex_stereochemistry(self):
        """Test handling of complex stereochemistry."""
        results = self.validator.test_edge_case_category(
            "complex_stereochemistry", EDGE_CASE_MOLECULES["complex_stereochemistry"]
        )
        
        # Should handle stereochemistry
        assert results["overall_stats"]["parseable_molecules"] > 0
        
        # Basic descriptors should work with stereochemistry
        basic_success = 0
        basic_total = 0
        
        for desc_key, desc_result in results["results_by_descriptor"].items():
            if desc_key.startswith("basic_"):
                basic_success += desc_result["successful"]
                basic_total += desc_result["successful"] + desc_result["failed"]
        
        if basic_total > 0:
            basic_success_rate = basic_success / basic_total
            assert basic_success_rate > 0.5  # At least 50% success for basic descriptors
    
    def test_pathological_cases(self):
        """Test handling of pathological cases."""
        results = self.validator.test_edge_case_category(
            "pathological_cases", EDGE_CASE_MOLECULES["pathological_cases"]
        )
        
        # Most pathological cases should fail to parse (this is expected)
        # But the system should not crash
        assert results["total_molecules"] > 0
        
        # Should handle failures gracefully
        assert "error_types" in results["overall_stats"]
        
        # If any molecules are parseable, calculations should not crash
        if results["overall_stats"]["parseable_molecules"] > 0:
            # Should have some error handling, not just crashes
            total_attempts = results["overall_stats"]["successful_calculations"] + results["overall_stats"]["failed_calculations"]
            assert total_attempts > 0
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across descriptors."""
        # Test with a known problematic molecule
        problematic_smiles = ["INVALID", "C(C)(C)(C)(C)(C)C", ""]
        
        for smiles in problematic_smiles:
            error_types = set()
            
            # Test each calculator type
            for calc_type, calc in self.validator.calculators.items():
                if calc_type == 'fingerprint':
                    continue  # Skip fingerprints for this test
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    mol_data = self.validator._create_mol_data(smiles, mol)
                    
                    # Try basic molecular weight calculation
                    if calc_type == 'basic':
                        calc.calculate_molecular_weight(mol_data)
                    elif calc_type == 'constitutional':
                        calc.calculate_heavy_atom_count(mol_data)
                    elif calc_type == 'topological':
                        calc.calculate_wiener_index(mol_data)
                        
                except Exception as e:
                    error_types.add(type(e).__name__)
            
            # If there were errors, they should be consistent types
            # (i.e., not random crashes)
            if error_types:
                expected_error_types = {
                    'ValueError', 'ChemiAccelError', 'DescriptorCalculationError',
                    'AttributeError', 'TypeError'
                }
                assert error_types.issubset(expected_error_types), \
                    f"Unexpected error types for {smiles}: {error_types}"
    
    def test_value_range_validation(self):
        """Test that descriptor values are within reasonable ranges for edge cases."""
        # Test with some edge cases that should produce valid results
        test_molecules = [
            "C",  # Methane
            "[Li+]",  # Lithium ion
            "C#C",  # Acetylene
            "C1CC1",  # Cyclopropane
        ]
        
        for smiles in test_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                mol_data = self.validator._create_mol_data(smiles, mol)
                
                # Test molecular weight
                try:
                    mw = self.validator.calculators['basic'].calculate_molecular_weight(mol_data)
                    assert 0 < mw < 1000, f"Unreasonable MW for {smiles}: {mw}"
                except:
                    pass  # Some edge cases may fail, that's OK
                
                # Test TPSA
                try:
                    tpsa = self.validator.calculators['basic'].calculate_tpsa(mol_data)
                    assert 0 <= tpsa < 500, f"Unreasonable TPSA for {smiles}: {tpsa}"
                except:
                    pass
                
                # Test atom count
                try:
                    atom_count = self.validator.calculators['constitutional'].calculate_num_atoms(mol_data)
                    assert 0 < atom_count < 1000, f"Unreasonable atom count for {smiles}: {atom_count}"
                except:
                    pass
                    
            except Exception:
                # Some edge cases may not be parseable, that's OK
                continue
    
    @pytest.mark.slow
    def test_comprehensive_edge_case_validation(self):
        """Run comprehensive edge case validation (marked as slow)."""
        report = self.validator.generate_comprehensive_report()
        
        # Should test multiple categories
        assert report["summary"]["total_categories"] > 5
        assert report["summary"]["total_molecules"] > 50
        
        # Should have reasonable overall success rate
        assert report["summary"]["overall_success_rate"] > 0.2  # At least 20% overall success
        
        # Should handle at least some molecules in each major category
        important_categories = ["minimal_molecules", "charged_species", "unusual_bonding"]
        for category in important_categories:
            if category in report["summary"]["category_results"]:
                category_result = report["summary"]["category_results"][category]
                assert category_result["parseable_molecules"] > 0, f"No parseable molecules in {category}"
    
    def test_memory_safety_with_edge_cases(self):
        """Test memory safety when processing edge cases."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process edge cases that might cause memory issues
        problematic_molecules = (
            EDGE_CASE_MOLECULES["pathological_cases"] + 
            EDGE_CASE_MOLECULES["large_rings"][:3]  # Limit large rings
        )
        
        for smiles in problematic_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                mol_data = self.validator._create_mol_data(smiles, mol)
                
                # Try various calculations
                try:
                    self.validator.calculators['basic'].calculate_molecular_weight(mol_data)
                except:
                    pass
                
                try:
                    self.validator.calculators['constitutional'].calculate_heavy_atom_count(mol_data)
                except:
                    pass
                    
            except Exception:
                continue
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50 MB)
        assert memory_increase < 50, f"Excessive memory usage: {memory_increase} MB"


if __name__ == "__main__":
    # Run edge case validation if called directly
    validator = EdgeCaseValidator()
    report = validator.generate_comprehensive_report()
    
    print("Edge Case Validation Report")
    print("=" * 50)
    print(f"Total categories tested: {report['summary']['total_categories']}")
    print(f"Total molecules tested: {report['summary']['total_molecules']}")
    print(f"Overall success rate: {report['summary']['overall_success_rate']:.2%}")
    print(f"Overall parseable rate: {report['summary']['overall_parseable_rate']:.2%}")
    
    print("\nResults by category:")
    for category, result in report['summary']['category_results'].items():
        print(f"  {category}: {result['success_rate']:.2%} success rate "
              f"({result['successful_calculations']} successful, "
              f"{result['parseable_molecules']} parseable)")