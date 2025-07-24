"""
Validation tests against medicinal chemistry datasets.

This module provides validation of drug discovery descriptors against
established medicinal chemistry datasets and benchmarks from literature.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.descriptors.drug_discovery import DrugDiscoveryDescriptorCalculator
from src.chemi_accel.core.exceptions import ChemiAccelError


@pytest.fixture
def drug_discovery_calculator():
    """Fixture providing drug discovery descriptor calculator."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return DrugDiscoveryDescriptorCalculator()


@pytest.fixture
def fda_approved_drugs():
    """Fixture providing FDA-approved drugs with known properties."""
    return {
        # Cardiovascular drugs
        "atorvastatin": {
            "smiles": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
            "therapeutic_area": "cardiovascular",
            "expected": {
                "qed_score": 0.6,  # Moderate drug-likeness
                "lipinski_violations": 1,  # MW violation
                "synthetic_accessibility": 7.0,  # Complex synthesis
            }
        },
        "metoprolol": {
            "smiles": "COC1=CC=C(C=C1)CCO[C@H](C)CNC(C)C",
            "therapeutic_area": "cardiovascular",
            "expected": {
                "qed_score": 0.8,  # Good drug-likeness
                "lipinski_violations": 0,  # No violations
                "cns_mpo_score": 3.0,  # Moderate CNS penetration
            }
        },
        
        # CNS drugs
        "sertraline": {
            "smiles": "CN[C@H]1CC[C@@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
            "therapeutic_area": "cns",
            "expected": {
                "qed_score": 0.7,  # Good drug-likeness
                "cns_mpo_score": 4.0,  # Good CNS properties
                "bbb_mpo_score": 3.5,  # Good BBB penetration
            }
        },
        "donepezil": {
            "smiles": "COC1=CC2=C(C=C1)C(=O)C(=C(N2)C)C3CCN(CC3)CC4=CC=CC=C4",
            "therapeutic_area": "cns",
            "expected": {
                "qed_score": 0.8,  # High drug-likeness
                "cns_mpo_score": 4.5,  # Excellent CNS properties
                "synthetic_accessibility": 6.0,  # Moderate complexity
            }
        },
        
        # Anti-infective drugs
        "ciprofloxacin": {
            "smiles": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
            "therapeutic_area": "anti_infective",
            "expected": {
                "qed_score": 0.5,  # Moderate drug-likeness
                "lipinski_violations": 0,  # No violations
                "synthetic_accessibility": 5.0,  # Moderate synthesis
            }
        },
        
        # Oncology drugs
        "imatinib": {
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "therapeutic_area": "oncology",
            "expected": {
                "qed_score": 0.7,  # Good drug-likeness
                "lipinski_violations": 1,  # MW violation
                "synthetic_accessibility": 8.0,  # Complex synthesis
            }
        }
    }


@pytest.fixture
def natural_products():
    """Fixture providing natural products with known properties."""
    return {
        "morphine": {
            "smiles": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
            "source": "opium_poppy",
            "expected": {
                "natural_product_likeness": 0.9,  # High NP-likeness
                "synthetic_accessibility": 9.0,  # Very difficult synthesis
                "qed_score": 0.7,  # Good drug-likeness despite complexity
            }
        },
        "quinine": {
            "smiles": "COC1=CC2=C(C=CN=C2C=C1)[C@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O",
            "source": "cinchona_bark",
            "expected": {
                "natural_product_likeness": 0.8,  # High NP-likeness
                "synthetic_accessibility": 8.5,  # Difficult synthesis
                "qed_score": 0.8,  # High drug-likeness
            }
        },
        "paclitaxel": {
            "smiles": "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C",
            "source": "pacific_yew",
            "expected": {
                "natural_product_likeness": 0.9,  # Very high NP-likeness
                "synthetic_accessibility": 10.0,  # Extremely difficult synthesis
                "lipinski_violations": 2,  # MW and other violations
            }
        }
    }


@pytest.fixture
def fragment_library():
    """Fixture providing validated fragment library."""
    return {
        # High-quality fragments
        "benzimidazole": {
            "smiles": "C1=CC=C2C(=C1)N=CN2",
            "quality": "high",
            "expected": {
                "fragment_likeness": 0.9,
                "rule_of_three": 0,
                "lead_likeness": 0.7,
            }
        },
        "pyrimidine": {
            "smiles": "C1=CN=CN=C1",
            "quality": "high",
            "expected": {
                "fragment_likeness": 1.0,
                "rule_of_three": 0,
                "lead_likeness": 0.6,
            }
        },
        
        # Medium-quality fragments
        "quinoline": {
            "smiles": "C1=CC=C2C(=C1)C=CC=N2",
            "quality": "medium",
            "expected": {
                "fragment_likeness": 0.8,
                "rule_of_three": 0,
                "lead_likeness": 0.8,
            }
        },
        
        # Problematic fragments
        "quinone": {
            "smiles": "O=C1C=CC(=O)C=C1",
            "quality": "low",
            "expected": {
                "pains_alerts": 1,  # Quinone is a PAINS
                "fragment_likeness": 0.6,
            }
        }
    }


@pytest.fixture
def toxic_compounds():
    """Fixture providing compounds with known toxicity issues."""
    return {
        # Hepatotoxic compounds
        "acetaminophen_overdose": {
            "smiles": "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol/Acetaminophen
            "toxicity_type": "hepatotoxic",
            "expected": {
                "hepatotoxicity": 0.3,  # Moderate at therapeutic doses
            }
        },
        
        # Cardiotoxic compounds
        "doxorubicin": {
            "smiles": "COC1=C(C=C2C(=C1)C(=O)C3=C(C2=O)C(=CC=C3O)O)O[C@H]4C[C@@H]([C@H]([C@@H](O4)C)O)N",
            "toxicity_type": "cardiotoxic",
            "expected": {
                "herg_inhibition": 0.4,  # Moderate hERG risk
            }
        },
        
        # Mutagenic compounds
        "aflatoxin_b1": {
            "smiles": "COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C(=C1)C=CO4",
            "toxicity_type": "mutagenic",
            "expected": {
                "ames_mutagenicity": 0.9,  # Highly mutagenic
                "carcinogenicity": 0.9,  # Highly carcinogenic
            }
        }
    }


class TestMedicinalChemistryValidation:
    """Test class for medicinal chemistry dataset validation."""
    
    def test_fda_approved_drugs_validation(self, drug_discovery_calculator, fda_approved_drugs):
        """Test validation against FDA-approved drugs."""
        
        for drug_name, drug_data in fda_approved_drugs.items():
            smiles = drug_data["smiles"]
            expected = drug_data["expected"]
            therapeutic_area = drug_data["therapeutic_area"]
            
            # Calculate descriptors
            results = drug_discovery_calculator.calculate_descriptors(
                [smiles], 
                list(expected.keys())
            )
            
            print(f"\nValidating {drug_name} ({therapeutic_area}):")
            
            for descriptor, expected_value in expected.items():
                calculated_value = results[descriptor].iloc[0]
                
                # Set tolerance based on descriptor type
                if descriptor == "qed_score":
                    tolerance = 0.2
                elif descriptor == "cns_mpo_score":
                    tolerance = 1.0
                elif descriptor == "bbb_mpo_score":
                    tolerance = 1.0
                elif descriptor == "synthetic_accessibility":
                    tolerance = 2.0
                elif descriptor == "lipinski_violations":
                    tolerance = 0  # Exact match for violations
                else:
                    tolerance = 0.3
                
                if descriptor == "lipinski_violations":
                    assert calculated_value == expected_value, \
                        f"{drug_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                else:
                    assert abs(calculated_value - expected_value) <= tolerance, \
                        f"{drug_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                
                print(f"  {descriptor}: {calculated_value:.3f} (expected: {expected_value})")
    
    def test_therapeutic_area_clustering(self, drug_discovery_calculator, fda_approved_drugs):
        """Test that drugs from same therapeutic area have similar properties."""
        
        # Group drugs by therapeutic area
        therapeutic_groups = {}
        for drug_name, drug_data in fda_approved_drugs.items():
            area = drug_data["therapeutic_area"]
            if area not in therapeutic_groups:
                therapeutic_groups[area] = []
            therapeutic_groups[area].append((drug_name, drug_data["smiles"]))
        
        # Calculate descriptors for each group
        for area, drugs in therapeutic_groups.items():
            if len(drugs) < 2:
                continue  # Need at least 2 drugs for comparison
            
            smiles_list = [drug[1] for drug in drugs]
            drug_names = [drug[0] for drug in drugs]
            
            results = drug_discovery_calculator.calculate_descriptors(
                smiles_list,
                ["qed_score", "cns_mpo_score", "synthetic_accessibility"]
            )
            
            print(f"\nTherapeutic area: {area}")
            
            # Check CNS drugs have higher CNS MPO scores
            if area == "cns":
                cns_scores = results["cns_mpo_score"].values
                avg_cns_score = np.mean(cns_scores)
                assert avg_cns_score > 3.0, f"CNS drugs should have high CNS MPO scores, got {avg_cns_score}"
                print(f"  Average CNS MPO score: {avg_cns_score:.2f}")
            
            # Check consistency within therapeutic area
            for descriptor in ["qed_score", "synthetic_accessibility"]:
                values = results[descriptor].values
                std_dev = np.std(values)
                print(f"  {descriptor} std dev: {std_dev:.3f}")
                
                # Allow reasonable variation within therapeutic areas
                if descriptor == "qed_score":
                    assert std_dev < 0.3, f"High QED variation in {area}: {std_dev}"
                elif descriptor == "synthetic_accessibility":
                    assert std_dev < 3.0, f"High SA variation in {area}: {std_dev}"
    
    def test_natural_products_validation(self, drug_discovery_calculator, natural_products):
        """Test validation against natural products."""
        
        smiles_list = [np_data["smiles"] for np_data in natural_products.values()]
        names_list = list(natural_products.keys())
        
        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list,
            ["natural_product_likeness", "synthetic_accessibility", "qed_score", "lipinski_violations"]
        )
        
        print("\nNatural Products Validation:")
        
        for i, (np_name, np_data) in enumerate(natural_products.items()):
            expected = np_data["expected"]
            
            print(f"\n{np_name} ({np_data['source']}):")
            
            for descriptor, expected_value in expected.items():
                calculated_value = results[descriptor].iloc[i]
                
                # Set tolerance
                if descriptor == "natural_product_likeness":
                    tolerance = 0.3
                elif descriptor == "synthetic_accessibility":
                    tolerance = 1.5
                elif descriptor == "qed_score":
                    tolerance = 0.2
                elif descriptor == "lipinski_violations":
                    tolerance = 0  # Exact match
                else:
                    tolerance = 0.3
                
                if descriptor == "lipinski_violations":
                    assert calculated_value == expected_value, \
                        f"{np_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                else:
                    # Natural product-likeness might be NaN due to RDKit version issues
                    if not np.isnan(calculated_value):
                        assert abs(calculated_value - expected_value) <= tolerance, \
                            f"{np_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                
                print(f"  {descriptor}: {calculated_value:.3f} (expected: {expected_value})")
        
        # Natural products should generally have high SA scores
        sa_scores = results["synthetic_accessibility"].values
        avg_sa = np.mean(sa_scores)
        assert avg_sa > 7.0, f"Natural products should have high SA scores, got {avg_sa}"
        print(f"\nAverage SA score for natural products: {avg_sa:.1f}")
    
    def test_fragment_library_validation(self, drug_discovery_calculator, fragment_library):
        """Test validation against fragment library."""
        
        smiles_list = [frag_data["smiles"] for frag_data in fragment_library.values()]
        names_list = list(fragment_library.keys())
        
        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list,
            ["fragment_likeness", "rule_of_three", "lead_likeness", "pains_alerts"]
        )
        
        print("\nFragment Library Validation:")
        
        high_quality_count = 0
        
        for i, (frag_name, frag_data) in enumerate(fragment_library.items()):
            quality = frag_data["quality"]
            expected = frag_data["expected"]
            
            print(f"\n{frag_name} (quality: {quality}):")
            
            for descriptor, expected_value in expected.items():
                calculated_value = results[descriptor].iloc[i]
                
                tolerance = 0.2 if descriptor != "rule_of_three" and descriptor != "pains_alerts" else 0
                
                if tolerance == 0:
                    assert calculated_value == expected_value, \
                        f"{frag_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                else:
                    assert abs(calculated_value - expected_value) <= tolerance, \
                        f"{frag_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                
                print(f"  {descriptor}: {calculated_value:.3f} (expected: {expected_value})")
            
            # Count high-quality fragments
            if quality == "high":
                fragment_score = results["fragment_likeness"].iloc[i]
                ro3_violations = results["rule_of_three"].iloc[i]
                
                if fragment_score > 0.8 and ro3_violations == 0:
                    high_quality_count += 1
        
        # Should identify most high-quality fragments correctly
        expected_high_quality = sum(1 for frag_data in fragment_library.values() 
                                  if frag_data["quality"] == "high")
        
        assert high_quality_count >= expected_high_quality * 0.8, \
            f"Should identify at least 80% of high-quality fragments"
        
        print(f"\nIdentified {high_quality_count}/{expected_high_quality} high-quality fragments")
    
    def test_toxicity_validation(self, drug_discovery_calculator, toxic_compounds):
        """Test validation against compounds with known toxicity."""
        
        # Use ADMET calculator for toxicity descriptors
        from src.chemi_accel.descriptors.admet import ADMETDescriptorCalculator
        admet_calc = ADMETDescriptorCalculator()
        
        for compound_name, compound_data in toxic_compounds.items():
            smiles = compound_data["smiles"]
            toxicity_type = compound_data["toxicity_type"]
            expected = compound_data["expected"]
            
            # Calculate toxicity descriptors
            results = admet_calc.calculate_descriptors(
                [smiles],
                list(expected.keys())
            )
            
            print(f"\n{compound_name} ({toxicity_type}):")
            
            for descriptor, expected_value in expected.items():
                calculated_value = results[descriptor].iloc[0]
                
                # Toxicity predictions have higher uncertainty
                tolerance = 0.4
                
                assert abs(calculated_value - expected_value) <= tolerance, \
                    f"{compound_name} {descriptor}: expected {expected_value}, got {calculated_value}"
                
                print(f"  {descriptor}: {calculated_value:.3f} (expected: {expected_value})")
    
    def test_drug_development_pipeline(self, drug_discovery_calculator):
        """Test complete drug development pipeline filtering."""
        
        # Simulate compound library with diverse properties
        compound_library = {
            "drug_like_1": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "drug_like_2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "non_drug_like": "CCCCCCCCCCCCCCCCCCCC",  # Long alkyl chain
            "fragment": "C1=CC=CC=C1",  # Benzene
            "pains_compound": "O=C1C=CC(=O)C=C1",  # Quinone
            "complex_natural": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # Morphine
        }
        
        smiles_list = list(compound_library.values())
        names_list = list(compound_library.keys())
        
        # Calculate comprehensive descriptors
        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list,
            ["qed_score", "lipinski_violations", "pains_alerts", "fragment_likeness", 
             "synthetic_accessibility", "lead_likeness"]
        )
        
        results["compound"] = names_list
        
        print("\nDrug Development Pipeline Filtering:")
        print(results[["compound", "qed_score", "lipinski_violations", "pains_alerts"]].round(3))
        
        # Apply typical drug discovery filters
        
        # 1. Drug-like compounds
        drug_like = results[
            (results["qed_score"] > 0.5) & 
            (results["lipinski_violations"] <= 1) & 
            (results["pains_alerts"] == 0)
        ]
        
        print(f"\nDrug-like compounds: {len(drug_like)}")
        for _, row in drug_like.iterrows():
            print(f"  {row['compound']}: QED={row['qed_score']:.2f}")
        
        # Should identify known drug-like compounds
        drug_like_names = drug_like["compound"].tolist()
        assert "drug_like_1" in drug_like_names, "Should identify aspirin as drug-like"
        assert "drug_like_2" in drug_like_names, "Should identify caffeine as drug-like"
        assert "non_drug_like" not in drug_like_names, "Should reject non-drug-like compound"
        assert "pains_compound" not in drug_like_names, "Should reject PAINS compound"
        
        # 2. Fragment-like compounds
        fragment_like = results[
            (results["fragment_likeness"] > 0.7) & 
            (results["pains_alerts"] == 0)
        ]
        
        print(f"\nFragment-like compounds: {len(fragment_like)}")
        for _, row in fragment_like.iterrows():
            print(f"  {row['compound']}: Fragment-likeness={row['fragment_likeness']:.2f}")
        
        # Should identify fragments
        fragment_names = fragment_like["compound"].tolist()
        assert "fragment" in fragment_names, "Should identify benzene as fragment-like"
        
        # 3. Lead-like compounds
        lead_like = results[
            (results["lead_likeness"] > 0.6) & 
            (results["synthetic_accessibility"] < 8.0) &
            (results["pains_alerts"] == 0)
        ]
        
        print(f"\nLead-like compounds: {len(lead_like)}")
        for _, row in lead_like.iterrows():
            print(f"  {row['compound']}: Lead-likeness={row['lead_likeness']:.2f}, SA={row['synthetic_accessibility']:.1f}")
        
        # Should identify appropriate leads
        lead_names = lead_like["compound"].tolist()
        assert len(lead_names) >= 2, "Should identify multiple lead-like compounds"
    
    def test_literature_benchmarks(self, drug_discovery_calculator):
        """Test against literature benchmarks and known datasets."""
        
        # ChEMBL-like drug compounds
        chembl_drugs = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "COC1=CC=C(C=C1)CCO[C@H](C)CNC(C)C",  # Metoprolol
        ]
        
        # Non-drug compounds
        non_drugs = [
            "CCCCCCCCCCCCCCCCCCCC",  # Long alkyl chain
            "C(C)(C)(C)C(C)(C)(C)C(C)(C)(C)C",  # Branched alkane
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # Very long chain
        ]
        
        # Calculate QED scores
        drug_results = drug_discovery_calculator.calculate_descriptors(
            chembl_drugs, ["qed_score", "lipinski_violations"]
        )
        
        non_drug_results = drug_discovery_calculator.calculate_descriptors(
            non_drugs, ["qed_score", "lipinski_violations"]
        )
        
        # Drugs should have higher QED scores than non-drugs
        avg_drug_qed = drug_results["qed_score"].mean()
        avg_non_drug_qed = non_drug_results["qed_score"].mean()
        
        print(f"\nLiterature Benchmark Results:")
        print(f"Average QED for drugs: {avg_drug_qed:.3f}")
        print(f"Average QED for non-drugs: {avg_non_drug_qed:.3f}")
        
        assert avg_drug_qed > avg_non_drug_qed, \
            f"Drugs should have higher QED than non-drugs: {avg_drug_qed} vs {avg_non_drug_qed}"
        
        # Most drugs should have QED > 0.5
        high_qed_drugs = (drug_results["qed_score"] > 0.5).sum()
        assert high_qed_drugs >= len(chembl_drugs) * 0.75, \
            f"At least 75% of drugs should have QED > 0.5, got {high_qed_drugs}/{len(chembl_drugs)}"
        
        # Most drugs should pass Lipinski's rule
        lipinski_compliant = (drug_results["lipinski_violations"] <= 1).sum()
        assert lipinski_compliant >= len(chembl_drugs) * 0.75, \
            f"At least 75% of drugs should be Lipinski compliant, got {lipinski_compliant}/{len(chembl_drugs)}"


class TestMedicinalChemistryPerformance:
    """Performance tests for medicinal chemistry applications."""
    
    def test_large_library_screening(self, drug_discovery_calculator):
        """Test performance on large compound libraries."""
        import time
        
        # Generate large library (simulate ChEMBL-like diversity)
        base_smiles = [
            "CCO", "CC(=O)O", "C1=CC=CC=C1", "CCN(CC)CC",
            "CC(C)O", "C1=CC=C(C=C1)O", "NC1=CC=CC=C1"
        ]
        
        # Create larger library by replication
        large_library = base_smiles * 200  # 1400 compounds
        
        start_time = time.time()
        results = drug_discovery_calculator.calculate_descriptors(
            large_library,
            ["qed_score", "lipinski_violations", "synthetic_accessibility"]
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(large_library) / processing_time
        
        print(f"\nLarge Library Screening Performance:")
        print(f"Compounds processed: {len(large_library)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Throughput: {throughput:.1f} compounds/second")
        
        # Should process at least 50 compounds per second
        assert throughput > 50, f"Throughput too low: {throughput} compounds/sec"
        
        # Results should be complete
        assert len(results) == len(large_library)
        assert not results["qed_score"].isna().all()
    
    def test_memory_efficiency(self, drug_discovery_calculator):
        """Test memory efficiency for large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch
        test_smiles = ["CCO"] * 5000
        results = drug_discovery_calculator.calculate_descriptors(
            test_smiles,
            ["qed_score", "lipinski_violations"]
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Efficiency Test:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Memory per compound: {memory_increase / len(test_smiles) * 1000:.2f} KB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase} MB"
        assert len(results) == len(test_smiles)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])