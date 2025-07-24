"""
ADMET and Drug Discovery Features Demo

This example demonstrates the use of ChemiAccel's ADMET and drug discovery
descriptors for pharmaceutical research and drug development workflows.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

# Import ChemiAccel modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chemi_accel.descriptors.admet import ADMETDescriptorCalculator
from src.chemi_accel.descriptors.drug_discovery import DrugDiscoveryDescriptorCalculator


def main():
    """Main demonstration function."""
    print("ChemiAccel ADMET and Drug Discovery Features Demo")
    print("=" * 50)
    
    # Initialize calculators
    admet_calc = ADMETDescriptorCalculator()
    drug_disc_calc = DrugDiscoveryDescriptorCalculator()
    
    # Demo 1: ADMET Profile Analysis
    print("\n1. ADMET Profile Analysis")
    print("-" * 30)
    demo_admet_analysis(admet_calc)
    
    # Demo 2: Drug-likeness Assessment
    print("\n2. Drug-likeness Assessment")
    print("-" * 30)
    demo_drug_likeness(drug_disc_calc)
    
    # Demo 3: Fragment Library Analysis
    print("\n3. Fragment Library Analysis")
    print("-" * 30)
    demo_fragment_analysis(drug_disc_calc)
    
    # Demo 4: Toxicity Screening
    print("\n4. Toxicity Screening")
    print("-" * 30)
    demo_toxicity_screening(admet_calc)
    
    # Demo 5: Lead Optimization Workflow
    print("\n5. Lead Optimization Workflow")
    print("-" * 30)
    demo_lead_optimization(admet_calc, drug_disc_calc)
    
    # Demo 6: Natural Product Analysis
    print("\n6. Natural Product Analysis")
    print("-" * 30)
    demo_natural_product_analysis(drug_disc_calc)
    
    print("\nDemo completed successfully!")


def demo_admet_analysis(admet_calc: ADMETDescriptorCalculator):
    """Demonstrate ADMET profile analysis for drug candidates."""
    
    # Drug candidates with different ADMET profiles
    drug_candidates = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"
    }
    
    smiles_list = list(drug_candidates.values())
    names_list = list(drug_candidates.keys())
    
    # Calculate comprehensive ADMET profile
    admet_descriptors = [
        "caco2_permeability", "hia_prediction", "bbb_permeability",
        "plasma_protein_binding", "cyp3a4_inhibition", "renal_clearance",
        "herg_inhibition", "hepatotoxicity"
    ]
    
    results = admet_calc.calculate_descriptors(smiles_list, admet_descriptors)
    results["compound"] = names_list
    
    print("ADMET Profiles:")
    print(results[["compound"] + admet_descriptors].round(3))
    
    # Analyze absorption properties
    print("\nAbsorption Analysis:")
    for i, name in enumerate(names_list):
        hia = results["hia_prediction"].iloc[i]
        caco2 = results["caco2_permeability"].iloc[i]
        
        absorption_class = "High" if hia > 0.7 else "Moderate" if hia > 0.4 else "Low"
        permeability_class = "High" if caco2 > -5.5 else "Moderate" if caco2 > -6.5 else "Low"
        
        print(f"  {name}: Absorption={absorption_class}, Permeability={permeability_class}")
    
    # Analyze CNS penetration
    print("\nCNS Penetration Analysis:")
    for i, name in enumerate(names_list):
        bbb = results["bbb_permeability"].iloc[i]
        
        cns_class = "High" if bbb > 0 else "Moderate" if bbb > -1 else "Low"
        print(f"  {name}: CNS penetration={cns_class} (log BB = {bbb:.2f})")
    
    # Safety assessment
    print("\nSafety Assessment:")
    for i, name in enumerate(names_list):
        herg = results["herg_inhibition"].iloc[i]
        hepato = results["hepatotoxicity"].iloc[i]
        
        cardiac_risk = "High" if herg > 0.5 else "Moderate" if herg > 0.3 else "Low"
        liver_risk = "High" if hepato > 0.5 else "Moderate" if hepato > 0.3 else "Low"
        
        print(f"  {name}: Cardiac risk={cardiac_risk}, Liver risk={liver_risk}")


def demo_drug_likeness(drug_disc_calc: DrugDiscoveryDescriptorCalculator):
    """Demonstrate drug-likeness assessment."""
    
    # Compounds with varying drug-likeness
    test_compounds = {
        "Drug-like (Aspirin)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Non-drug-like (Large)": "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC2=CC=C(C=C2)O)C(=O)O",
        "Fragment-like": "C1=CC=C(C=C1)O",
        "Natural product": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "PAINS compound": "O=C1C=CC(=O)C=C1"
    }
    
    smiles_list = list(test_compounds.values())
    names_list = list(test_compounds.keys())
    
    # Calculate drug-likeness descriptors
    drug_descriptors = [
        "qed_score", "cns_mpo_score", "synthetic_accessibility",
        "lipinski_violations", "veber_violations", "pains_alerts",
        "lead_likeness", "fragment_likeness", "natural_product_likeness"
    ]
    
    results = drug_disc_calc.calculate_descriptors(smiles_list, drug_descriptors)
    results["compound"] = names_list
    
    print("Drug-likeness Assessment:")
    print(results[["compound", "qed_score", "lipinski_violations", "pains_alerts"]].round(3))
    
    # Detailed analysis
    print("\nDetailed Drug-likeness Analysis:")
    for i, name in enumerate(names_list):
        qed = results["qed_score"].iloc[i]
        lipinski = results["lipinski_violations"].iloc[i]
        pains = results["pains_alerts"].iloc[i]
        sa = results["synthetic_accessibility"].iloc[i]
        
        drug_like = "Yes" if qed > 0.5 and lipinski <= 1 and pains == 0 else "No"
        synthesis = "Easy" if sa < 4 else "Moderate" if sa < 7 else "Difficult"
        
        print(f"  {name}:")
        print(f"    Drug-like: {drug_like} (QED={qed:.2f})")
        print(f"    Lipinski violations: {lipinski}")
        print(f"    PAINS alerts: {pains}")
        print(f"    Synthesis: {synthesis} (SA={sa:.1f})")
    
    # CNS drug assessment
    print("\nCNS Drug Assessment:")
    for i, name in enumerate(names_list):
        cns_mpo = results["cns_mpo_score"].iloc[i]
        
        cns_suitability = "Excellent" if cns_mpo > 4 else "Good" if cns_mpo > 3 else "Poor"
        print(f"  {name}: CNS suitability={cns_suitability} (MPO={cns_mpo:.1f})")


def demo_fragment_analysis(drug_disc_calc: DrugDiscoveryDescriptorCalculator):
    """Demonstrate fragment library analysis."""
    
    # Fragment library
    fragments = {
        "Benzene": "C1=CC=CC=C1",
        "Phenol": "C1=CC=C(C=C1)O",
        "Aniline": "NC1=CC=CC=C1",
        "Pyridine": "C1=CC=NC=C1",
        "Imidazole": "C1=CN=CN1",
        "Thiophene": "C1=CSC=C1",
        "Furan": "C1=COC=C1",
        "Indole": "C1=CC=C2C(=C1)C=CN2"
    }
    
    smiles_list = list(fragments.values())
    names_list = list(fragments.keys())
    
    # Calculate fragment descriptors
    fragment_descriptors = [
        "fragment_likeness", "rule_of_three", "lead_likeness",
        "pains_alerts", "brenk_alerts"
    ]
    
    results = drug_disc_calc.calculate_descriptors(smiles_list, fragment_descriptors)
    results["fragment"] = names_list
    
    print("Fragment Library Analysis:")
    print(results[["fragment"] + fragment_descriptors].round(3))
    
    # Fragment quality assessment
    print("\nFragment Quality Assessment:")
    high_quality_fragments = []
    
    for i, name in enumerate(names_list):
        frag_like = results["fragment_likeness"].iloc[i]
        ro3_violations = results["rule_of_three"].iloc[i]
        pains = results["pains_alerts"].iloc[i]
        brenk = results["brenk_alerts"].iloc[i]
        
        quality = "High" if frag_like > 0.7 and ro3_violations == 0 and pains == 0 else "Medium" if frag_like > 0.5 else "Low"
        
        if quality == "High":
            high_quality_fragments.append(name)
        
        print(f"  {name}: Quality={quality}")
        print(f"    Fragment-likeness: {frag_like:.2f}")
        print(f"    Rule of 3 violations: {ro3_violations}")
        print(f"    Structural alerts: PAINS={pains}, Brenk={brenk}")
    
    print(f"\nHigh-quality fragments identified: {', '.join(high_quality_fragments)}")


def demo_toxicity_screening(admet_calc: ADMETDescriptorCalculator):
    """Demonstrate toxicity screening workflow."""
    
    # Compounds with known toxicity issues
    test_compounds = {
        "Safe compound": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "Hepatotoxic": "C(Cl)(Cl)Cl",  # Chloroform
        "Mutagenic": "NC1=CC=CC=C1",  # Aniline
        "Carcinogenic": "C1=CC=CC=C1",  # Benzene
        "hERG liability": "CCN(CC)CC"  # Triethylamine (simplified)
    }
    
    smiles_list = list(test_compounds.values())
    names_list = list(test_compounds.keys())
    
    # Calculate toxicity descriptors
    toxicity_descriptors = [
        "herg_inhibition", "ames_mutagenicity", "hepatotoxicity",
        "skin_sensitization", "carcinogenicity"
    ]
    
    results = admet_calc.calculate_descriptors(smiles_list, toxicity_descriptors)
    results["compound"] = names_list
    
    print("Toxicity Screening Results:")
    print(results[["compound"] + toxicity_descriptors].round(3))
    
    # Risk assessment
    print("\nToxicity Risk Assessment:")
    for i, name in enumerate(names_list):
        risks = []
        
        if results["herg_inhibition"].iloc[i] > 0.5:
            risks.append("Cardiac (hERG)")
        if results["ames_mutagenicity"].iloc[i] > 0.5:
            risks.append("Mutagenic")
        if results["hepatotoxicity"].iloc[i] > 0.5:
            risks.append("Hepatotoxic")
        if results["skin_sensitization"].iloc[i] > 0.5:
            risks.append("Skin sensitization")
        if results["carcinogenicity"].iloc[i] > 0.5:
            risks.append("Carcinogenic")
        
        risk_level = "High" if len(risks) > 2 else "Medium" if len(risks) > 0 else "Low"
        
        print(f"  {name}: Risk level={risk_level}")
        if risks:
            print(f"    Specific risks: {', '.join(risks)}")
    
    # Safety ranking
    safety_scores = []
    for i in range(len(names_list)):
        # Calculate overall safety score (lower is better)
        safety_score = (
            results["herg_inhibition"].iloc[i] +
            results["ames_mutagenicity"].iloc[i] +
            results["hepatotoxicity"].iloc[i] +
            results["carcinogenicity"].iloc[i]
        ) / 4
        safety_scores.append(safety_score)
    
    # Sort by safety
    safety_ranking = sorted(zip(names_list, safety_scores), key=lambda x: x[1])
    
    print("\nSafety Ranking (safest first):")
    for i, (name, score) in enumerate(safety_ranking, 1):
        print(f"  {i}. {name} (safety score: {score:.2f})")


def demo_lead_optimization(admet_calc: ADMETDescriptorCalculator, 
                          drug_disc_calc: DrugDiscoveryDescriptorCalculator):
    """Demonstrate lead optimization workflow."""
    
    # Lead compound and analogs
    lead_series = {
        "Original lead": "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "Analog 1 (F-sub)": "CC1=CC=C(C=C1)C(=O)NC2=CC=C(F)C=C2",
        "Analog 2 (OH-sub)": "CC1=CC=C(C=C1)C(=O)NC2=CC=C(O)C=C2",
        "Analog 3 (N-methyl)": "CC1=CC=C(C=C1)C(=O)N(C)C2=CC=CC=C2",
        "Analog 4 (extended)": "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)CC"
    }
    
    smiles_list = list(lead_series.values())
    names_list = list(lead_series.keys())
    
    # Calculate optimization-relevant descriptors
    optimization_descriptors = [
        "qed_score", "cns_mpo_score", "synthetic_accessibility",
        "hia_prediction", "bbb_permeability", "cyp3a4_inhibition",
        "herg_inhibition", "lipinski_violations"
    ]
    
    # Calculate ADMET properties
    admet_results = admet_calc.calculate_descriptors(
        smiles_list, 
        ["hia_prediction", "bbb_permeability", "cyp3a4_inhibition", "herg_inhibition"]
    )
    
    # Calculate drug discovery properties
    drug_results = drug_disc_calc.calculate_descriptors(
        smiles_list,
        ["qed_score", "cns_mpo_score", "synthetic_accessibility", "lipinski_violations"]
    )
    
    # Combine results
    combined_results = pd.merge(admet_results, drug_results, on="smiles")
    combined_results["compound"] = names_list
    
    print("Lead Optimization Analysis:")
    print(combined_results[["compound", "qed_score", "cns_mpo_score", "hia_prediction", "herg_inhibition"]].round(3))
    
    # Multi-parameter optimization scoring
    print("\nMulti-Parameter Optimization Scoring:")
    mpo_scores = []
    
    for i, name in enumerate(names_list):
        # Calculate composite score (higher is better)
        qed = combined_results["qed_score"].iloc[i]
        cns_mpo = combined_results["cns_mpo_score"].iloc[i] / 6.0  # Normalize to 0-1
        hia = combined_results["hia_prediction"].iloc[i]
        safety = 1 - combined_results["herg_inhibition"].iloc[i]  # Invert for scoring
        
        mpo_score = (qed + cns_mpo + hia + safety) / 4
        mpo_scores.append(mpo_score)
        
        print(f"  {name}: MPO score = {mpo_score:.3f}")
    
    # Rank compounds
    ranking = sorted(zip(names_list, mpo_scores), key=lambda x: x[1], reverse=True)
    
    print("\nOptimization Ranking (best first):")
    for i, (name, score) in enumerate(ranking, 1):
        print(f"  {i}. {name} (MPO score: {score:.3f})")
    
    # Identify best analog
    best_compound = ranking[0][0]
    print(f"\nRecommended compound for further development: {best_compound}")


def demo_natural_product_analysis(drug_disc_calc: DrugDiscoveryDescriptorCalculator):
    """Demonstrate natural product analysis."""
    
    # Natural products and synthetic drugs
    compounds = {
        "Morphine (NP)": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "Quinine (NP)": "COC1=CC2=C(C=CN=C2C=C1)[C@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O",
        "Taxol (NP)": "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C",
        "Aspirin (Synthetic)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen (Synthetic)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    }
    
    smiles_list = list(compounds.values())
    names_list = list(compounds.keys())
    
    # Calculate natural product descriptors
    np_descriptors = [
        "natural_product_likeness", "synthetic_accessibility", "qed_score",
        "lipinski_violations", "lead_likeness"
    ]
    
    results = drug_disc_calc.calculate_descriptors(smiles_list, np_descriptors)
    results["compound"] = names_list
    
    print("Natural Product Analysis:")
    print(results[["compound"] + np_descriptors].round(3))
    
    # Classify compounds
    print("\nCompound Classification:")
    for i, name in enumerate(names_list):
        np_like = results["natural_product_likeness"].iloc[i]
        sa_score = results["synthetic_accessibility"].iloc[i]
        
        np_class = "High" if np_like > 0.6 else "Medium" if np_like > 0.3 else "Low"
        synthesis = "Easy" if sa_score < 4 else "Moderate" if sa_score < 7 else "Difficult"
        
        print(f"  {name}:")
        print(f"    NP-likeness: {np_class} ({np_like:.2f})")
        print(f"    Synthetic accessibility: {synthesis} ({sa_score:.1f})")
    
    # Compare NP vs synthetic drugs
    np_compounds = [name for name in names_list if "(NP)" in name]
    synthetic_compounds = [name for name in names_list if "(Synthetic)" in name]
    
    np_avg_sa = results[results["compound"].isin(np_compounds)]["synthetic_accessibility"].mean()
    synthetic_avg_sa = results[results["compound"].isin(synthetic_compounds)]["synthetic_accessibility"].mean()
    
    print(f"\nComparison:")
    print(f"  Natural products avg SA score: {np_avg_sa:.1f}")
    print(f"  Synthetic drugs avg SA score: {synthetic_avg_sa:.1f}")
    print(f"  Natural products are {'more' if np_avg_sa > synthetic_avg_sa else 'less'} complex to synthesize")


if __name__ == "__main__":
    main()