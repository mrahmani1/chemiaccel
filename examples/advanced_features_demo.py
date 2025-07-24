#!/usr/bin/env python3
"""
ChemiAccel Advanced Features Demo

This script demonstrates the advanced RDKit integration features including:
- Parallel substructure searching with SMARTS pattern matching
- Maximum Common Substructure (MCS) calculation
- Pharmacophore matching and analysis
- Reaction enumeration
- Drug-likeness filters (Lipinski, Veber, PAINS)
"""

import time
from pathlib import Path

from chemi_accel.utils.rdkit_integration import RDKitMolecularParser
from chemi_accel.descriptors.advanced import (
    SubstructureSearcher,
    MaximumCommonSubstructure,
    PharmacophoreAnalyzer,
    ReactionEnumerator,
    DrugLikenessFilters,
    search_substructures,
    calculate_mcs,
    analyze_pharmacophores,
    filter_drug_like_compounds
)

# Sample molecules for demonstration
DEMO_MOLECULES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CCN(CC)CC",  # triethylamine
    "CC(C)C(=O)O",  # isobutyric acid
    "c1ccc2c(c1)cccn2",  # quinoline
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
    "CN1CCC[C@H]1c2cccnc2",  # nicotine
    "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
    "CC1=CC(=O)C=C(C1=O)C",  # 2,5-dimethyl-1,4-benzoquinone
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
]


def demo_substructure_search():
    """Demonstrate parallel substructure searching."""
    print("=" * 60)
    print("SUBSTRUCTURE SEARCH DEMO")
    print("=" * 60)
    
    # Parse molecules
    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(DEMO_MOLECULES, error_handling="skip")
    print(f"Loaded {len(molecules)} molecules")
    
    # Define search patterns
    patterns = {
        "alcohol": "[OH]",
        "aromatic_ring": "c1ccccc1",
        "carbonyl": "[C]=O",
        "amine": "[N]",
        "ester": "[C](=O)[O][C]",
        "amide": "[C](=O)[N]",
        "nitro": "[N+](=O)[O-]",
        "halogen": "[F,Cl,Br,I]"
    }
    
    # Perform search using convenience function
    print(f"\nSearching for {len(patterns)} patterns...")
    start_time = time.time()
    results = search_substructures(molecules, patterns, n_jobs=2)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Results shape: {results.shape}")
    
    # Display results summary
    print("\nPattern occurrence summary:")
    for pattern_name in patterns.keys():
        count_col = f"{pattern_name}_count"
        present_col = f"{pattern_name}_present"
        
        if count_col in results.columns:
            total_matches = results[count_col].sum()
            molecules_with_pattern = results[present_col].sum()
            print(f"  {pattern_name:15}: {total_matches:3d} matches in {molecules_with_pattern:2d} molecules")
    
    # Show specific examples
    print("\nExamples:")
    for _, row in results.head(3).iterrows():
        mol_id = row['mol_id']
        smiles = row['smiles']
        total = row['total_matches']
        print(f"  {mol_id}: {smiles} -> {total} total matches")
    
    return results


def demo_mcs_calculation():
    """Demonstrate Maximum Common Substructure calculation."""
    print("\n" + "=" * 60)
    print("MAXIMUM COMMON SUBSTRUCTURE (MCS) DEMO")
    print("=" * 60)
    
    # Parse molecules
    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(DEMO_MOLECULES[:5], error_handling="skip")
    print(f"Loaded {len(molecules)} molecules for MCS calculation")
    
    # Calculate MCS for all molecules
    print("\nCalculating MCS for all molecules...")
    start_time = time.time()
    mcs_result = calculate_mcs(molecules, timeout=10)
    mcs_time = time.time() - start_time
    
    print(f"MCS calculation completed in {mcs_time:.4f} seconds")
    print(f"MCS SMARTS: {mcs_result['mcs_smarts']}")
    print(f"MCS atoms: {mcs_result['num_atoms']}")
    print(f"MCS bonds: {mcs_result['num_bonds']}")
    print(f"Valid MCS: {mcs_result['valid_mcs']}")
    
    if 'average_coverage' in mcs_result:
        print(f"Average coverage: {mcs_result['average_coverage']:.2%}")
    
    # Calculate pairwise MCS for smaller set
    print("\nCalculating pairwise MCS...")
    mcs_calc = MaximumCommonSubstructure(timeout=5)
    pairwise_results = mcs_calc.calculate_pairwise_mcs(molecules[:3])
    
    print(f"Pairwise MCS results: {len(pairwise_results)} pairs")
    if not pairwise_results.empty:
        print("\nTop pairwise MCS results:")
        for _, row in pairwise_results.head(3).iterrows():
            mol1 = row['smiles_1']
            mol2 = row['smiles_2']
            atoms = row['mcs_atoms']
            print(f"  {mol1} vs {mol2}: {atoms} atoms")
    
    return mcs_result


def demo_pharmacophore_analysis():
    """Demonstrate pharmacophore analysis."""
    print("\n" + "=" * 60)
    print("PHARMACOPHORE ANALYSIS DEMO")
    print("=" * 60)
    
    # Parse molecules
    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(DEMO_MOLECULES, error_handling="skip")
    print(f"Loaded {len(molecules)} molecules for pharmacophore analysis")
    
    # Analyze pharmacophore features
    print("\nAnalyzing pharmacophore features...")
    start_time = time.time()
    results = analyze_pharmacophores(molecules)
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.4f} seconds")
    
    if not results.empty:
        print(f"Results shape: {results.shape}")
        
        # Show feature summary
        feature_cols = [col for col in results.columns if col.endswith('_count') or col == 'total_features']
        if feature_cols:
            print("\nPharmacophore feature summary:")
            for col in feature_cols:
                if col in results.columns:
                    total = results[col].sum()
                    avg = results[col].mean()
                    print(f"  {col:20}: total={total:3d}, avg={avg:.1f}")
        
        # Show examples
        print("\nExamples:")
        for _, row in results.head(3).iterrows():
            mol_id = row['mol_id']
            smiles = row['smiles']
            total = row.get('total_features', 0)
            print(f"  {mol_id}: {smiles} -> {total} features")
    else:
        print("No pharmacophore results (feature extraction may not be available)")
    
    return results


def demo_reaction_enumeration():
    """Demonstrate reaction enumeration."""
    print("\n" + "=" * 60)
    print("REACTION ENUMERATION DEMO")
    print("=" * 60)
    
    # Parse molecules with reactive groups
    reactive_smiles = [
        "CC(=O)O",  # acetic acid (carboxylic acid)
        "CCN",  # ethylamine (primary amine)
        "CCO",  # ethanol (alcohol)
        "c1ccc(Br)cc1",  # bromobenzene (aryl halide)
    ]
    
    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(reactive_smiles, error_handling="skip")
    print(f"Loaded {len(molecules)} reactive molecules")
    
    # Define simple reaction templates
    reaction_templates = {
        "amide_formation": "[C:1](=[O:2])[OH:3].[N:4]>>[C:1](=[O:2])[N:4]",
        "ester_formation": "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5]",
    }
    
    # Enumerate reactions
    print(f"\nEnumerating reactions with {len(reaction_templates)} templates...")
    enumerator = ReactionEnumerator(n_jobs=1)
    
    start_time = time.time()
    products = enumerator.enumerate_reactions(
        molecules, 
        reaction_templates=reaction_templates,
        max_products_per_reaction=3
    )
    enum_time = time.time() - start_time
    
    print(f"Reaction enumeration completed in {enum_time:.4f} seconds")
    print(f"Generated {len(products)} products")
    
    if not products.empty:
        print("\nReaction products:")
        for _, row in products.head(5).iterrows():
            reactant = row['reactant_smiles']
            product = row['product_smiles']
            reaction = row['reaction_name']
            print(f"  {reaction}: {reactant} -> {product}")
    
    return products


def demo_drug_likeness_filters():
    """Demonstrate drug-likeness filtering."""
    print("\n" + "=" * 60)
    print("DRUG-LIKENESS FILTERING DEMO")
    print("=" * 60)
    
    # Parse molecules
    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(DEMO_MOLECULES, error_handling="skip")
    print(f"Loaded {len(molecules)} molecules for drug-likeness assessment")
    
    # Apply drug-likeness filters
    print("\nApplying drug-likeness filters...")
    start_time = time.time()
    results = filter_drug_like_compounds(
        molecules,
        min_score=0.6,
        require_lipinski=True,
        require_veber=True,
        exclude_pains=False  # Skip PAINS for demo (may not be available)
    )
    filter_time = time.time() - start_time
    
    print(f"Filtering completed in {filter_time:.4f} seconds")
    print(f"Results shape: {results.shape}")
    
    if not results.empty:
        # Summary statistics
        passed = results['passes_drug_likeness_filter'].sum()
        total = len(results)
        avg_score = results['drug_likeness_score'].mean()
        
        print(f"\nDrug-likeness summary:")
        print(f"  Molecules passed: {passed}/{total} ({passed/total:.1%})")
        print(f"  Average score: {avg_score:.3f}")
        
        # Lipinski violations
        lipinski_violations = results['lipinski_violations'].value_counts().sort_index()
        print(f"\nLipinski violations distribution:")
        for violations, count in lipinski_violations.items():
            print(f"  {violations} violations: {count} molecules")
        
        # Show examples
        print(f"\nTop drug-like molecules:")
        top_molecules = results.nlargest(3, 'drug_likeness_score')
        for _, row in top_molecules.iterrows():
            mol_id = row['mol_id']
            smiles = row['smiles']
            score = row['drug_likeness_score']
            mw = row['molecular_weight']
            logp = row['logp']
            print(f"  {mol_id}: {smiles}")
            print(f"    Score: {score:.3f}, MW: {mw:.1f}, LogP: {logp:.2f}")
    
    return results


def main():
    """Run all advanced features demos."""
    print("ChemiAccel Advanced Features Demo")
    print("=" * 60)
    print(f"Demonstrating advanced cheminformatics capabilities")
    print(f"Dataset: {len(DEMO_MOLECULES)} diverse molecules")
    
    try:
        # Run all demos
        substructure_results = demo_substructure_search()
        mcs_results = demo_mcs_calculation()
        pharmacophore_results = demo_pharmacophore_analysis()
        reaction_results = demo_reaction_enumeration()
        drug_likeness_results = demo_drug_likeness_filters()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("All advanced features demonstrated successfully!")
        print("\nFeatures tested:")
        print("  ✓ Parallel substructure searching")
        print("  ✓ Maximum Common Substructure (MCS) calculation")
        print("  ✓ Pharmacophore analysis")
        print("  ✓ Reaction enumeration")
        print("  ✓ Drug-likeness filtering")
        
        return True
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)