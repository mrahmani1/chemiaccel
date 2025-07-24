"""
Tests for advanced RDKit integration features.

This module tests the advanced cheminformatics capabilities including
substructure searching, MCS calculation, pharmacophore analysis,
reaction enumeration, and drug-likeness filtering.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict
import time

from chemi_accel.core.molecular import MolecularData
from chemi_accel.utils.rdkit_integration import parse_smiles_batch
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

# Test molecules with known properties
TEST_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CCN(CC)CC",  # triethylamine
    "CC(C)C(=O)O",  # isobutyric acid
    "c1ccc2c(c1)cccn2",  # quinoline
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
    "CN1CCC[C@H]1c2cccnc2",  # nicotine
    "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
    "CC1=CC(=O)C=C(C1=O)C"  # 2,5-dimethyl-1,4-benzoquinone
]


@pytest.fixture
def test_molecules():
    """Create test molecules from SMILES."""
    from chemi_accel.utils.rdkit_integration import RDKitMolecularParser
    parser = RDKitMolecularParser()
    return parser.parse_smiles_batch(TEST_SMILES, error_handling="skip")


class TestSubstructureSearcher:
    """Test substructure searching functionality."""
    
    def test_initialization(self):
        """Test SubstructureSearcher initialization."""
        searcher = SubstructureSearcher(n_jobs=2)
        assert searcher.n_jobs == 2
    
    def test_search_patterns(self, test_molecules):
        """Test pattern searching."""
        searcher = SubstructureSearcher(n_jobs=1)
        
        patterns = {
            "alcohol": "[OH]",
            "aromatic": "c1ccccc1",
            "carbonyl": "[C]=O"
        }
        
        results = searcher.search_patterns(test_molecules, patterns)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(test_molecules)
        assert "alcohol_count" in results.columns
        assert "aromatic_present" in results.columns
        
        # Check specific results
        ethanol_row = results[results["smiles"] == "CCO"].iloc[0]
        assert ethanol_row["alcohol_count"] == 1
        
        benzene_row = results[results["smiles"] == "c1ccccc1"].iloc[0]
        assert benzene_row["aromatic_present"] == True
    
    def test_functional_groups(self, test_molecules):
        """Test functional group detection."""
        searcher = SubstructureSearcher(n_jobs=1)
        results = searcher.search_functional_groups(test_molecules)
        
        assert isinstance(results, pd.DataFrame)
        assert "alcohol_count" in results.columns
        assert "carboxylic_acid_count" in results.columns


class TestMaximumCommonSubstructure:
    """Test MCS calculation functionality."""
    
    def test_initialization(self):
        """Test MCS calculator initialization."""
        mcs_calc = MaximumCommonSubstructure(timeout=30)
        assert mcs_calc.timeout == 30
    
    def test_calculate_mcs(self, test_molecules):
        """Test MCS calculation."""
        mcs_calc = MaximumCommonSubstructure(timeout=10)
        
        # Test with similar molecules
        similar_mols = [mol for mol in test_molecules if "CC" in mol.smiles][:3]
        
        if len(similar_mols) >= 2:
            result = mcs_calc.calculate_mcs(similar_mols)
            
            assert isinstance(result, dict)
            assert "mcs_smarts" in result
            assert "num_atoms" in result
            assert "calculation_time" in result
            assert result["num_molecules"] == len(similar_mols)
    
    def test_pairwise_mcs(self, test_molecules):
        """Test pairwise MCS calculation."""
        mcs_calc = MaximumCommonSubstructure(timeout=5)
        
        # Use first 3 molecules for speed
        small_set = test_molecules[:3]
        results = mcs_calc.calculate_pairwise_mcs(small_set)
        
        assert isinstance(results, pd.DataFrame)
        expected_pairs = len(small_set) * (len(small_set) - 1) // 2
        assert len(results) == expected_pairs


class TestPharmacophoreAnalyzer:
    """Test pharmacophore analysis functionality."""
    
    def test_initialization(self):
        """Test PharmacophoreAnalyzer initialization."""
        analyzer = PharmacophoreAnalyzer()
        assert hasattr(analyzer, 'feature_definitions')
    
    def test_extract_features(self, test_molecules):
        """Test pharmacophore feature extraction."""
        analyzer = PharmacophoreAnalyzer()
        
        # Test with a molecule that has known features
        ethanol_mol = next(mol for mol in test_molecules if mol.smiles == "CCO")
        features = analyzer.extract_pharmacophore_features(ethanol_mol)
        
        assert isinstance(features, dict)
        assert "donor" in features
        assert "acceptor" in features
        
        # Ethanol should have donor and acceptor features
        assert len(features["donor"]) > 0 or len(features["acceptor"]) > 0
    
    def test_pharmacophore_fingerprint(self, test_molecules):
        """Test pharmacophore fingerprint calculation."""
        analyzer = PharmacophoreAnalyzer()
        
        mol = test_molecules[0]
        fingerprint = analyzer.calculate_pharmacophore_fingerprint(mol)
        
        assert isinstance(fingerprint, dict)
        assert "feature_counts" in fingerprint
        assert "total_features" in fingerprint
    
    def test_pattern_matching(self, test_molecules):
        """Test pharmacophore pattern matching."""
        analyzer = PharmacophoreAnalyzer()
        
        # Define a simple pattern
        pattern = {"donor": 1}  # At least 1 donor
        
        results = analyzer.match_pharmacophore_pattern(test_molecules, pattern)
        
        assert isinstance(results, pd.DataFrame)
        assert "matches_pattern" in results.columns
        assert "donor_count" in results.columns


class TestReactionEnumerator:
    """Test reaction enumeration functionality."""
    
    def test_initialization(self):
        """Test ReactionEnumerator initialization."""
        enumerator = ReactionEnumerator(n_jobs=2)
        assert enumerator.n_jobs == 2
        assert len(enumerator.reaction_templates) > 0
    
    def test_enumerate_reactions(self, test_molecules):
        """Test reaction enumeration."""
        enumerator = ReactionEnumerator(n_jobs=1)
        
        # Use a subset for faster testing
        small_set = test_molecules[:3]
        
        # Use only one reaction template for speed
        templates = {"test_reaction": "[OH:1]>>[OH:1]"}  # Identity reaction
        
        results = enumerator.enumerate_reactions(
            small_set, 
            reaction_templates=templates,
            max_products_per_reaction=2
        )
        
        assert isinstance(results, pd.DataFrame)
        # Results might be empty if no molecules match the reaction pattern
    
    def test_filter_novel_products(self):
        """Test novel product filtering."""
        enumerator = ReactionEnumerator(n_jobs=1)
        
        # Create mock products DataFrame
        products_df = pd.DataFrame({
            "product_smiles": ["CCO", "CCC", "CCCC"],
            "reaction_name": ["test"] * 3
        })
        
        known_molecules = {"CCO", "CCC"}
        
        novel = enumerator.filter_novel_products(products_df, known_molecules)
        
        assert len(novel) == 1
        assert novel.iloc[0]["product_smiles"] == "CCCC"


class TestDrugLikenessFilters:
    """Test drug-likeness filtering functionality."""
    
    def test_initialization(self):
        """Test DrugLikenessFilters initialization."""
        filters = DrugLikenessFilters()
        # PAINS might not be available in all RDKit installations
        assert hasattr(filters, 'pains_available')
    
    def test_lipinski_descriptors(self, test_molecules):
        """Test Lipinski descriptor calculation."""
        filters = DrugLikenessFilters()
        
        mol = test_molecules[0]
        descriptors = filters.calculate_lipinski_descriptors(mol)
        
        assert isinstance(descriptors, dict)
        assert "molecular_weight" in descriptors
        assert "logp" in descriptors
        assert "hbd" in descriptors
        assert "hba" in descriptors
        assert "lipinski_violations" in descriptors
        assert "lipinski_compliant" in descriptors
    
    def test_veber_descriptors(self, test_molecules):
        """Test Veber descriptor calculation."""
        filters = DrugLikenessFilters()
        
        mol = test_molecules[0]
        descriptors = filters.calculate_veber_descriptors(mol)
        
        assert isinstance(descriptors, dict)
        assert "tpsa" in descriptors
        assert "rotatable_bonds" in descriptors
        assert "veber_violations" in descriptors
        assert "veber_compliant" in descriptors
    
    def test_pains_filter(self, test_molecules):
        """Test PAINS filter."""
        filters = DrugLikenessFilters()
        
        mol = test_molecules[0]
        result = filters.check_pains_filter(mol)
        
        assert isinstance(result, dict)
        assert "pains_alert" in result
        assert "pains_matches" in result
        assert "num_pains_matches" in result
        assert "pains_available" in result
    
    def test_drug_likeness_score(self, test_molecules):
        """Test comprehensive drug-likeness scoring."""
        filters = DrugLikenessFilters()
        
        mol = test_molecules[0]
        result = filters.calculate_drug_likeness_score(mol)
        
        assert isinstance(result, dict)
        assert "drug_likeness_score" in result
        assert 0 <= result["drug_likeness_score"] <= 1
    
    def test_filter_molecules(self, test_molecules):
        """Test drug-likeness filtering."""
        filters = DrugLikenessFilters()
        
        results = filters.filter_drug_like_molecules(
            test_molecules,
            min_score=0.5,
            require_lipinski=False,
            require_veber=False,
            exclude_pains=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(test_molecules)
        assert "passes_drug_likeness_filter" in results.columns
        assert "filter_reasons" in results.columns


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_search_substructures(self, test_molecules):
        """Test substructure search convenience function."""
        patterns = {"alcohol": "[OH]", "aromatic": "c1ccccc1"}
        
        results = search_substructures(test_molecules, patterns, n_jobs=1)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(test_molecules)
    
    def test_calculate_mcs_convenience(self, test_molecules):
        """Test MCS calculation convenience function."""
        # Use first 2 molecules
        small_set = test_molecules[:2]
        
        result = calculate_mcs(small_set, timeout=5)
        
        assert isinstance(result, dict)
        assert "mcs_smarts" in result
    
    def test_analyze_pharmacophores_convenience(self, test_molecules):
        """Test pharmacophore analysis convenience function."""
        results = analyze_pharmacophores(test_molecules)
        
        assert isinstance(results, pd.DataFrame)
        # Results might be empty if analysis fails, but should be DataFrame
    
    def test_filter_drug_like_convenience(self, test_molecules):
        """Test drug-likeness filtering convenience function."""
        results = filter_drug_like_compounds(
            test_molecules,
            min_score=0.5,
            require_lipinski=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(test_molecules)


class TestPerformanceComparison:
    """Test performance comparisons against RDKit CPU implementations."""
    
    def test_substructure_search_performance(self, test_molecules):
        """Compare substructure search performance."""
        patterns = {
            "alcohol": "[OH]",
            "aromatic": "c1ccccc1",
            "carbonyl": "[C]=O",
            "amine": "[N]"
        }
        
        # Test ChemiAccel implementation
        searcher = SubstructureSearcher(n_jobs=1)
        
        start_time = time.time()
        chemi_results = searcher.search_patterns(test_molecules, patterns, use_multiprocessing=False)
        chemi_time = time.time() - start_time
        
        # Test direct RDKit implementation
        start_time = time.time()
        rdkit_results = []
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol:
                row = {"mol_id": mol_data.mol_id, "smiles": mol_data.smiles}
                for pattern_name, smarts in patterns.items():
                    try:
                        from rdkit import Chem
                        pattern_mol = Chem.MolFromSmarts(smarts)
                        matches = rdkit_mol.GetSubstructMatches(pattern_mol) if pattern_mol else []
                        row[f"{pattern_name}_count"] = len(matches)
                    except:
                        row[f"{pattern_name}_count"] = 0
                rdkit_results.append(row)
        rdkit_df = pd.DataFrame(rdkit_results)
        rdkit_time = time.time() - start_time
        
        # Compare results
        assert len(chemi_results) == len(rdkit_df)
        
        # Log performance comparison
        print(f"ChemiAccel substructure search: {chemi_time:.4f}s")
        print(f"Direct RDKit search: {rdkit_time:.4f}s")
        
        # Verify results are similar (allowing for small differences)
        for pattern_name in patterns.keys():
            chemi_col = f"{pattern_name}_count"
            if chemi_col in chemi_results.columns and chemi_col in rdkit_df.columns:
                chemi_counts = chemi_results[chemi_col].sum()
                rdkit_counts = rdkit_df[chemi_col].sum()
                assert abs(chemi_counts - rdkit_counts) <= 1  # Allow small differences
    
    def test_drug_likeness_performance(self, test_molecules):
        """Compare drug-likeness calculation performance."""
        filters = DrugLikenessFilters()
        
        # Test ChemiAccel batch processing
        start_time = time.time()
        chemi_results = filters.filter_drug_like_molecules(
            test_molecules,
            require_lipinski=False,
            require_veber=False,
            exclude_pains=False
        )
        chemi_time = time.time() - start_time
        
        # Test individual RDKit calculations
        start_time = time.time()
        rdkit_results = []
        for mol_data in test_molecules:
            try:
                lipinski = filters.calculate_lipinski_descriptors(mol_data)
                rdkit_results.append({
                    "mol_id": mol_data.mol_id,
                    "molecular_weight": lipinski["molecular_weight"],
                    "logp": lipinski["logp"]
                })
            except:
                pass
        rdkit_time = time.time() - start_time
        
        print(f"ChemiAccel drug-likeness batch: {chemi_time:.4f}s")
        print(f"Individual RDKit calculations: {rdkit_time:.4f}s")
        
        # ChemiAccel should be comparable or faster for batch processing
        assert len(chemi_results) == len(test_molecules)


if __name__ == "__main__":
    pytest.main([__file__])