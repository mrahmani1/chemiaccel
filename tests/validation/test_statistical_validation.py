"""
Statistical validation for large-scale descriptor calculations.

This module provides statistical validation of descriptor calculations across
large datasets, including distribution analysis, correlation testing, and
outlier detection to ensure robust performance at scale.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kstest
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import ConstitutionalDescriptorCalculator
from src.chemi_accel.descriptors.topological import TopologicalDescriptorCalculator
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator


# Large test dataset for statistical validation
LARGE_VALIDATION_DATASET = [
    # Alkanes
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
    "CC(C)C", "CC(C)CC", "CC(C)CCC", "CC(C)(C)C", "CC(C)C(C)C",
    
    # Alkenes and alkynes
    "C=C", "C=CC", "C=CCC", "C=CCCC", "CC=CC", "CC=CCC", "CCC=CCC",
    "C#C", "C#CC", "C#CCC", "CC#CC", "CC#CCC",
    
    # Alcohols
    "CO", "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(O)C", "CCC(O)C",
    "c1ccc(cc1)O", "Cc1ccc(cc1)O", "CCc1ccc(cc1)O",
    
    # Ethers
    "COC", "CCOC", "CCOCC", "CCCOCC", "c1ccc(cc1)OC", "COc1ccccc1",
    
    # Aldehydes and ketones
    "C=O", "CC=O", "CCC=O", "CCCC=O", "CC(=O)C", "CC(=O)CC", "CCC(=O)CCC",
    "c1ccc(cc1)C=O", "CC(=O)c1ccccc1",
    
    # Carboxylic acids and esters
    "C(=O)O", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "c1ccc(cc1)C(=O)O",
    "CC(=O)OC", "CC(=O)OCC", "CCC(=O)OCC", "CC(=O)Oc1ccccc1",
    
    # Amines
    "CN", "CCN", "CCCN", "CC(C)N", "CCN(CC)CC", "c1ccc(cc1)N", "Cc1ccc(cc1)N",
    
    # Amides
    "C(=O)N", "CC(=O)N", "CCC(=O)N", "CC(=O)NC", "CC(=O)NCC",
    
    # Aromatic compounds
    "c1ccccc1", "c1ccc2ccccc2c1", "c1ccc2c(c1)ccc3ccccc32",
    "Cc1ccccc1", "CCc1ccccc1", "Cc1ccc(cc1)C", "c1ccc(cc1)c2ccccc2",
    
    # Heterocycles
    "c1ccncc1", "c1cccnc1", "c1cnc2ccccc2n1", "c1coc2ccccc12",
    "c1csc2ccccc12", "c1cc2ccccc2nc1", "c1ccc2[nH]ccc2c1",
    
    # Halogenated compounds
    "CCl", "CCCl", "CCBr", "CCI", "CF", "CCF", "c1ccc(cc1)Cl", "c1ccc(cc1)F",
    
    # Sulfur compounds
    "CS", "CCS", "CCCS", "c1ccc(cc1)S", "CC(=O)SC", "CSC",
    
    # Phosphorus compounds
    "CP", "CCP", "c1ccc(cc1)P", "CC(=O)PC",
    
    # Complex molecules
    "CC(C)C(=O)Nc1ccc(cc1)C(=O)O",  # Ibuprofen-like
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CCN(CC)CC(=O)Nc1c(cccc1C)C",  # Lidocaine-like
    "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone derivative
]


class StatisticalValidator:
    """Statistical validation framework for descriptor calculations."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize statistical validator.
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
        self.calculators = {
            'basic': BasicDescriptorCalculator(),
            'constitutional': ConstitutionalDescriptorCalculator(use_gpu=False),
            'topological': TopologicalDescriptorCalculator(use_gpu=False),
            'fingerprint': FingerprintGenerator(use_gpu=False)
        }
        
    def calculate_descriptor_statistics(self, smiles_list: List[str], 
                                      descriptor_name: str,
                                      calculator_type: str) -> Dict[str, Any]:
        """Calculate statistical properties of a descriptor across molecules.
        
        Args:
            smiles_list: List of SMILES strings
            descriptor_name: Name of descriptor to calculate
            calculator_type: Type of calculator ('basic', 'constitutional', etc.)
            
        Returns:
            Dictionary with statistical properties
        """
        calculator = self.calculators[calculator_type]
        values = []
        failed_molecules = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_molecules.append(smiles)
                    continue
                
                mol_data = self._create_mol_data(smiles, mol)
                
                # Get descriptor calculation method
                if calculator_type == 'basic':
                    if descriptor_name == 'molecular_weight':
                        value = calculator.calculate_molecular_weight(mol_data)
                    elif descriptor_name == 'tpsa':
                        value = calculator.calculate_tpsa(mol_data)
                    elif descriptor_name == 'logp':
                        value = calculator.calculate_logp(mol_data)
                    elif descriptor_name == 'num_rotatable_bonds':
                        value = calculator.calculate_num_rotatable_bonds(mol_data)
                    else:
                        raise ValueError(f"Unknown basic descriptor: {descriptor_name}")
                        
                elif calculator_type == 'constitutional':
                    if descriptor_name == 'num_heavy_atoms':
                        value = calculator.calculate_heavy_atom_count(mol_data)
                    elif descriptor_name == 'bertz_complexity':
                        value = calculator.calculate_bertz_complexity(mol_data)
                    elif descriptor_name == 'carbon_count':
                        value = calculator.calculate_carbon_count(mol_data)
                    else:
                        raise ValueError(f"Unknown constitutional descriptor: {descriptor_name}")
                        
                elif calculator_type == 'topological':
                    if descriptor_name == 'wiener_index':
                        value = calculator.calculate_wiener_index(mol_data)
                    elif descriptor_name == 'zagreb_index_1':
                        value = calculator.calculate_zagreb_index_1(mol_data)
                    elif descriptor_name == 'randic_index':
                        value = calculator.calculate_randic_index(mol_data)
                    else:
                        raise ValueError(f"Unknown topological descriptor: {descriptor_name}")
                else:
                    raise ValueError(f"Unknown calculator type: {calculator_type}")
                
                if np.isfinite(value):
                    values.append(value)
                else:
                    failed_molecules.append(smiles)
                    
            except Exception as e:
                failed_molecules.append(smiles)
                continue
        
        if not values:
            return {"error": "No valid values calculated"}
        
        values = np.array(values)
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "skewness": stats.skew(values),
            "kurtosis": stats.kurtosis(values),
            "failed_molecules": failed_molecules,
            "success_rate": len(values) / len(smiles_list),
            "values": values.tolist()
        }
    
    def compare_with_reference(self, smiles_list: List[str],
                             descriptor_name: str,
                             chemi_accel_func: callable,
                             reference_func: callable) -> Dict[str, Any]:
        """Compare descriptor distributions with reference implementation.
        
        Args:
            smiles_list: List of SMILES strings
            descriptor_name: Name of descriptor
            chemi_accel_func: ChemiAccel calculation function
            reference_func: Reference calculation function
            
        Returns:
            Statistical comparison results
        """
        chemi_accel_values = []
        reference_values = []
        failed_comparisons = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                mol_data = self._create_mol_data(smiles, mol)
                
                # Calculate with both implementations
                ca_value = chemi_accel_func(mol_data)
                ref_value = reference_func(mol)
                
                if np.isfinite(ca_value) and np.isfinite(ref_value):
                    chemi_accel_values.append(ca_value)
                    reference_values.append(ref_value)
                else:
                    failed_comparisons.append(smiles)
                    
            except Exception as e:
                failed_comparisons.append(smiles)
                continue
        
        if len(chemi_accel_values) < 10:
            return {"error": "Insufficient valid comparisons"}
        
        ca_values = np.array(chemi_accel_values)
        ref_values = np.array(reference_values)
        
        # Statistical tests
        correlation_pearson, p_pearson = pearsonr(ca_values, ref_values)
        correlation_spearman, p_spearman = spearmanr(ca_values, ref_values)
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, ks_p_value = kstest(ca_values, ref_values)
        
        # Calculate errors
        absolute_errors = np.abs(ca_values - ref_values)
        relative_errors = np.abs(ca_values - ref_values) / np.abs(ref_values)
        relative_errors = relative_errors[np.isfinite(relative_errors)]
        
        return {
            "descriptor": descriptor_name,
            "sample_size": len(ca_values),
            "correlation_pearson": correlation_pearson,
            "correlation_p_value_pearson": p_pearson,
            "correlation_spearman": correlation_spearman,
            "correlation_p_value_spearman": p_spearman,
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "mean_absolute_error": np.mean(absolute_errors),
            "max_absolute_error": np.max(absolute_errors),
            "mean_relative_error": np.mean(relative_errors) if len(relative_errors) > 0 else np.nan,
            "max_relative_error": np.max(relative_errors) if len(relative_errors) > 0 else np.nan,
            "rmse": np.sqrt(np.mean((ca_values - ref_values) ** 2)),
            "failed_comparisons": failed_comparisons,
            "success_rate": len(ca_values) / len(smiles_list)
        }
    
    def detect_outliers(self, values: np.ndarray, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in descriptor values.
        
        Args:
            values: Array of descriptor values
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Outlier detection results
        """
        if method == 'iqr':
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = (values < lower_bound) | (values > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outliers = z_scores > 3
            
        elif method == 'modified_zscore':
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return {
            "method": method,
            "num_outliers": np.sum(outliers),
            "outlier_percentage": np.sum(outliers) / len(values) * 100,
            "outlier_indices": np.where(outliers)[0].tolist(),
            "outlier_values": values[outliers].tolist()
        }
    
    def _create_mol_data(self, smiles: str, mol) -> 'MolecularData':
        """Create MolecularData object."""
        from src.chemi_accel.core.molecular import MolecularData
        
        return MolecularData(
            smiles=smiles,
            mol_id=f"stat_val_{hash(smiles)}",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=Descriptors.MolWt(mol),
            backend_data={"rdkit_mol": mol}
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestStatisticalValidation:
    """Test statistical properties of descriptor calculations."""
    
    def setup_method(self):
        """Set up statistical validator."""
        self.validator = StatisticalValidator()
        self.test_dataset = LARGE_VALIDATION_DATASET[:50]  # Use subset for testing
    
    def test_molecular_weight_distribution(self):
        """Test statistical properties of molecular weight calculations."""
        stats = self.validator.calculate_descriptor_statistics(
            self.test_dataset, 'molecular_weight', 'basic'
        )
        
        assert "error" not in stats
        assert stats["count"] > 0
        assert stats["success_rate"] > 0.8  # At least 80% success rate
        
        # Molecular weights should be positive and reasonable
        assert stats["min"] > 0
        assert stats["max"] < 1000  # Reasonable upper bound for test molecules
        assert stats["mean"] > 0
        assert stats["std"] > 0
        
        # Distribution should not be too skewed for diverse molecules
        assert abs(stats["skewness"]) < 5
    
    def test_tpsa_distribution(self):
        """Test statistical properties of TPSA calculations."""
        stats = self.validator.calculate_descriptor_statistics(
            self.test_dataset, 'tpsa', 'basic'
        )
        
        assert "error" not in stats
        assert stats["count"] > 0
        assert stats["success_rate"] > 0.8
        
        # TPSA should be non-negative
        assert stats["min"] >= 0
        assert stats["max"] < 300  # Reasonable upper bound
        assert stats["mean"] >= 0
        
        # Many molecules should have TPSA = 0 (hydrocarbons)
        zero_tpsa_count = sum(1 for v in stats["values"] if v == 0)
        assert zero_tpsa_count > 0
    
    def test_logp_distribution(self):
        """Test statistical properties of LogP calculations."""
        stats = self.validator.calculate_descriptor_statistics(
            self.test_dataset, 'logp', 'basic'
        )
        
        assert "error" not in stats
        assert stats["count"] > 0
        assert stats["success_rate"] > 0.8
        
        # LogP can be negative or positive, but should be reasonable
        assert -5 < stats["min"] < 10
        assert -5 < stats["max"] < 10
        assert -2 < stats["mean"] < 5
    
    def test_heavy_atom_count_distribution(self):
        """Test statistical properties of heavy atom counts."""
        stats = self.validator.calculate_descriptor_statistics(
            self.test_dataset, 'num_heavy_atoms', 'constitutional'
        )
        
        assert "error" not in stats
        assert stats["count"] > 0
        assert stats["success_rate"] > 0.9  # Should be very reliable
        
        # Heavy atom counts should be positive integers
        assert stats["min"] >= 1  # At least one heavy atom
        assert stats["max"] < 100  # Reasonable upper bound
        assert all(isinstance(v, (int, np.integer)) or v.is_integer() for v in stats["values"])
    
    def test_correlation_with_rdkit(self):
        """Test correlation of descriptor values with RDKit reference."""
        # Test molecular weight correlation
        comparison = self.validator.compare_with_reference(
            self.test_dataset,
            'molecular_weight',
            lambda mol_data: self.validator.calculators['basic'].calculate_molecular_weight(mol_data),
            Descriptors.MolWt
        )
        
        assert "error" not in comparison
        assert comparison["sample_size"] > 10
        assert comparison["success_rate"] > 0.8
        
        # Should have perfect correlation with RDKit
        assert comparison["correlation_pearson"] > 0.999
        assert comparison["correlation_p_value_pearson"] < 0.001
        assert comparison["mean_absolute_error"] < 1e-6
        assert comparison["rmse"] < 1e-6
    
    def test_tpsa_correlation_with_rdkit(self):
        """Test TPSA correlation with RDKit reference."""
        comparison = self.validator.compare_with_reference(
            self.test_dataset,
            'tpsa',
            lambda mol_data: self.validator.calculators['basic'].calculate_tpsa(mol_data),
            Descriptors.TPSA
        )
        
        assert "error" not in comparison
        assert comparison["correlation_pearson"] > 0.999
        assert comparison["mean_absolute_error"] < 1e-4
    
    def test_outlier_detection(self):
        """Test outlier detection in descriptor values."""
        # Calculate molecular weights
        values = []
        for smiles in self.test_dataset:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol_data = self.validator._create_mol_data(smiles, mol)
                    mw = self.validator.calculators['basic'].calculate_molecular_weight(mol_data)
                    if np.isfinite(mw):
                        values.append(mw)
            except:
                continue
        
        values = np.array(values)
        
        # Test IQR method
        outliers_iqr = self.validator.detect_outliers(values, 'iqr')
        assert outliers_iqr["outlier_percentage"] < 20  # Less than 20% outliers expected
        
        # Test Z-score method
        outliers_zscore = self.validator.detect_outliers(values, 'zscore')
        assert outliers_zscore["outlier_percentage"] < 10  # Less than 10% outliers expected
    
    def test_descriptor_value_ranges(self):
        """Test that descriptor values fall within expected ranges."""
        descriptors_to_test = [
            ('molecular_weight', 'basic', 1, 1000),
            ('tpsa', 'basic', 0, 300),
            ('logp', 'basic', -10, 15),
            ('num_heavy_atoms', 'constitutional', 1, 100),
        ]
        
        for desc_name, calc_type, min_val, max_val in descriptors_to_test:
            stats = self.validator.calculate_descriptor_statistics(
                self.test_dataset, desc_name, calc_type
            )
            
            if "error" not in stats:
                assert min_val <= stats["min"], f"{desc_name} min value {stats['min']} below expected {min_val}"
                assert stats["max"] <= max_val, f"{desc_name} max value {stats['max']} above expected {max_val}"
                
                # Check individual values
                for value in stats["values"]:
                    assert min_val <= value <= max_val, \
                        f"{desc_name} value {value} outside expected range [{min_val}, {max_val}]"
    
    def test_distribution_normality(self):
        """Test normality of descriptor distributions."""
        # Test molecular weight distribution
        stats = self.validator.calculate_descriptor_statistics(
            self.test_dataset, 'molecular_weight', 'basic'
        )
        
        if "error" not in stats and len(stats["values"]) > 10:
            values = np.array(stats["values"])
            
            # Shapiro-Wilk test for normality (for small samples)
            if len(values) <= 50:
                shapiro_stat, shapiro_p = stats.shapiro(values)
                # Don't require normality, just check that test runs
                assert 0 <= shapiro_stat <= 1
                assert 0 <= shapiro_p <= 1
    
    def test_consistency_across_subsets(self):
        """Test consistency of statistics across different subsets."""
        # Split dataset into two halves
        mid_point = len(self.test_dataset) // 2
        subset1 = self.test_dataset[:mid_point]
        subset2 = self.test_dataset[mid_point:]
        
        # Calculate statistics for both subsets
        stats1 = self.validator.calculate_descriptor_statistics(
            subset1, 'molecular_weight', 'basic'
        )
        stats2 = self.validator.calculate_descriptor_statistics(
            subset2, 'molecular_weight', 'basic'
        )
        
        if "error" not in stats1 and "error" not in stats2:
            # Means should be reasonably close (within 50% of each other)
            mean_ratio = stats1["mean"] / stats2["mean"] if stats2["mean"] != 0 else 1
            assert 0.5 < mean_ratio < 2.0, f"Inconsistent means: {stats1['mean']} vs {stats2['mean']}"
            
            # Standard deviations should be reasonably close
            std_ratio = stats1["std"] / stats2["std"] if stats2["std"] != 0 else 1
            assert 0.2 < std_ratio < 5.0, f"Inconsistent stds: {stats1['std']} vs {stats2['std']}"
    
    def test_performance_statistics(self):
        """Test performance statistics of descriptor calculations."""
        import time
        
        # Time molecular weight calculations
        start_time = time.time()
        
        successful_calculations = 0
        for smiles in self.test_dataset:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol_data = self.validator._create_mol_data(smiles, mol)
                    mw = self.validator.calculators['basic'].calculate_molecular_weight(mol_data)
                    if np.isfinite(mw):
                        successful_calculations += 1
            except:
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance metrics
        molecules_per_second = successful_calculations / total_time if total_time > 0 else 0
        
        assert molecules_per_second > 10, f"Performance too slow: {molecules_per_second} molecules/sec"
        assert successful_calculations > len(self.test_dataset) * 0.8, "Too many failed calculations"
    
    def test_error_rate_analysis(self):
        """Test error rates across different molecule types."""
        molecule_categories = {
            'simple': ['C', 'CC', 'CCC', 'CCO'],
            'aromatic': ['c1ccccc1', 'Cc1ccccc1', 'c1ccc2ccccc2c1'],
            'heterocycles': ['c1ccncc1', 'c1coc2ccccc12'],
            'complex': ['CC(C)C(=O)Nc1ccc(cc1)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
        }
        
        for category, molecules in molecule_categories.items():
            stats = self.validator.calculate_descriptor_statistics(
                molecules, 'molecular_weight', 'basic'
            )
            
            if "error" not in stats:
                # Error rate should be low for all categories
                error_rate = 1 - stats["success_rate"]
                assert error_rate < 0.2, f"High error rate for {category}: {error_rate:.2%}"
                
                # Should have some successful calculations
                assert stats["count"] > 0, f"No successful calculations for {category}"


@pytest.mark.slow
class TestLargeScaleStatisticalValidation:
    """Large-scale statistical validation tests (marked as slow)."""
    
    def setup_method(self):
        """Set up for large-scale testing."""
        self.validator = StatisticalValidator()
        # Use full dataset for large-scale tests
        self.large_dataset = LARGE_VALIDATION_DATASET
    
    def test_large_scale_molecular_weight_statistics(self):
        """Test molecular weight statistics on large dataset."""
        stats = self.validator.calculate_descriptor_statistics(
            self.large_dataset, 'molecular_weight', 'basic'
        )
        
        assert "error" not in stats
        assert stats["count"] > 50  # Should process most molecules
        assert stats["success_rate"] > 0.9  # High success rate expected
        
        # Large dataset should show good statistical properties
        assert stats["std"] > 0  # Should have variation
        assert 10 < stats["mean"] < 200  # Reasonable mean for diverse molecules
    
    def test_large_scale_correlation_analysis(self):
        """Test correlation analysis on large dataset."""
        comparison = self.validator.compare_with_reference(
            self.large_dataset,
            'molecular_weight',
            lambda mol_data: self.validator.calculators['basic'].calculate_molecular_weight(mol_data),
            Descriptors.MolWt
        )
        
        assert "error" not in comparison
        assert comparison["sample_size"] > 50
        assert comparison["correlation_pearson"] > 0.9999  # Very high correlation expected
        assert comparison["mean_absolute_error"] < 1e-5  # Very low error expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])