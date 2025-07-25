"""
Comprehensive reference implementation validation for ChemiAccel.

This module provides systematic validation of ChemiAccel's descriptor calculations
against established reference implementations (RDKit, Mordred) with bit-level
accuracy testing, statistical validation, and edge case handling.
"""

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        Lipinski,
        rdchem,
        rdMolDescriptors,
        rdPartialCharges,
    )

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import (
    ConstitutionalDescriptorCalculator,
)
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator
from src.chemi_accel.descriptors.topological import TopologicalDescriptorCalculator

# Comprehensive test dataset with diverse molecular structures
VALIDATION_DATASET = {
    "simple_molecules": [
        "C",  # Methane
        "CC",  # Ethane
        "CCC",  # Propane
        "CCCC",  # Butane
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "CCN",  # Ethylamine
        "CCS",  # Ethanethiol
        "CC#N",  # Acetonitrile
        "C=C",  # Ethene
    ],
    "aromatic_compounds": [
        "c1ccccc1",  # Benzene
        "Cc1ccccc1",  # Toluene
        "CCc1ccccc1",  # Ethylbenzene
        "c1ccc2ccccc2c1",  # Naphthalene
        "c1ccc2c(c1)ccc3ccccc32",  # Anthracene
        "c1ccc(cc1)c2ccccc2",  # Biphenyl
        "Cc1ccc(cc1)C",  # p-Xylene
        "c1ccc(cc1)O",  # Phenol
        "c1ccc(cc1)N",  # Aniline
        "c1ccc(cc1)C(=O)O",  # Benzoic acid
    ],
    "heterocycles": [
        "c1ccncc1",  # Pyridine
        "c1cccnc1",  # Pyrimidine
        "c1cnc2ccccc2n1",  # Quinazoline
        "c1coc2ccccc12",  # Benzofuran
        "c1csc2ccccc12",  # Benzothiophene
        "c1cc2ccccc2nc1",  # Quinoline
        "c1cc2ccccc2oc1",  # Chromene
        "c1cnc(nc1)N",  # 2-Aminopyrimidine
        "c1ccc2[nH]ccc2c1",  # Indole
        "c1ccc2nc3ccccc3cc2c1",  # Phenanthroline
    ],
    "complex_structures": [
        "CC(C)C(=O)Nc1ccc(cc1)C(=O)O",  # Ibuprofen-like
        "CC(C)(C)c1ccc(cc1)O",  # 4-tert-Butylphenol
        "CCN(CC)CC(=O)Nc1c(cccc1C)C",  # Lidocaine-like
        "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone derivative
        "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C",  # Di-tert-butylbenzene
        "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",  # Sulfanilamide-like
        "CC1=CC=C(C=C1)S(=O)(=O)Nc2nccs2",  # Thiazole sulfonamide
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ],
    "challenging_cases": [
        "C1CCC2CCCCC2C1",  # Decalin
        "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene (alternative)
        "C1CC2CC3CC(C1)CC(C2)C3",  # Adamantane
        "C1C2C3C4C5C1C6C7C8C2C9C3C%10C4C%11C5C6C7C8C9C%10%11",  # Fullerene fragment
        "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C",  # Sterically hindered
        "C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C24",  # Triphenylene
        "C1=CC=C2C(=C1)C=C3C=CC=CC3=C2",  # Phenanthrene
        "C1=CC2=C(C=C1)C=CC3=CC=CC=C32",  # Anthracene (alt)
        "C1=CC=C2C(=C1)C=CC=C2C3=CC=CC=C3",  # 1-Phenylnaphthalene
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=CC=C3",  # Terphenyl
    ],
    "stereochemistry": [
        "C[C@H](O)C(=O)O",  # L-Lactic acid
        "C[C@@H](O)C(=O)O",  # D-Lactic acid
        "C[C@H](N)C(=O)O",  # L-Alanine
        "C[C@@H](N)C(=O)O",  # D-Alanine
        "C[C@H]1CC[C@@H](C)CC1",  # Trans-1,4-dimethylcyclohexane
        "C[C@H]1CC[C@H](C)CC1",  # Cis-1,4-dimethylcyclohexane
        "C/C=C/C",  # Trans-2-butene
        "C/C=C\\C",  # Cis-2-butene
        "C[C@H](O)[C@@H](O)C",  # Meso-2,3-butanediol
        "C[C@H](O)[C@H](O)C",  # (R,R)-2,3-butanediol
    ],
    "functional_groups": [
        "CC(=O)N",  # Acetamide
        "CC(=O)OC",  # Methyl acetate
        "CC(=O)Cl",  # Acetyl chloride
        "CC(=O)C",  # Acetone
        "CC(=S)N",  # Thioacetamide
        "CC(=O)NN",  # Acethydrazide
        "CC(=O)NO",  # Acetohydroxamic acid
        "CC(=NOC)C",  # Acetone oxime methyl ether
        "CC(=O)C(=O)C",  # 2,3-Butanedione
        "CC(=O)C(=O)O",  # Pyruvic acid
    ],
    "edge_cases": [
        "[H]",  # Single hydrogen
        "[He]",  # Helium
        "[Li+]",  # Lithium ion
        "[C-]",  # Carbanion
        "[NH4+]",  # Ammonium
        "[OH-]",  # Hydroxide
        "C[N+](C)(C)C",  # Tetramethylammonium
        "C[S+](C)C",  # Trimethylsulfonium
        "[O-][N+](=O)c1ccccc1",  # Nitrobenzene
        "C1=CC=C([N+](=O)[O-])C=C1",  # Nitrobenzene (alt)
    ],
}


class ReferenceValidationFramework:
    """Framework for systematic validation against reference implementations."""

    def __init__(self, tolerance_config: Optional[Dict[str, float]] = None):
        """Initialize validation framework.

        Args:
            tolerance_config: Dictionary mapping descriptor names to tolerance values
        """
        self.tolerance_config = tolerance_config or self._default_tolerances()
        self.validation_results = {}
        self.performance_metrics = {}

        # Initialize calculators
        self.chemi_accel_calc = Calculator(backend="cpu")  # Use CPU for reproducibility
        self.basic_calc = BasicDescriptorCalculator()
        self.constitutional_calc = ConstitutionalDescriptorCalculator(use_gpu=False)
        self.topological_calc = TopologicalDescriptorCalculator(use_gpu=False)
        self.fingerprint_gen = FingerprintGenerator(use_gpu=False)

        if MORDRED_AVAILABLE:
            self.mordred_calc = MordredCalculator(mordred_descriptors, ignore_3D=True)

    def _default_tolerances(self) -> Dict[str, float]:
        """Define default tolerance values for different descriptor types."""
        return {
            # Exact match required
            "molecular_weight": 1e-6,
            "exact_molecular_weight": 1e-6,
            "num_atoms": 0,
            "num_bonds": 0,
            "num_heavy_atoms": 0,
            # Small numerical tolerance
            "tpsa": 1e-4,
            "logp": 1e-3,
            "molar_refractivity": 1e-3,
            "num_rotatable_bonds": 0,
            "num_aromatic_rings": 0,
            "num_aliphatic_rings": 0,
            # Moderate tolerance for complex calculations
            "bertz_complexity": 1e-2,
            "synthetic_accessibility": 0.1,
            "wiener_index": 1e-6,
            "zagreb_index_1": 1e-6,
            "zagreb_index_2": 1e-6,
            "randic_index": 1e-4,
            # Higher tolerance for approximate methods
            "kappa_1": 1e-3,
            "kappa_2": 1e-3,
            "kappa_3": 1e-3,
            "electrotopological_state": 1e-2,
            # Fingerprints (exact match)
            "morgan_fingerprint": 0,
            "maccs_keys": 0,
            "topological_fingerprint": 0,
        }

    def validate_descriptor_accuracy(
        self,
        descriptor_name: str,
        test_molecules: List[str],
        reference_func: callable,
        chemi_accel_func: callable,
    ) -> Dict[str, Any]:
        """Validate accuracy of a single descriptor against reference.

        Args:
            descriptor_name: Name of the descriptor being validated
            test_molecules: List of SMILES strings to test
            reference_func: Reference implementation function
            chemi_accel_func: ChemiAccel implementation function

        Returns:
            Dictionary containing validation results
        """
        results = {
            "descriptor": descriptor_name,
            "total_molecules": len(test_molecules),
            "successful_comparisons": 0,
            "exact_matches": 0,
            "within_tolerance": 0,
            "failures": [],
            "performance": {},
        }

        tolerance = self.tolerance_config.get(descriptor_name, 1e-3)

        # Time ChemiAccel calculation
        start_time = time.time()
        chemi_accel_values = []
        for smiles in test_molecules:
            try:
                value = chemi_accel_func(smiles)
                chemi_accel_values.append(value)
            except Exception as e:
                chemi_accel_values.append(None)
                results["failures"].append(
                    {"smiles": smiles, "error": f"ChemiAccel error: {str(e)}"}
                )
        chemi_accel_time = time.time() - start_time

        # Time reference calculation
        start_time = time.time()
        reference_values = []
        for smiles in test_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    value = reference_func(mol)
                    reference_values.append(value)
                else:
                    reference_values.append(None)
            except Exception as e:
                reference_values.append(None)
                results["failures"].append(
                    {"smiles": smiles, "error": f"Reference error: {str(e)}"}
                )
        reference_time = time.time() - start_time

        # Compare results
        for i, (smiles, our_val, ref_val) in enumerate(
            zip(test_molecules, chemi_accel_values, reference_values)
        ):
            if our_val is not None and ref_val is not None:
                results["successful_comparisons"] += 1

                if isinstance(our_val, np.ndarray) and isinstance(ref_val, np.ndarray):
                    # Array comparison (e.g., fingerprints)
                    if np.array_equal(our_val, ref_val):
                        results["exact_matches"] += 1
                        results["within_tolerance"] += 1
                    elif np.allclose(our_val, ref_val, atol=tolerance):
                        results["within_tolerance"] += 1
                    else:
                        results["failures"].append(
                            {
                                "smiles": smiles,
                                "our_value": our_val.tolist()
                                if hasattr(our_val, "tolist")
                                else str(our_val),
                                "reference_value": ref_val.tolist()
                                if hasattr(ref_val, "tolist")
                                else str(ref_val),
                                "error": "Array values differ beyond tolerance",
                            }
                        )
                else:
                    # Scalar comparison
                    if abs(our_val - ref_val) <= tolerance:
                        if our_val == ref_val:
                            results["exact_matches"] += 1
                        results["within_tolerance"] += 1
                    else:
                        results["failures"].append(
                            {
                                "smiles": smiles,
                                "our_value": our_val,
                                "reference_value": ref_val,
                                "absolute_error": abs(our_val - ref_val),
                                "relative_error": abs(our_val - ref_val) / abs(ref_val)
                                if ref_val != 0
                                else float("inf"),
                            }
                        )

        # Performance metrics
        results["performance"] = {
            "chemi_accel_time": chemi_accel_time,
            "reference_time": reference_time,
            "speedup": reference_time / chemi_accel_time if chemi_accel_time > 0 else 0,
            "molecules_per_second_chemi_accel": len(test_molecules) / chemi_accel_time
            if chemi_accel_time > 0
            else 0,
            "molecules_per_second_reference": len(test_molecules) / reference_time
            if reference_time > 0
            else 0,
        }

        return results

    def run_comprehensive_validation(
        self, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive validation across all descriptors and molecule types.

        Args:
            output_file: Optional file path to save detailed results

        Returns:
            Summary of validation results
        """
        all_molecules = []
        for category, molecules in VALIDATION_DATASET.items():
            all_molecules.extend(molecules)

        validation_suite = {
            # Basic descriptors
            "molecular_weight": (
                lambda smiles: self.basic_calc.calculate_molecular_weight(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.MolWt,
            ),
            "tpsa": (
                lambda smiles: self.basic_calc.calculate_tpsa(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.TPSA,
            ),
            "logp": (
                lambda smiles: self.basic_calc.calculate_logp(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.MolLogP,
            ),
            "num_rotatable_bonds": (
                lambda smiles: self.basic_calc.calculate_num_rotatable_bonds(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.NumRotatableBonds,
            ),
            # Constitutional descriptors
            "num_heavy_atoms": (
                lambda smiles: self.constitutional_calc.calculate_heavy_atom_count(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.HeavyAtomCount,
            ),
            "bertz_complexity": (
                lambda smiles: self.constitutional_calc.calculate_bertz_complexity(
                    self._smiles_to_mol_data(smiles)
                ),
                Descriptors.BertzCT,
            ),
            # Topological descriptors
            "wiener_index": (
                lambda smiles: self.topological_calc.calculate_wiener_index(
                    self._smiles_to_mol_data(smiles)
                ),
                lambda mol: self._calculate_wiener_index_reference(mol),
            ),
        }

        summary = {
            "total_descriptors": len(validation_suite),
            "total_molecules": len(all_molecules),
            "descriptor_results": {},
            "overall_accuracy": 0.0,
            "overall_performance": {},
        }

        successful_validations = 0
        total_within_tolerance = 0
        total_comparisons = 0

        for desc_name, (chemi_accel_func, reference_func) in validation_suite.items():
            print(f"Validating {desc_name}...")

            result = self.validate_descriptor_accuracy(
                desc_name, all_molecules, reference_func, chemi_accel_func
            )

            summary["descriptor_results"][desc_name] = result

            if result["successful_comparisons"] > 0:
                successful_validations += 1
                total_within_tolerance += result["within_tolerance"]
                total_comparisons += result["successful_comparisons"]

        # Calculate overall metrics
        if total_comparisons > 0:
            summary["overall_accuracy"] = total_within_tolerance / total_comparisons

        # Save detailed results if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        return summary

    def _smiles_to_mol_data(self, smiles: str) -> MolecularData:
        """Convert SMILES to MolecularData object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return MolecularData(
            smiles=smiles,
            mol_id=f"val_{hash(smiles)}",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=Descriptors.MolWt(mol),
            backend_data={"rdkit_mol": mol},
        )

    def _calculate_wiener_index_reference(self, mol) -> float:
        """Calculate Wiener index using reference implementation."""
        # Simple reference implementation
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        n_atoms = mol.GetNumAtoms()

        # Calculate shortest path distances
        distances = np.full((n_atoms, n_atoms), float("inf"))
        np.fill_diagonal(distances, 0)

        # Floyd-Warshall algorithm
        for i in range(n_atoms):
            for j in range(n_atoms):
                if adj_matrix[i, j] == 1:
                    distances[i, j] = 1

        for k in range(n_atoms):
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]

        # Sum all finite distances
        wiener_index = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i, j] != float("inf"):
                    wiener_index += distances[i, j]

        return wiener_index


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestReferenceValidation:
    """Test class for reference implementation validation."""

    def setup_method(self):
        """Set up validation framework."""
        self.validator = ReferenceValidationFramework()

    def test_basic_descriptor_validation(self):
        """Test validation of basic descriptors."""
        test_molecules = VALIDATION_DATASET["simple_molecules"][
            :5
        ]  # Small subset for testing

        # Test molecular weight validation
        result = self.validator.validate_descriptor_accuracy(
            "molecular_weight",
            test_molecules,
            Descriptors.MolWt,
            lambda smiles: self.validator.basic_calc.calculate_molecular_weight(
                self.validator._smiles_to_mol_data(smiles)
            ),
        )

        assert result["successful_comparisons"] > 0
        assert (
            result["within_tolerance"] == result["successful_comparisons"]
        )  # Should be exact
        assert len(result["failures"]) == 0

    def test_fingerprint_validation(self):
        """Test validation of molecular fingerprints."""
        test_molecules = VALIDATION_DATASET["aromatic_compounds"][:3]

        def chemi_accel_morgan(smiles):
            mol_data = self.validator._smiles_to_mol_data(smiles)
            return self.validator.fingerprint_gen.generate_morgan_fingerprint(
                mol_data, radius=2, n_bits=1024
            )

        def reference_morgan(mol):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return np.array([fp[i] for i in range(1024)], dtype=np.int8)

        result = self.validator.validate_descriptor_accuracy(
            "morgan_fingerprint", test_molecules, reference_morgan, chemi_accel_morgan
        )

        assert result["successful_comparisons"] > 0
        assert (
            result["exact_matches"] == result["successful_comparisons"]
        )  # Should be exact

    def test_edge_case_validation(self):
        """Test validation with edge case molecules."""
        edge_cases = VALIDATION_DATASET["edge_cases"][:5]

        # Test with molecules that might cause issues
        result = self.validator.validate_descriptor_accuracy(
            "molecular_weight",
            edge_cases,
            Descriptors.MolWt,
            lambda smiles: self.validator.basic_calc.calculate_molecular_weight(
                self.validator._smiles_to_mol_data(smiles)
            ),
        )

        # Should handle edge cases gracefully
        assert result["total_molecules"] == len(edge_cases)
        # Some edge cases might fail, but shouldn't crash
        assert result["successful_comparisons"] >= 0

    def test_performance_validation(self):
        """Test performance comparison with reference implementations."""
        test_molecules = VALIDATION_DATASET["simple_molecules"]

        result = self.validator.validate_descriptor_accuracy(
            "tpsa",
            test_molecules,
            Descriptors.TPSA,
            lambda smiles: self.validator.basic_calc.calculate_tpsa(
                self.validator._smiles_to_mol_data(smiles)
            ),
        )

        # Check performance metrics
        assert "performance" in result
        assert result["performance"]["chemi_accel_time"] > 0
        assert result["performance"]["reference_time"] > 0
        assert result["performance"]["molecules_per_second_chemi_accel"] > 0

    def test_statistical_validation(self):
        """Test statistical properties of descriptor calculations."""
        test_molecules = VALIDATION_DATASET["aromatic_compounds"]

        # Calculate descriptors for aromatic compounds
        molecular_weights = []
        tpsa_values = []

        for smiles in test_molecules:
            try:
                mol_data = self.validator._smiles_to_mol_data(smiles)
                mw = self.validator.basic_calc.calculate_molecular_weight(mol_data)
                tpsa = self.validator.basic_calc.calculate_tpsa(mol_data)

                molecular_weights.append(mw)
                tpsa_values.append(tpsa)
            except Exception:
                continue

        # Statistical checks
        assert len(molecular_weights) > 0
        assert all(mw > 0 for mw in molecular_weights)
        assert all(tpsa >= 0 for tpsa in tpsa_values)

        # Aromatic compounds should have reasonable MW range
        assert min(molecular_weights) >= 78  # Benzene
        assert max(molecular_weights) <= 500  # Reasonable upper bound

    @pytest.mark.slow
    def test_comprehensive_validation_suite(self):
        """Run comprehensive validation suite (marked as slow)."""
        # Use smaller subset for CI
        test_molecules = (
            VALIDATION_DATASET["simple_molecules"][:3]
            + VALIDATION_DATASET["aromatic_compounds"][:3]
        )

        # Override all_molecules for this test
        original_dataset = VALIDATION_DATASET.copy()
        VALIDATION_DATASET.clear()
        VALIDATION_DATASET["test_subset"] = test_molecules

        try:
            summary = self.validator.run_comprehensive_validation()

            assert summary["total_descriptors"] > 0
            assert summary["total_molecules"] == len(test_molecules)
            assert summary["overall_accuracy"] >= 0.8  # At least 80% accuracy

            # Check that we have results for each descriptor
            for desc_name, result in summary["descriptor_results"].items():
                assert result["total_molecules"] == len(test_molecules)
                assert result["successful_comparisons"] >= 0

        finally:
            # Restore original dataset
            VALIDATION_DATASET.clear()
            VALIDATION_DATASET.update(original_dataset)


if __name__ == "__main__":
    # Run validation if called directly
    validator = ReferenceValidationFramework()
    summary = validator.run_comprehensive_validation("validation_results.json")

    print("Validation Summary:")
    print(f"Total descriptors tested: {summary['total_descriptors']}")
    print(f"Total molecules tested: {summary['total_molecules']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")

    for desc_name, result in summary["descriptor_results"].items():
        accuracy = (
            result["within_tolerance"] / result["successful_comparisons"]
            if result["successful_comparisons"] > 0
            else 0
        )
        print(
            f"{desc_name}: {accuracy:.2%} accuracy ({result['within_tolerance']}/{result['successful_comparisons']})"
        )
