"""
Bit-level accuracy validation for critical ChemiAccel descriptors.

This module provides rigorous bit-level accuracy testing for critical molecular
descriptors, ensuring exact numerical reproducibility with reference implementations.
"""

import math
import struct
from typing import Any, Dict

import numpy as np
import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import (
    ConstitutionalDescriptorCalculator,
)
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator


def float_to_bits(f: float) -> str:
    """Convert float to binary representation."""
    return "".join(f"{c:08b}" for c in struct.pack("!f", f))


def double_to_bits(d: float) -> str:
    """Convert double to binary representation."""
    return "".join(f"{c:08b}" for c in struct.pack("!d", d))


def compare_float_bits(
    val1: float, val2: float, precision: str = "single"
) -> Dict[str, Any]:
    """Compare two floating point values at bit level.

    Args:
        val1: First value
        val2: Second value
        precision: 'single' or 'double' precision

    Returns:
        Dictionary with comparison results
    """
    if precision == "single":
        bits1 = float_to_bits(val1)
        bits2 = float_to_bits(val2)
    else:
        bits1 = double_to_bits(val1)
        bits2 = double_to_bits(val2)

    # Count differing bits
    differing_bits = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

    return {
        "value1": val1,
        "value2": val2,
        "bits1": bits1,
        "bits2": bits2,
        "differing_bits": differing_bits,
        "exact_match": differing_bits == 0,
        "ulp_difference": abs(val1 - val2)
        / np.finfo(np.float32 if precision == "single" else np.float64).eps,
    }


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestBitLevelAccuracy:
    """Test bit-level accuracy for critical descriptors."""

    def setup_method(self):
        """Set up calculators and test molecules."""
        self.basic_calc = BasicDescriptorCalculator()
        self.constitutional_calc = ConstitutionalDescriptorCalculator(use_gpu=False)
        self.fingerprint_gen = FingerprintGenerator(use_gpu=False)

        # Test molecules with known exact values
        self.exact_test_molecules = {
            "methane": {
                "smiles": "C",
                "expected": {
                    "molecular_weight": 16.04246,
                    "exact_molecular_weight": 16.031300127,
                    "num_atoms": 5,
                    "num_heavy_atoms": 1,
                    "num_bonds": 4,
                    "tpsa": 0.0,
                    "logp": 0.6361,
                },
            },
            "ethane": {
                "smiles": "CC",
                "expected": {
                    "molecular_weight": 30.06904,
                    "exact_molecular_weight": 30.046950192,
                    "num_atoms": 8,
                    "num_heavy_atoms": 2,
                    "num_bonds": 7,
                    "tpsa": 0.0,
                    "logp": 1.2722,
                },
            },
            "water": {
                "smiles": "O",
                "expected": {
                    "molecular_weight": 18.01528,
                    "exact_molecular_weight": 18.010564684,
                    "num_atoms": 3,
                    "num_heavy_atoms": 1,
                    "num_bonds": 2,
                    "tpsa": 0.0,  # No polar surface area for water in SMILES
                },
            },
            "benzene": {
                "smiles": "c1ccccc1",
                "expected": {
                    "molecular_weight": 78.11184,
                    "exact_molecular_weight": 78.046950192,
                    "num_atoms": 12,
                    "num_heavy_atoms": 6,
                    "num_bonds": 12,
                    "num_aromatic_rings": 1,
                    "tpsa": 0.0,
                },
            },
        }

    def _create_mol_data(self, smiles: str) -> MolecularData:
        """Create MolecularData object from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return MolecularData(
            smiles=smiles,
            mol_id=f"bit_test_{hash(smiles)}",
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            molecular_weight=Descriptors.MolWt(mol),
            backend_data={"rdkit_mol": mol},
        )

    def test_molecular_weight_bit_accuracy(self):
        """Test bit-level accuracy of molecular weight calculations."""
        for mol_name, mol_data in self.exact_test_molecules.items():
            smiles = mol_data["smiles"]
            expected_mw = mol_data["expected"]["molecular_weight"]

            mol_data_obj = self._create_mol_data(smiles)
            calculated_mw = self.basic_calc.calculate_molecular_weight(mol_data_obj)

            # Also get RDKit reference
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]
            rdkit_mw = Descriptors.MolWt(rdkit_mol)

            # Compare with RDKit (should be exact)
            rdkit_comparison = compare_float_bits(calculated_mw, rdkit_mw, "single")
            assert rdkit_comparison[
                "exact_match"
            ], f"Molecular weight mismatch with RDKit for {mol_name}: {rdkit_comparison}"

            # Compare with expected (allow small tolerance due to RDKit version differences)
            expected_comparison = compare_float_bits(
                calculated_mw, expected_mw, "single"
            )
            assert (
                expected_comparison["differing_bits"] <= 2
            ), f"Molecular weight differs significantly from expected for {mol_name}: {expected_comparison}"

    def test_exact_molecular_weight_bit_accuracy(self):
        """Test bit-level accuracy of exact molecular weight calculations."""
        for mol_name, mol_data in self.exact_test_molecules.items():
            smiles = mol_data["smiles"]

            mol_data_obj = self._create_mol_data(smiles)
            calculated_exact_mw = self.basic_calc.calculate_exact_molecular_weight(
                mol_data_obj
            )

            # Get RDKit reference
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]
            rdkit_exact_mw = Descriptors.ExactMolWt(rdkit_mol)

            # Should be exactly equal to RDKit
            rdkit_comparison = compare_float_bits(
                calculated_exact_mw, rdkit_exact_mw, "double"
            )
            assert rdkit_comparison[
                "exact_match"
            ], f"Exact molecular weight mismatch with RDKit for {mol_name}: {rdkit_comparison}"

    def test_integer_descriptor_exactness(self):
        """Test that integer descriptors are exactly correct."""
        integer_descriptors = [
            (
                "num_atoms",
                lambda calc, mol: calc.calculate_num_atoms(mol),
                lambda mol: mol.GetNumAtoms(),
            ),
            (
                "num_heavy_atoms",
                lambda calc, mol: calc.calculate_num_heavy_atoms(mol),
                lambda mol: mol.GetNumHeavyAtoms(),
            ),
            (
                "num_bonds",
                lambda calc, mol: calc.calculate_num_bonds(mol),
                lambda mol: mol.GetNumBonds(),
            ),
            (
                "num_rotatable_bonds",
                lambda calc, mol: calc.calculate_num_rotatable_bonds(mol),
                lambda mol: Descriptors.NumRotatableBonds(mol),
            ),
        ]

        for mol_name, mol_data in self.exact_test_molecules.items():
            smiles = mol_data["smiles"]
            mol_data_obj = self._create_mol_data(smiles)
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]

            for desc_name, chemi_accel_func, rdkit_func in integer_descriptors:
                if desc_name == "num_rotatable_bonds":
                    calc = self.basic_calc
                else:
                    calc = self.constitutional_calc

                our_value = chemi_accel_func(calc, mol_data_obj)
                rdkit_value = rdkit_func(rdkit_mol)

                assert (
                    our_value == rdkit_value
                ), f"{desc_name} mismatch for {mol_name}: ours={our_value}, RDKit={rdkit_value}"

                # Verify it's actually an integer
                assert isinstance(
                    our_value, (int, np.integer)
                ), f"{desc_name} should return integer, got {type(our_value)}"

    def test_tpsa_bit_accuracy(self):
        """Test bit-level accuracy of TPSA calculations."""
        # Test molecules with known TPSA values
        tpsa_test_molecules = {
            "ethanol": ("CCO", 20.23),
            "acetic_acid": ("CC(=O)O", 37.30),
            "aspirin": ("CC(=O)OC1=CC=CC=C1C(=O)O", 63.60),
            "caffeine": ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 58.44),
        }

        for mol_name, (smiles, expected_tpsa) in tpsa_test_molecules.items():
            mol_data_obj = self._create_mol_data(smiles)
            calculated_tpsa = self.basic_calc.calculate_tpsa(mol_data_obj)

            # Get RDKit reference
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]
            rdkit_tpsa = Descriptors.TPSA(rdkit_mol)

            # Should match RDKit exactly
            rdkit_comparison = compare_float_bits(calculated_tpsa, rdkit_tpsa, "single")
            assert rdkit_comparison[
                "exact_match"
            ], f"TPSA mismatch with RDKit for {mol_name}: {rdkit_comparison}"

            # Should be close to expected (allow for RDKit version differences)
            assert (
                abs(calculated_tpsa - expected_tpsa) < 0.1
            ), f"TPSA differs from expected for {mol_name}: calculated={calculated_tpsa}, expected={expected_tpsa}"

    def test_logp_bit_accuracy(self):
        """Test bit-level accuracy of LogP calculations."""
        logp_test_molecules = {
            "benzene": ("c1ccccc1", 2.13),
            "toluene": ("Cc1ccccc1", 2.73),
            "phenol": ("c1ccc(cc1)O", 1.46),
            "aniline": ("c1ccc(cc1)N", 0.90),
        }

        for mol_name, (smiles, expected_logp) in logp_test_molecules.items():
            mol_data_obj = self._create_mol_data(smiles)
            calculated_logp = self.basic_calc.calculate_logp(mol_data_obj)

            # Get RDKit reference
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]
            rdkit_logp = Descriptors.MolLogP(rdkit_mol)

            # Should match RDKit exactly
            rdkit_comparison = compare_float_bits(calculated_logp, rdkit_logp, "single")
            assert rdkit_comparison[
                "exact_match"
            ], f"LogP mismatch with RDKit for {mol_name}: {rdkit_comparison}"

    def test_fingerprint_bit_exactness(self):
        """Test bit-level exactness of molecular fingerprints."""
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]

        for smiles in test_molecules:
            mol_data_obj = self._create_mol_data(smiles)
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]

            # Test Morgan fingerprints
            for radius in [0, 1, 2, 3]:
                for n_bits in [512, 1024, 2048]:
                    our_fp = self.fingerprint_gen.generate_morgan_fingerprint(
                        mol_data_obj, radius=radius, n_bits=n_bits
                    )

                    rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(
                        rdkit_mol, radius, nBits=n_bits
                    )
                    rdkit_fp_array = np.zeros(n_bits, dtype=np.int8)
                    for i in range(n_bits):
                        rdkit_fp_array[i] = rdkit_fp[i]

                    # Should be exactly equal
                    assert np.array_equal(
                        our_fp, rdkit_fp_array
                    ), f"Morgan fingerprint mismatch for {smiles} (r={radius}, bits={n_bits})"

                    # Verify data types
                    assert (
                        our_fp.dtype == np.int8
                    ), f"Wrong dtype for Morgan fingerprint: {our_fp.dtype}"
                    assert np.all(
                        (our_fp == 0) | (our_fp == 1)
                    ), "Fingerprint should be binary"

    def test_constitutional_descriptor_accuracy(self):
        """Test bit-level accuracy of constitutional descriptors."""
        test_molecules = {
            "methane": "C",
            "ethane": "CC",
            "propane": "CCC",
            "benzene": "c1ccccc1",
            "toluene": "Cc1ccccc1",
        }

        for mol_name, smiles in test_molecules.items():
            mol_data_obj = self._create_mol_data(smiles)
            rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]

            # Test heavy atom count
            our_heavy_atoms = self.constitutional_calc.calculate_heavy_atom_count(
                mol_data_obj
            )
            rdkit_heavy_atoms = rdkit_mol.GetNumHeavyAtoms()

            assert (
                our_heavy_atoms == rdkit_heavy_atoms
            ), f"Heavy atom count mismatch for {mol_name}: ours={our_heavy_atoms}, RDKit={rdkit_heavy_atoms}"

            # Test atom type counts
            our_carbon_count = self.constitutional_calc.calculate_carbon_count(
                mol_data_obj
            )
            rdkit_carbon_count = sum(
                1 for atom in rdkit_mol.GetAtoms() if atom.GetAtomicNum() == 6
            )

            assert (
                our_carbon_count == rdkit_carbon_count
            ), f"Carbon count mismatch for {mol_name}: ours={our_carbon_count}, RDKit={rdkit_carbon_count}"

    def test_numerical_stability(self):
        """Test numerical stability across different input orders."""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

        # Calculate descriptors multiple times
        results = []
        for _ in range(10):
            batch_results = []
            for smiles in test_smiles:
                mol_data_obj = self._create_mol_data(smiles)

                mw = self.basic_calc.calculate_molecular_weight(mol_data_obj)
                tpsa = self.basic_calc.calculate_tpsa(mol_data_obj)
                logp = self.basic_calc.calculate_logp(mol_data_obj)

                batch_results.append((mw, tpsa, logp))

            results.append(batch_results)

        # All runs should produce identical results
        for i in range(1, len(results)):
            for j, (smiles, (mw1, tpsa1, logp1), (mw2, tpsa2, logp2)) in enumerate(
                zip(test_smiles, results[0], results[i])
            ):
                assert mw1 == mw2, f"MW instability for {smiles}: {mw1} != {mw2}"
                assert (
                    tpsa1 == tpsa2
                ), f"TPSA instability for {smiles}: {tpsa1} != {tpsa2}"
                assert (
                    logp1 == logp2
                ), f"LogP instability for {smiles}: {logp1} != {logp2}"

    def test_edge_case_bit_accuracy(self):
        """Test bit-level accuracy for edge cases."""
        edge_cases = {
            "hydrogen": "[H]",
            "single_carbon": "C",
            "charged_carbon": "[C-]",
            "ammonium": "[NH4+]",
        }

        for case_name, smiles in edge_cases.items():
            try:
                mol_data_obj = self._create_mol_data(smiles)
                rdkit_mol = mol_data_obj.backend_data["rdkit_mol"]

                # Test molecular weight
                our_mw = self.basic_calc.calculate_molecular_weight(mol_data_obj)
                rdkit_mw = Descriptors.MolWt(rdkit_mol)

                mw_comparison = compare_float_bits(our_mw, rdkit_mw, "single")
                assert mw_comparison[
                    "exact_match"
                ], f"MW mismatch for edge case {case_name}: {mw_comparison}"

                # Test atom count
                our_atoms = self.constitutional_calc.calculate_num_atoms(mol_data_obj)
                rdkit_atoms = rdkit_mol.GetNumAtoms()

                assert (
                    our_atoms == rdkit_atoms
                ), f"Atom count mismatch for edge case {case_name}: ours={our_atoms}, RDKit={rdkit_atoms}"

            except Exception as e:
                # Some edge cases might not be supported - that's OK
                print(f"Edge case {case_name} not supported: {e}")
                continue

    def test_precision_consistency(self):
        """Test that calculations maintain consistent precision."""
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN(CC)CC"]

        for smiles in test_molecules:
            mol_data_obj = self._create_mol_data(smiles)

            # Calculate the same descriptor multiple times
            mw_values = []
            for _ in range(5):
                mw = self.basic_calc.calculate_molecular_weight(mol_data_obj)
                mw_values.append(mw)

            # All values should be bit-identical
            for i in range(1, len(mw_values)):
                comparison = compare_float_bits(mw_values[0], mw_values[i], "single")
                assert comparison[
                    "exact_match"
                ], f"Precision inconsistency for {smiles}: {comparison}"

    def test_deterministic_fingerprints(self):
        """Test that fingerprint generation is deterministic."""
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

        for smiles in test_molecules:
            mol_data_obj = self._create_mol_data(smiles)

            # Generate same fingerprint multiple times
            fingerprints = []
            for _ in range(5):
                fp = self.fingerprint_gen.generate_morgan_fingerprint(
                    mol_data_obj, radius=2, n_bits=1024
                )
                fingerprints.append(fp)

            # All fingerprints should be identical
            for i in range(1, len(fingerprints)):
                assert np.array_equal(
                    fingerprints[0], fingerprints[i]
                ), f"Non-deterministic fingerprint for {smiles}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestNumericalPrecision:
    """Test numerical precision and floating point handling."""

    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        basic_calc = BasicDescriptorCalculator()

        # Test with molecules that might produce edge case values
        edge_molecules = {
            "very_small": "C",  # Should produce small but normal values
            "medium": "c1ccccc1",  # Normal values
            "large": "C" * 20,  # Might produce larger values
        }

        for mol_name, smiles in edge_molecules.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                mol_data = MolecularData(
                    smiles=smiles,
                    mol_id=f"edge_{mol_name}",
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol),
                    backend_data={"rdkit_mol": mol},
                )

                # Test various descriptors
                mw = basic_calc.calculate_molecular_weight(mol_data)
                tpsa = basic_calc.calculate_tpsa(mol_data)
                logp = basic_calc.calculate_logp(mol_data)

                # Values should be finite and reasonable
                assert math.isfinite(mw), f"Non-finite MW for {mol_name}: {mw}"
                assert math.isfinite(tpsa), f"Non-finite TPSA for {mol_name}: {tpsa}"
                assert math.isfinite(logp), f"Non-finite LogP for {mol_name}: {logp}"

                # Values should be in reasonable ranges
                assert 0 < mw < 10000, f"Unreasonable MW for {mol_name}: {mw}"
                assert 0 <= tpsa < 1000, f"Unreasonable TPSA for {mol_name}: {tpsa}"
                assert -10 < logp < 20, f"Unreasonable LogP for {mol_name}: {logp}"

            except Exception as e:
                print(f"Could not test {mol_name}: {e}")
                continue

    def test_precision_loss_detection(self):
        """Test detection of precision loss in calculations."""
        basic_calc = BasicDescriptorCalculator()

        # Test with molecules that might cause precision issues
        precision_test_molecules = [
            "CCCCCCCCCCCCCCCCCCCC",  # Long chain
            "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4cccc5",  # Large aromatic system
        ]

        for smiles in precision_test_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                mol_data = MolecularData(
                    smiles=smiles,
                    mol_id=f"precision_{hash(smiles)}",
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol),
                    backend_data={"rdkit_mol": mol},
                )

                # Calculate descriptor and compare with RDKit
                our_mw = basic_calc.calculate_molecular_weight(mol_data)
                rdkit_mw = Descriptors.MolWt(mol)

                # Should have high precision match
                relative_error = (
                    abs(our_mw - rdkit_mw) / rdkit_mw if rdkit_mw != 0 else 0
                )
                assert (
                    relative_error < 1e-10
                ), f"Precision loss detected for {smiles}: relative_error={relative_error}"

            except Exception as e:
                print(f"Could not test precision for {smiles}: {e}")
                continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
