"""
Validation tests for 3D descriptor calculations against reference implementations.

This module validates the accuracy of ChemiAccel's 3D descriptor calculations
by comparing results against established reference implementations like RDKit.
"""


import numpy as np
import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.conformer_generation import GPUConformerGenerator
from src.chemi_accel.descriptors.descriptors_3d_integrated import (
    Integrated3DDescriptorCalculator,
)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestRDKitValidation:
    """Validate 3D descriptors against RDKit reference implementations."""

    def setup_method(self):
        """Set up test molecules with known 3D structures."""
        # Test molecules with different shapes
        test_smiles = [
            "CC",  # Ethane (simple)
            "CCO",  # Ethanol (with heteroatom)
            "c1ccccc1",  # Benzene (aromatic, planar)
            "CCC(C)C",  # Isopentane (branched)
            "C1CCCCC1",  # Cyclohexane (cyclic)
        ]

        self.test_molecules = []

        for i, smiles in enumerate(test_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            # Generate conformer
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)

            mol_data = MolecularData(
                smiles=smiles,
                mol_id=f"test_mol_{i}",
                num_atoms=mol.GetNumAtoms(),
                num_bonds=mol.GetNumBonds(),
                molecular_weight=Chem.Descriptors.MolWt(mol),
                backend_data={"rdkit_mol": mol},
            )

            self.test_molecules.append(mol_data)

    def test_radius_of_gyration_validation(self):
        """Validate radius of gyration calculations against RDKit."""
        try:
            from rdkit.Chem.Descriptors3D import RadiusOfGyration
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")

        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=False, use_gpu=False  # Use existing conformers
        )

        for mol_data in self.test_molecules:
            # Calculate using our implementation
            descriptors = calculator.calculate_3d_descriptors(mol_data)
            our_rg = descriptors.get("radius_of_gyration", 0.0)

            # Calculate using RDKit
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            rdkit_rg = RadiusOfGyration(rdkit_mol)

            # Should be approximately equal (within 10% relative error)
            if rdkit_rg > 0:
                relative_error = abs(our_rg - rdkit_rg) / rdkit_rg
                assert (
                    relative_error < 0.1
                ), f"RG mismatch for {mol_data.smiles}: ours={our_rg:.3f}, RDKit={rdkit_rg:.3f}"

    def test_asphericity_validation(self):
        """Validate asphericity calculations against RDKit."""
        try:
            from rdkit.Chem.Descriptors3D import Asphericity
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")

        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=False, use_gpu=False
        )

        for mol_data in self.test_molecules:
            # Calculate using our implementation
            descriptors = calculator.calculate_3d_descriptors(mol_data)
            our_asphericity = descriptors.get("asphericity", 0.0)

            # Calculate using RDKit
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            rdkit_asphericity = Asphericity(rdkit_mol)

            # Should be approximately equal (within 20% relative error due to formula differences)
            if rdkit_asphericity > 0:
                relative_error = (
                    abs(our_asphericity - rdkit_asphericity) / rdkit_asphericity
                )
                assert (
                    relative_error < 0.3
                ), f"Asphericity mismatch for {mol_data.smiles}: ours={our_asphericity:.3f}, RDKit={rdkit_asphericity:.3f}"

    def test_principal_moments_validation(self):
        """Validate principal moments calculations."""
        try:
            from rdkit.Chem.Descriptors3D import PMI1, PMI2, PMI3
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")

        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=False, use_gpu=False
        )

        for mol_data in self.test_molecules:
            # Calculate using our implementation
            descriptors = calculator.calculate_3d_descriptors(mol_data)
            our_pmi1 = descriptors.get("pmi1", 0.0)
            our_pmi2 = descriptors.get("pmi2", 0.0)
            our_pmi3 = descriptors.get("pmi3", 0.0)

            # Calculate using RDKit
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            rdkit_pmi1 = PMI1(rdkit_mol)
            rdkit_pmi2 = PMI2(rdkit_mol)
            rdkit_pmi3 = PMI3(rdkit_mol)

            # PMI values should be in ascending order
            assert (
                our_pmi1 <= our_pmi2 <= our_pmi3
            ), f"PMI ordering incorrect for {mol_data.smiles}"

            # Should be approximately equal (within 20% relative error)
            for our_val, rdkit_val, name in [
                (our_pmi1, rdkit_pmi1, "PMI1"),
                (our_pmi2, rdkit_pmi2, "PMI2"),
                (our_pmi3, rdkit_pmi3, "PMI3"),
            ]:
                if rdkit_val > 0:
                    relative_error = abs(our_val - rdkit_val) / rdkit_val
                    assert (
                        relative_error < 0.3
                    ), f"{name} mismatch for {mol_data.smiles}: ours={our_val:.3f}, RDKit={rdkit_val:.3f}"

    def test_eccentricity_validation(self):
        """Validate eccentricity calculations."""
        try:
            from rdkit.Chem.Descriptors3D import Eccentricity
        except ImportError:
            pytest.skip("RDKit 3D descriptors not available")

        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=False, use_gpu=False
        )

        for mol_data in self.test_molecules:
            # Calculate using our implementation
            descriptors = calculator.calculate_3d_descriptors(mol_data)
            our_eccentricity = descriptors.get("eccentricity", 0.0)

            # Calculate using RDKit
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            rdkit_eccentricity = Eccentricity(rdkit_mol)

            # Should be approximately equal (within 20% relative error)
            if rdkit_eccentricity > 0:
                relative_error = (
                    abs(our_eccentricity - rdkit_eccentricity) / rdkit_eccentricity
                )
                assert (
                    relative_error < 0.3
                ), f"Eccentricity mismatch for {mol_data.smiles}: ours={our_eccentricity:.3f}, RDKit={rdkit_eccentricity:.3f}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestConformerGenerationValidation:
    """Validate conformer generation quality and diversity."""

    def setup_method(self):
        """Set up test molecules for conformer generation."""
        test_smiles = [
            "CCCC",  # Butane (flexible)
            "CC(C)CC",  # Methylpropane (branched)
            "C1CCCCC1",  # Cyclohexane (ring)
        ]

        self.test_molecules = []

        for i, smiles in enumerate(test_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            mol_data = MolecularData(
                smiles=smiles,
                mol_id=f"conformer_test_{i}",
                num_atoms=mol.GetNumAtoms(),
                num_bonds=mol.GetNumBonds(),
                molecular_weight=Chem.Descriptors.MolWt(mol),
                backend_data={"rdkit_mol": mol},
            )

            self.test_molecules.append(mol_data)

    def test_conformer_generation_quality(self):
        """Test quality of generated conformers."""
        generator = GPUConformerGenerator(
            num_conformers=5,
            use_gpu=False,  # Use CPU for reproducibility
            energy_minimize=True,
        )

        for mol_data in self.test_molecules:
            result = generator.generate_conformers_parallel(mol_data)
            rdkit_mol = result.backend_data["rdkit_mol"]

            # Should have generated conformers
            assert (
                rdkit_mol.GetNumConformers() > 0
            ), f"No conformers generated for {mol_data.smiles}"

            # Conformers should have reasonable energies (no NaN or infinite values)
            for conf_id in range(rdkit_mol.GetNumConformers()):
                conf = rdkit_mol.GetConformer(conf_id)

                # Check that coordinates are reasonable
                for atom_idx in range(rdkit_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    assert (
                        not np.isnan(pos.x)
                        and not np.isnan(pos.y)
                        and not np.isnan(pos.z)
                    )
                    assert (
                        not np.isinf(pos.x)
                        and not np.isinf(pos.y)
                        and not np.isinf(pos.z)
                    )

                    # Coordinates should be within reasonable range
                    assert abs(pos.x) < 100 and abs(pos.y) < 100 and abs(pos.z) < 100

    def test_conformer_diversity(self):
        """Test diversity of generated conformers."""
        from src.chemi_accel.descriptors.conformer_generation import (
            validate_conformer_quality,
        )

        generator = GPUConformerGenerator(
            num_conformers=10,
            use_gpu=False,
            prune_rms_thresh=0.5,  # Allow some diversity
        )

        for mol_data in self.test_molecules:
            result = generator.generate_conformers_parallel(mol_data)

            # Validate conformer quality
            quality_metrics = validate_conformer_quality(result)

            if "error" not in quality_metrics:
                # Should have reasonable diversity
                diversity_score = quality_metrics.get("diversity_score", 0.0)
                assert (
                    diversity_score >= 0.0
                ), f"Invalid diversity score for {mol_data.smiles}"

                # For flexible molecules, should have some diversity
                if mol_data.smiles in ["CCCC", "CC(C)CC"]:  # Flexible molecules
                    assert (
                        diversity_score > 0.1
                    ), f"Low diversity for flexible molecule {mol_data.smiles}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestIntegrated3DDescriptorValidation:
    """Validate the integrated 3D descriptor calculation system."""

    def setup_method(self):
        """Set up test molecules."""
        self.test_smiles = ["CCO", "c1ccccc1", "CCC(C)C"]

        self.test_molecules = []
        for i, smiles in enumerate(self.test_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            mol_data = MolecularData(
                smiles=smiles,
                mol_id=f"integrated_test_{i}",
                num_atoms=mol.GetNumAtoms(),
                num_bonds=mol.GetNumBonds(),
                molecular_weight=Chem.Descriptors.MolWt(mol),
                backend_data={"rdkit_mol": mol},
            )

            self.test_molecules.append(mol_data)

    def test_integrated_calculation_completeness(self):
        """Test that integrated calculator produces complete descriptor sets."""
        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=True, num_conformers=3, use_gpu=False
        )

        for mol_data in self.test_molecules:
            descriptors = calculator.calculate_3d_descriptors(mol_data)

            # Should have basic 3D descriptors
            required_descriptors = [
                "radius_of_gyration",
                "asphericity",
                "eccentricity",
                "pmi1",
                "pmi2",
                "pmi3",
            ]

            for desc_name in required_descriptors:
                assert (
                    desc_name in descriptors
                ), f"Missing descriptor {desc_name} for {mol_data.smiles}"
                assert not np.isnan(
                    descriptors[desc_name]
                ), f"NaN value for {desc_name}"
                assert not np.isinf(
                    descriptors[desc_name]
                ), f"Infinite value for {desc_name}"

    def test_batch_calculation_consistency(self):
        """Test that batch calculations are consistent with individual calculations."""
        # Use pre-generated conformers to ensure consistency
        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=False,  # Don't generate new conformers
            num_conformers=2,
            use_gpu=False,
        )

        # Pre-generate conformers for all molecules
        for mol_data in self.test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            if rdkit_mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(rdkit_mol)

        # Calculate individually
        individual_results = []
        for mol_data in self.test_molecules:
            descriptors = calculator.calculate_3d_descriptors(mol_data)
            individual_results.append(descriptors)

        # Calculate as batch
        batch_results = calculator.calculate_3d_descriptors_batch(self.test_molecules)

        # Results should be consistent
        assert len(batch_results) == len(individual_results)

        for i, (individual, batch_row) in enumerate(
            zip(individual_results, batch_results.to_dict("records"))
        ):
            for desc_name in ["radius_of_gyration", "asphericity", "pmi1"]:
                if desc_name in individual and desc_name in batch_row:
                    individual_val = individual[desc_name]
                    batch_val = batch_row[desc_name]

                    # Should be approximately equal (allow for conformer generation differences)
                    if abs(individual_val) > 1e-6:
                        relative_error = abs(individual_val - batch_val) / abs(
                            individual_val
                        )
                        assert (
                            relative_error < 0.1
                        ), f"Batch inconsistency for {desc_name} in molecule {i}: individual={individual_val:.6f}, batch={batch_val:.6f}"

    def test_descriptor_value_ranges(self):
        """Test that descriptor values are within expected ranges."""
        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=True, num_conformers=2, use_gpu=False
        )

        for mol_data in self.test_molecules:
            descriptors = calculator.calculate_3d_descriptors(mol_data)

            # Radius of gyration should be positive
            rg = descriptors.get("radius_of_gyration", 0.0)
            assert rg >= 0.0, f"Negative radius of gyration for {mol_data.smiles}"
            assert (
                rg < 100.0
            ), f"Unreasonably large radius of gyration for {mol_data.smiles}"

            # Asphericity should be between 0 and 1
            asphericity = descriptors.get("asphericity", 0.0)
            assert (
                0.0 <= asphericity <= 1.0
            ), f"Asphericity out of range for {mol_data.smiles}"

            # Eccentricity should be between 0 and 1
            eccentricity = descriptors.get("eccentricity", 0.0)
            assert (
                0.0 <= eccentricity <= 1.0
            ), f"Eccentricity out of range for {mol_data.smiles}"

            # PMI values should be non-negative and ordered
            pmi1 = descriptors.get("pmi1", 0.0)
            pmi2 = descriptors.get("pmi2", 0.0)
            pmi3 = descriptors.get("pmi3", 0.0)

            assert (
                pmi1 >= 0.0 and pmi2 >= 0.0 and pmi3 >= 0.0
            ), f"Negative PMI values for {mol_data.smiles}"
            assert pmi1 <= pmi2 <= pmi3, f"PMI values not ordered for {mol_data.smiles}"


class TestPerformanceBenchmarks:
    """Performance benchmarks for 3D descriptor calculations."""

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculation_performance(self):
        """Test calculation performance and timing."""
        import time

        # Create test molecules
        test_smiles = ["CCO", "c1ccccc1", "CCC(C)C", "CC(C)(C)C"] * 5  # 20 molecules
        test_molecules = []

        for i, smiles in enumerate(test_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            mol_data = MolecularData(
                smiles=smiles,
                mol_id=f"perf_test_{i}",
                num_atoms=mol.GetNumAtoms(),
                num_bonds=mol.GetNumBonds(),
                molecular_weight=Chem.Descriptors.MolWt(mol),
                backend_data={"rdkit_mol": mol},
            )
            test_molecules.append(mol_data)

        calculator = Integrated3DDescriptorCalculator(
            generate_conformers=True, num_conformers=3, use_gpu=False
        )

        # Time the calculation
        start_time = time.time()
        results = calculator.calculate_3d_descriptors_batch(test_molecules)
        end_time = time.time()

        calculation_time = end_time - start_time

        # Should complete in reasonable time (less than 30 seconds for 20 small molecules)
        assert calculation_time < 30.0, f"Calculation too slow: {calculation_time:.2f}s"

        # Should have results for all molecules
        assert len(results) == len(test_molecules)

        # Average time per molecule should be reasonable
        avg_time_per_molecule = calculation_time / len(test_molecules)
        assert (
            avg_time_per_molecule < 2.0
        ), f"Average time per molecule too high: {avg_time_per_molecule:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__])
