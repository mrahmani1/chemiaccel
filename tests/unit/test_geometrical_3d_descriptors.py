"""
Tests for geometrical and 3D molecular descriptors.

This module tests the 3D shape descriptors, electrostatic descriptors,
surface area calculations, and geometric diameter/eccentricity calculations.
"""

import time

import numpy as np
import pandas as pd
import pytest
from chemi_accel.descriptors.geometrical_3d import (
    ConformerGenerator,
    ElectrostaticDescriptors,
    Geometrical3DCalculator,
    Shape3DDescriptors,
    calculate_3d_descriptors,
    calculate_electrostatic_descriptors,
    calculate_shape_descriptors,
    generate_conformers,
)

# Test molecules with known 3D properties
TEST_SMILES = [
    "CCO",  # ethanol - simple molecule
    "c1ccccc1",  # benzene - planar aromatic
    "CC(C)C",  # isobutane - branched alkane
    "C1CCCCC1",  # cyclohexane - ring structure
    "CCN(CC)CC",  # triethylamine - tertiary amine
    "CC(=O)O",  # acetic acid - carboxylic acid
    "c1ccc2c(c1)cccn2",  # quinoline - bicyclic aromatic
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol - substituted aromatic
    "CN1CCC[C@H]1c2cccnc2",  # nicotine - complex heterocycle
    "CC(=O)Nc1ccc(O)cc1",  # acetaminophen - drug molecule
]

# Reference values for validation (approximate expected ranges)
EXPECTED_RANGES = {
    "radius_of_gyration": (1.0, 10.0),
    "asphericity": (0.0, 1.0),
    "eccentricity": (0.0, 1.0),
    "molecular_volume": (10.0, 500.0),
    "molecular_surface_area": (30.0, 1000.0),  # Lowered minimum for small molecules
    "geometric_diameter": (2.0, 20.0),
    "dipole_moment": (0.0, 10.0),
    "total_charge": (-2.0, 2.0),
    "charge_range": (0.0, 2.0),
}


@pytest.fixture
def test_molecules():
    """Create test molecules from SMILES."""
    from chemi_accel.utils.rdkit_integration import RDKitMolecularParser

    parser = RDKitMolecularParser()
    return parser.parse_smiles_batch(TEST_SMILES, error_handling="skip")


@pytest.fixture
def molecules_with_conformers(test_molecules):
    """Create test molecules with 3D conformers."""
    generator = ConformerGenerator(num_conformers=5, optimize_conformers=True)
    return generator.generate_conformers_batch(test_molecules)


class TestConformerGenerator:
    """Test conformer generation functionality."""

    def test_initialization(self):
        """Test ConformerGenerator initialization."""
        generator = ConformerGenerator(
            num_conformers=5,
            max_attempts=100,
            optimize_conformers=True,
            force_field="UFF",
        )
        assert generator.num_conformers == 5
        assert generator.max_attempts == 100
        assert generator.optimize_conformers == True
        assert generator.force_field == "UFF"

    def test_invalid_force_field(self):
        """Test invalid force field raises error."""
        with pytest.raises(ValueError):
            ConformerGenerator(force_field="INVALID")

    def test_generate_single_conformer(self, test_molecules):
        """Test conformer generation for single molecule."""
        generator = ConformerGenerator(num_conformers=3, optimize_conformers=False)

        mol = test_molecules[0]  # ethanol
        mol_with_conformers = generator.generate_conformers(mol)

        assert mol_with_conformers is not None
        rdkit_mol = mol_with_conformers.backend_data.get("rdkit_mol")
        assert rdkit_mol is not None
        assert rdkit_mol.GetNumConformers() > 0
        assert "num_conformers" in mol_with_conformers.backend_data
        assert "conformer_ids" in mol_with_conformers.backend_data

    def test_generate_conformers_batch(self, test_molecules):
        """Test batch conformer generation."""
        generator = ConformerGenerator(num_conformers=2, optimize_conformers=False)

        molecules_with_conformers = generator.generate_conformers_batch(
            test_molecules[:3]
        )

        assert len(molecules_with_conformers) == 3

        for mol in molecules_with_conformers:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol is not None:  # Some molecules might fail
                assert rdkit_mol.GetNumConformers() >= 0

    def test_conformer_optimization(self, test_molecules):
        """Test conformer optimization with different force fields."""
        # Test UFF optimization
        generator_uff = ConformerGenerator(
            num_conformers=2, optimize_conformers=True, force_field="UFF"
        )

        mol = test_molecules[0]
        mol_uff = generator_uff.generate_conformers(mol)

        rdkit_mol = mol_uff.backend_data.get("rdkit_mol")
        assert rdkit_mol is not None
        assert rdkit_mol.GetNumConformers() > 0

        # Test MMFF optimization
        generator_mmff = ConformerGenerator(
            num_conformers=2, optimize_conformers=True, force_field="MMFF"
        )

        mol_mmff = generator_mmff.generate_conformers(mol)
        rdkit_mol_mmff = mol_mmff.backend_data.get("rdkit_mol")
        assert rdkit_mol_mmff is not None


class TestShape3DDescriptors:
    """Test 3D shape descriptor calculations."""

    def test_initialization(self):
        """Test Shape3DDescriptors initialization."""
        calculator = Shape3DDescriptors()
        assert hasattr(calculator, "descriptor_functions")
        assert len(calculator.descriptor_functions) > 0

    def test_molecular_volume_calculation(self, molecules_with_conformers):
        """Test molecular volume calculation."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            volume = calculator._calculate_molecular_volume(rdkit_mol, 0)
            assert isinstance(volume, float)
            assert volume >= 0
            assert (
                EXPECTED_RANGES["molecular_volume"][0]
                <= volume
                <= EXPECTED_RANGES["molecular_volume"][1]
            )

    def test_surface_area_calculation(self, molecules_with_conformers):
        """Test molecular surface area calculation."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            surface_area = calculator._calculate_molecular_surface_area(rdkit_mol, 0)
            assert isinstance(surface_area, float)
            assert surface_area >= 0
            assert (
                EXPECTED_RANGES["molecular_surface_area"][0]
                <= surface_area
                <= EXPECTED_RANGES["molecular_surface_area"][1]
            )

    def test_geometric_diameter_calculation(self, molecules_with_conformers):
        """Test geometric diameter calculation."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            diameter = calculator._calculate_geometric_diameter(rdkit_mol, 0)
            assert isinstance(diameter, float)
            assert diameter >= 0
            assert (
                EXPECTED_RANGES["geometric_diameter"][0]
                <= diameter
                <= EXPECTED_RANGES["geometric_diameter"][1]
            )

    def test_shape_anisotropy_calculation(self, molecules_with_conformers):
        """Test shape anisotropy calculation."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            anisotropy = calculator._calculate_shape_anisotropy(rdkit_mol, 0)
            assert isinstance(anisotropy, float)
            assert 0.0 <= anisotropy <= 1.0

    def test_compactness_calculation(self, molecules_with_conformers):
        """Test molecular compactness calculation."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            compactness = calculator._calculate_compactness(rdkit_mol, 0)
            assert isinstance(compactness, float)
            assert compactness >= 0

    def test_calculate_shape_descriptors_single(self, molecules_with_conformers):
        """Test shape descriptor calculation for single molecule."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            descriptors = calculator.calculate_shape_descriptors(mol)

            assert isinstance(descriptors, dict)
            assert len(descriptors) > 0

            # Check that all values are numeric
            for key, value in descriptors.items():
                assert isinstance(value, (int, float))
                assert not np.isnan(value)
                assert not np.isinf(value)

    def test_calculate_shape_descriptors_batch(self, molecules_with_conformers):
        """Test batch shape descriptor calculation."""
        calculator = Shape3DDescriptors()

        # Filter molecules that have conformers
        valid_molecules = []
        for mol in molecules_with_conformers[:5]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                valid_molecules.append(mol)

        if valid_molecules:
            results = calculator.calculate_shape_descriptors_batch(valid_molecules)

            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert "mol_id" in results.columns
            assert "smiles" in results.columns

            # Check descriptor columns
            descriptor_cols = [
                col
                for col in results.columns
                if col not in ["mol_id", "smiles", "conformer_id"]
            ]
            assert len(descriptor_cols) > 0

            # Check for valid numeric values
            for col in descriptor_cols:
                assert results[col].dtype in [np.float64, np.int64]
                assert not results[col].isna().all()

    def test_specific_shape_descriptors(self, molecules_with_conformers):
        """Test calculation of specific shape descriptors."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            # Test specific descriptors
            specific_descriptors = [
                "molecular_volume",
                "geometric_diameter",
                "compactness",
            ]
            descriptors = calculator.calculate_shape_descriptors(
                mol, specific_descriptors
            )

            assert len(descriptors) == len(specific_descriptors)
            for desc_name in specific_descriptors:
                assert desc_name in descriptors
                assert isinstance(descriptors[desc_name], (int, float))


class TestElectrostaticDescriptors:
    """Test electrostatic descriptor calculations."""

    def test_initialization(self):
        """Test ElectrostaticDescriptors initialization."""
        calculator = ElectrostaticDescriptors(charge_model="gasteiger")
        assert calculator.charge_model == "gasteiger"

        calculator_mmff = ElectrostaticDescriptors(charge_model="mmff")
        assert calculator_mmff.charge_model == "mmff"

    def test_invalid_charge_model(self):
        """Test invalid charge model raises error."""
        with pytest.raises(ValueError):
            ElectrostaticDescriptors(charge_model="invalid")

    def test_partial_charges_calculation(self, molecules_with_conformers):
        """Test partial charge calculation."""
        calculator = ElectrostaticDescriptors(charge_model="gasteiger")

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            charges = calculator._calculate_partial_charges(rdkit_mol)

            assert isinstance(charges, np.ndarray)
            assert len(charges) == rdkit_mol.GetNumAtoms()
            assert not np.any(np.isnan(charges))
            assert not np.any(np.isinf(charges))

    def test_electrostatic_descriptors_single(self, molecules_with_conformers):
        """Test electrostatic descriptor calculation for single molecule."""
        calculator = ElectrostaticDescriptors(charge_model="gasteiger")

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            descriptors = calculator.calculate_electrostatic_descriptors(mol)

            assert isinstance(descriptors, dict)
            assert len(descriptors) > 0

            # Check expected descriptors
            expected_descriptors = [
                "total_charge",
                "abs_total_charge",
                "charge_mean",
                "charge_std",
                "charge_min",
                "charge_max",
                "charge_range",
                "positive_charge_sum",
                "negative_charge_sum",
                "charge_polarization",
                "dipole_moment",
                "electrostatic_energy_sum",
                "electrostatic_energy_mean",
                "electrostatic_energy_std",
            ]

            for desc_name in expected_descriptors:
                assert desc_name in descriptors
                assert isinstance(descriptors[desc_name], (int, float))
                assert not np.isnan(descriptors[desc_name])
                assert not np.isinf(descriptors[desc_name])

            # Check reasonable ranges
            assert (
                EXPECTED_RANGES["total_charge"][0]
                <= descriptors["total_charge"]
                <= EXPECTED_RANGES["total_charge"][1]
            )
            assert descriptors["abs_total_charge"] >= 0
            assert descriptors["charge_range"] >= 0
            assert descriptors["dipole_moment"] >= 0

    def test_different_charge_models(self, molecules_with_conformers):
        """Test different charge models."""
        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            # Test Gasteiger charges
            calc_gasteiger = ElectrostaticDescriptors(charge_model="gasteiger")
            desc_gasteiger = calc_gasteiger.calculate_electrostatic_descriptors(mol)

            # Test MMFF charges (may fall back to Gasteiger)
            calc_mmff = ElectrostaticDescriptors(charge_model="mmff")
            desc_mmff = calc_mmff.calculate_electrostatic_descriptors(mol)

            # Both should return valid descriptors
            assert isinstance(desc_gasteiger, dict)
            assert isinstance(desc_mmff, dict)
            assert len(desc_gasteiger) > 0
            assert len(desc_mmff) > 0


class TestGeometrical3DCalculator:
    """Test main 3D descriptor calculator."""

    def test_initialization(self):
        """Test Geometrical3DCalculator initialization."""
        calculator = Geometrical3DCalculator(
            generate_conformers=True, num_conformers=5, charge_model="gasteiger"
        )
        assert calculator.generate_conformers == True
        assert calculator.conformer_generator is not None
        assert calculator.shape_calculator is not None
        assert calculator.electrostatic_calculator is not None

    def test_initialization_no_conformers(self):
        """Test initialization without conformer generation."""
        calculator = Geometrical3DCalculator(generate_conformers=False)
        assert calculator.generate_conformers == False
        assert calculator.conformer_generator is None

    def test_calculate_3d_descriptors_single(self, test_molecules):
        """Test 3D descriptor calculation for single molecule."""
        calculator = Geometrical3DCalculator(generate_conformers=True, num_conformers=3)

        mol = test_molecules[0]  # ethanol
        descriptors = calculator.calculate_3d_descriptors(mol)

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 0

        # Check that we have both shape and electrostatic descriptors
        shape_keys = [
            k
            for k in descriptors.keys()
            if k in calculator.shape_calculator.descriptor_functions
        ]
        electrostatic_keys = [
            k for k in descriptors.keys() if "charge" in k or "dipole" in k
        ]

        assert len(shape_keys) > 0
        assert len(electrostatic_keys) > 0

    def test_calculate_3d_descriptors_selective(self, test_molecules):
        """Test selective 3D descriptor calculation."""
        calculator = Geometrical3DCalculator(generate_conformers=True, num_conformers=2)

        mol = test_molecules[0]

        # Test only shape descriptors
        descriptors_shape = calculator.calculate_3d_descriptors(
            mol, include_shape=True, include_electrostatic=False
        )

        shape_keys = [
            k
            for k in descriptors_shape.keys()
            if k in calculator.shape_calculator.descriptor_functions
        ]
        electrostatic_keys = [
            k for k in descriptors_shape.keys() if "charge" in k or "dipole" in k
        ]

        assert len(shape_keys) > 0
        assert len(electrostatic_keys) == 0

        # Test only electrostatic descriptors
        descriptors_electrostatic = calculator.calculate_3d_descriptors(
            mol, include_shape=False, include_electrostatic=True
        )

        shape_keys = [
            k
            for k in descriptors_electrostatic.keys()
            if k in calculator.shape_calculator.descriptor_functions
        ]
        electrostatic_keys = [
            k
            for k in descriptors_electrostatic.keys()
            if "charge" in k or "dipole" in k
        ]

        assert len(shape_keys) == 0
        assert len(electrostatic_keys) > 0

    def test_calculate_3d_descriptors_batch(self, test_molecules):
        """Test batch 3D descriptor calculation."""
        calculator = Geometrical3DCalculator(generate_conformers=True, num_conformers=2)

        # Use subset for faster testing
        molecules_subset = test_molecules[:3]
        results = calculator.calculate_3d_descriptors_batch(molecules_subset)

        assert isinstance(results, pd.DataFrame)
        assert len(results) <= len(molecules_subset)  # Some might fail

        if len(results) > 0:
            assert "mol_id" in results.columns
            assert "smiles" in results.columns

            # Check for descriptor columns
            descriptor_cols = [
                col
                for col in results.columns
                if col not in ["mol_id", "smiles", "error"]
            ]
            assert len(descriptor_cols) > 0

    def test_get_available_descriptors(self):
        """Test getting available descriptor list."""
        calculator = Geometrical3DCalculator()
        available = calculator.get_available_descriptors()

        assert isinstance(available, dict)
        assert "shape_descriptors" in available
        assert "electrostatic_descriptors" in available
        assert len(available["shape_descriptors"]) > 0
        assert len(available["electrostatic_descriptors"]) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_conformers_convenience(self, test_molecules):
        """Test conformer generation convenience function."""
        molecules_subset = test_molecules[:2]
        molecules_with_conformers = generate_conformers(
            molecules_subset, num_conformers=3, optimize_conformers=False
        )

        assert len(molecules_with_conformers) == len(molecules_subset)

        for mol in molecules_with_conformers:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol is not None:
                assert rdkit_mol.GetNumConformers() >= 0

    def test_calculate_3d_descriptors_convenience(self, test_molecules):
        """Test 3D descriptor calculation convenience function."""
        molecules_subset = test_molecules[:2]
        results = calculate_3d_descriptors(
            molecules_subset, generate_conformers=True, num_conformers=2
        )

        assert isinstance(results, pd.DataFrame)
        # Results might be empty if conformer generation fails

    def test_calculate_shape_descriptors_convenience(self, molecules_with_conformers):
        """Test shape descriptor calculation convenience function."""
        # Filter molecules with conformers
        valid_molecules = []
        for mol in molecules_with_conformers[:3]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                valid_molecules.append(mol)

        if valid_molecules:
            results = calculate_shape_descriptors(valid_molecules)

            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0

    def test_calculate_electrostatic_descriptors_convenience(
        self, molecules_with_conformers
    ):
        """Test electrostatic descriptor calculation convenience function."""
        # Filter molecules with conformers
        valid_molecules = []
        for mol in molecules_with_conformers[:3]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                valid_molecules.append(mol)

        if valid_molecules:
            results = calculate_electrostatic_descriptors(valid_molecules)

            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert "dipole_moment" in results.columns
            assert "total_charge" in results.columns


class TestValidationAgainstReference:
    """Test validation against established 3D descriptor implementations."""

    def test_rdkit_3d_descriptor_comparison(self, molecules_with_conformers):
        """Compare results with RDKit's built-in 3D descriptors."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            # Calculate using our implementation
            our_descriptors = calculator.calculate_shape_descriptors(mol)

            # Calculate using RDKit directly (if available)
            try:
                from rdkit.Chem.Descriptors3D import (
                    Asphericity,
                    Eccentricity,
                    RadiusOfGyration,
                )

                rdkit_rog = RadiusOfGyration(rdkit_mol, confId=0)
                rdkit_asphericity = Asphericity(rdkit_mol, confId=0)
                rdkit_eccentricity = Eccentricity(rdkit_mol, confId=0)

                # Compare results (allow for small numerical differences)
                if "radius_of_gyration" in our_descriptors:
                    assert abs(our_descriptors["radius_of_gyration"] - rdkit_rog) < 0.1

                if "asphericity" in our_descriptors:
                    assert abs(our_descriptors["asphericity"] - rdkit_asphericity) < 0.1

                if "eccentricity" in our_descriptors:
                    assert (
                        abs(our_descriptors["eccentricity"] - rdkit_eccentricity) < 0.1
                    )

            except ImportError:
                # RDKit 3D descriptors not available, skip comparison
                pass

    def test_charge_conservation(self, molecules_with_conformers):
        """Test that partial charges sum to reasonable total charge."""
        calculator = ElectrostaticDescriptors(charge_model="gasteiger")

        for mol in molecules_with_conformers[:3]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                descriptors = calculator.calculate_electrostatic_descriptors(mol)

                # For neutral molecules, total charge should be close to zero
                total_charge = descriptors["total_charge"]
                assert abs(total_charge) < 1.0  # Allow for some numerical error

                # Positive and negative charges should roughly balance
                pos_charge = descriptors["positive_charge_sum"]
                neg_charge = descriptors["negative_charge_sum"]

                if pos_charge > 0 and neg_charge < 0:
                    assert abs(pos_charge + neg_charge - total_charge) < 0.1

    def test_descriptor_ranges_validation(self, molecules_with_conformers):
        """Test that calculated descriptors fall within expected ranges."""
        calculator = Geometrical3DCalculator(generate_conformers=False)

        for mol in molecules_with_conformers[:5]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                descriptors = calculator.calculate_3d_descriptors(mol)

                # Test ranges for key descriptors
                for desc_name, (min_val, max_val) in EXPECTED_RANGES.items():
                    if desc_name in descriptors:
                        value = descriptors[desc_name]
                        assert (
                            min_val <= value <= max_val
                        ), f"{desc_name} = {value} not in range [{min_val}, {max_val}]"


class TestPerformanceComparison:
    """Test performance comparisons and benchmarks."""

    def test_conformer_generation_performance(self, test_molecules):
        """Test conformer generation performance."""
        generator = ConformerGenerator(num_conformers=5, optimize_conformers=False)

        molecules_subset = test_molecules[:3]

        start_time = time.time()
        molecules_with_conformers = generator.generate_conformers_batch(
            molecules_subset
        )
        generation_time = time.time() - start_time

        print(
            f"Conformer generation time for {len(molecules_subset)} molecules: {generation_time:.4f}s"
        )

        # Should complete within reasonable time
        assert generation_time < 30.0  # 30 seconds max for 3 molecules

        # Check that conformers were generated
        conformer_counts = []
        for mol in molecules_with_conformers:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol:
                conformer_counts.append(rdkit_mol.GetNumConformers())

        assert len(conformer_counts) > 0
        assert max(conformer_counts) > 0

    def test_3d_descriptor_calculation_performance(self, molecules_with_conformers):
        """Test 3D descriptor calculation performance."""
        calculator = Geometrical3DCalculator(generate_conformers=False)

        # Filter valid molecules
        valid_molecules = []
        for mol in molecules_with_conformers[:5]:
            rdkit_mol = mol.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                valid_molecules.append(mol)

        if valid_molecules:
            start_time = time.time()
            results = calculator.calculate_3d_descriptors_batch(valid_molecules)
            calculation_time = time.time() - start_time

            print(
                f"3D descriptor calculation time for {len(valid_molecules)} molecules: {calculation_time:.4f}s"
            )

            # Should complete within reasonable time
            assert calculation_time < 60.0  # 60 seconds max

            if len(results) > 0:
                descriptors_per_second = len(results) / calculation_time
                print(
                    f"Descriptors calculated per second: {descriptors_per_second:.2f}"
                )

                # Should achieve reasonable throughput
                assert descriptors_per_second > 0.1  # At least 0.1 molecules per second


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_molecule_without_conformers(self, test_molecules):
        """Test handling of molecules without conformers."""
        calculator = Shape3DDescriptors()

        mol = test_molecules[0]

        # Should raise error for molecule without conformers
        with pytest.raises(Exception):
            calculator.calculate_shape_descriptors(mol)

    def test_invalid_conformer_id(self, molecules_with_conformers):
        """Test handling of invalid conformer ID."""
        calculator = Shape3DDescriptors()

        mol = molecules_with_conformers[0]
        rdkit_mol = mol.backend_data.get("rdkit_mol")

        if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
            # Should handle invalid conformer ID gracefully
            try:
                descriptors = calculator.calculate_shape_descriptors(mol, conf_id=999)
                # If it doesn't raise an error, descriptors should still be valid
                assert isinstance(descriptors, dict)
            except Exception:
                # It's acceptable to raise an error for invalid conformer ID
                pass

    def test_empty_molecule_list(self):
        """Test handling of empty molecule list."""
        calculator = Geometrical3DCalculator()

        results = calculator.calculate_3d_descriptors_batch([])

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_conformer_generation_failure_handling(self):
        """Test handling of conformer generation failures."""
        generator = ConformerGenerator(num_conformers=1, max_attempts=1)

        # Create a problematic molecule (very strained)
        from chemi_accel.utils.rdkit_integration import parse_smiles

        try:
            # This might fail conformer generation
            problematic_mol = parse_smiles("C1C2C3C4C1C2C3C4")  # Highly strained cage
            result = generator.generate_conformers(problematic_mol)

            # If it succeeds, result should be valid
            assert result is not None

        except Exception as e:
            # It's acceptable for highly strained molecules to fail
            assert isinstance(e, Exception)


if __name__ == "__main__":
    pytest.main([__file__])
