"""
Crystal structure file parsers for CIF and POSCAR formats.

This module provides parsers for common crystallographic file formats,
with support for periodic boundary conditions and integration with
pymatgen and ASE.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifParser
    from pymatgen.io.vasp import Poscar

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("pymatgen not available. Some functionality will be limited.")

try:
    from ase import Atoms
    from ase.io import read as ase_read

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available. Some functionality will be limited.")


@dataclass
class CrystalStructure:
    """Standardized crystal structure representation."""

    # Lattice parameters
    lattice_vectors: np.ndarray  # (3, 3) matrix
    lattice_parameters: np.ndarray  # [a, b, c, alpha, beta, gamma]

    # Atomic information
    atomic_numbers: np.ndarray
    atomic_symbols: List[str]
    fractional_coords: np.ndarray  # (n_atoms, 3)
    cartesian_coords: np.ndarray  # (n_atoms, 3)

    # Metadata
    space_group: Optional[str] = None
    space_group_number: Optional[int] = None
    formula: Optional[str] = None
    density: Optional[float] = None
    volume: Optional[float] = None

    # Additional properties
    magnetic_moments: Optional[np.ndarray] = None
    occupancies: Optional[np.ndarray] = None
    thermal_factors: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and compute derived properties."""
        self._validate_structure()
        self._compute_derived_properties()

    def _validate_structure(self):
        """Validate the crystal structure data."""
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError("Lattice vectors must be a 3x3 matrix")

        n_atoms = len(self.atomic_numbers)
        if self.fractional_coords.shape != (n_atoms, 3):
            raise ValueError("Fractional coordinates shape mismatch")

        if len(self.atomic_symbols) != n_atoms:
            raise ValueError("Number of atomic symbols must match number of atoms")

    def _compute_derived_properties(self):
        """Compute derived structural properties."""
        # Volume
        if self.volume is None:
            self.volume = np.abs(np.linalg.det(self.lattice_vectors))

        # Lattice parameters if not provided
        if self.lattice_parameters is None:
            self.lattice_parameters = self._compute_lattice_parameters()

        # Density if not provided
        if self.density is None:
            self.density = self._compute_density()

    def _compute_lattice_parameters(self) -> np.ndarray:
        """Compute lattice parameters from lattice vectors."""
        a_vec, b_vec, c_vec = self.lattice_vectors

        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)

        alpha = np.arccos(np.dot(b_vec, c_vec) / (b * c)) * 180 / np.pi
        beta = np.arccos(np.dot(a_vec, c_vec) / (a * c)) * 180 / np.pi
        gamma = np.arccos(np.dot(a_vec, b_vec) / (a * b)) * 180 / np.pi

        return np.array([a, b, c, alpha, beta, gamma])

    def _compute_density(self) -> float:
        """Compute crystal density in g/cm³."""
        # Atomic masses (simplified - using average atomic masses)
        atomic_mass_map = {
            1: 1.008,
            6: 12.011,
            7: 14.007,
            8: 15.999,
            9: 18.998,
            11: 22.990,
            12: 24.305,
            13: 26.982,
            14: 28.085,
            15: 30.974,
            16: 32.065,
            17: 35.453,
            19: 39.098,
            20: 40.078,
        }

        total_mass = sum(atomic_mass_map.get(z, z * 1.8) for z in self.atomic_numbers)
        volume_cm3 = self.volume * 1e-24  # Convert Ų to cm³
        avogadro = 6.022e23

        return total_mass / (volume_cm3 * avogadro)


class CIFParser:
    """Parser for Crystallographic Information Files (CIF)."""

    def __init__(self, use_pymatgen: bool = True):
        """Initialize CIF parser.

        Args:
            use_pymatgen: Whether to use pymatgen for parsing (recommended)
        """
        self.use_pymatgen = use_pymatgen and PYMATGEN_AVAILABLE

        if not self.use_pymatgen and not PYMATGEN_AVAILABLE:
            warnings.warn("pymatgen not available, using basic CIF parser")

    def parse_file(self, filepath: Union[str, Path]) -> List[CrystalStructure]:
        """Parse CIF file and return crystal structures.

        Args:
            filepath: Path to CIF file

        Returns:
            List of CrystalStructure objects
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CIF file not found: {filepath}")

        if self.use_pymatgen:
            return self._parse_with_pymatgen(filepath)
        else:
            return self._parse_basic(filepath)

    def _parse_with_pymatgen(self, filepath: Path) -> List[CrystalStructure]:
        """Parse CIF using pymatgen."""
        try:
            parser = CifParser(str(filepath))
            structures = []

            for structure in parser.get_structures():
                crystal_struct = self._pymatgen_to_crystal_structure(structure)
                structures.append(crystal_struct)

            return structures

        except Exception as e:
            raise ValueError(f"Failed to parse CIF file with pymatgen: {e}")

    def _pymatgen_to_crystal_structure(
        self, structure: "Structure"
    ) -> CrystalStructure:
        """Convert pymatgen Structure to CrystalStructure."""
        lattice = structure.lattice

        return CrystalStructure(
            lattice_vectors=lattice.matrix,
            lattice_parameters=np.array(
                [
                    lattice.a,
                    lattice.b,
                    lattice.c,
                    lattice.alpha,
                    lattice.beta,
                    lattice.gamma,
                ]
            ),
            atomic_numbers=np.array([site.specie.Z for site in structure]),
            atomic_symbols=[str(site.specie) for site in structure],
            fractional_coords=np.array([site.frac_coords for site in structure]),
            cartesian_coords=np.array([site.coords for site in structure]),
            formula=structure.formula,
            volume=structure.volume,
            density=structure.density,
        )

    def _parse_basic(self, filepath: Path) -> List[CrystalStructure]:
        """Basic CIF parser without pymatgen."""
        # This is a simplified parser for basic CIF files
        # In production, pymatgen should be used for robust parsing

        with open(filepath) as f:
            content = f.read()

        # Extract basic information (simplified implementation)
        lines = content.split("\n")

        # Find lattice parameters
        lattice_params = self._extract_lattice_parameters(lines)

        # Find atomic positions
        atomic_data = self._extract_atomic_positions(lines)

        if not lattice_params or not atomic_data:
            raise ValueError("Could not parse essential CIF data")

        # Create lattice vectors from parameters
        lattice_vectors = self._create_lattice_vectors(*lattice_params)

        return [
            CrystalStructure(
                lattice_vectors=lattice_vectors,
                lattice_parameters=np.array(lattice_params),
                atomic_numbers=atomic_data["atomic_numbers"],
                atomic_symbols=atomic_data["symbols"],
                fractional_coords=atomic_data["frac_coords"],
                cartesian_coords=self._frac_to_cart(
                    atomic_data["frac_coords"], lattice_vectors
                ),
            )
        ]

    def _extract_lattice_parameters(self, lines: List[str]) -> Optional[List[float]]:
        """Extract lattice parameters from CIF lines."""
        params = {}
        param_keys = [
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        ]

        for line in lines:
            line = line.strip()
            for key in param_keys:
                if line.startswith(key):
                    try:
                        value = float(line.split()[1].split("(")[0])
                        params[key] = value
                    except (IndexError, ValueError):
                        continue

        if len(params) == 6:
            return [params[key] for key in param_keys]
        return None

    def _extract_atomic_positions(self, lines: List[str]) -> Optional[Dict]:
        """Extract atomic positions from CIF lines."""
        # Simplified extraction - in practice, this would be more robust
        in_atom_site = False
        atomic_data = {"symbols": [], "frac_coords": [], "atomic_numbers": []}

        element_to_z = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
        }

        for line in lines:
            line = line.strip()
            if line.startswith("_atom_site_"):
                in_atom_site = True
                continue

            if in_atom_site and line and not line.startswith("_"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        symbol = parts[0].split("_")[0]  # Remove site labels
                        symbol = "".join(c for c in symbol if c.isalpha())

                        x = float(parts[1].split("(")[0])
                        y = float(parts[2].split("(")[0])
                        z = float(parts[3].split("(")[0])

                        atomic_data["symbols"].append(symbol)
                        atomic_data["frac_coords"].append([x, y, z])
                        atomic_data["atomic_numbers"].append(
                            element_to_z.get(symbol, 6)
                        )
                    except (ValueError, IndexError):
                        continue

        if atomic_data["symbols"]:
            atomic_data["frac_coords"] = np.array(atomic_data["frac_coords"])
            atomic_data["atomic_numbers"] = np.array(atomic_data["atomic_numbers"])
            return atomic_data

        return None

    def _create_lattice_vectors(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> np.ndarray:
        """Create lattice vectors from parameters."""
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Create lattice vectors
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_gamma = np.sin(gamma_rad)

        # Volume calculation
        volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                + 2 * cos_alpha * cos_beta * cos_gamma
                - cos_alpha**2
                - cos_beta**2
                - cos_gamma**2
            )
        )

        # Lattice vectors
        v1 = np.array([a, 0, 0])
        v2 = np.array([b * cos_gamma, b * sin_gamma, 0])
        v3 = np.array(
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                volume / (a * b * sin_gamma),
            ]
        )

        return np.array([v1, v2, v3])

    def _frac_to_cart(
        self, frac_coords: np.ndarray, lattice_vectors: np.ndarray
    ) -> np.ndarray:
        """Convert fractional to cartesian coordinates."""
        return np.dot(frac_coords, lattice_vectors)


class POSCARParser:
    """Parser for VASP POSCAR/CONTCAR files."""

    def __init__(self, use_pymatgen: bool = True):
        """Initialize POSCAR parser.

        Args:
            use_pymatgen: Whether to use pymatgen for parsing (recommended)
        """
        self.use_pymatgen = use_pymatgen and PYMATGEN_AVAILABLE

    def parse_file(self, filepath: Union[str, Path]) -> CrystalStructure:
        """Parse POSCAR file and return crystal structure.

        Args:
            filepath: Path to POSCAR file

        Returns:
            CrystalStructure object
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"POSCAR file not found: {filepath}")

        if self.use_pymatgen:
            return self._parse_with_pymatgen(filepath)
        else:
            return self._parse_basic(filepath)

    def _parse_with_pymatgen(self, filepath: Path) -> CrystalStructure:
        """Parse POSCAR using pymatgen."""
        try:
            poscar = Poscar.from_file(str(filepath))
            structure = poscar.structure

            return CrystalStructure(
                lattice_vectors=structure.lattice.matrix,
                lattice_parameters=np.array(
                    [
                        structure.lattice.a,
                        structure.lattice.b,
                        structure.lattice.c,
                        structure.lattice.alpha,
                        structure.lattice.beta,
                        structure.lattice.gamma,
                    ]
                ),
                atomic_numbers=np.array([site.specie.Z for site in structure]),
                atomic_symbols=[str(site.specie) for site in structure],
                fractional_coords=np.array([site.frac_coords for site in structure]),
                cartesian_coords=np.array([site.coords for site in structure]),
                formula=structure.formula,
                volume=structure.volume,
                density=structure.density,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse POSCAR file with pymatgen: {e}")

    def _parse_basic(self, filepath: Path) -> CrystalStructure:
        """Basic POSCAR parser without pymatgen."""
        with open(filepath) as f:
            lines = [line.strip() for line in f.readlines()]

        # Parse POSCAR format
        comment = lines[0]
        scaling_factor = float(lines[1])

        # Lattice vectors
        lattice_vectors = []
        for i in range(2, 5):
            vector = [float(x) for x in lines[i].split()]
            lattice_vectors.append(vector)
        lattice_vectors = np.array(lattice_vectors) * scaling_factor

        # Element symbols and counts
        if lines[5].split()[0].isalpha():
            # VASP 5.x format with element symbols
            elements = lines[5].split()
            counts = [int(x) for x in lines[6].split()]
            coord_start = 7
        else:
            # VASP 4.x format without element symbols
            counts = [int(x) for x in lines[5].split()]
            elements = [f"X{i}" for i in range(len(counts))]  # Generic symbols
            coord_start = 6

        # Coordinate type
        coord_type = lines[coord_start].lower()
        is_direct = coord_type.startswith("d")

        # Atomic positions
        atomic_symbols = []
        atomic_numbers = []
        coords = []

        element_to_z = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
        }

        coord_line = coord_start + 1
        for element, count in zip(elements, counts):
            for _ in range(count):
                coord_data = [float(x) for x in lines[coord_line].split()[:3]]
                coords.append(coord_data)
                atomic_symbols.append(element)
                atomic_numbers.append(element_to_z.get(element, 6))
                coord_line += 1

        coords = np.array(coords)

        # Convert coordinates if necessary
        if is_direct:
            fractional_coords = coords
            cartesian_coords = np.dot(coords, lattice_vectors)
        else:
            cartesian_coords = coords * scaling_factor
            # Convert to fractional
            fractional_coords = np.dot(cartesian_coords, np.linalg.inv(lattice_vectors))

        return CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=None,  # Will be computed in __post_init__
            atomic_numbers=np.array(atomic_numbers),
            atomic_symbols=atomic_symbols,
            fractional_coords=fractional_coords,
            cartesian_coords=cartesian_coords,
        )
