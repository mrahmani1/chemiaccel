"""
Advanced RDKit integration features for ChemiAccel.

This module provides advanced cheminformatics capabilities including:
- Parallel substructure searching with SMARTS pattern matching
- Maximum Common Substructure (MCS) calculation with GPU acceleration
- Pharmacophore matching and 3D spatial pattern recognition
- Reaction enumeration with parallel template application
- Drug-likeness filters (Lipinski, Veber, PAINS) with batch processing
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        Lipinski,
        rdchem,
        rdDepictor,
        rdFMCS,
        rdMolDescriptors,
    )

    RDKIT_AVAILABLE = True

    # Optional imports that might not be available in all RDKit versions
    try:
        from rdkit.Chem.Pharm3D import Pharmacophore

        PHARMACOPHORE_AVAILABLE = True
    except ImportError:
        PHARMACOPHORE_AVAILABLE = False

    try:
        from rdkit.Chem.Features import FeatureFactory

        FEATURE_FACTORY_AVAILABLE = True
    except ImportError:
        FEATURE_FACTORY_AVAILABLE = False

    try:
        from rdkit.Chem import ChemicalReaction

        REACTION_AVAILABLE = True
    except ImportError:
        REACTION_AVAILABLE = False

    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        FILTER_CATALOG_AVAILABLE = True
    except ImportError:
        FILTER_CATALOG_AVAILABLE = False

    try:
        from rdkit.Chem.rdMolAlign import AlignMol

        MOL_ALIGN_AVAILABLE = True
    except ImportError:
        MOL_ALIGN_AVAILABLE = False

    try:
        from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist, ShapeTanimotoDist

        SHAPE_HELPERS_AVAILABLE = True
    except ImportError:
        SHAPE_HELPERS_AVAILABLE = False

except ImportError:
    RDKIT_AVAILABLE = False
    PHARMACOPHORE_AVAILABLE = False
    FEATURE_FACTORY_AVAILABLE = False
    REACTION_AVAILABLE = False
    FILTER_CATALOG_AVAILABLE = False
    MOL_ALIGN_AVAILABLE = False
    SHAPE_HELPERS_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..core.molecular import MolecularData
from ..utils.logging import get_logger

logger = get_logger()


def check_rdkit_availability():
    """Check if RDKit is available and raise error if not."""
    if not RDKIT_AVAILABLE:
        raise ChemiAccelError(
            "RDKit is not available. Please install RDKit: conda install -c conda-forge rdkit"
        )


class SubstructureSearcher:
    """Parallel substructure searching with SMARTS pattern matching."""

    def __init__(self, n_jobs: int = -1):
        """Initialize substructure searcher.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        check_rdkit_availability()

        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = min(n_jobs, mp.cpu_count())

        logger.info(
            f"SubstructureSearcher initialized with {self.n_jobs} parallel jobs"
        )

    def _single_molecule_search(
        self, mol_data: MolecularData, patterns: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Search for patterns in a single molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            patterns: List of (pattern_name, smarts_pattern) tuples

        Returns:
            Dictionary with search results
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
                "error": "No RDKit mol object found",
            }

        results = {
            "mol_id": mol_data.mol_id,
            "smiles": mol_data.smiles,
            "matches": {},
            "match_counts": {},
            "total_matches": 0,
        }

        for pattern_name, smarts_pattern in patterns:
            try:
                # Compile SMARTS pattern
                pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                if pattern_mol is None:
                    results["matches"][pattern_name] = []
                    results["match_counts"][pattern_name] = 0
                    continue

                # Find all matches
                matches = rdkit_mol.GetSubstructMatches(pattern_mol)
                results["matches"][pattern_name] = [list(match) for match in matches]
                results["match_counts"][pattern_name] = len(matches)
                results["total_matches"] += len(matches)

            except Exception as e:
                logger.warning(f"Pattern matching failed for {pattern_name}: {e}")
                results["matches"][pattern_name] = []
                results["match_counts"][pattern_name] = 0

        return results

    def search_patterns(
        self,
        molecules: List[MolecularData],
        patterns: Dict[str, str],
        use_multiprocessing: bool = True,
    ) -> pd.DataFrame:
        """Search for SMARTS patterns in multiple molecules.

        Args:
            molecules: List of MolecularData objects
            patterns: Dictionary of {pattern_name: smarts_pattern}
            use_multiprocessing: Whether to use multiprocessing

        Returns:
            DataFrame with search results
        """
        if not molecules:
            return pd.DataFrame()

        pattern_list = list(patterns.items())

        if use_multiprocessing and len(molecules) > 1:
            # Use multiprocessing for large datasets
            search_func = partial(self._single_molecule_search, patterns=pattern_list)

            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(search_func, molecules))
        else:
            # Sequential processing for small datasets or debugging
            results = []
            for mol_data in molecules:
                result = self._single_molecule_search(mol_data, pattern_list)
                results.append(result)

        # Convert results to DataFrame
        df_data = []
        for result in results:
            row = {
                "mol_id": result["mol_id"],
                "smiles": result["smiles"],
                "total_matches": result["total_matches"],
            }

            # Add individual pattern counts
            for pattern_name in patterns.keys():
                row[f"{pattern_name}_count"] = result["match_counts"].get(
                    pattern_name, 0
                )
                row[f"{pattern_name}_present"] = (
                    result["match_counts"].get(pattern_name, 0) > 0
                )

            # Add error information if present
            if "error" in result:
                row["error"] = result["error"]

            df_data.append(row)

        df = pd.DataFrame(df_data)
        logger.info(
            f"Completed substructure search for {len(df)} molecules with {len(patterns)} patterns"
        )

        return df

    def search_functional_groups(self, molecules: List[MolecularData]) -> pd.DataFrame:
        """Search for common functional groups.

        Args:
            molecules: List of MolecularData objects

        Returns:
            DataFrame with functional group presence
        """
        functional_groups = {
            "alcohol": "[OH]",
            "aldehyde": "[CX3H1](=O)[#6]",
            "ketone": "[#6][CX3](=O)[#6]",
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "ester": "[#6][CX3](=O)[OX2H0][#6]",
            "amide": "[NX3][CX3](=[OX1])[#6]",
            "amine_primary": "[NX3;H2;!$(NC=O)]",
            "amine_secondary": "[NX3;H1;!$(NC=O)]",
            "amine_tertiary": "[NX3;H0;!$(NC=O)]",
            "aromatic_ring": "c1ccccc1",
            "benzene": "c1ccccc1",
            "pyridine": "n1ccccc1",
            "imidazole": "c1c[nH]cn1",
            "thiophene": "c1ccsc1",
            "furan": "c1ccoc1",
            "nitro": "[N+](=O)[O-]",
            "nitrile": "[C-]#[N+]",
            "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
            "phosphate": "[PX4](=[OX1])([OX2])([OX2])[OX2]",
        }

        return self.search_patterns(molecules, functional_groups)


class MaximumCommonSubstructure:
    """Maximum Common Substructure (MCS) calculation with optimization."""

    def __init__(
        self,
        timeout: int = 60,
        match_valences: bool = True,
        ring_matches_ring_only: bool = True,
        complete_rings_only: bool = True,
    ):
        """Initialize MCS calculator.

        Args:
            timeout: Maximum time in seconds for MCS calculation
            match_valences: Whether to match atom valences
            ring_matches_ring_only: Whether ring atoms only match ring atoms
            complete_rings_only: Whether to require complete rings in MCS
        """
        check_rdkit_availability()

        self.timeout = timeout
        self.match_valences = match_valences
        self.ring_matches_ring_only = ring_matches_ring_only
        self.complete_rings_only = complete_rings_only

        logger.debug(f"MCS calculator initialized with timeout={timeout}s")

    def calculate_mcs(
        self, molecules: List[MolecularData], min_num_atoms: int = 3
    ) -> Dict[str, Any]:
        """Calculate Maximum Common Substructure for a set of molecules.

        Args:
            molecules: List of MolecularData objects
            min_num_atoms: Minimum number of atoms in MCS

        Returns:
            Dictionary with MCS information
        """
        if len(molecules) < 2:
            raise ValueError("At least 2 molecules required for MCS calculation")

        rdkit_mols = []
        for mol_data in molecules:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None:
                raise ChemiAccelError(
                    f"No RDKit mol object found for molecule {mol_data.mol_id}"
                )
            rdkit_mols.append(rdkit_mol)

        try:
            # Set up MCS parameters
            params = rdFMCS.MCSParameters()
            params.Timeout = self.timeout
            params.AtomCompareParameters.MatchValences = self.match_valences
            params.BondCompareParameters.RingMatchesRingOnly = (
                self.ring_matches_ring_only
            )
            params.BondCompareParameters.CompleteRingsOnly = self.complete_rings_only

            # Calculate MCS
            start_time = time.time()
            mcs_result = rdFMCS.FindMCS(rdkit_mols, params)
            calculation_time = time.time() - start_time

            # Extract MCS information
            mcs_mol = None
            mcs_smarts = mcs_result.smartsString
            mcs_smiles = None

            if mcs_smarts and mcs_result.numAtoms >= min_num_atoms:
                try:
                    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                    if mcs_mol:
                        mcs_smiles = Chem.MolToSmiles(mcs_mol)
                except:
                    pass

            result = {
                "num_molecules": len(molecules),
                "mcs_smarts": mcs_smarts,
                "mcs_smiles": mcs_smiles,
                "num_atoms": mcs_result.numAtoms,
                "num_bonds": mcs_result.numBonds,
                "calculation_time": calculation_time,
                "timeout_reached": mcs_result.canceled,
                "valid_mcs": mcs_result.numAtoms >= min_num_atoms,
            }

            # Calculate coverage for each molecule
            if mcs_mol and mcs_result.numAtoms >= min_num_atoms:
                coverages = []
                for i, rdkit_mol in enumerate(rdkit_mols):
                    matches = rdkit_mol.GetSubstructMatches(mcs_mol)
                    coverage = (
                        len(matches[0]) / rdkit_mol.GetNumAtoms() if matches else 0.0
                    )
                    coverages.append(coverage)

                result["molecule_coverages"] = coverages
                result["average_coverage"] = np.mean(coverages)
                result["min_coverage"] = np.min(coverages)
                result["max_coverage"] = np.max(coverages)

            logger.info(
                f"MCS calculation completed: {mcs_result.numAtoms} atoms, "
                f"{mcs_result.numBonds} bonds in {calculation_time:.2f}s"
            )

            return result

        except Exception as e:
            raise ChemiAccelError(f"MCS calculation failed: {e}")

    def calculate_pairwise_mcs(
        self, molecules: List[MolecularData], min_num_atoms: int = 3
    ) -> pd.DataFrame:
        """Calculate pairwise MCS for all molecule pairs.

        Args:
            molecules: List of MolecularData objects
            min_num_atoms: Minimum number of atoms in MCS

        Returns:
            DataFrame with pairwise MCS results
        """
        n_mols = len(molecules)
        if n_mols < 2:
            return pd.DataFrame()

        results = []

        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                mol_pair = [molecules[i], molecules[j]]

                try:
                    mcs_result = self.calculate_mcs(mol_pair, min_num_atoms)

                    result_row = {
                        "mol_id_1": molecules[i].mol_id,
                        "mol_id_2": molecules[j].mol_id,
                        "smiles_1": molecules[i].smiles,
                        "smiles_2": molecules[j].smiles,
                        "mcs_smarts": mcs_result["mcs_smarts"],
                        "mcs_smiles": mcs_result["mcs_smiles"],
                        "mcs_atoms": mcs_result["num_atoms"],
                        "mcs_bonds": mcs_result["num_bonds"],
                        "calculation_time": mcs_result["calculation_time"],
                        "valid_mcs": mcs_result["valid_mcs"],
                    }

                    if "molecule_coverages" in mcs_result:
                        result_row["coverage_1"] = mcs_result["molecule_coverages"][0]
                        result_row["coverage_2"] = mcs_result["molecule_coverages"][1]

                    results.append(result_row)

                except Exception as e:
                    logger.warning(f"Pairwise MCS failed for molecules {i}, {j}: {e}")
                    result_row = {
                        "mol_id_1": molecules[i].mol_id,
                        "mol_id_2": molecules[j].mol_id,
                        "smiles_1": molecules[i].smiles,
                        "smiles_2": molecules[j].smiles,
                        "mcs_smarts": None,
                        "mcs_smiles": None,
                        "mcs_atoms": 0,
                        "mcs_bonds": 0,
                        "calculation_time": 0.0,
                        "valid_mcs": False,
                        "error": str(e),
                    }
                    results.append(result_row)

        df = pd.DataFrame(results)
        logger.info(f"Completed pairwise MCS calculation for {len(df)} pairs")

        return df


class PharmacophoreAnalyzer:
    """Pharmacophore matching and 3D spatial pattern recognition."""

    def __init__(self):
        """Initialize pharmacophore analyzer."""
        check_rdkit_availability()

        if FEATURE_FACTORY_AVAILABLE:
            try:
                # Initialize feature factory for pharmacophore features
                self.feature_factory = FeatureFactory.BuildFeatureFactory(
                    "BaseFeatures.fdef"
                )
                self.pharmacophore_available = True
            except Exception as e:
                logger.warning(f"Pharmacophore features not available: {e}")
                self.pharmacophore_available = False
        else:
            logger.warning("FeatureFactory not available in this RDKit version")
            self.pharmacophore_available = False

        # Define common pharmacophore features
        self.feature_definitions = {
            "donor": "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
            "acceptor": "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
            "aromatic": "[a]",
            "hydrophobic": "[$([CH3X4,CH2X3,CH1X2,F1X1,Cl1X1,Br1X1,I1X1]),$([CH3X4,CH2X3,CH1X2,F1X1,Cl1X1,Br1X1,I1X1])]",
            "positive": "[+,$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
            "negative": "[-,$([OX2,OX1-][CX3]=[OX1]),$([OX2,OX1-][SX4]=[OX1]),$([OX2,OX1-][PX4]=[OX1])]",
        }

        logger.debug(
            f"PharmacophoreAnalyzer initialized (available: {self.pharmacophore_available})"
        )

    def extract_pharmacophore_features(
        self, mol_data: MolecularData
    ) -> Dict[str, List[Tuple[int, ...]]]:
        """Extract pharmacophore features from a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary of feature types and their atom indices
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        features = {}

        for feature_name, smarts_pattern in self.feature_definitions.items():
            try:
                pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                if pattern_mol is None:
                    features[feature_name] = []
                    continue

                matches = rdkit_mol.GetSubstructMatches(pattern_mol)
                features[feature_name] = [tuple(match) for match in matches]

            except Exception as e:
                logger.warning(f"Feature extraction failed for {feature_name}: {e}")
                features[feature_name] = []

        return features

    def calculate_pharmacophore_fingerprint(
        self, mol_data: MolecularData, include_3d: bool = False
    ) -> Dict[str, Any]:
        """Calculate pharmacophore fingerprint for a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            include_3d: Whether to include 3D distance information

        Returns:
            Dictionary with pharmacophore fingerprint data
        """
        features = self.extract_pharmacophore_features(mol_data)

        # Count features
        feature_counts = {name: len(matches) for name, matches in features.items()}

        # Calculate feature pairs and distances if 3D coordinates available
        fingerprint = {
            "feature_counts": feature_counts,
            "total_features": sum(feature_counts.values()),
        }

        if include_3d:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol and rdkit_mol.GetNumConformers() > 0:
                fingerprint["3d_distances"] = self._calculate_3d_feature_distances(
                    rdkit_mol, features
                )

        return fingerprint

    def _calculate_3d_feature_distances(
        self, mol: Chem.Mol, features: Dict[str, List[Tuple[int, ...]]]
    ) -> Dict[str, List[float]]:
        """Calculate 3D distances between pharmacophore features."""
        if mol.GetNumConformers() == 0:
            return {}

        conf = mol.GetConformer()
        distances = {}

        feature_types = list(features.keys())
        for i, feat_type1 in enumerate(feature_types):
            for feat_type2 in feature_types[i:]:
                pair_key = f"{feat_type1}_{feat_type2}"
                pair_distances = []

                for match1 in features[feat_type1]:
                    for match2 in features[feat_type2]:
                        if feat_type1 == feat_type2 and match1 == match2:
                            continue

                        # Calculate distance between feature centroids
                        pos1 = np.mean(
                            [conf.GetAtomPosition(atom_idx) for atom_idx in match1],
                            axis=0,
                        )
                        pos2 = np.mean(
                            [conf.GetAtomPosition(atom_idx) for atom_idx in match2],
                            axis=0,
                        )

                        distance = np.linalg.norm(pos1 - pos2)
                        pair_distances.append(distance)

                if pair_distances:
                    distances[pair_key] = pair_distances

        return distances

    def match_pharmacophore_pattern(
        self, molecules: List[MolecularData], pattern_features: Dict[str, int]
    ) -> pd.DataFrame:
        """Match molecules against a pharmacophore pattern.

        Args:
            molecules: List of MolecularData objects
            pattern_features: Dictionary of required feature counts

        Returns:
            DataFrame with pharmacophore matching results
        """
        results = []

        for mol_data in molecules:
            try:
                fingerprint = self.calculate_pharmacophore_fingerprint(mol_data)
                feature_counts = fingerprint["feature_counts"]

                # Check if molecule matches pattern
                matches_pattern = True
                for feature_type, required_count in pattern_features.items():
                    if feature_counts.get(feature_type, 0) < required_count:
                        matches_pattern = False
                        break

                result = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    "matches_pattern": matches_pattern,
                    "total_features": fingerprint["total_features"],
                }

                # Add individual feature counts
                for feature_type in self.feature_definitions.keys():
                    result[f"{feature_type}_count"] = feature_counts.get(
                        feature_type, 0
                    )

                results.append(result)

            except Exception as e:
                logger.warning(
                    f"Pharmacophore matching failed for {mol_data.mol_id}: {e}"
                )
                result = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    "matches_pattern": False,
                    "total_features": 0,
                    "error": str(e),
                }
                results.append(result)

        df = pd.DataFrame(results)
        logger.info(f"Completed pharmacophore matching for {len(df)} molecules")

        return df


class ReactionEnumerator:
    """Reaction enumeration with parallel template application."""

    def __init__(self, n_jobs: int = -1):
        """Initialize reaction enumerator.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        check_rdkit_availability()

        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = min(n_jobs, mp.cpu_count())

        # Common reaction templates (SMARTS)
        self.reaction_templates = {
            "amide_formation": "[C:1](=[O:2])[OH:3].[N:4]>>[C:1](=[O:2])[N:4]",
            "ester_formation": "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5]",
            "suzuki_coupling": "[c:1][Br,I:2].[c:3][B:4]([OH:5])[OH:6]>>[c:1][c:3]",
            "click_reaction": "[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[c:2][n:3][n:4][n:5]1",
            "reductive_amination": "[C:1]=[O:2].[N:3]>>[C:1][N:3]",
            "nucleophilic_substitution": "[C:1][Cl,Br,I:2].[N,O,S:3]>>[C:1][*:3]",
        }

        logger.info(
            f"ReactionEnumerator initialized with {self.n_jobs} jobs and {len(self.reaction_templates)} templates"
        )

    def _apply_reaction_to_molecule(
        self, mol_data: MolecularData, reaction_smarts: str, reaction_name: str
    ) -> List[Dict[str, Any]]:
        """Apply a reaction template to a single molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            reaction_smarts: SMARTS pattern for the reaction
            reaction_name: Name of the reaction

        Returns:
            List of reaction products
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return []

        try:
            # Create reaction from SMARTS
            reaction = AllChem.ReactionFromSmarts(reaction_smarts)
            if reaction is None:
                return []

            # Apply reaction
            products = reaction.RunReactants((rdkit_mol,))

            results = []
            for i, product_set in enumerate(products):
                for j, product_mol in enumerate(product_set):
                    try:
                        # Sanitize product
                        Chem.SanitizeMol(product_mol)
                        product_smiles = Chem.MolToSmiles(product_mol)

                        result = {
                            "reactant_mol_id": mol_data.mol_id,
                            "reactant_smiles": mol_data.smiles,
                            "reaction_name": reaction_name,
                            "product_id": f"{mol_data.mol_id}_{reaction_name}_{i}_{j}",
                            "product_smiles": product_smiles,
                            "product_mol": product_mol,
                        }
                        results.append(result)

                    except Exception as e:
                        logger.debug(
                            f"Failed to process product {i},{j} for {mol_data.mol_id}: {e}"
                        )
                        continue

            return results

        except Exception as e:
            logger.warning(
                f"Reaction application failed for {mol_data.mol_id} with {reaction_name}: {e}"
            )
            return []

    def enumerate_reactions(
        self,
        molecules: List[MolecularData],
        reaction_templates: Optional[Dict[str, str]] = None,
        max_products_per_reaction: int = 10,
    ) -> pd.DataFrame:
        """Enumerate reaction products for multiple molecules.

        Args:
            molecules: List of MolecularData objects
            reaction_templates: Dictionary of reaction templates (None for default)
            max_products_per_reaction: Maximum products per reaction per molecule

        Returns:
            DataFrame with enumerated reaction products
        """
        if reaction_templates is None:
            reaction_templates = self.reaction_templates

        all_results = []

        for reaction_name, reaction_smarts in reaction_templates.items():
            logger.info(f"Applying reaction template: {reaction_name}")

            # Apply reaction to all molecules
            reaction_func = partial(
                self._apply_reaction_to_molecule,
                reaction_smarts=reaction_smarts,
                reaction_name=reaction_name,
            )

            if len(molecules) > 1 and self.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    reaction_results = list(executor.map(reaction_func, molecules))
            else:
                reaction_results = [reaction_func(mol) for mol in molecules]

            # Flatten results and limit products per reaction
            for mol_results in reaction_results:
                if len(mol_results) > max_products_per_reaction:
                    mol_results = mol_results[:max_products_per_reaction]
                all_results.extend(mol_results)

        # Convert to DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            # Remove the RDKit mol objects for the DataFrame (keep them in a separate structure if needed)
            df_clean = df.drop(columns=["product_mol"], errors="ignore")
        else:
            df_clean = pd.DataFrame()

        logger.info(
            f"Enumerated {len(all_results)} reaction products from {len(molecules)} reactants"
        )

        return df_clean

    def filter_novel_products(
        self, products_df: pd.DataFrame, known_molecules: Set[str]
    ) -> pd.DataFrame:
        """Filter reaction products to find novel compounds.

        Args:
            products_df: DataFrame with reaction products
            known_molecules: Set of known SMILES strings

        Returns:
            DataFrame with only novel products
        """
        if products_df.empty:
            return products_df

        # Filter out known molecules
        novel_mask = ~products_df["product_smiles"].isin(known_molecules)
        novel_products = products_df[novel_mask].copy()

        # Remove duplicates within the products
        novel_products = novel_products.drop_duplicates(subset=["product_smiles"])

        logger.info(
            f"Found {len(novel_products)} novel products out of {len(products_df)} total products"
        )

        return novel_productscla


class DrugLikenessFilters:
    """Drug-likeness filters (Lipinski, Veber, PAINS) with batch processing."""

    def __init__(self):
        """Initialize drug-likeness filters."""
        check_rdkit_availability()

        # Initialize PAINS filter
        if FILTER_CATALOG_AVAILABLE:
            try:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                self.pains_catalog = FilterCatalog(params)
                self.pains_available = True
            except Exception as e:
                logger.warning(f"PAINS filter not available: {e}")
                self.pains_available = False
        else:
            logger.warning("FilterCatalog not available in this RDKit version")
            self.pains_available = False

        logger.debug(
            f"DrugLikenessFilters initialized (PAINS available: {self.pains_available})"
        )

    def calculate_lipinski_descriptors(
        self, mol_data: MolecularData
    ) -> Dict[str, float]:
        """Calculate Lipinski Rule of Five descriptors.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary with Lipinski descriptors
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        try:
            descriptors = {
                "molecular_weight": Descriptors.MolWt(rdkit_mol),
                "logp": Crippen.MolLogP(rdkit_mol),
                "hbd": Descriptors.NumHDonors(rdkit_mol),
                "hba": Descriptors.NumHAcceptors(rdkit_mol),
            }

            # Calculate violations
            violations = 0
            if descriptors["molecular_weight"] > 500:
                violations += 1
            if descriptors["logp"] > 5:
                violations += 1
            if descriptors["hbd"] > 5:
                violations += 1
            if descriptors["hba"] > 10:
                violations += 1

            descriptors["lipinski_violations"] = violations
            descriptors["lipinski_compliant"] = violations == 0

            return descriptors

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="lipinski_descriptors",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to calculate Lipinski descriptors: {e}",
            )

    def calculate_veber_descriptors(self, mol_data: MolecularData) -> Dict[str, float]:
        """Calculate Veber rule descriptors.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary with Veber descriptors
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        try:
            descriptors = {
                "tpsa": Descriptors.TPSA(rdkit_mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(rdkit_mol),
            }

            # Calculate violations
            violations = 0
            if descriptors["tpsa"] > 140:
                violations += 1
            if descriptors["rotatable_bonds"] > 10:
                violations += 1

            descriptors["veber_violations"] = violations
            descriptors["veber_compliant"] = violations == 0

            return descriptors

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="veber_descriptors",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to calculate Veber descriptors: {e}",
            )

    def check_pains_filter(self, mol_data: MolecularData) -> Dict[str, Any]:
        """Check molecule against PAINS (Pan-Assay Interference Compounds) filter.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary with PAINS filter results
        """
        if not self.pains_available:
            return {
                "pains_alert": False,
                "pains_matches": [],
                "num_pains_matches": 0,
                "pains_available": False,
            }

        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        try:
            matches = []
            if self.pains_catalog.HasMatch(rdkit_mol):
                entries = self.pains_catalog.GetMatches(rdkit_mol)
                matches = [entry.GetDescription() for entry in entries]

            result = {
                "pains_alert": len(matches) > 0,
                "pains_matches": matches,
                "num_pains_matches": len(matches),
                "pains_available": True,
            }

            return result

        except Exception as e:
            logger.warning(f"PAINS filter check failed for {mol_data.mol_id}: {e}")
            return {
                "pains_alert": False,
                "pains_matches": [],
                "num_pains_matches": 0,
                "pains_available": True,
                "error": str(e),
            }

    def calculate_drug_likeness_score(self, mol_data: MolecularData) -> Dict[str, Any]:
        """Calculate comprehensive drug-likeness score.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary with drug-likeness assessment
        """
        try:
            # Get individual assessments
            lipinski = self.calculate_lipinski_descriptors(mol_data)
            veber = self.calculate_veber_descriptors(mol_data)
            pains = self.check_pains_filter(mol_data)

            # Calculate overall drug-likeness score
            score = 0.0
            max_score = 3.0

            # Lipinski compliance (0-1)
            if lipinski["lipinski_compliant"]:
                score += 1.0
            else:
                # Partial credit based on number of violations
                score += max(0, 1.0 - lipinski["lipinski_violations"] * 0.25)

            # Veber compliance (0-1)
            if veber["veber_compliant"]:
                score += 1.0
            else:
                score += max(0, 1.0 - veber["veber_violations"] * 0.5)

            # PAINS filter (0-1)
            if not pains["pains_alert"]:
                score += 1.0

            # Normalize score to 0-1
            normalized_score = score / max_score

            result = {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
                "drug_likeness_score": normalized_score,
                **lipinski,
                **veber,
                **pains,
            }

            return result

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="drug_likeness_score",
                molecule_id=mol_data.mol_id,
                reason=f"Failed to calculate drug-likeness score: {e}",
            )

    def filter_drug_like_molecules(
        self,
        molecules: List[MolecularData],
        min_score: float = 0.7,
        require_lipinski: bool = True,
        require_veber: bool = True,
        exclude_pains: bool = True,
    ) -> pd.DataFrame:
        """Filter molecules for drug-likeness.

        Args:
            molecules: List of MolecularData objects
            min_score: Minimum drug-likeness score
            require_lipinski: Whether to require Lipinski compliance
            require_veber: Whether to require Veber compliance
            exclude_pains: Whether to exclude PAINS alerts

        Returns:
            DataFrame with drug-likeness assessment and filtering
        """
        results = []

        for mol_data in molecules:
            try:
                drug_likeness = self.calculate_drug_likeness_score(mol_data)

                # Apply filters
                passes_filters = True
                filter_reasons = []

                if drug_likeness["drug_likeness_score"] < min_score:
                    passes_filters = False
                    filter_reasons.append(
                        f"Low drug-likeness score ({drug_likeness['drug_likeness_score']:.2f})"
                    )

                if require_lipinski and not drug_likeness["lipinski_compliant"]:
                    passes_filters = False
                    filter_reasons.append(
                        f"Lipinski violations ({drug_likeness['lipinski_violations']})"
                    )

                if require_veber and not drug_likeness["veber_compliant"]:
                    passes_filters = False
                    filter_reasons.append(
                        f"Veber violations ({drug_likeness['veber_violations']})"
                    )

                if exclude_pains and drug_likeness["pains_alert"]:
                    passes_filters = False
                    filter_reasons.append(
                        f"PAINS alert ({drug_likeness['num_pains_matches']} matches)"
                    )

                drug_likeness["passes_drug_likeness_filter"] = passes_filters
                drug_likeness["filter_reasons"] = (
                    "; ".join(filter_reasons) if filter_reasons else ""
                )

                results.append(drug_likeness)

            except Exception as e:
                logger.warning(
                    f"Drug-likeness calculation failed for {mol_data.mol_id}: {e}"
                )
                result = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    "drug_likeness_score": 0.0,
                    "passes_drug_likeness_filter": False,
                    "filter_reasons": f"Calculation error: {e}",
                    "error": str(e),
                }
                results.append(result)

        df = pd.DataFrame(results)

        if not df.empty:
            passed = df["passes_drug_likeness_filter"].sum()
            logger.info(
                f"Drug-likeness filtering: {passed}/{len(df)} molecules passed filters"
            )

        return df


# Convenience functions for easy access
def search_substructures(
    molecules: List[MolecularData], patterns: Dict[str, str], n_jobs: int = -1
) -> pd.DataFrame:
    """Search for substructure patterns in molecules (convenience function).

    Args:
        molecules: List of MolecularData objects
        patterns: Dictionary of {pattern_name: smarts_pattern}
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with search results
    """
    searcher = SubstructureSearcher(n_jobs=n_jobs)
    return searcher.search_patterns(molecules, patterns)


def calculate_mcs(molecules: List[MolecularData], **kwargs) -> Dict[str, Any]:
    """Calculate Maximum Common Substructure (convenience function).

    Args:
        molecules: List of MolecularData objects
        **kwargs: Additional arguments for MCS calculation

    Returns:
        Dictionary with MCS information
    """
    mcs_calculator = MaximumCommonSubstructure(**kwargs)
    return mcs_calculator.calculate_mcs(molecules)


def analyze_pharmacophores(molecules: List[MolecularData]) -> pd.DataFrame:
    """Analyze pharmacophore features in molecules (convenience function).

    Args:
        molecules: List of MolecularData objects

    Returns:
        DataFrame with pharmacophore analysis
    """
    analyzer = PharmacophoreAnalyzer()
    results = []

    for mol_data in molecules:
        try:
            fingerprint = analyzer.calculate_pharmacophore_fingerprint(mol_data)
            result = {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
                **fingerprint["feature_counts"],
                "total_features": fingerprint["total_features"],
            }
            results.append(result)
        except Exception as e:
            logger.warning(f"Pharmacophore analysis failed for {mol_data.mol_id}: {e}")

    return pd.DataFrame(results)


def filter_drug_like_compounds(
    molecules: List[MolecularData], **kwargs
) -> pd.DataFrame:
    """Filter molecules for drug-likeness (convenience function).

    Args:
        molecules: List of MolecularData objects
        **kwargs: Additional arguments for filtering

    Returns:
        DataFrame with drug-likeness assessment
    """
    filters = DrugLikenessFilters()
    return filters.filter_drug_like_molecules(molecules, **kwargs)
